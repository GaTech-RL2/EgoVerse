"""A bounded Python-subset interpreter for searched harnesses.

No Python exec/eval, imports, attributes, host objects or ambient capabilities
are available. Candidates see JSON copies of public inputs only. Evaluator
results and source files never enter this interpreter. This deliberately small
language supports retrieval, evidence memory, prompts and decision scheduling.
"""

import ast
import copy
import json
import operator
import re


class ContractError(ValueError):
    pass


class Harness:
    def __init__(self, source):
        if not isinstance(source, str) or len(source.encode()) > 20000:
            raise ContractError("Candidate source exceeds 20KB")
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ContractError("Expected exactly one harness function")
        self.function = tree.body[0]
        args = self.function.args
        if (
            self.function.name != "harness"
            or self.function.decorator_list
            or self.function.returns
            or args.defaults
            or args.kw_defaults
            or args.vararg
            or args.kwarg
            or args.posonlyargs
            or args.kwonlyargs
            or any(a.annotation for a in args.args)
            or [a.arg for a in args.args]
            != ["observation", "history", "cards", "memory"]
        ):
            raise ContractError("Expected harness(observation, history, cards, memory)")
        allowed = (
            ast.Module,
            ast.FunctionDef,
            ast.arguments,
            ast.arg,
            ast.Return,
            ast.Assign,
            ast.If,
            ast.For,
            ast.Expr,
            ast.Name,
            ast.Load,
            ast.Store,
            ast.Constant,
            ast.Dict,
            ast.List,
            ast.Tuple,
            ast.Subscript,
            ast.Slice,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mod,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Gt,
            ast.GtE,
            ast.Lt,
            ast.LtE,
            ast.In,
            ast.NotIn,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.UnaryOp,
            ast.Not,
            ast.USub,
            ast.Call,
            ast.IfExp,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise ContractError(f"Forbidden syntax: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise ContractError("Private names are unavailable")
            if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name)
                or node.func.id not in {"get", "len", "min", "max", "rank", "append"}
            ):
                raise ContractError("Only fixed pure JSON helpers may be called")
        self.source = source

    def run(self, observation, history, cards, memory):
        values = json.loads(
            json.dumps([observation, history, cards, memory], allow_nan=False)
        )
        self.variables = dict(
            zip(("observation", "history", "cards", "memory"), values)
        )
        self.fuel = 10000
        try:
            result = self.block(self.function.body)
            if len(json.dumps(result, allow_nan=False).encode()) > 16000:
                raise ContractError("Harness output exceeds 16KB")
            return result
        except (
            KeyError,
            IndexError,
            TypeError,
            ZeroDivisionError,
            RecursionError,
        ) as error:
            raise ContractError(f"Candidate failed: {type(error).__name__}") from error

    def tick(self, value=None):
        self.fuel -= 1
        if self.fuel < 0:
            raise ContractError("Candidate exceeded its instruction budget")
        if isinstance(value, (list, tuple, dict)) and len(value) > 256:
            raise ContractError("Candidate container exceeds 256 entries")
        if (
            isinstance(value, (list, tuple, dict))
            and len(json.dumps(value, allow_nan=False).encode()) > 131072
        ):
            raise ContractError("Candidate intermediate exceeds 128KB")
        if isinstance(value, str) and len(value) > 8192:
            raise ContractError("Candidate string exceeds 8192 characters")
        return value

    def expression(self, node):
        self.tick()
        read = self.expression
        if isinstance(node, ast.Constant):
            result = node.value
        elif isinstance(node, ast.Name):
            result = self.variables[node.id]
        elif isinstance(node, (ast.List, ast.Tuple)):
            result = [read(n) for n in node.elts]
        elif isinstance(node, ast.Dict):
            result = {read(k): read(v) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Slice):
            result = slice(
                *(
                    read(n) if n is not None else None
                    for n in (node.lower, node.upper, node.step)
                )
            )
        elif isinstance(node, ast.Subscript):
            result = read(node.value)[read(node.slice)]
        elif isinstance(node, ast.BinOp):
            op = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mod: operator.mod}[
                type(node.op)
            ]
            left, right = read(node.left), read(node.right)
            if isinstance(node.op, ast.Mod) and (
                type(left) is not int or type(right) is not int
            ):
                raise ContractError("Modulo supports integers only")
            result = op(left, right)
        elif isinstance(node, ast.UnaryOp):
            result = (operator.not_ if isinstance(node.op, ast.Not) else operator.neg)(
                read(node.operand)
            )
        elif isinstance(node, ast.BoolOp):
            result = read(node.values[0])
            for value in node.values[1:]:
                if (isinstance(node.op, ast.And) and not result) or (
                    isinstance(node.op, ast.Or) and result
                ):
                    break
                result = read(value)
        elif isinstance(node, ast.IfExp):
            result = read(node.body if read(node.test) else node.orelse)
        elif isinstance(node, ast.Compare):
            ops = {
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.In: lambda a, b: a in b,
                ast.NotIn: lambda a, b: a not in b,
            }
            left, result = read(node.left), True
            for op, value in zip(node.ops, node.comparators):
                right = read(value)
                if not ops[type(op)](left, right):
                    result = False
                    break
                left = right
        elif isinstance(node, ast.Call):
            args = [read(n) for n in node.args]

            def rank(cards, query):
                words = set(re.findall(r"[a-z]+", query.lower()))
                return sorted(
                    cards,
                    key=lambda c: (
                        -len(
                            words & set(re.findall(r"[a-z]+", c["description"].lower()))
                        ),
                        c["skill_id"],
                    ),
                )

            helpers = {
                "get": lambda d, k, default=None: d.get(k, default),
                "len": len,
                "min": min,
                "max": max,
                "rank": rank,
                "append": lambda items, item: items + [item],
            }
            result = helpers[node.func.id](*args)
        else:
            raise ContractError("Unsupported expression")
        return self.tick(result)

    def assign(self, target, value):
        if isinstance(target, ast.Name):
            self.variables[target.id] = value
        elif isinstance(target, ast.Subscript):
            self.expression(target.value)[self.expression(target.slice)] = value
        else:
            raise ContractError("Unsupported assignment target")

    def block(self, statements):
        for node in statements:
            self.tick()
            if isinstance(node, ast.Return):
                return self.expression(node.value)
            if isinstance(node, ast.Assign):
                value = self.expression(node.value)
                for target in node.targets:
                    self.assign(target, copy.deepcopy(value))
            elif isinstance(node, ast.If):
                result = self.block(
                    node.body if self.expression(node.test) else node.orelse
                )
                if result is not None:
                    return result
            elif isinstance(node, ast.For):
                values = self.expression(node.iter)
                if not isinstance(values, list) or len(values) > 256 or node.orelse:
                    raise ContractError(
                        "Loops require a bounded list and no else clause"
                    )
                for value in values:
                    self.assign(node.target, copy.deepcopy(value))
                    result = self.block(node.body)
                    if result is not None:
                        return result
            elif isinstance(node, ast.Expr):
                self.expression(node.value)
            else:
                raise ContractError("Unsupported statement")
        return None
