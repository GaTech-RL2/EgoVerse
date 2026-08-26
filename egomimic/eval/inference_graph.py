"""Small keyed inference graph with episode-scoped action-cache state.

The evaluator calls one graph.  A graph either serves an already committed
action from ``check_cache`` or runs ``preprocess -> model -> update_cache``.
Nodes expose explicit ``in``/``out`` port maps so the same node (or subgraph)
can be reused under different key names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Sequence


class GraphContractError(RuntimeError):
    """Raised when a node violates its declared keyed interface."""


class KeyedNode:
    """Adapt a callable to a shared key-value context.

    ``inputs`` and ``outputs`` use ``{callable_port: graph_key}``.  Config
    loaders may pass literal YAML keys ``in`` and ``out`` through ``ports``.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        inputs: Mapping[str, str] | None = None,
        outputs: Mapping[str, str] | None = None,
        **ports: Any,
    ) -> None:
        self.fn = fn
        self.inputs = dict(ports.pop("in", inputs or {}))
        self.outputs = dict(ports.pop("out", outputs or {}))
        if ports:
            raise TypeError(f"unknown node fields: {sorted(ports)}")

    def __call__(self, context: MutableMapping[str, Any]) -> None:
        missing = [key for key in self.inputs.values() if key not in context]
        if missing:
            raise GraphContractError(f"node input keys missing: {missing}")
        result = self.fn(**{
            port: context[key] for port, key in self.inputs.items()
        })
        if not self.outputs:
            if result is not None:
                raise GraphContractError("node with no outputs returned a value")
            return
        if len(self.outputs) == 1:
            port = next(iter(self.outputs))
            # A mapping is commonly the value transported through one port;
            # treat it as an output record only when it names that port.
            if not isinstance(result, Mapping) or port not in result:
                result = {port: result}
        if not isinstance(result, Mapping):
            raise GraphContractError("multi-output node must return a mapping")
        missing_ports = [port for port in self.outputs if port not in result]
        if missing_ports:
            raise GraphContractError(
                f"node output ports missing: {missing_ports}")
        for port, key in self.outputs.items():
            context[key] = result[port]


class Subgraph(KeyedNode):
    """A sequence of keyed nodes presented as one single-entry/exit node."""

    def __init__(self, nodes: Sequence[KeyedNode], **ports: Any) -> None:
        self.nodes = tuple(nodes)
        super().__init__(self._run, **ports)

    def _run(self, **values: Any) -> Any:
        local: dict[str, Any] = dict(values)
        for node in self.nodes:
            node(local)
        if len(self.outputs) == 1:
            port = next(iter(self.outputs))
            if port not in local:
                raise GraphContractError(f"subgraph endpoint {port!r} missing")
            return local[port]
        return {port: local[port] for port in self.outputs}


@dataclass
class ActionCacheState:
    """Mutable state owned by one policy instance and reset per episode."""

    actions: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        self.actions.clear()

    def replace(self, actions: Sequence[Any]) -> None:
        self.actions[:] = list(actions)

    def pop(self) -> Any:
        if not self.actions:
            raise GraphContractError("attempted to pop an empty action cache")
        return self.actions.pop(0)

    def __bool__(self) -> bool:
        return bool(self.actions)


class InferenceGraph:
    """Fixed controller topology with an early terminal action edge."""

    def __init__(
        self,
        *,
        check_cache: KeyedNode,
        inference_preprocess: KeyedNode,
        model: KeyedNode,
        update_cache: KeyedNode,
        terminal_key: str = "policy.action",
    ) -> None:
        self.check_cache = check_cache
        self.inference_preprocess = inference_preprocess
        self.model = model
        self.update_cache = update_cache
        self.terminal_key = terminal_key

    def __call__(self, **inputs: Any) -> Any:
        context: dict[str, Any] = dict(inputs)
        self.check_cache(context)
        if context.get(self.terminal_key) is not None:
            return context[self.terminal_key]
        self.inference_preprocess(context)
        self.model(context)
        self.update_cache(context)
        if self.terminal_key not in context:
            raise GraphContractError(
                f"graph completed without terminal key {self.terminal_key!r}")
        return context[self.terminal_key]
