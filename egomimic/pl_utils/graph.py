from typing import Any

import torch.nn as nn

from egomimic.pl_utils.nodes import Node


class Graph(nn.Module):
    """A DAG of :class:`Node` modules.

    Wires nodes by output→input name. Each output name must be produced by
    exactly one node. Cycles are detected at construction.

    On ``forward(batch, global_mode)`` the batch is copied into a global dict;
    nodes execute in topological order, reading inputs from and writing outputs
    to the dict. The full dict is returned.
    """

    def __init__(self, nodes: list[Node], compile: bool | str | None = None):
        super().__init__()
        self.nodes_list = nn.ModuleList(nodes)
        self._order = self._topo_sort(nodes)
        for node in nodes:
            node.maybe_compile(compile)

    @staticmethod
    def _topo_sort(nodes: list[Node]) -> list[Node]:
        producers: dict[str, Node] = {}
        for node in nodes:
            for out in node.output_names:
                if out in producers:
                    raise ValueError(
                        f"output '{out}' produced by both "
                        f"'{producers[out].name}' and '{node.name}'"
                    )
                producers[out] = node

        # edges: producer node -> list of consumer nodes
        consumers: dict[int, list[Node]] = {id(n): [] for n in nodes}
        indeg: dict[int, int] = {id(n): 0 for n in nodes}
        for node in nodes:
            upstream_ids = set()
            for inp in node.input_names:
                prod = producers.get(inp)
                if prod is not None and id(prod) != id(node):
                    upstream_ids.add(id(prod))
            for uid in upstream_ids:
                consumers[uid].append(node)
            indeg[id(node)] = len(upstream_ids)

        ready = [n for n in nodes if indeg[id(n)] == 0]
        order: list[Node] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for c in consumers[id(n)]:
                indeg[id(c)] -= 1
                if indeg[id(c)] == 0:
                    ready.append(c)
        if len(order) != len(nodes):
            unresolved = [n.name for n in nodes if indeg[id(n)] > 0]
            raise ValueError(f"cycle detected among nodes: {unresolved}")
        return order

    def forward(self, batch: dict[str, Any], global_mode: str) -> dict[str, Any]:
        global_dict = dict(batch)
        for node in self._order:
            global_dict.update(node.run(global_dict, global_mode))
        return global_dict
