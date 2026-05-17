from typing import Any, Callable

import torch
import torch.nn as nn


class Node(nn.Module):
    """A single computation unit in the model DAG.

    Reads named inputs from the graph's global dict, calls a per-mode function
    (or the underlying module by default), and writes named outputs back.

    mode_fns values are callables with signature ``(node, inputs: dict) -> dict``
    (typically Hydra-instantiated). A value of ``None`` skips the node in that mode.
    """

    def __init__(
        self,
        module: nn.Module,
        inputs: list[str],
        outputs: list[str],
        mode_fns: dict[str, Callable | None] | None = None,
        compile: bool | str | None = None,
        name: str | None = None,
    ):
        super().__init__()
        self.module = module
        self.input_names = list(inputs)
        self.output_names = list(outputs)
        self.mode_fns = dict(mode_fns) if mode_fns else {}
        self.name = name or self.__class__.__name__
        self._compile_spec = compile

    def maybe_compile(self, default: bool | str | None) -> None:
        spec = self._compile_spec if self._compile_spec is not None else default
        if not spec:
            return
        mode = spec if isinstance(spec, str) else "default"
        self.module = torch.compile(self.module, mode=mode)

    def _pack(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result
        if len(self.output_names) == 1:
            return {self.output_names[0]: result}
        return dict(zip(self.output_names, result, strict=True))

    def run(self, global_dict: dict[str, Any], global_mode: str) -> dict[str, Any]:
        if global_mode in self.mode_fns:
            fn = self.mode_fns[global_mode]
            if fn is None:
                return {}
            inputs = {k: global_dict[k] for k in self.input_names}
            return self._pack(fn(self, inputs))
        inputs = {k: global_dict[k] for k in self.input_names}
        return self._pack(self.module(**inputs))


class EvalNode(Node):
    """Node that only runs when ``global_mode == 'eval'``."""

    def run(self, global_dict, global_mode):
        if global_mode != "eval":
            return {}
        return super().run(global_dict, global_mode)


class FreezeWrapper(nn.Module):
    """Wraps a module and optionally disables grad + forces eval()."""

    def __init__(self, module: nn.Module, freeze: bool = False):
        super().__init__()
        self.module = module
        self.freeze = freeze
        if freeze:
            for p in self.module.parameters():
                p.requires_grad = False
            self.module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.module.eval()
        return self

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
