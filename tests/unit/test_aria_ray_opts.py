"""The Aria Ray driver builds the args aria_to_lance.convert reads by hand, so nothing but
this test notices when convert starts reading a new one."""

import ast
from pathlib import Path

import egomimic.scripts.aria_process as pkg

HERE = Path(next(iter(pkg.__path__)))


def _tree(name):
    return ast.parse((HERE / name).read_text())


def test_the_ray_driver_supplies_everything_convert_reads():
    convert = next(
        n
        for n in ast.walk(_tree("aria_to_lance.py"))
        if isinstance(n, ast.FunctionDef) and n.name == "convert"
    )
    read = {
        n.attr
        for n in ast.walk(convert)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "args"
    }
    supplied = set()
    for n in ast.walk(_tree("aria_to_lance_ray.py")):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") in (
            "dict",
            "SimpleNamespace",
        ):
            supplied |= {k.arg for k in n.keywords if k.arg}
    assert read and read <= supplied, read - supplied
