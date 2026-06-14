import inspect
from Tsimulation.pushshapes import PushShapesEnv
print("ENVFILE:", inspect.getsourcefile(PushShapesEnv))
src = inspect.getsource(PushShapesEnv)
for i, line in enumerate(src.splitlines()):
    s = line.strip().lower()
    if s.startswith("def ") or "goal" in s or "coverage" in s or "draw" in s or "blit" in s or "_obs" in s:
        print(f"{i:4d}: {line[:130]}")
