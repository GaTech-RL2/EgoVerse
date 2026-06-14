import ast
for f in ["Tsimulation/pushshapes/shapes.py","Tsimulation/pushshapes/env.py","Tsimulation/pushshapes/render.py"]:
    ast.parse(open(f).read())
print("SYNTAX_OK")
