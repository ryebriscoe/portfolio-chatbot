import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"


def _import_lines(tree):
    for node in tree.body:
        if isinstance(node, ast.Try):  # optional imports wrapped in try/except
            yield from _import_lines(ast.Module(body=node.body, type_ignores=[]))
        elif isinstance(node, ast.Import):
            yield node.lineno, [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            yield node.lineno, [node.module]


def test_spaces_is_imported_before_torch_or_rag():
    # ZeroGPU only lets the model move to "cuda" at startup if `spaces` is
    # imported before anything initializes torch (decision D21).
    imports = list(_import_lines(ast.parse(APP.read_text(encoding="utf-8"))))
    spaces_line = next(line for line, names in imports if "spaces" in names)
    later = [names for line, names in imports if line < spaces_line and any(
        n.split(".")[0] in ("torch", "rag", "gradio", "transformers", "sentence_transformers") for n in names)]
    assert later == []
