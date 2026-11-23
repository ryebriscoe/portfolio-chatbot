import subprocess
import sys

import pytest

pytest.importorskip("fastembed")

SCRIPT = """
import sys
from rag.index import FastEmbedEmbedder
v = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", "Q: ").embed(["hello"], is_query=True)
assert v.shape == (1, 384), v.shape
print("torch" in sys.modules)
"""


def test_embedding_never_loads_torch():
    # On ZeroGPU, using torch in the main process (even on CPU) breaks the GPU
    # workers, so search embeddings must run without torch (decision D21).
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, cwd=".")
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "False"
