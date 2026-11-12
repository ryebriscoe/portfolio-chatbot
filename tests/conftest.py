import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

FIXTURE = ROOT / "tests" / "fixtures" / "content.json"


@pytest.fixture
def fixture_path():
    return str(FIXTURE)


@pytest.fixture
def fixture_text():
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def docs(fixture_path):
    from rag.content import load_documents

    return load_documents(fixture_path)
