from dataclasses import replace

import pytest
from helpers import FakeEmbedder, FakeGenerator

from rag.chunker import chunk_documents
from rag.config import load_settings
from rag.index import Index
from rag.pipeline import Bot, build_index
from rag.prompt import NOT_COVERED_REPLY


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.delenv("GEN_MODEL", raising=False)
    return replace(load_settings(), min_score=0.2)


@pytest.fixture
def index(docs):
    return Index(chunk_documents(docs), FakeEmbedder())


def test_ask_streams_cumulative_then_done_with_sources(settings, index):
    events = list(Bot(settings, index, FakeGenerator()).ask("What has he built with Java?", []))
    assert [e["answer"] for e in events[:3]] == ["Ariya ", "Ariya built ", "Ariya built it."]
    assert all(e["done"] is False and e["sources"] == [] for e in events[:3])
    final = events[-1]
    assert final["done"] is True
    assert final["answer"] == "Ariya built it."
    assert {"cst-438", "cst-338"} <= {s["id"] for s in final["sources"]}
    assert len(final["sources"]) <= settings.max_sources


def test_ask_rejects_empty_question_without_calling_model(settings, index):
    gen = FakeGenerator()
    assert list(Bot(settings, index, gen).ask("   ", [])) == [{"error": "empty", "done": True}]
    assert gen.calls == []


def test_ask_rejects_too_long_question(settings, index):
    gen = FakeGenerator()
    assert list(Bot(settings, index, gen).ask("a" * 501, [])) == [{"error": "too long", "done": True}]
    assert gen.calls == []


def test_ask_unavailable_when_index_none(settings):
    gen = FakeGenerator()
    assert list(Bot(settings, None, gen).ask("What has he built?", [])) == [
        {"error": "unavailable", "done": True}
    ]
    assert gen.calls == []


def test_no_match_replies_not_covered_without_the_model(settings, index):
    gen = FakeGenerator()
    events = list(Bot(settings, index, gen).ask("What's the weather tomorrow?", []))
    assert events == [{"answer": NOT_COVERED_REPLY, "sources": [], "done": True}]
    assert gen.calls == []


def test_refusal_answer_shows_no_sources(settings, index):
    gen = FakeGenerator(pieces=["That isn’t covered on Ariya’s portfolio. ", "You can reach him by email."])
    events = list(Bot(settings, index, gen).ask("Has he worked at Google with Java?", []))
    assert gen.calls  # the model did run: chunks matched, but it declined
    assert events[-1]["sources"] == []


def test_build_index_adds_overviews_and_uses_settings(settings, fixture_path):
    index = build_index(replace(settings, content_url=fixture_path), embedder=FakeEmbedder())
    assert "overview-skills" in {c.doc_id for c in index.chunks}
    assert index.keyword_weight == settings.keyword_weight


def test_ask_uses_sanitized_history_and_retrieval_query(settings, index):
    gen = FakeGenerator()
    history = [["junk"], ["Tell me about his Java work", "He built two apps."], 7]
    events = list(Bot(settings, index, gen).ask("Which one used JavaFX?", history))
    roles = [m["role"] for m in gen.calls[0]]
    assert roles == ["system", "user", "assistant", "user"]
    assert gen.calls[0][1]["content"] == "Tell me about his Java work"
    assert "cst-338" in {s["id"] for s in events[-1]["sources"]}


def test_warmup_reports_readiness(settings, index):
    assert Bot(settings, index, FakeGenerator()).warmup() == {"ready": True}
    assert Bot(settings, None, FakeGenerator()).warmup() == {"ready": False}
