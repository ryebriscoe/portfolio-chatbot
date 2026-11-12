from dataclasses import replace

from helpers import FakeEmbedder

from rag.chunker import chunk_documents
from rag.config import load_settings
from rag.explain import explain
from rag.index import Index


def make(docs, **overrides):
    settings = replace(load_settings(), min_score=0.2, **overrides)
    return Index(chunk_documents(docs), FakeEmbedder(), keyword_weight=settings.keyword_weight), settings


def test_explain_shows_ranked_chunks_and_the_context(docs):
    index, settings = make(docs)
    out = explain(index, settings, "What has he built with Java?")
    assert "Question: What has he built with Java?" in out
    assert "cst-438#0" in out and "kept" in out
    assert "meaning" in out and "keyword" in out
    assert "[CST 438 Software Engineering | completed]" in out


def test_explain_marks_chunks_below_min_score_as_dropped(docs):
    index, settings = make(docs, top_k=2)
    out = explain(index, settings, "What has he built with Java?")
    assert "dropped" in out


def test_explain_says_when_nothing_matches(docs):
    index, settings = make(docs)
    out = explain(index, settings, "What's the weather tomorrow?")
    assert "No chunk passed min_score" in out
