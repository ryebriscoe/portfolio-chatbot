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


def test_warmup_reports_readiness_and_the_running_model(settings, index):
    assert Bot(settings, index, FakeGenerator()).warmup() == {"ready": True, "model": settings.gen_model}
    assert Bot(settings, None, FakeGenerator()).warmup() == {"ready": False, "model": settings.gen_model}


def test_sources_follow_what_the_answer_talks_about(settings, index):
    # Retrieval for "java" ranks CST 438 first, but the answer is about the JavaFX app.
    gen = FakeGenerator(pieces=["He built a JavaFX ", "app with JUnit tests."])
    events = list(Bot(settings, index, gen).ask("What has he built with Java?", []))
    assert events[-1]["sources"][0]["id"] == "cst-338"


class FailingGenerator:
    def __init__(self, status, after=()):
        self.status, self.after = status, list(after)

    def stream(self, messages, temperature, max_new_tokens):
        from rag.generate import GeneratorError

        yield from self.after
        raise GeneratorError(self.status, "boom")


def test_rate_limited_model_api_reports_busy(settings, index):
    events = list(Bot(settings, index, FailingGenerator(429)).ask("What has he built with Java?", []))
    assert events[-1] == {"error": "busy", "done": True}


def test_failing_model_api_reports_unavailable_even_mid_answer(settings, index):
    events = list(Bot(settings, index, FailingGenerator(500, after=["He built "])).ask("What has he built with Java?", []))
    assert events[0]["answer"] == "He built "
    assert events[-1] == {"error": "unavailable", "done": True}


def test_bot_uses_a_refreshing_index(settings, docs):
    from rag.refresh import RefreshingIndex

    built = Index(chunk_documents(docs), FakeEmbedder())
    holder = RefreshingIndex(lambda: built, max_age=60, retry_after=10, background=False)
    bot = Bot(settings, holder, FakeGenerator())
    assert bot.warmup()["ready"] is True
    assert list(bot.ask("What has he built with Java?", []))[-1]["done"] is True


def test_visitor_over_the_limit_is_told_to_wait_and_the_model_is_not_called(settings, index):
    from rag.limits import RateLimiter

    gen = FakeGenerator()
    bot = Bot(settings, index, gen, limiter=RateLimiter(per_minute=1, per_day=10))
    assert list(bot.ask("What has he built with Java?", [], client="1.2.3.4"))[-1]["done"] is True
    assert list(bot.ask("What has he built with Java?", [], client="1.2.3.4")) == [{"error": "rate_limited", "done": True}]
    assert len(gen.calls) == 1
    assert list(bot.ask("What has he built with Java?", [], client="5.6.7.8"))[-1]["done"] is True


def test_not_covered_replies_do_not_count_against_the_limit(settings, index):
    from rag.limits import RateLimiter

    bot = Bot(settings, index, FakeGenerator(), limiter=RateLimiter(per_minute=1, per_day=10))
    for _ in range(3):
        assert list(bot.ask("What's the weather tomorrow?", [], client="1.2.3.4"))[-1]["sources"] == []
    assert list(bot.ask("What has he built with Java?", [], client="1.2.3.4"))[-1]["done"] is True
