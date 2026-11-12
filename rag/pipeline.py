"""The question-answering pipeline: validate → retrieve → prompt → stream."""

from typing import Iterator

from rag.config import Settings
from rag.generate import Generator
from rag.index import Index
from rag.prompt import (
    InvalidQuestion,
    build_messages,
    retrieval_query,
    sanitize_history,
    select_sources,
    validate_question,
)


class Bot:
    def __init__(self, settings: Settings, index: Index | None, generator: Generator):
        self.settings = settings
        self.index = index  # None means the content export failed to load (D12)
        self.generator = generator

    def warmup(self) -> dict:
        return {"ready": self.index is not None}

    def ask(self, question: object, history: object) -> Iterator[dict]:
        s = self.settings
        try:
            question = validate_question(question, s.max_question_chars)
        except InvalidQuestion as exc:
            yield {"error": str(exc), "done": True}
            return
        if self.index is None:
            yield {"error": "unavailable", "done": True}
            return

        turns = sanitize_history(history, s.history_turns)
        hits = self.index.search(retrieval_query(question, turns), s.top_k, s.min_score)
        messages = build_messages(question, turns, hits)

        answer = ""
        for piece in self.generator.stream(messages, s.temperature, s.max_new_tokens):
            answer += piece
            yield {"answer": answer, "sources": [], "done": False}
        yield {"answer": answer.strip(), "sources": select_sources(hits, s.max_sources), "done": True}
