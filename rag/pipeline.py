"""The question-answering pipeline: validate → retrieve → prompt → stream."""

from typing import Iterator

from rag.chunker import chunk_documents
from rag.config import Settings
from rag.content import load_documents
from rag.generate import Generator
from rag.index import Embedder, FastEmbedEmbedder, Index
from rag.overview import overview_documents
from rag.prompt import (
    NOT_COVERED_REPLY,
    InvalidQuestion,
    build_messages,
    is_not_covered,
    retrieval_query,
    sanitize_history,
    select_sources,
    validate_question,
)


def build_index(settings: Settings, embedder: Embedder | None = None) -> Index:
    """Load the site content and turn it into a searchable index."""
    documents = load_documents(settings.content_url)
    documents += overview_documents(documents)
    embedder = embedder or FastEmbedEmbedder(settings.embed_model, settings.query_prefix)
    return Index(chunk_documents(documents), embedder, settings.keyword_weight)


class Bot:
    def __init__(self, settings: Settings, index: Index | None, generator: Generator):
        self.settings = settings
        self.index = index  # None means the content export failed to load (D12)
        self.generator = generator

    def warmup(self) -> dict:
        """Readiness plus the model actually running, so deploys can be verified."""
        return {"ready": self.index is not None, "model": self.settings.gen_model}

    def ask(self, question: object, history: object) -> Iterator[dict]:
        s = self.settings

        # 1. Validate the input.
        try:
            question = validate_question(question, s.max_question_chars)
        except InvalidQuestion as exc:
            yield {"error": str(exc), "done": True}
            return
        if self.index is None:
            yield {"error": "unavailable", "done": True}
            return

        # 2. Retrieve the chunks that best match the question.
        turns = sanitize_history(history, s.history_turns)
        hits = self.index.search(retrieval_query(question, turns), s.top_k, s.min_score)
        if not hits:  # nothing on the site matches: answer without the model (D17)
            yield {"answer": NOT_COVERED_REPLY, "sources": [], "done": True}
            return

        # 3. Build the prompt and stream the model's answer.
        answer = ""
        for piece in self.generator.stream(build_messages(question, turns, hits), s.temperature, s.max_new_tokens):
            answer += piece
            yield {"answer": answer, "sources": [], "done": False}

        # 4. Attach sources, unless the model said the site doesn't cover it.
        answer = answer.strip()
        if is_not_covered(answer):
            sources = []
        else:  # link the retrieved pages closest to what the answer says (D26)
            sources = select_sources(self.index.rank_by_answer(answer, hits, s.source_retrieval_weight), s.max_sources, s.source_margin)
        yield {"answer": answer, "sources": sources, "done": True}
