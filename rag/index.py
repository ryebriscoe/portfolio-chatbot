"""In-memory search index: NumPy cosine similarity plus a keyword bonus.

The index is a (n_chunks, dim) array of L2-normalized embeddings and the
parallel list of chunks; row i belongs to chunk i (decision D10). With
normalized rows, the dot product of a row and the query is their cosine
similarity ("meaning" score). A chunk's final score is:

    meaning + keyword_weight * keyword        (decision D16)
"""

from typing import Protocol

import numpy as np

from rag.chunker import Chunk
from rag.keywords import KeywordScorer


class Embedder(Protocol):
    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Return a (len(texts), dim) float32 array with L2-normalized rows."""


def with_prefix(texts: list[str], prefix: str, is_query: bool) -> list[str]:
    """Some embedding models want an instruction in front of questions (D15)."""
    return [prefix + t for t in texts] if is_query and prefix else list(texts)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.query_prefix = query_prefix

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        texts = with_prefix(texts, self.query_prefix, is_query)
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


class Index:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, keyword_weight: float = 0.0):
        self.chunks = chunks
        self.embedder = embedder
        self.keyword_weight = keyword_weight
        texts = [c.text for c in chunks]
        self.vectors = embedder.embed(texts)
        self.keywords = KeywordScorer(texts)

    def score_parts(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (meaning, keyword) scores for every chunk."""
        meaning = self.vectors @ self.embedder.embed([query], is_query=True)[0]
        return meaning, self.keywords.score(query)

    def search(self, query: str, k: int, min_score: float) -> list[tuple[Chunk, float]]:
        meaning, keyword = self.score_parts(query)
        scores = meaning + self.keyword_weight * keyword
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] >= min_score]
