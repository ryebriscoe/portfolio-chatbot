"""In-memory embedding index with NumPy cosine search (decision D10).

The index is a (n_chunks, dim) array of L2-normalized embeddings plus the
parallel list of chunks; row i belongs to chunk i. With normalized rows, the
dot product of a row and the query is their cosine similarity.
"""

from typing import Protocol

import numpy as np

from rag.chunker import Chunk


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dim) float32 array with L2-normalized rows."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


class Index:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = embedder.embed([c.text for c in chunks])

    def search(self, query: str, k: int, min_score: float) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed([query])[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] >= min_score]
