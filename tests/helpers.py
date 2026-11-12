"""Test doubles shared across test modules."""

import re

import numpy as np


class FakeEmbedder:
    """Deterministic bag-of-words embedder over a fixed vocabulary."""

    VOCAB = [
        "java", "spring", "javafx", "junit", "aws", "capstone", "basketball",
        "computer", "vision", "certification", "ibm", "deep", "learning", "rust",
        "sort", "google",
    ]

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), len(self.VOCAB)), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = re.findall(r"[a-z]+", text.lower())
            for j, word in enumerate(self.VOCAB):
                out[i, j] = tokens.count(word)
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class FakeGenerator:
    """Streams a fixed answer and records what it was asked."""

    def __init__(self, pieces=("Ariya ", "built ", "it.")):
        self.pieces = list(pieces)
        self.calls = []

    def stream(self, messages, temperature, max_new_tokens):
        self.calls.append(messages)
        yield from self.pieces
