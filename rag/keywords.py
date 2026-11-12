"""Keyword matching to complement meaning-based search (decision D16).

Embeddings are good at meaning ("built a web backend" ~ "server-side app")
but can rank an exact term like "PostgreSQL" or "computer vision" too low.
This scorer gives each chunk a 0..1 score for how many of the question's
important words it contains. Rare words count more than common ones (IDF).
"""

import math
import re

import numpy as np

STOPWORDS = frozenset(
    "a an and any anything are as at be been by can could did do does for from give has have he "
    "her him his how i in is it me my of on or she tell that the their them they this to was were "
    "what whats when where which who with would you your about".split()
)
_WORD = re.compile(r"[a-z0-9+#]+")


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


class KeywordScorer:
    def __init__(self, texts: list[str]):
        self.token_sets = [set(tokenize(t)) for t in texts]
        self.n = len(texts)

    def _idf(self, word: str) -> float:
        """Rarer words get a higher weight; a word in no chunk gets the highest."""
        containing = sum(word in tokens for tokens in self.token_sets)
        return math.log(1 + self.n / (1 + containing))

    def score(self, query: str) -> np.ndarray:
        weights = {w: self._idf(w) for w in set(tokenize(query))}
        total = sum(weights.values())
        if not total:
            return np.zeros(self.n, dtype=np.float32)
        return np.array(
            [sum(wt for w, wt in weights.items() if w in tokens) / total for tokens in self.token_sets],
            dtype=np.float32,
        )
