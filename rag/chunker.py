"""Split Documents into retrieval chunks (spec §4.1).

Every document gets a summary chunk (#0) built from its front matter, then
body chunks of roughly min_words..max_words words, each prefixed with the
document title so it stands on its own.
"""

import re
from dataclasses import dataclass

from rag.content import Document

_SPACES = re.compile(r"[ \t ]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    url: str
    type: str
    text: str
    status: str | None


def _norm(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def summary_text(doc: Document) -> str:
    meta = [m for m in (doc.status and doc.status.replace("_", " "), doc.term) if m]
    parts = [f"{doc.title} ({', '.join(meta)})." if meta else f"{doc.title}."]
    if doc.project:
        parts.append(f"Project: {doc.project}.")
    if doc.tech:
        parts.append(f"Tech: {', '.join(doc.tech)}.")
    if doc.catalog and not doc.text.strip():
        parts.append(f"Course description: {_norm(doc.catalog)}")
    return " ".join(parts)


def _pieces(paragraph: str, max_words: int) -> list[str]:
    """Split one paragraph into pieces of at most max_words, at sentence ends when possible."""
    if len(paragraph.split()) <= max_words:
        return [paragraph]
    pieces, current = [], []
    for sentence in _SENTENCE_END.split(paragraph):
        words = sentence.split()
        while len(words) > max_words:  # a single over-long sentence: hard split
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if current and len(current) + len(words) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _body_groups(doc: Document, min_words: int, max_words: int) -> list[str]:
    paragraphs = [_norm(p) for p in re.split(r"\n\s*\n", doc.text)]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return []
    if doc.catalog:
        paragraphs.insert(0, f"Course description: {_norm(doc.catalog)}")

    pieces = [piece for p in paragraphs for piece in _pieces(p, max_words)]
    groups, current = [], []
    for piece in pieces:
        if current and len(" ".join(current).split()) + len(piece.split()) > max_words:
            groups.append(" ".join(current))
            current = []
        current.append(piece)
    if current:
        last = " ".join(current)
        if groups and len(last.split()) < min_words and len(f"{groups[-1]} {last}".split()) <= max_words:
            groups[-1] = f"{groups[-1]} {last}"
        else:
            groups.append(last)
    return groups


def chunk_documents(docs: list[Document], min_words: int = 120, max_words: int = 200) -> list[Chunk]:
    chunks = []
    for doc in docs:
        texts = [summary_text(doc)] + [f"{doc.title}: {g}" for g in _body_groups(doc, min_words, max_words)]
        for i, text in enumerate(texts):
            chunks.append(Chunk(
                id=f"{doc.id}#{i}", doc_id=doc.id, title=doc.title, url=doc.url,
                type=doc.type, text=text, status=doc.status,
            ))
    return chunks
