from helpers import FakeEmbedder

from rag.chunker import chunk_documents
from rag.index import Index


def build(docs):
    embedder = FakeEmbedder()
    return Index(chunk_documents(docs), embedder), embedder


def test_search_ranks_matching_chunk_first(docs):
    index, _ = build(docs)
    hits = index.search("java spring", k=4, min_score=0.0)
    assert hits[0][0].doc_id == "cst-438"


def test_search_respects_k(docs):
    index, _ = build(docs)
    assert len(index.search("java", k=2, min_score=-1.0)) == 2


def test_search_drops_below_min_score(docs):
    index, _ = build(docs)
    assert index.search("java spring", k=4, min_score=0.99) == []


def test_unrelated_query_returns_nothing(docs):
    index, _ = build(docs)
    assert index.search("weather tomorrow", k=4, min_score=0.1) == []


def test_scores_descending(docs):
    index, _ = build(docs)
    scores = [s for _, s in index.search("java computer vision", k=4, min_score=-1.0)]
    assert scores == sorted(scores, reverse=True)


def test_index_embeds_chunks_once(docs):
    index, embedder = build(docs)
    assert embedder.calls == 1
    index.search("java", k=4, min_score=0.0)
    index.search("aws", k=4, min_score=0.0)
    assert embedder.calls == 3
