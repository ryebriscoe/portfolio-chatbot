from helpers import FakeEmbedder

from rag.chunker import chunk_documents
from rag.index import Index, with_prefix


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


def test_chunks_and_queries_are_embedded_differently(docs):
    index, embedder = build(docs)
    index.search("java", k=4, min_score=0.0)
    assert embedder.query_flags == [False, True]


def test_with_prefix_only_changes_queries():
    assert with_prefix(["a", "b"], "Q: ", is_query=True) == ["Q: a", "Q: b"]
    assert with_prefix(["a"], "Q: ", is_query=False) == ["a"]
    assert with_prefix(["a"], "", is_query=True) == ["a"]


def test_keyword_weight_adds_a_bonus_for_exact_terms(docs):
    chunks = chunk_documents(docs)
    plain = Index(chunks, FakeEmbedder())
    hybrid = Index(chunks, FakeEmbedder(), keyword_weight=0.5)
    # "gradle" is not in the fake embedder's vocabulary, so meaning search
    # scores every chunk 0; only the keyword bonus can find CST 338.
    assert plain.search("gradle", k=1, min_score=0.01) == []
    hits = hybrid.search("gradle", k=1, min_score=0.01)
    assert hits[0][0].doc_id == "cst-338"


def test_score_parts_add_up_to_search_score(docs):
    index = Index(chunk_documents(docs), FakeEmbedder(), keyword_weight=0.2)
    meaning, keyword = index.score_parts("java spring")
    chunk, score = index.search("java spring", k=1, min_score=0.0)[0]
    i = index.chunks.index(chunk)
    assert abs(score - (meaning[i] + 0.2 * keyword[i])) < 1e-6
