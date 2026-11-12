import numpy as np

from rag.keywords import KeywordScorer, tokenize

TEXTS = [
    "CST 438 Software Engineering. Tech: Java, Spring Boot, AWS EC2.",
    "CST 338 Software Design. Tech: Java, JavaFX, JUnit.",
    "B.A.D. basketball project using a computer vision model.",
    "CST 363 Databases. Tech: PostgreSQL, SQL.",
]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What has he built with Java?") == ["built", "java"]


def test_tokenize_keeps_language_names_with_symbols():
    assert tokenize("C++ and C# code") == ["c++", "c#", "code"]


def test_scores_are_between_zero_and_one():
    scores = KeywordScorer(TEXTS).score("java spring vision")
    assert scores.shape == (4,)
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_chunk_with_all_query_terms_scores_one():
    scores = KeywordScorer(TEXTS).score("computer vision")
    assert scores[2] == 1.0
    assert scores[0] == 0.0


def test_rare_terms_count_more_than_common_ones():
    # "java" is in two chunks, "postgresql" in one: matching only the rare
    # term must beat matching only the common one.
    scores = KeywordScorer(TEXTS).score("java postgresql")
    assert scores[3] > scores[0] > 0


def test_unknown_terms_lower_every_score():
    scorer = KeywordScorer(TEXTS)
    assert scorer.score("java")[0] == 1.0
    assert scorer.score("java kubernetes")[0] < 0.5


def test_query_with_only_stopwords_scores_zero():
    assert not KeywordScorer(TEXTS).score("what is it?").any()
