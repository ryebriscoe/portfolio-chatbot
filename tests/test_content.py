import pytest

from rag.content import ContentUnavailable, load_documents


def test_load_documents_from_file(docs):
    assert len(docs) == 5
    capstone = next(d for d in docs if d.id == "cst-499")
    assert capstone.text == ""
    assert capstone.catalog


def test_missing_optional_fields_become_none_or_empty(docs):
    capstone = next(d for d in docs if d.id == "cst-499")
    assert capstone.tech == ()
    assert capstone.repo is None
    assert capstone.project is None


def test_html_entities_are_decoded(docs):
    bad = next(d for d in docs if d.id.startswith("project-b-a-d"))
    assert "Ariya’s" in bad.text
    assert "&rsquo;" not in bad.text


def test_retries_then_raises():
    calls, sleeps = [], []

    def fetch(url):
        calls.append(url)
        raise OSError("down")

    with pytest.raises(ContentUnavailable):
        load_documents("https://example.test/c.json", fetch=fetch, sleep=sleeps.append)
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_recovers_on_second_attempt(fixture_text):
    attempts = []

    def fetch(url):
        attempts.append(url)
        if len(attempts) == 1:
            raise OSError("blip")
        return fixture_text

    docs = load_documents("https://example.test/c.json", fetch=fetch, sleep=lambda s: None)
    assert len(docs) == 5


def test_invalid_json_raises_unavailable():
    with pytest.raises(ContentUnavailable):
        load_documents("https://example.test/c.json", fetch=lambda u: "not json", sleep=lambda s: None)
