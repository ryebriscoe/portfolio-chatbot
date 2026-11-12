import pytest

from rag.chunker import Chunk
from rag.prompt import (
    SYSTEM_PROMPT,
    NOT_COVERED_REPLY,
    InvalidQuestion,
    build_messages,
    format_context,
    is_not_covered,
    retrieval_query,
    sanitize_history,
    select_sources,
    validate_question,
)


def chunk(doc_id, title="CST 438 Software Engineering", status="completed", n=0, url="/courses/cst-438/"):
    return Chunk(id=f"{doc_id}#{n}", doc_id=doc_id, title=title, url=url, type="course",
                 text=f"{title}: text {n}", status=status)


def test_system_prompt_rules_verbatim():
    assert "Use ONLY the facts in the CONTEXT below." in SYSTEM_PROMPT
    assert ("That isn't covered on\n   Ariya's portfolio. You can reach him at ariyabriscoe@gmail.com or LinkedIn."
            in SYSTEM_PROMPT)
    assert 'Refer to him as "he" or "Ariya".' in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.rstrip().endswith("CONTEXT:")


def test_validate_question_trims():
    assert validate_question("  hi  ", 500) == "hi"


def test_validate_question_rejects_whitespace_only():
    with pytest.raises(InvalidQuestion, match="empty"):
        validate_question("   \n", 500)


@pytest.mark.parametrize("bad", [None, 123, ["q"]])
def test_validate_question_rejects_non_string(bad):
    with pytest.raises(InvalidQuestion, match="empty"):
        validate_question(bad, 500)


def test_validate_question_length_counts_after_trim():
    assert validate_question("  " + "a" * 500 + "  ", 500) == "a" * 500
    with pytest.raises(InvalidQuestion, match="too long"):
        validate_question("a" * 501, 500)


def test_sanitize_history_keeps_last_turns():
    raw = [[f"q{i}", f"a{i}"] for i in range(5)]
    assert sanitize_history(raw, 2) == [("q3", "a3"), ("q4", "a4")]


def test_sanitize_history_drops_malformed():
    raw = [["a", "b"], "x", ["c"], [1, 2], None, ["d", "e"]]
    assert sanitize_history(raw, 5) == [("a", "b"), ("d", "e")]


@pytest.mark.parametrize("raw", ["oops", None, 42, {"a": "b"}])
def test_sanitize_history_non_list_returns_empty(raw):
    assert sanitize_history(raw, 2) == []


def test_sanitize_history_truncates_long_turns():
    out = sanitize_history([["q" * 5000, "a" * 5000]], 2)
    assert len(out[0][0]) == 2000 and len(out[0][1]) == 2000


def test_retrieval_query_includes_previous_question():
    assert retrieval_query("what tech?", [("Tell me about CST 438", "...")]) == "Tell me about CST 438\nwhat tech?"
    assert retrieval_query("what tech?", []) == "what tech?"


def test_format_context_labels_and_status_fallback():
    cert = Chunk(id="c#0", doc_id="c", title="IBM AI", url="/certifications/", type="certification",
                 text="IBM AI: x", status=None)
    ctx = format_context([(chunk("cst-438"), 0.9), (cert, 0.5)])
    assert "[CST 438 Software Engineering | completed]\nCST 438 Software Engineering: text 0" in ctx
    assert "[IBM AI | certification]" in ctx


def test_build_messages_order_and_context_labels():
    msgs = build_messages("What did he build?", [("q1", "a1")], [(chunk("cst-438"), 0.9)])
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith(SYSTEM_PROMPT)
    assert "[CST 438 Software Engineering | completed]" in msgs[0]["content"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1] == {"role": "user", "content": "What did he build?"}


def test_build_messages_empty_hits_says_no_content():
    msgs = build_messages("weather?", [], [])
    assert msgs[0]["content"].endswith("(no relevant content found on the site)")


def test_select_sources_dedupes_by_doc_and_caps():
    hits = [
        (chunk("cst-438", n=0), 0.9), (chunk("cst-438", n=1), 0.85), (chunk("cst-438", n=2), 0.8),
        (chunk("cst-338", title="CST 338 Software Design", url="/courses/cst-338/"), 0.7),
        (chunk("cst-499", title="CST 499 Capstone", url="/courses/cst-499/"), 0.6),
        (chunk("cst-200", title="Other", url="/courses/cst-200/"), 0.5),
    ]
    assert select_sources(hits, 3) == [
        {"id": "cst-438", "title": "CST 438 Software Engineering", "url": "/courses/cst-438/"},
        {"id": "cst-338", "title": "CST 338 Software Design", "url": "/courses/cst-338/"},
        {"id": "cst-499", "title": "CST 499 Capstone", "url": "/courses/cst-499/"},
    ]


def test_not_covered_reply_matches_the_prompt_rule():
    assert NOT_COVERED_REPLY == (
        "That isn't covered on Ariya's portfolio. You can reach him at ariyabriscoe@gmail.com or LinkedIn."
    )
    assert is_not_covered(NOT_COVERED_REPLY)
    assert is_not_covered("Sorry, that isn’t covered on Ariya’s portfolio.")
    assert not is_not_covered("He built a Spring Boot app.")
