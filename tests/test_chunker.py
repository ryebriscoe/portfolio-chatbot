from rag.chunker import chunk_documents, summary_text
from rag.content import Document


def by_id(docs, doc_id):
    return next(d for d in docs if d.id == doc_id)


def words(chunk):
    return len(chunk.text.split(": ", 1)[1].split())


def test_summary_text_for_course(docs):
    assert summary_text(by_id(docs, "cst-438")) == (
        "CST 438 Software Engineering (completed, Summer 2026, Term B). "
        "Project: Canvas Style Course Management Platform. "
        "Tech: Java, Spring Boot, JavaScript, REST APIs, AWS EC2."
    )


def test_summary_humanizes_status(docs):
    assert summary_text(by_id(docs, "cert-ibm-ai-engineering")).startswith(
        "IBM AI Engineering (in progress)."
    )


def test_summary_includes_catalog_when_no_body(docs):
    capstone = by_id(docs, "cst-499")
    assert summary_text(capstone).endswith(capstone.catalog)


def test_planned_course_without_body_yields_summary_only(docs):
    chunks = [c for c in chunk_documents(docs) if c.doc_id == "cst-499"]
    assert len(chunks) == 1
    assert chunks[0].id == "cst-499#0"
    for c in chunk_documents(docs):
        assert c.text.strip()
        assert "None" not in c.text


def test_catalog_becomes_body_paragraph_when_body_exists(docs):
    chunks = [c for c in chunk_documents(docs) if c.doc_id == "cst-438"]
    assert any("Course description: Prepares students" in c.text for c in chunks[1:])


def test_body_chunks_respect_word_limits():
    para = " ".join(["word"] * 59) + " end."
    doc = Document(id="d", type="page", title="T", url="/", text="\n\n".join([para] * 10))
    body = [c for c in chunk_documents([doc]) if not c.id.endswith("#0")]
    assert all(words(c) <= 200 for c in body)
    assert all(words(c) >= 120 for c in body[:-1])
    assert body[0].text.startswith("T: ")


def test_long_single_paragraph_is_split():
    sentence = " ".join(["word"] * 29) + " stop."
    doc = Document(id="d", type="page", title="T", url="/", text=" ".join([sentence] * 15))
    body = [c for c in chunk_documents([doc]) if not c.id.endswith("#0")]
    assert len(body) >= 3
    assert all(words(c) <= 200 for c in body)
    assert all(c.text.endswith("stop.") for c in body)


def test_chunk_ids_are_sequential(docs):
    chunks = [c for c in chunk_documents(docs) if c.doc_id == "cst-438"]
    assert [c.id for c in chunks] == [f"cst-438#{i}" for i in range(len(chunks))]


def test_chunk_metadata_preserved(docs):
    for c in chunk_documents(docs):
        d = by_id(docs, c.doc_id)
        assert (c.url, c.status, c.type, c.title) == (d.url, d.status, d.type, d.title)


def test_whitespace_normalized():
    doc = Document(id="d", type="page", title="T", url="/", text="a  \t b    c")
    body = [c for c in chunk_documents([doc]) if not c.id.endswith("#0")]
    assert body[0].text == "T: a b c"
