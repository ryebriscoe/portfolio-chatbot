from rag.overview import overview_documents


def by_id(docs, doc_id):
    return next(d for d in docs if d.id == doc_id)


def test_builds_two_overview_documents(docs):
    extra = overview_documents(docs)
    assert [(d.id, d.type, d.url) for d in extra] == [
        ("overview-skills", "page", "/coursework/"),
        ("overview-coursework", "page", "/coursework/"),
    ]


def test_skills_overview_lists_courses_per_technology(docs):
    text = by_id(overview_documents(docs), "overview-skills").text
    assert "Java: CST 338, CST 438." in text
    assert "JavaFX: CST 338." in text
    # certifications list topics, not hands-on tech, so they are left out
    assert "Deep learning" not in text


def test_coursework_overview_groups_by_status(docs):
    text = by_id(overview_documents(docs), "overview-coursework").text
    assert "Completed courses: CST 338 Software Design, CST 438 Software Engineering." in text
    assert "Planned courses: CST 499 Computer Science Capstone." in text
    assert "Certifications in progress: IBM AI Engineering." in text
    assert "In progress courses" not in text  # no empty groups


def test_no_documents_gives_no_overviews():
    assert overview_documents([]) == []


def test_overviews_open_with_the_words_recruiters_use(docs):
    extra = overview_documents(docs)
    assert by_id(extra, "overview-skills").text.startswith("Programming languages, frameworks, tools")
    assert "currently taking right now" in by_id(extra, "overview-coursework").text.split("\n\n")[0]
