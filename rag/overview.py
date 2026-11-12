"""Build two summary documents that answer "big picture" questions (D18).

Search returns only the top few chunks, so a question like "What languages
does he know?" or "What is he taking right now?" can't be answered from
single course pages. These documents gather that information in one place:

  overview-skills      each technology and the course codes that used it
  overview-coursework  courses and certifications grouped by status
"""

from rag.content import Document

# The intros use the words recruiters use, so these documents match their questions.
SKILLS_INTRO = (
    "Programming languages, frameworks, tools, and technologies Ariya knows and has used "
    "in his coursework, each with the course codes that used it."
)
COURSEWORK_INTRO = (
    "Courses Ariya has completed, courses he is currently taking right now, and courses "
    "he plans to take, plus the status of his certifications."
)
STATUS_LABELS = (("completed", "Completed"), ("in_progress", "In progress"), ("planned", "Planned"))
PARAGRAPH = "\n\n"


def _skills(courses: list[Document]) -> str:
    by_tech: dict[str, list[str]] = {}
    for course in sorted(courses, key=lambda d: d.title):
        code = " ".join(course.title.split()[:2])  # "CST 438"
        for tech in course.tech:
            by_tech.setdefault(tech, []).append(code)
    lines = [f"{tech}: {', '.join(codes)}." for tech, codes in sorted(by_tech.items(), key=lambda kv: kv[0].lower())]
    return PARAGRAPH.join([SKILLS_INTRO] + lines)


def _coursework(courses: list[Document], certs: list[Document]) -> str:
    lines = [COURSEWORK_INTRO]
    for status, label in STATUS_LABELS:
        titles = sorted(d.title for d in courses if d.status == status)
        if titles:
            lines.append(f"{label} courses: {', '.join(titles)}.")
    for status, label in STATUS_LABELS:
        titles = sorted(d.title for d in certs if d.status == status)
        if titles:
            lines.append(f"Certifications {label.lower()}: {', '.join(titles)}.")
    return PARAGRAPH.join(lines)


def overview_documents(docs: list[Document]) -> list[Document]:
    courses = [d for d in docs if d.type == "course"]
    certs = [d for d in docs if d.type == "certification"]
    if not courses and not certs:
        return []
    return [
        Document(id="overview-skills", type="page", title="Skills overview", url="/coursework/", text=_skills(courses)),
        Document(id="overview-coursework", type="page", title="Coursework overview", url="/coursework/",
                 text=_coursework(courses, certs)),
    ]
