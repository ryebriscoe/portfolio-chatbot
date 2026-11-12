"""Prompt assembly, input validation, and source selection (decision D11)."""

from rag.chunker import Chunk

SYSTEM_PROMPT = """You answer questions from recruiters about Ariya Briscoe, a Computer Science
(Software Engineering) student at CSU Monterey Bay graduating Fall 2026.
Refer to him as "he" or "Ariya".

Rules:
1. Use ONLY the facts in the CONTEXT below. Never add skills, jobs, projects,
   or technologies that are not in the context.
2. If the context does not answer the question, say: "That isn't covered on
   Ariya's portfolio. You can reach him at ariyabriscoe@gmail.com or LinkedIn."
3. Keep answers to 2-4 sentences. Name specific projects, courses, and tech.
4. Respect status: completed work is past tense; in-progress courses are
   "currently taking"; planned courses are "plans to take".
5. Only discuss Ariya's education, projects, skills, and certifications.
   Politely decline anything else (general coding help, opinions, salary).
6. Do not include URLs; sources are shown separately.

CONTEXT:"""

NO_CONTENT = "(no relevant content found on the site)"
MAX_TURN_CHARS = 2000

Hit = tuple[Chunk, float]


class InvalidQuestion(ValueError):
    """Raised with message "empty" or "too long"."""


def validate_question(question: object, max_chars: int) -> str:
    if not isinstance(question, str) or not question.strip():
        raise InvalidQuestion("empty")
    question = question.strip()
    if len(question) > max_chars:
        raise InvalidQuestion("too long")
    return question


def sanitize_history(raw: object, turns: int, max_turn_chars: int = MAX_TURN_CHARS) -> list[tuple[str, str]]:
    if not isinstance(raw, list):
        return []
    clean = [
        (item[0][:max_turn_chars], item[1][:max_turn_chars])
        for item in raw
        if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, str) for x in item)
    ]
    return clean[-turns:] if turns > 0 else []


def retrieval_query(question: str, history: list[tuple[str, str]]) -> str:
    return f"{history[-1][0]}\n{question}" if history else question


def format_context(hits: list[Hit]) -> str:
    if not hits:
        return NO_CONTENT
    return "\n\n".join(f"[{c.title} | {c.status or c.type}]\n{c.text}" for c, _ in hits)


def build_messages(question: str, history: list[tuple[str, str]], hits: list[Hit]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n{format_context(hits)}"}]
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": question})
    return messages


def select_sources(hits: list[Hit], max_sources: int) -> list[dict[str, str]]:
    sources, seen = [], set()
    for c, _ in hits:
        if c.doc_id in seen:
            continue
        seen.add(c.doc_id)
        sources.append({"id": c.doc_id, "title": c.title, "url": c.url})
        if len(sources) == max_sources:
            break
    return sources
