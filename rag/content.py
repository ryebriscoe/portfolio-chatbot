"""Load the site's content export (chatbot-content.json) into Documents."""

import html
import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable


class ContentUnavailable(Exception):
    """The content export could not be loaded after all retries."""


@dataclass(frozen=True)
class Document:
    id: str
    type: str
    title: str
    url: str
    text: str
    status: str | None = None
    term: str | None = None
    project: str | None = None
    repo: str | None = None
    catalog: str | None = None
    tech: tuple[str, ...] = ()


def _fetch(source: str) -> str:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=10) as resp:
            return resp.read().decode("utf-8")
    with open(source, encoding="utf-8") as f:
        return f.read()


def _clean(value):
    return html.unescape(value) if isinstance(value, str) else value


def _to_document(raw: dict) -> Document:
    optional = {k: _clean(raw.get(k)) or None for k in ("status", "term", "project", "repo", "catalog")}
    return Document(
        id=raw["id"],
        type=raw["type"],
        title=_clean(raw["title"]),
        url=raw["url"],
        text=_clean(raw.get("text") or ""),
        tech=tuple(_clean(t) for t in raw.get("tech") or ()),
        **optional,
    )


def load_documents(
    source: str,
    *,
    retries: int = 3,
    fetch: Callable[[str], str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Document]:
    fetch = fetch or _fetch
    last_error = None
    for attempt in range(retries):
        try:
            data = json.loads(fetch(source))
            return [_to_document(d) for d in data["documents"]]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            last_error = exc
            if attempt < retries - 1:
                sleep(2 ** attempt)
    raise ContentUnavailable(f"could not load {source}: {last_error}")
