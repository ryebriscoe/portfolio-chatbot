"""Keep the search index in step with the site (decision D27).

The Space loads chatbot-content.json at startup. With answers coming from
Groq (D23) the Space rarely restarts, so the index is rebuilt in the
background once it is older than `max_age`. Visitors keep getting answers
from the old index while it rebuilds. If a rebuild fails (for example the
site is briefly down), the old index stays and another try happens after
`retry_after`. The same retry also recovers from a failed first build.
"""

import logging
import threading
import time
from typing import Callable, Generic, TypeVar

log = logging.getLogger("portfolio-chatbot")
T = TypeVar("T")


class RefreshingIndex(Generic[T]):
    def __init__(self, build: Callable[[], T], max_age: float, retry_after: float,
                 clock: Callable[[], float] = time.monotonic, background: bool = True):
        self._build = build
        self._max_age = max_age
        self._retry_after = retry_after
        self._clock = clock
        self._background = background
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._index: T | None = None
        self._next_build = 0.0
        self._rebuild()  # first build at startup, in the foreground

    def get(self) -> T | None:
        """The current index (None until a build has succeeded)."""
        with self._lock:
            due = self._clock() >= self._next_build and not (self._thread and self._thread.is_alive())
            if due and self._background:
                self._thread = threading.Thread(target=self._rebuild, daemon=True)
                self._thread.start()
        if due and not self._background:
            self._rebuild()
        return self._index

    def wait(self) -> None:
        """Block until a background rebuild finishes (used by tests)."""
        if self._thread:
            self._thread.join()

    def _rebuild(self) -> None:
        try:
            index = self._build()
        except Exception:  # keep serving the old index; try again soon
            log.exception("content refresh failed; retrying in %ss", self._retry_after)
            self._next_build = self._clock() + self._retry_after
            return
        self._index = index
        self._next_build = self._clock() + self._max_age
        log.info("search index rebuilt from the site content")
