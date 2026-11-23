"""Per-visitor question limits (decision D28).

All visitors share one Groq account with a daily request limit, and the
Space's /ask API is public. Without a per-visitor limit, one script could
use up the whole day's allowance and make the bot unavailable for everyone.
Each visitor (by IP address) gets `per_minute` and `per_day` questions that
reach the model; refused questions don't count.
"""

import threading
import time
from collections import deque
from typing import Callable, Mapping

MINUTE, DAY = 60.0, 86_400.0


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int, clock: Callable[[], float] = time.monotonic):
        self.per_minute = per_minute
        self.per_day = per_day
        self._clock = clock
        self._seen: dict[str, deque[float]] = {}  # visitor -> times of allowed questions (last 24 h)
        self._lock = threading.Lock()

    def allow(self, visitor: str) -> bool:
        """Record and allow a question, or refuse it if the visitor is over a limit."""
        now = self._clock()
        with self._lock:
            self._forget_inactive(now)
            times = self._seen.setdefault(visitor, deque())
            while times and now - times[0] >= DAY:
                times.popleft()
            last_minute = sum(1 for t in times if now - t < MINUTE)
            if len(times) >= self.per_day or last_minute >= self.per_minute:
                return False
            times.append(now)
            return True

    def tracked_visitors(self) -> int:
        return len(self._seen)

    def _forget_inactive(self, now: float) -> None:
        stale = [v for v, times in self._seen.items() if not times or now - times[-1] >= DAY]
        for visitor in stale:
            del self._seen[visitor]


def client_key(headers: Mapping[str, str], host: str | None) -> str:
    """The visitor's address. Behind Hugging Face's proxy, the first entry of
    X-Forwarded-For is the real client; otherwise use the connection's host."""
    forwarded = headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    return first or host or "unknown"
