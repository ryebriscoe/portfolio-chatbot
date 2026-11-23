import threading

from rag.refresh import RefreshingIndex


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def builder(*results):
    """Returns a build function that yields each result in turn (raising exceptions)."""
    calls = []

    def build():
        calls.append(len(calls))
        result = results[min(len(calls), len(results)) - 1]
        if isinstance(result, Exception):
            raise result
        return result

    return build, calls


def test_builds_once_at_start():
    build, calls = builder("a")
    holder = RefreshingIndex(build, max_age=60, retry_after=10, clock=Clock(), background=False)
    assert holder.get() == "a"
    assert holder.get() == "a"
    assert len(calls) == 1


def test_rebuilds_after_max_age():
    clock = Clock()
    build, calls = builder("a", "b")
    holder = RefreshingIndex(build, max_age=60, retry_after=10, clock=clock, background=False)
    clock.t = 59
    assert holder.get() == "a"
    clock.t = 61
    assert holder.get() == "b"
    assert len(calls) == 2


def test_failed_rebuild_keeps_the_old_index_and_retries_later():
    clock = Clock()
    build, calls = builder("a", OSError("site down"), "c")
    holder = RefreshingIndex(build, max_age=60, retry_after=10, clock=clock, background=False)
    clock.t = 61
    assert holder.get() == "a"  # rebuild failed: keep serving "a"
    clock.t = 65
    assert holder.get() == "a" and len(calls) == 2  # waits retry_after before trying again
    clock.t = 72
    assert holder.get() == "c"


def test_failed_first_build_recovers_after_retry():
    clock = Clock()
    build, calls = builder(OSError("site down"), "a")
    holder = RefreshingIndex(build, max_age=60, retry_after=10, clock=clock, background=False)
    assert holder.get() is None
    clock.t = 11
    assert holder.get() == "a"


def test_background_rebuild_serves_the_old_index_meanwhile():
    clock = Clock()
    release = threading.Event()
    results = iter(["a", "b"])

    def build():
        value = next(results)
        if value == "b":
            release.wait(5)
        return value

    holder = RefreshingIndex(build, max_age=60, retry_after=10, clock=clock, background=True)
    clock.t = 61
    assert holder.get() == "a"  # rebuild started in the background
    assert holder.get() == "a"  # still rebuilding: no second rebuild starts
    release.set()
    holder.wait()
    assert holder.get() == "b"
