from rag.limits import RateLimiter, client_key


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_allows_up_to_the_per_minute_limit():
    limiter = RateLimiter(per_minute=3, per_day=100, clock=Clock())
    assert [limiter.allow("a") for _ in range(4)] == [True, True, True, False]


def test_minute_window_slides():
    clock = Clock()
    limiter = RateLimiter(per_minute=2, per_day=100, clock=clock)
    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    clock.t = 61
    assert limiter.allow("a")


def test_daily_limit_resets_after_a_day():
    clock = Clock()
    limiter = RateLimiter(per_minute=100, per_day=5, clock=clock)
    for i in range(5):
        clock.t = i * 120  # spread out so only the daily limit applies
        assert limiter.allow("a")
    clock.t = 1000
    assert not limiter.allow("a")
    clock.t = 86_400 + 1000
    assert limiter.allow("a")


def test_visitors_are_counted_separately():
    limiter = RateLimiter(per_minute=1, per_day=100, clock=Clock())
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")


def test_refused_questions_do_not_count():
    clock = Clock()
    limiter = RateLimiter(per_minute=1, per_day=2, clock=clock)
    assert limiter.allow("a")
    for _ in range(5):
        assert not limiter.allow("a")  # refused: per-minute
    clock.t = 61
    assert limiter.allow("a")  # the 5 refusals didn't use up the daily allowance


def test_inactive_visitors_are_forgotten():
    clock = Clock()
    limiter = RateLimiter(per_minute=5, per_day=5, clock=clock)
    for i in range(50):
        limiter.allow(f"visitor-{i}")
    clock.t = 86_401
    limiter.allow("new")
    assert limiter.tracked_visitors() == 1


def test_client_key_prefers_the_first_forwarded_address():
    assert client_key({"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, "10.0.0.9") == "1.2.3.4"
    assert client_key({}, "10.0.0.9") == "10.0.0.9"
    assert client_key({}, None) == "unknown"
