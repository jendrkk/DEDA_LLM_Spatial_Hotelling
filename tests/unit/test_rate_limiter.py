import pytest

import hotelling.llm.client as mod
from hotelling.llm.client import RateLimiter, RateLimitExceeded


def test_daily_limit_raises():
    rl = RateLimiter(requests_per_minute=0, requests_per_day=3)  # rpm disabled
    for _ in range(3):
        rl.acquire()
    with pytest.raises(RateLimitExceeded):
        rl.acquire()


def test_minute_limit_sleeps(monkeypatch):
    clock = {"t": 1000.0}
    slept = []
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])

    def fake_sleep(s):
        slept.append(s)
        clock["t"] += s

    monkeypatch.setattr(mod.time, "sleep", fake_sleep)

    rl = RateLimiter(requests_per_minute=2, requests_per_day=0)  # rpd disabled
    rl.acquire()
    rl.acquire()
    assert slept == []          # first 2 calls in the window: no throttle
    rl.acquire()                # 3rd must wait until the oldest ages out (~60 s)
    assert len(slept) == 1 and abs(slept[0] - 60.05) < 1e-6
