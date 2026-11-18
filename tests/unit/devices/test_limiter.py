"""Tests for `backend/devices/limiter.py`: how fast a guessed token can be tried.

Docker Desktop's port publishing hides the client's address, so the limiter keys on the device
id the token claims, plus one global cap. Every test runs on a fake clock.
"""

from __future__ import annotations

import logging

import pytest

from backend.devices.limiter import AuthLimiter

LIMITER_LOGGER = "backend.devices.limiter"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def limiter(clock) -> AuthLimiter:
    return AuthLimiter(clock)


def fail(limiter: AuthLimiter, device_id: str | None, times: int) -> None:
    for _ in range(times):
        limiter.failed(device_id)


class TestPerId:
    def test_four_failures_leave_the_id_allowed(self, limiter):
        fail(limiter, "example-room", 4)

        assert limiter.allowed("example-room")

    def test_five_failures_in_60_s_block_that_id(self, limiter, clock):
        for _ in range(5):
            limiter.failed("example-room")
            clock.now += 14

        assert not limiter.allowed("example-room")

    def test_another_id_is_still_allowed(self, limiter):
        fail(limiter, "example-room", 5)

        assert limiter.allowed("other-room")

    def test_the_block_lasts_300_s(self, limiter, clock):
        fail(limiter, "example-room", 5)

        clock.now += 299.9
        blocked_late = limiter.allowed("example-room")
        clock.now += 0.1

        assert not blocked_late
        assert limiter.allowed("example-room")

    def test_failures_older_than_60_s_do_not_count(self, limiter, clock):
        fail(limiter, "example-room", 4)
        clock.now += 60.1

        limiter.failed("example-room")

        assert limiter.allowed("example-room")

    def test_an_unclaimed_id_counts_only_globally(self, limiter):
        fail(limiter, None, 5)

        assert limiter.allowed(None)
        assert limiter.allowed("example-room")


class TestGlobalPause:
    def test_30_failures_in_a_minute_pause_nothing(self, limiter):
        for index in range(30):
            limiter.failed(f"made-up-{index}")

        assert not limiter.paused()
        assert limiter.allowed("example-room")

    def test_31_failures_across_made_up_ids_pause_every_id_for_60_s(self, limiter, clock):
        for index in range(31):
            limiter.failed(f"made-up-{index}")

        paused = (limiter.paused(), limiter.allowed("example-room"), limiter.allowed(None))
        clock.now += 59.9
        still = limiter.allowed("example-room")
        clock.now += 0.1

        assert paused == (True, False, False)
        assert not still
        assert limiter.allowed("example-room")
        assert not limiter.paused()

    def test_the_pause_is_logged_once_with_its_failure_count(self, limiter, caplog):
        with caplog.at_level(logging.WARNING, logger=LIMITER_LOGGER):
            for index in range(40):
                limiter.failed(f"made-up-{index}")

        lines = [r.getMessage() for r in caplog.records if "device_auth_paused" in r.getMessage()]
        assert lines == ["device_auth_paused failures=31"]

    def test_failures_spread_over_more_than_a_minute_do_not_pause(self, limiter, clock):
        for index in range(60):
            limiter.failed(f"made-up-{index}")
            clock.now += 2.1

        assert not limiter.paused()

    def test_a_second_pause_after_the_first_ends_is_logged_again(self, limiter, clock, caplog):
        with caplog.at_level(logging.WARNING, logger=LIMITER_LOGGER):
            fail_many(limiter, 31, prefix="first")
            clock.now += 61
            fail_many(limiter, 31, prefix="second")

        lines = [r for r in caplog.records if "device_auth_paused" in r.getMessage()]
        assert len(lines) == 2


def fail_many(limiter: AuthLimiter, count: int, *, prefix: str) -> None:
    for index in range(count):
        limiter.failed(f"{prefix}-{index}")


def test_old_ids_are_forgotten_so_made_up_ids_cannot_grow_memory(limiter, clock):
    for index in range(20):
        limiter.failed(f"made-up-{index}")
    clock.now += 400

    limiter.failed("example-room")

    assert limiter.tracked_ids() == 1
