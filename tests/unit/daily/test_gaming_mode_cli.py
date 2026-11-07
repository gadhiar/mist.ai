"""Tests for `instance.py gaming-mode on|off|status` (Interfaces and contracts, item 17, CLI).

The runner records every Docker call and the clock records every sleep in one shared event
list, so the tests assert the order of marker writes, sleeps and stops without sleeping.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.daily import instance

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "daily" / "instance.py"
FIXED_NOW = datetime(2026, 10, 23, 9, 30, 5, tzinfo=UTC)
FIXED_STAMP = "2026-10-23T09:30:05Z"
EARLIER = "2026-10-23T07:00:00Z"
TAG = "daily-2026.10.23"
MARKER = ".mist-live-instance"


class FakeRunner:
    """Answers each exact argv with a canned result; a list answers successive calls."""

    def __init__(self, responses, events: list, on_call=None):
        self.responses = dict(responses)
        self.calls: list[list[str]] = []
        self.events = events
        self.on_call = on_call

    def __call__(self, argv, *, stdin=None):
        self.calls.append(list(argv))
        self.events.append(("run", list(argv)))
        if self.on_call is not None:
            self.on_call(list(argv))
        key = tuple(argv)
        if key not in self.responses:
            raise AssertionError(f"unexpected command: {list(argv)}")
        answer = self.responses[key]
        if isinstance(answer, list):
            return answer.pop(0) if len(answer) > 1 else answer[0]
        return answer


class FakeClock:
    def __init__(self, events: list):
        self.now = 500.0
        self.sleeps: list[float] = []
        self.events = events

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.events.append(("sleep", seconds))
        self.now += seconds


def ok(stdout: str = "") -> instance.CommandResult:
    return instance.CommandResult(0, stdout.encode(), b"")


def fail(stderr: bytes = b"boom") -> instance.CommandResult:
    return instance.CommandResult(1, b"", stderr)


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "registry" / "live-instances"
    monkeypatch.setenv("MIST_LIVE_INSTANCES_FILE", str(path))
    return path


@pytest.fixture
def root(tmp_path) -> Path:
    path = (tmp_path / "instances" / "daily").resolve()
    for sub in ("state", "mist-memory", "logs", "ops", "bin"):
        (path / sub).mkdir(parents=True)
    marker = json.dumps({"v": 1, "instance": "daily", "created_at": "2026-10-01T00:00:00Z"})
    for directory in (path, path / "state", path / "mist-memory"):
        (directory / MARKER).write_text(marker + "\n", encoding="utf-8")
    compose = path / "release" / TAG / "deploy" / "daily" / "compose.yml"
    compose.parent.mkdir(parents=True)
    compose.write_text("name: mist-daily\n", encoding="utf-8")
    (path / "daily.env").write_text(
        f"MIST_RELEASE_TAG={TAG}\nMIST_DAILY_ROOT={path.as_posix()}\n", encoding="utf-8"
    )
    return path


def base(root: Path) -> list[str]:
    compose = root / "release" / TAG / "deploy" / "daily" / "compose.yml"
    return [
        "docker",
        "compose",
        "-p",
        "mist-daily",
        "-f",
        str(compose),
        "--env-file",
        str(root / "daily.env"),
    ]


def stop_llm(root: Path) -> tuple[str, ...]:
    return (*base(root), "stop", "mist-llm")


def start_llm(root: Path) -> tuple[str, ...]:
    return (*base(root), "up", "-d", "mist-llm")


def ps_llm(root: Path) -> tuple[str, ...]:
    return (*base(root), "ps", "--format", "json", "mist-llm")


def llm_ps(health: str) -> instance.CommandResult:
    return ok(json.dumps([{"Service": "mist-llm", "State": "running", "Health": health}]))


def marker_path(root: Path) -> Path:
    return root / "ops" / "gpu-yielded"


def read_marker(root: Path) -> dict:
    return json.loads(marker_path(root).read_text(encoding="utf-8"))


def write_marker(root: Path, by: str, since: str = EARLIER) -> None:
    marker_path(root).write_text(json.dumps({"v": 1, "since": since, "by": by}), encoding="utf-8")


class Harness:
    def __init__(self, root: Path, responses: dict, on_call=None):
        self.events: list = []
        self.runner = FakeRunner(responses, self.events, on_call)
        self.clock = FakeClock(self.events)
        self.ctx = instance.Context(
            runner=self.runner,
            clock=lambda: FIXED_NOW,
            home=root.parent,
            script_path=SCRIPT,
            sleep=self.clock.sleep,
            monotonic=self.clock.monotonic,
        )

    def run(self, *argv: str) -> int:
        return instance.main(list(argv), self.ctx)


# --- on --------------------------------------------------------------------------------------


def test_on_writes_the_marker_atomically_then_sleeps_two_polls_then_stops(root, monkeypatch):
    seen_at_stop: list[dict] = []

    def on_call(argv):
        seen_at_stop.append(read_marker(root))

    h = Harness(root, {stop_llm(root): ok()}, on_call)
    real_replace = instance.os.replace

    def recording_replace(src, dst):
        # The rename lands in the same event list as the sleeps and the runner calls, so the
        # order write, sleep, stop is pinned, not just each step on its own.
        h.events.append(("replace", Path(src).parent, Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(instance.os, "replace", recording_replace)
    assert h.run("gaming-mode", "on", "--root", str(root)) == 0

    assert h.events == [
        ("replace", root / "ops", marker_path(root)),
        ("sleep", 4.0),
        ("run", list(stop_llm(root))),
    ]
    assert read_marker(root) == {"v": 1, "since": FIXED_STAMP, "by": "manual"}
    assert seen_at_stop == [{"v": 1, "since": FIXED_STAMP, "by": "manual"}]


def test_on_sleeps_twice_the_poll_interval_daily_env_sets(root):
    with (root / "daily.env").open("a", encoding="utf-8") as handle:
        handle.write("MIST_GPU_YIELD_POLL_S=3.5\n")
    h = Harness(root, {stop_llm(root): ok()})
    assert h.run("gaming-mode", "on", "--root", str(root)) == 0
    assert h.clock.sleeps == [7.0]


def test_on_by_watcher_records_the_watcher(root):
    h = Harness(root, {stop_llm(root): ok()})
    assert h.run("gaming-mode", "on", "--by", "watcher", "--root", str(root)) == 0
    assert read_marker(root)["by"] == "watcher"


def test_a_second_on_keeps_since_and_by_skips_the_sleep_and_still_stops(root):
    write_marker(root, "manual")
    before = marker_path(root).read_bytes()
    h = Harness(root, {stop_llm(root): ok()})
    assert h.run("gaming-mode", "on", "--root", str(root)) == 0
    assert marker_path(root).read_bytes() == before
    assert h.clock.sleeps == []
    assert h.runner.calls == [list(stop_llm(root))]


def test_manual_on_over_a_watcher_marker_takes_it_over_and_keeps_since(root):
    write_marker(root, "watcher")
    h = Harness(root, {stop_llm(root): ok()})
    assert h.run("gaming-mode", "on", "--by", "manual", "--root", str(root)) == 0
    assert read_marker(root) == {"v": 1, "since": EARLIER, "by": "manual"}
    assert h.clock.sleeps == []


def test_watcher_on_over_a_manual_marker_leaves_it_unchanged(root):
    write_marker(root, "manual")
    before = marker_path(root).read_bytes()
    h = Harness(root, {stop_llm(root): ok()})
    assert h.run("gaming-mode", "on", "--by", "watcher", "--root", str(root)) == 0
    assert marker_path(root).read_bytes() == before
    assert h.runner.calls == [list(stop_llm(root))]


def test_a_stop_that_fails_once_is_retried_and_exits_0(root):
    h = Harness(root, {stop_llm(root): [fail(), ok()]})
    assert h.run("gaming-mode", "on", "--root", str(root)) == 0
    assert h.runner.calls == [list(stop_llm(root))] * 2
    assert marker_path(root).exists()


def test_a_stop_that_fails_twice_removes_the_marker_this_call_wrote(root, capsys):
    h = Harness(root, {stop_llm(root): [fail(b"daemon gone"), fail(b"daemon gone")]})
    assert h.run("gaming-mode", "on", "--root", str(root)) == 1
    assert h.runner.calls == [list(stop_llm(root))] * 2
    assert not marker_path(root).exists()
    assert "daemon gone" in capsys.readouterr().err


def test_a_stop_that_fails_twice_leaves_a_marker_that_was_already_there(root):
    write_marker(root, "watcher")
    before = marker_path(root).read_bytes()
    h = Harness(root, {stop_llm(root): fail()})
    assert h.run("gaming-mode", "on", "--by", "watcher", "--root", str(root)) == 1
    assert marker_path(root).read_bytes() == before


def test_on_refuses_without_a_release_and_writes_nothing(root):
    (root / "daily.env").write_text("MIST_RELEASE_TAG=daily-2026.09.01\n", encoding="utf-8")
    h = Harness(root, {})
    assert h.run("gaming-mode", "on", "--root", str(root)) == 2
    assert not marker_path(root).exists()
    assert h.runner.calls == []


# --- off -------------------------------------------------------------------------------------


def test_off_starts_mist_llm_polls_until_healthy_then_removes_the_marker(root):
    write_marker(root, "manual")
    present_at_poll: list[bool] = []

    def on_call(argv):
        if "ps" in argv:
            present_at_poll.append(marker_path(root).exists())

    responses = {
        start_llm(root): ok(),
        ps_llm(root): [llm_ps("starting"), llm_ps("starting"), llm_ps("healthy")],
    }
    h = Harness(root, responses, on_call)
    assert h.run("gaming-mode", "off", "--root", str(root)) == 0
    assert h.runner.calls == [list(start_llm(root))] + [list(ps_llm(root))] * 3
    assert present_at_poll == [True, True, True]
    assert not marker_path(root).exists()
    assert len(h.clock.sleeps) == 2


def test_off_reads_newline_delimited_ps_output(root):
    write_marker(root, "manual")
    line = json.dumps({"Service": "mist-llm", "State": "running", "Health": "healthy"})
    h = Harness(root, {start_llm(root): ok(), ps_llm(root): ok(line + "\n")})
    assert h.run("gaming-mode", "off", "--root", str(root)) == 0
    assert not marker_path(root).exists()


def test_off_with_mist_llm_never_healthy_exits_1_and_keeps_the_marker(root, capsys):
    write_marker(root, "manual")
    h = Harness(root, {start_llm(root): ok(), ps_llm(root): llm_ps("unhealthy")})
    code = h.run("gaming-mode", "off", "--root", str(root), "--wait-minutes", "3")
    assert code == 1
    assert marker_path(root).exists()
    assert 180 <= sum(h.clock.sleeps) < 190
    assert "mist-llm" in capsys.readouterr().err


def test_off_waits_five_minutes_by_default(root):
    write_marker(root, "manual")
    h = Harness(root, {start_llm(root): ok(), ps_llm(root): llm_ps("starting")})
    assert h.run("gaming-mode", "off", "--root", str(root)) == 1
    assert 300 <= sum(h.clock.sleeps) < 310


def test_off_with_a_failed_start_exits_1_and_keeps_the_marker(root):
    write_marker(root, "manual")
    h = Harness(root, {start_llm(root): fail()})
    assert h.run("gaming-mode", "off", "--root", str(root)) == 1
    assert marker_path(root).exists()


def test_off_with_no_marker_starts_mist_llm_and_exits_0(root):
    h = Harness(root, {start_llm(root): ok(), ps_llm(root): llm_ps("healthy")})
    assert h.run("gaming-mode", "off", "--root", str(root)) == 0
    assert h.runner.calls[0] == list(start_llm(root))
    assert not marker_path(root).exists()


def test_off_in_process_aborts_stops_mist_llm_and_keeps_the_marker(root):
    write_marker(root, "watcher")
    responses = {
        start_llm(root): ok(),
        ps_llm(root): llm_ps("starting"),
        stop_llm(root): ok(),
    }
    h = Harness(root, responses)
    checks = iter([False, True])
    outcome = instance.gaming_mode_off(
        root, h.ctx, wait_minutes=5, should_abort=lambda: next(checks)
    )
    assert outcome == "aborted"
    assert h.runner.calls[-1] == list(stop_llm(root))
    assert marker_path(root).exists()


def test_on_and_off_are_callable_in_process(root):
    h = Harness(
        root, {stop_llm(root): ok(), start_llm(root): ok(), ps_llm(root): llm_ps("healthy")}
    )
    instance.gaming_mode_on(root, h.ctx, by="watcher")
    assert read_marker(root)["by"] == "watcher"
    assert instance.gaming_mode_off(root, h.ctx, wait_minutes=1) == "done"
    assert not marker_path(root).exists()


# --- status ----------------------------------------------------------------------------------


def test_status_prints_normal_without_a_marker(root, capsys):
    h = Harness(root, {})
    assert h.run("gaming-mode", "status", "--root", str(root)) == 0
    assert capsys.readouterr().out == "gaming-mode: normal\n"
    assert h.runner.calls == []


def test_status_prints_yielded_with_since_and_by(root, capsys):
    write_marker(root, "watcher")
    h = Harness(root, {})
    assert h.run("gaming-mode", "status", "--root", str(root)) == 0
    assert capsys.readouterr().out == f"gaming-mode: yielded since {EARLIER} by watcher\n"


@pytest.mark.parametrize("body", ["{not json", "[]", '{"v": 1}', ""])
def test_status_reads_an_unparsable_marker_as_yielded_since_unknown(root, capsys, body):
    marker_path(root).write_text(body, encoding="utf-8")
    h = Harness(root, {})
    assert h.run("gaming-mode", "status", "--root", str(root)) == 0
    assert capsys.readouterr().out == "gaming-mode: yielded since unknown by unknown\n"
