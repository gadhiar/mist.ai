"""Tests for the gaming watcher: `scripts/daily/gamewatch.py` and `instance.py gaming-watch`.

`classify` and `decide` are pure, so their rules and sequences run on plain values; the case
table is in `test_gamewatch_cases.py`. The loop runs with the injected runner, clock and
process lister, so nothing lists real processes, runs Docker or sleeps. Every path is
synthetic, on the drives `X:` and `Y:` or under the shared system folders.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.daily import gamewatch, instance

REPO_ROOT = Path(__file__).resolve().parents[3]
DAILY_SCRIPTS = REPO_ROOT / "scripts" / "daily"
SCRIPT = DAILY_SCRIPTS / "instance.py"
GAMEWATCH = DAILY_SCRIPTS / "gamewatch.py"
ENV_EXAMPLE = REPO_ROOT / "deploy" / "daily" / "daily.env.example"
README = REPO_ROOT / "deploy" / "daily" / "README.md"
FIXTURE_FILES = (Path(__file__), Path(__file__).with_name("test_gamewatch_cases.py"))

FIXED_NOW = datetime(2026, 10, 23, 9, 30, 5, tzinfo=UTC)
TAG = "daily-2026.10.23"
MARKER = ".mist-live-instance"

LIBRARIES_LINE = "X:\\Library;X:\\Other Library"
RUNTIME_LINE = "Y:\\Servers"
GAME = gamewatch.ProcessRecord("X:\\Other Library\\Example Game\\examplegame.exe")
OTHER = gamewatch.ProcessRecord("C:\\Windows\\System32\\svchost.exe")
SENTINEL = "zq9sentinel7token"
JAVA = "C:\\Program Files\\Java\\jdk-21\\bin\\java.exe"
SERVER = gamewatch.ProcessRecord(
    JAVA,
    cwd="Y:\\Servers\\Example Server",
    args=("-Xmx4G", "--accessToken", SENTINEL, "-jar", "server.jar", "nogui"),
)
DELAY = 120.0


def rules(libraries=(), runtime_dirs=(), exclude_dirs=(), exclude_names=()) -> gamewatch.GameRules:
    return gamewatch.GameRules(
        libraries=tuple(libraries),
        runtime_dirs=tuple(runtime_dirs),
        exclude_dirs=tuple(exclude_dirs),
        exclude_names=tuple(exclude_names),
    )


def state(last=None, present=False, suppressed=False) -> gamewatch.WatchState:
    return gamewatch.WatchState(
        last_seen_game_at=last, marker_was_present=present, suppressed=suppressed
    )


def decide(previous, game, marker, now=1000.0, delay=DELAY):
    return gamewatch.decide(previous, game, marker, now, delay)


# --- Module rules ----------------------------------------------------------------------------


def test_gamewatch_imports_only_the_standard_library():
    imported = []
    for node in ast.walk(ast.parse(GAMEWATCH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    outside = [name for name in imported if name.split(".")[0] not in sys.stdlib_module_names]
    assert outside == []


def test_list_processes_raises_off_windows(monkeypatch):
    monkeypatch.setattr(gamewatch.os, "name", "posix")
    with pytest.raises(gamewatch.GameWatchError):
        gamewatch.list_processes()


def test_the_v1_path_matching_api_is_gone():
    for name in ("list_image_paths", "match_index", "game_dirs_from_env"):
        assert not hasattr(gamewatch, name), name


def test_is_64_bit_reads_the_pointer_size(monkeypatch):
    monkeypatch.setattr(gamewatch.struct, "calcsize", lambda fmt: 8 if fmt == "P" else 0)
    assert gamewatch.is_64_bit() is True
    monkeypatch.setattr(gamewatch.struct, "calcsize", lambda fmt: 4 if fmt == "P" else 0)
    assert gamewatch.is_64_bit() is False


@pytest.mark.parametrize(
    "value",
    [
        state(),
        gamewatch.ProcessRecord("X:\\a.exe"),
        rules(),
        gamewatch.Verdict(counts=False, variable=None, index=None, folder=None, reason="outside"),
    ],
    ids=["watch-state", "process-record", "game-rules", "verdict"],
)
def test_the_data_types_are_frozen_with_slots(value):
    field = dataclasses.fields(value)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, field, None)
    assert not hasattr(value, "__dict__")


def test_a_process_record_reads_nothing_beyond_its_image_by_default():
    assert gamewatch.ProcessRecord("X:\\a.exe") == gamewatch.ProcessRecord(
        image="X:\\a.exe", cwd=None, args=None
    )


# --- The shipped exclusion table -------------------------------------------------------------


def test_the_shared_runtimes_are_the_two_java_launchers():
    assert gamewatch.SHARED_RUNTIMES == ("java.exe", "javaw.exe")


@pytest.mark.parametrize("pattern", gamewatch.SHIPPED_EXCLUDED_NAMES)
def test_every_shipped_name_is_a_lower_case_exe_pattern(pattern):
    assert pattern == pattern.lower()
    assert pattern.endswith(".exe")
    assert not re.search(r"[\\/:]", pattern)


@pytest.mark.parametrize("folder", gamewatch.SHIPPED_EXCLUDED_FOLDERS)
def test_no_shipped_folder_holds_a_drive_or_a_users_component(folder):
    assert ":" not in folder
    assert folder == folder.lower()
    assert "users" not in folder.split("\\")
    assert all(part.strip() == part and part for part in folder.split("\\"))


DRIVE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
SHARED_SYSTEM_PATHS = ("c:\\program files", "c:\\windows")


def string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def foreign_drives(texts: list[str]) -> list[str]:
    offending = []
    for text in texts:
        for match in DRIVE.finditer(text):
            if match.group(0)[0].upper() in "XY":
                continue
            if text[match.start() :].lower().startswith(SHARED_SYSTEM_PATHS):
                continue
            offending.append(text)
    return offending


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda path: path.name)
def test_fixtures_use_only_the_synthetic_drives_and_the_shared_system_paths(path):
    """A fixture on another drive could only have come from a real machine."""
    assert foreign_drives(string_constants(path)) == []


def test_the_drive_check_catches_a_path_on_another_drive():
    # Built at run time, so this file's own constants stay clean.
    real_looking = "D" + ":\\Some Folder\\x.exe"
    assert foreign_drives([real_looking]) == [real_looking]
    assert foreign_drives(["C" + ":\\Program Files (x86)\\x.exe"]) == []
    assert foreign_drives(["C" + ":\\Games\\x.exe", "[TimeSpan]::Zero"]) == [
        "C" + ":\\Games\\x.exe"
    ]


# --- classify: matching, beyond the case table -----------------------------------------------


def test_a_sibling_directory_with_a_longer_name_is_outside():
    verdict = gamewatch.classify(
        gamewatch.ProcessRecord("X:\\Games2\\x.exe"), rules(libraries=["X:\\Games"])
    )
    assert (verdict.counts, verdict.reason) == (False, "outside")


def test_matching_ignores_case_and_a_trailing_backslash():
    verdict = gamewatch.classify(
        gamewatch.ProcessRecord("x:\\games\\X\\y.EXE"), rules(libraries=["X:\\Games\\"])
    )
    assert (verdict.counts, verdict.variable, verdict.index) == (True, "MIST_GAME_DIRS", 0)


def test_a_drive_root_holds_everything_on_its_drive():
    root_rules = rules(libraries=["X:\\"])
    assert gamewatch.classify(gamewatch.ProcessRecord("X:\\any\\game.exe"), root_rules).counts
    assert not gamewatch.classify(gamewatch.ProcessRecord("Y:\\any\\game.exe"), root_rules).counts


def test_forward_slashes_and_dot_segments_are_normalised():
    verdict = gamewatch.classify(
        gamewatch.ProcessRecord("X:/Games/./a/../b/game.exe"), rules(libraries=["X:\\Games"])
    )
    assert verdict.counts is True


def test_a_path_under_two_nested_libraries_names_the_lower_index():
    verdict = gamewatch.classify(
        gamewatch.ProcessRecord("X:\\Games\\Inner\\game.exe"),
        rules(libraries=["X:\\Games\\Inner", "X:\\Games"]),
    )
    assert verdict.index == 0


def test_a_runtime_folder_under_two_nested_entries_names_the_lower_index():
    nested = rules(runtime_dirs=["Y:\\Servers", "Y:\\Packs\\Instances", "Y:\\Other", "Y:\\Packs"])
    record = gamewatch.ProcessRecord(JAVA, args=("--gameDir", "Y:\\Packs\\Instances\\a"))
    verdict = gamewatch.classify(record, nested)
    assert (verdict.variable, verdict.index) == ("MIST_GAME_RUNTIME_DIRS", 1)


def test_the_runtime_directory_itself_is_a_runtime_folder():
    record = gamewatch.ProcessRecord(JAVA, cwd="Y:\\Servers")
    assert gamewatch.classify(record, rules(runtime_dirs=["Y:\\Servers"])).counts is True


def test_a_runtime_folder_under_an_excluded_directory_does_not_count():
    excluded = rules(runtime_dirs=["Y:\\Servers"], exclude_dirs=["", "Y:\\Servers\\Test"])
    record = gamewatch.ProcessRecord(JAVA, args=("--gameDir", "Y:\\Servers\\Test\\inner"))
    assert gamewatch.classify(record, excluded).reason == "MIST_GAME_EXCLUDE[1]"


def test_a_runtime_under_a_library_removed_by_a_host_exclusion_reports_the_exclusion():
    lib_rules = rules(libraries=["X:\\Games"], exclude_dirs=["X:\\Games\\Tool"])
    record = gamewatch.ProcessRecord("X:\\Games\\Tool\\jre\\bin\\java.exe", args=())
    verdict = gamewatch.classify(record, lib_rules)
    assert (verdict.counts, verdict.reason) == (False, "MIST_GAME_EXCLUDE[0]")


def test_a_non_runtime_program_is_never_judged_by_its_folder():
    """Only `java.exe` and `javaw.exe` are opened for a working directory or arguments."""
    record = gamewatch.ProcessRecord("Y:\\Servers\\a\\run.exe", cwd="Y:\\Servers\\a")
    assert gamewatch.classify(record, rules(runtime_dirs=["Y:\\Servers"])).reason == "outside"


def test_the_first_matching_exclusion_wins_shipped_folders_then_names_then_host():
    host = rules(
        libraries=["X:\\Games"],
        exclude_dirs=["X:\\Games\\SteamVR"],
        exclude_names=["", "*launcher*.exe"],
    )
    record = gamewatch.ProcessRecord("X:\\Games\\SteamVR\\Launcher.exe")
    assert gamewatch.classify(record, host).reason == "shipped folder steamvr"
    record = gamewatch.ProcessRecord("X:\\Games\\Tool\\Launcher.exe")
    assert gamewatch.classify(record, host).reason == "shipped name *launcher*.exe"


def test_a_shipped_folder_must_match_whole_consecutive_components():
    lib = rules(libraries=["X:\\Games"])
    for image in ("X:\\Games\\riot clients\\a.exe", "X:\\Games\\epic games\\x\\launcher\\a.exe"):
        assert gamewatch.classify(gamewatch.ProcessRecord(image), lib).counts is True, image


# --- find_game -------------------------------------------------------------------------------

FIND_RULES = rules(libraries=["X:\\Library", "X:\\Other Library"], runtime_dirs=["Y:\\Servers"])


def test_find_game_prefers_a_library_game_to_a_runtime_game():
    verdict = gamewatch.find_game([OTHER, SERVER, GAME], FIND_RULES)
    assert verdict is not None
    assert (verdict.variable, verdict.index) == ("MIST_GAME_DIRS", 1)


def test_find_game_takes_the_lowest_index():
    first = gamewatch.ProcessRecord("X:\\Library\\a\\a.exe")
    verdict = gamewatch.find_game([GAME, first], FIND_RULES)
    assert verdict is not None
    assert verdict.index == 0


def test_find_game_finds_a_runtime_game_alone():
    verdict = gamewatch.find_game([OTHER, SERVER], FIND_RULES)
    assert verdict is not None
    assert (verdict.variable, verdict.index) == ("MIST_GAME_RUNTIME_DIRS", 0)


def test_find_game_is_none_when_nothing_counts():
    launcher = gamewatch.ProcessRecord("X:\\Library\\a\\Launcher.exe")
    assert gamewatch.find_game([OTHER, launcher], FIND_RULES) is None
    assert gamewatch.find_game([], FIND_RULES) is None


# --- decide, one case per rule of item 17 ----------------------------------------------------


def test_a_game_sets_last_seen_to_now():
    _action, new = decide(state(last=10.0), True, None, now=50.0)
    assert new.last_seen_game_at == 50.0


def test_no_game_keeps_last_seen():
    _action, new = decide(state(last=10.0, present=True), False, "manual", now=50.0)
    assert new.last_seen_game_at == 10.0


def test_marker_gone_since_the_last_tick_with_a_game_suppresses():
    action, new = decide(state(last=10.0, present=True), True, None)
    assert action is None
    assert new.suppressed is True


def test_marker_gone_since_the_last_tick_without_a_game_does_not_suppress():
    action, new = decide(state(last=10.0, present=True), False, None)
    assert action is None
    assert new.suppressed is False


def test_no_marker_a_game_and_not_suppressed_turns_on():
    action, new = decide(state(), True, None)
    assert action == "on"
    assert new == state(last=1000.0, present=False, suppressed=False)


def test_no_marker_a_game_and_suppressed_stays_off():
    action, new = decide(state(last=990.0, suppressed=True), True, None)
    assert action is None
    assert new.suppressed is True


@pytest.mark.parametrize("marker", [None, "manual", "watcher"])
def test_no_game_clears_suppressed(marker):
    _action, new = decide(
        state(last=999.0, present=marker is not None, suppressed=True), False, marker
    )
    assert new.suppressed is False


def test_watcher_marker_no_game_and_last_seen_older_than_the_delay_turns_off():
    action, new = decide(state(last=1000.0 - DELAY, present=True), False, "watcher")
    assert action == "off"
    assert new.marker_was_present is False


def test_watcher_marker_no_game_and_never_seen_a_game_turns_off():
    action, new = decide(state(present=True), False, "watcher")
    assert action == "off"
    assert new.marker_was_present is False


def test_watcher_marker_no_game_within_the_delay_waits():
    action, new = decide(state(last=1000.0 - DELAY + 1, present=True), False, "watcher")
    assert action is None
    assert new.marker_was_present is True


def test_watcher_marker_with_a_game_does_nothing():
    action, new = decide(state(last=1.0, present=True), True, "watcher")
    assert action is None
    assert new == state(last=1000.0, present=True, suppressed=False)


@pytest.mark.parametrize("game", [True, False])
@pytest.mark.parametrize("last", [None, 0.0, 999.0])
def test_manual_marker_never_acts(game, last):
    action, new = decide(state(last=last, present=True), game, "manual")
    assert action is None
    assert new.marker_was_present is True


@pytest.mark.parametrize("game", [True, False])
def test_an_unreadable_marker_never_acts(game):
    """A marker that does not parse still yields the backend, so it is nobody's to end."""
    action, new = decide(state(present=True), game, "unknown")
    assert action is None
    assert new.marker_was_present is True


@pytest.mark.parametrize(
    ("marker", "game", "present"),
    [(None, False, False), (None, True, False), ("manual", False, True), ("watcher", True, True)],
)
def test_marker_was_present_is_whether_a_marker_exists_now(marker, game, present):
    for previous in (state(present=True), state(present=False)):
        _action, new = decide(previous, game, marker)
        assert new.marker_was_present is present


# --- decide, sequences -----------------------------------------------------------------------


def run_ticks(ticks, *, marker=None, on_works=True, delay=DELAY):
    """Feed `decide` one tick at a time, applying the watcher's own commands to the marker.

    Each tick is `(now, game)` or `(now, game, marker_override)`; an override is what someone
    else did to the marker since the last tick.
    """
    current = state()
    actions = []
    for tick in ticks:
        now, game = tick[0], tick[1]
        if len(tick) == 3:
            marker = tick[2]
        action, current = gamewatch.decide(current, game, marker, now, delay)
        actions.append(action)
        if action == "on" and on_works and marker is None:
            marker = "watcher"
        elif action == "off":
            marker = None
    return actions


def test_a_game_starts_runs_and_closes():
    ticks = [(0.0, True), (10.0, True), (20.0, True)]
    ticks += [(t, False) for t in range(30, 140, 10)]
    ticks += [(140.0, False), (150.0, False)]
    actions = run_ticks(ticks)
    assert actions[0] == "on"
    assert actions[1:-2] == [None] * (len(ticks) - 3)
    # The last game was seen at 20, so the delay of 120 s has passed at 140.
    assert actions[-2:] == ["off", None]


def test_a_manual_on_then_the_game_closes_never_turns_off():
    ticks = [(0.0, True), (10.0, True)] + [(t, False) for t in range(20, 1000, 10)]
    assert set(run_ticks(ticks, marker="manual")) == {None}


def test_a_manual_off_while_the_game_runs_holds_until_a_tick_with_no_game():
    actions = run_ticks(
        [
            (0.0, True),
            (10.0, True),
            (20.0, True, None),  # Someone ran gaming-mode off.
            (30.0, True),
            (40.0, True),
            (50.0, False),
            (60.0, True),
        ]
    )
    assert actions == ["on", None, None, None, None, None, "on"]


def test_a_failed_on_is_tried_again_on_the_next_tick():
    actions = run_ticks([(0.0, True), (10.0, True), (20.0, True)], on_works=False)
    assert actions == ["on", "on", "on"]


# --- Parsing ---------------------------------------------------------------------------------

VARIABLES = ("MIST_GAME_DIRS", "MIST_GAME_RUNTIME_DIRS", "MIST_GAME_EXCLUDE")
TWO_DIRS = ["C:\\Program Files\\Games", "X:\\Steam Library\\steamapps\\common"]
TWO_DIRS_LINE = "C:\\Program Files\\Games;X:\\Steam Library\\steamapps\\common"


def parsed(rules_value: gamewatch.GameRules, variable: str) -> list[str]:
    if variable == "MIST_GAME_DIRS":
        return list(rules_value.libraries)
    if variable == "MIST_GAME_RUNTIME_DIRS":
        return list(rules_value.runtime_dirs)
    return [entry for entry in rules_value.exclude_dirs if entry]


@pytest.mark.parametrize("variable", VARIABLES)
@pytest.mark.parametrize(
    "value",
    [TWO_DIRS_LINE, f'"{TWO_DIRS_LINE}"', f"{TWO_DIRS_LINE};", f" {TWO_DIRS_LINE} ; "],
    ids=["unquoted", "quoted", "trailing-separator", "spaced"],
)
def test_each_variable_keeps_spaces_and_backslashes(variable, value):
    text = f"# {variable}=<commented>\nMIST_RELEASE_TAG={TAG}\n  {variable} = {value}\n"
    assert parsed(gamewatch.rules_from_env(text), variable) == TWO_DIRS


@pytest.mark.parametrize("variable", VARIABLES)
@pytest.mark.parametrize("line", ["{v}=", "{v}=;", '{v}=""', "OTHER=1", "", "# {v}=X:\\a"])
def test_an_empty_unset_or_commented_variable_gives_none(variable, line):
    result = gamewatch.rules_from_env(line.format(v=variable) + "\n")
    assert result == rules()


@pytest.mark.parametrize("variable", VARIABLES)
def test_the_last_line_of_a_variable_wins(variable):
    text = f"{variable}=X:\\Old\n{variable}={TWO_DIRS_LINE}\n"
    assert parsed(gamewatch.rules_from_env(text), variable) == TWO_DIRS


def test_only_one_pair_of_outer_quotes_is_stripped():
    assert gamewatch.parse_game_dirs("\"'X:\\Games'\"") == ["'X:\\Games'"]


def test_an_exclude_entry_with_a_separator_or_colon_is_a_directory_and_others_are_names():
    text = "MIST_GAME_EXCLUDE=X:\\Tools;Mod_Tool.exe;tools/editor;Y:;Save*Editor.EXE\n"
    result = gamewatch.rules_from_env(text)
    assert result.exclude_dirs == ("X:\\Tools", "", "tools/editor", "Y:", "")
    assert result.exclude_names == ("", "mod_tool.exe", "", "", "save*editor.exe")
    assert result.libraries == () and result.runtime_dirs == ()


def test_an_exclude_name_pattern_matches_ignoring_case():
    host = gamewatch.rules_from_env("MIST_GAME_DIRS=X:\\Games\nMIST_GAME_EXCLUDE=*Editor*.exe\n")
    verdict = gamewatch.classify(gamewatch.ProcessRecord("X:\\Games\\a\\SaveEDITOR.exe"), host)
    assert verdict.reason == "MIST_GAME_EXCLUDE[0]"


# --- gaming-watch ----------------------------------------------------------------------------


class FakeRunner:
    """Answers each exact argv with a canned result; a list answers successive calls."""

    def __init__(self, responses, events: list):
        self.responses = dict(responses)
        self.calls: list[list[str]] = []
        self.events = events

    def __call__(self, argv, *, stdin=None):
        self.calls.append(list(argv))
        self.events.append(("run", list(argv)))
        key = tuple(argv)
        if key not in self.responses:
            raise AssertionError(f"unexpected command: {list(argv)}")
        answer = self.responses[key]
        if isinstance(answer, list):
            return answer.pop(0) if len(answer) > 1 else answer[0]
        return answer


class FakeClock:
    def __init__(self, events: list):
        self.now = 1000.0
        self.sleeps: list[float] = []
        self.events = events

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.events.append(("sleep", seconds))
        self.now += seconds


class FakeLister:
    """Returns the next canned process list on each call, then repeats the last."""

    def __init__(self, lists, events: list):
        self.lists = list(lists)
        self.calls = 0
        self.events = events

    def __call__(self) -> list[gamewatch.ProcessRecord]:
        self.calls += 1
        self.events.append(("list",))
        return self.lists.pop(0) if len(self.lists) > 1 else self.lists[0]


def ok(stdout: str = "") -> instance.CommandResult:
    return instance.CommandResult(0, stdout.encode(), b"")


def fail(stderr: bytes = b"boom") -> instance.CommandResult:
    return instance.CommandResult(1, b"", stderr)


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "registry" / "live-instances"
    monkeypatch.setenv("MIST_LIVE_INSTANCES_FILE", str(path))
    return path


def make_root(path: Path, env_extra: str, *, tag: str = TAG, release: bool = True) -> Path:
    for sub in ("state", "mist-memory", "logs", "ops", "bin"):
        (path / sub).mkdir(parents=True)
    marker = json.dumps({"v": 1, "instance": "daily", "created_at": "2026-10-01T00:00:00Z"})
    for directory in (path, path / "state", path / "mist-memory"):
        (directory / MARKER).write_text(marker + "\n", encoding="utf-8")
    if release:
        compose = path / "release" / tag / "deploy" / "daily" / "compose.yml"
        compose.parent.mkdir(parents=True)
        compose.write_text("name: mist-daily\n", encoding="utf-8")
    (path / "daily.env").write_text(
        f"MIST_RELEASE_TAG={tag}\nMIST_DAILY_ROOT={path.as_posix()}\n{env_extra}",
        encoding="utf-8",
    )
    return path


WATCH_ENV = (
    f"MIST_GAME_DIRS={LIBRARIES_LINE}\nMIST_GAME_RUNTIME_DIRS={RUNTIME_LINE}\n"
    "MIST_GAME_POLL_S=10\nMIST_GAME_RESUME_DELAY_S=15\n"
)


@pytest.fixture
def root(tmp_path) -> Path:
    return make_root((tmp_path / "instances" / "daily").resolve(), WATCH_ENV)


def base(root: Path) -> list[str]:
    compose = root / "release" / TAG / "deploy" / "daily" / "compose.yml"
    return ["docker", "compose", "-p", "mist-daily", "-f", str(compose)] + [
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


def write_marker(root: Path, by: str) -> None:
    marker_path(root).write_text(
        json.dumps({"v": 1, "since": "2026-10-23T07:00:00Z", "by": by}), encoding="utf-8"
    )


def log_text(root: Path) -> str:
    return (root / "logs" / "gaming-watch.log").read_text(encoding="utf-8")


class Harness:
    def __init__(self, root: Path, responses: dict, lists):
        self.root = root
        self.events: list = []
        self.runner = FakeRunner(responses, self.events)
        self.clock = FakeClock(self.events)
        self.lister = FakeLister(lists, self.events)
        self.ctx = instance.Context(
            runner=self.runner,
            clock=lambda: FIXED_NOW,
            home=root.parent,
            script_path=SCRIPT,
            sleep=self.clock.sleep,
            monotonic=self.clock.monotonic,
            processes=self.lister,
        )

    def watch(self, ticks: int) -> None:
        instance.gaming_watch(self.root, self.ctx, max_ticks=ticks)


@pytest.mark.parametrize(
    "env_extra",
    [
        "",
        "MIST_GAME_DIRS=\n",
        "MIST_GAME_DIRS=;\nMIST_GAME_RUNTIME_DIRS=\n",
        'MIST_GAME_DIRS=""\nMIST_GAME_EXCLUDE=X:\\Games\\Tool\n',
    ],
    ids=["unset", "empty", "both-empty", "only-exclude"],
)
def test_gaming_watch_refuses_without_either_folder_variable(tmp_path, env_extra, capsys):
    root = make_root(tmp_path / "daily", env_extra)
    h = Harness(root, {}, [[GAME]])
    assert instance.main(["gaming-watch", "--root", str(root)], h.ctx) == 2
    err = capsys.readouterr().err
    assert "MIST_GAME_DIRS" in err and "MIST_GAME_RUNTIME_DIRS" in err
    assert h.runner.calls == []
    assert h.lister.calls == 0


def test_gaming_watch_runs_with_only_runtime_dirs_set(tmp_path):
    root = make_root(tmp_path / "daily", f"MIST_GAME_RUNTIME_DIRS={RUNTIME_LINE}\n")
    h = Harness(root, {}, [[OTHER]])
    h.watch(1)
    assert h.lister.calls == 1


def test_gaming_watch_refuses_on_a_python_that_is_not_64_bit(root, monkeypatch, capsys):
    monkeypatch.setattr(instance._load_gamewatch(), "is_64_bit", lambda: False)
    h = Harness(root, {}, [[GAME]])
    assert instance.main(["gaming-watch", "--root", str(root)], h.ctx) == 2
    assert "64-bit Python" in capsys.readouterr().err
    assert h.lister.calls == 0


@pytest.mark.parametrize(
    "line", ["MIST_GAME_POLL_S=0", "MIST_GAME_POLL_S=soon", "MIST_GAME_RESUME_DELAY_S=-1"]
)
def test_gaming_watch_refuses_a_bad_interval(tmp_path, line):
    root = make_root(tmp_path / "daily", f"MIST_GAME_DIRS={LIBRARIES_LINE}\n{line}\n")
    h = Harness(root, {}, [[GAME]])
    assert instance.main(["gaming-watch", "--root", str(root)], h.ctx) == 2
    assert h.runner.calls == []


def test_gaming_watch_defaults_to_a_10_s_poll(tmp_path):
    root = make_root(tmp_path / "daily", f"MIST_GAME_DIRS={LIBRARIES_LINE}\n")
    h = Harness(root, {}, [[OTHER]])
    h.watch(3)
    assert h.clock.sleeps == [10.0, 10.0]


@pytest.mark.parametrize("game", [GAME, SERVER], ids=["library-game", "runtime-game"])
def test_three_ticks_drive_on_then_off_through_the_gaming_mode_argv(root, game):
    responses = {stop_llm(root): ok(), start_llm(root): ok(), ps_llm(root): llm_ps("healthy")}
    h = Harness(root, responses, [[OTHER, game], [OTHER], [OTHER]])
    h.watch(3)
    # Tick 1 at 1000 sees the game: on, which waits two backend polls before the stop. Tick 2
    # at 1014 is within the 15 s delay; tick 3 at 1024 is past it: off.
    assert h.runner.calls == [list(stop_llm(root)), list(start_llm(root)), list(ps_llm(root))]
    assert h.clock.sleeps == [4.0, 10.0, 10.0]
    assert not marker_path(root).exists()


def test_the_watchers_on_records_the_watcher(root):
    h = Harness(root, {stop_llm(root): ok()}, [[GAME]])
    h.watch(1)
    assert json.loads(marker_path(root).read_text(encoding="utf-8"))["by"] == "watcher"


@pytest.mark.parametrize(
    ("game", "named"),
    [(GAME, "MIST_GAME_DIRS[1]"), (SERVER, "MIST_GAME_RUNTIME_DIRS[0]")],
    ids=["library-game", "runtime-game"],
)
def test_the_log_names_the_variable_and_index_and_nothing_of_the_program(root, game, named):
    responses = {stop_llm(root): ok(), start_llm(root): ok(), ps_llm(root): llm_ps("healthy")}
    h = Harness(root, responses, [[OTHER, game], [OTHER], [OTHER]])
    h.watch(3)
    text = log_text(root)
    assert f"game seen under {named}" in text
    assert "no game running" in text
    assert "gaming-mode on" in text and "gaming-mode off" in text
    lowered = text.lower()
    fragments = ["examplegame", ".exe", "other library", "svchost", "system32", "java"]
    fragments += ["example server", "servers", "server.jar", "accesstoken", "-xmx4g", "nogui"]
    fragments.append(SENTINEL)
    for fragment in fragments:
        assert fragment not in lowered, fragment


def test_the_first_log_line_gives_the_entry_counts_and_the_missing_indexes(tmp_path):
    present = tmp_path / "library"
    present.mkdir()
    env = (
        f"MIST_GAME_DIRS={present};X:\\Absent Library\n"
        f"MIST_GAME_RUNTIME_DIRS={present}\n"
        f"MIST_GAME_EXCLUDE=tool.exe;X:\\Absent Library\\Tool;{present}\n"
    )
    root = make_root(tmp_path / "daily", env)
    h = Harness(root, {}, [[OTHER]])
    h.watch(1)
    first = log_text(root).splitlines()[0]
    assert "2 MIST_GAME_DIRS, 1 MIST_GAME_RUNTIME_DIRS and 3 MIST_GAME_EXCLUDE entries" in first
    assert "missing: MIST_GAME_DIRS[1], MIST_GAME_EXCLUDE[1]" in first
    assert "absent" not in first.lower()
    assert str(present) not in first


def test_the_first_log_line_says_when_nothing_is_missing(tmp_path):
    present = tmp_path / "library"
    present.mkdir()
    root = make_root(tmp_path / "daily", f"MIST_GAME_DIRS={present}\n")
    Harness(root, {}, [[OTHER]]).watch(1)
    assert "missing: none" in log_text(root).splitlines()[0]


def test_a_failed_on_is_logged_and_decided_again_next_tick(root):
    # `on` retries its stop once, so two failures make one failed `on`.
    gone = fail(b"daemon gone")
    h = Harness(root, {stop_llm(root): [gone, gone, ok()]}, [[GAME]])
    h.watch(2)
    assert h.runner.calls == [list(stop_llm(root))] * 3
    assert marker_path(root).exists()
    text = log_text(root)
    assert "daemon gone" in text
    assert "failed" in text


def test_a_failed_off_is_logged_and_tried_again_next_tick(root):
    write_marker(root, "watcher")
    responses = {start_llm(root): [fail(b"no daemon"), ok()], ps_llm(root): llm_ps("healthy")}
    h = Harness(root, responses, [[OTHER]])
    h.watch(2)
    assert h.runner.calls == [list(start_llm(root)), list(start_llm(root)), list(ps_llm(root))]
    assert "no daemon" in log_text(root)
    assert not marker_path(root).exists()


@pytest.mark.parametrize("game", [GAME, SERVER], ids=["library-game", "runtime-game"])
def test_the_watchers_off_aborts_when_find_game_sees_a_game(root, game):
    write_marker(root, "watcher")
    responses = {
        start_llm(root): ok(),
        ps_llm(root): llm_ps("starting"),
        stop_llm(root): ok(),
    }
    # Tick 1 lists no game, so off starts mist-llm; the abort check lists again and sees one.
    h = Harness(root, responses, [[OTHER], [OTHER], [OTHER, game]])
    h.watch(1)
    assert h.runner.calls == [
        list(start_llm(root)),
        list(ps_llm(root)),
        list(stop_llm(root)),
    ]
    assert h.lister.calls == 3
    assert marker_path(root).exists()
    assert "aborted" in log_text(root)


def test_the_watchers_off_is_not_aborted_by_a_launcher(root):
    write_marker(root, "watcher")
    launcher = gamewatch.ProcessRecord("X:\\Library\\Example Game\\Launcher.exe")
    responses = {start_llm(root): ok(), ps_llm(root): [llm_ps("starting"), llm_ps("healthy")]}
    h = Harness(root, responses, [[launcher]])
    h.watch(1)
    assert list(stop_llm(root)) not in h.runner.calls
    assert not marker_path(root).exists()


def test_a_manual_marker_is_never_ended_by_the_watcher(root):
    write_marker(root, "manual")
    h = Harness(root, {}, [[GAME], [], [], []])
    h.watch(4)
    assert h.runner.calls == []
    assert marker_path(root).exists()


def test_a_lister_failure_ends_the_watch_with_exit_1_and_is_logged(root, capsys):
    def broken() -> list[gamewatch.ProcessRecord]:
        raise instance._load_gamewatch().GameWatchError("listing needs Windows")

    h = Harness(root, {}, [[]])
    ctx = dataclasses.replace(h.ctx, processes=broken)
    assert instance.main(["gaming-watch", "--root", str(root)], ctx) == 1
    assert "listing needs Windows" in capsys.readouterr().err
    assert "listing needs Windows" in log_text(root)


# --- The installed layout --------------------------------------------------------------------


def run_watch(directory: Path, root: Path, registry: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(directory / "instance.py"), "gaming-watch", "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(directory),
        env={"MIST_LIVE_INSTANCES_FILE": str(registry), "PATH": ""},
        timeout=60,
    )


def test_gaming_watch_finds_gamewatch_beside_it_and_names_it_when_missing(tmp_path, registry):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("instance.py", "gamewatch.py", "volume_safety.py"):
        shutil.copy2(DAILY_SCRIPTS / name, bin_dir / name)
    root = make_root(tmp_path / "daily", "")

    beside = run_watch(bin_dir, root, registry)
    assert beside.returncode == 2, beside.stderr
    assert "MIST_GAME_DIRS" in beside.stderr
    assert "MIST_GAME_RUNTIME_DIRS" in beside.stderr

    (bin_dir / "gamewatch.py").unlink()
    without = run_watch(bin_dir, root, registry)
    assert without.returncode == 1, without.stderr
    assert "gamewatch.py" in without.stderr
    assert "Traceback" not in without.stderr


def test_other_commands_run_without_gamewatch_beside_them(tmp_path):
    for name in ("instance.py", "volume_safety.py"):
        shutil.copy2(DAILY_SCRIPTS / name, tmp_path / name)
    completed = subprocess.run(
        [sys.executable, str(tmp_path / "instance.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    assert "gaming-watch" in completed.stdout


# --- No console window under pythonw ---------------------------------------------------------


@pytest.mark.parametrize(("os_name", "flags"), [("nt", 0x08000000), ("posix", 0)])
def test_every_subprocess_gets_create_no_window_on_windows_only(monkeypatch, os_name, flags):
    recorded = []

    def run(argv, **kwargs):
        recorded.append(kwargs.get("creationflags"))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(instance.subprocess, "run", run)
    monkeypatch.setattr(instance.os, "name", os_name)
    try:
        instance.subprocess_runner(["docker", "info"])
        instance.subprocess_runner(["docker", "info"], stdin=b"x")
    finally:
        monkeypatch.undo()
    assert recorded == [flags, flags]


# --- Documentation ---------------------------------------------------------------------------


def example_block(name: str) -> tuple[str, str]:
    """The comment lines directly above `name`'s assignment in the env example, joined, and
    the assignment itself.
    """
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    index = next(i for i, line in enumerate(lines) if re.match(rf"^#?\s*{name}=", line))
    start = index
    while start > 0 and lines[start - 1].startswith("# ") and "=" not in lines[start - 1]:
        start -= 1
    comment = " ".join(line[2:] for line in lines[start:index])
    return comment, lines[index]


@pytest.mark.parametrize(
    ("name", "default"),
    [
        ("MIST_GAME_DIRS", None),
        ("MIST_GAME_RUNTIME_DIRS", None),
        ("MIST_GAME_EXCLUDE", None),
        ("MIST_GAME_POLL_S", "10"),
        ("MIST_GAME_RESUME_DELAY_S", "120"),
    ],
)
def test_env_example_documents_the_watcher_variables(name, default):
    comment, assignment = example_block(name)
    assert "only the host CLI reads it" in comment
    # Commented out: unset is the default, and unset folder variables refuse.
    assert assignment.startswith(f"# {name}=")
    value = assignment.split("=", 1)[1]
    assert re.fullmatch(r"<[a-z0-9][a-z0-9-]*>", value), assignment
    if default is not None:
        assert f"{default} when unset" in comment
    else:
        assert "semicolon" in comment and "no quotes" in comment


def readme_section(title: str) -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match is not None, f"README has no '## {title}' section"
    return " ".join(match.group(1).split())


def test_readme_gaming_mode_section_describes_the_watcher():
    section = readme_section("Gaming mode")
    for phrase in (
        "gaming-watch",
        "`MIST_GAME_DIRS`",
        "`MIST_GAME_RUNTIME_DIRS`",
        "`MIST_GAME_EXCLUDE`",
        "`MIST_GAME_POLL_S`",
        "`MIST_GAME_RESUME_DELAY_S`",
        "`--gameDir`",
        "never ends a manual gaming mode",
        "may go unseen",
        "64-bit Python",
        "32-bit Java",
        "logs/gaming-watch.log",
        "MIST_GAME_DIRS[",
    ):
        assert phrase in section, phrase


# --- The sign-in task scripts ----------------------------------------------------------------
#
# The runner has no PowerShell, so the scripts are checked by parsing their text, with the
# helpers `tests/unit/daily/test_instance_operations.py` uses for the first task.

WINDOWS_DIR = REPO_ROOT / "deploy" / "daily" / "windows"
REGISTER = WINDOWS_DIR / "register-signin-task.ps1"
UNREGISTER = WINDOWS_DIR / "unregister-signin-task.ps1"
INSTANCE_TASK = "MIST daily instance"
WATCH_TASK = "MIST gaming watch"
SETTINGS = (
    "-Settings (New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) "
    "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries)"
)


def ps_code(path: Path) -> str:
    """The script without comments, with each backtick line continuation joined."""
    text = re.sub(r"<#.*?#>", "", path.read_text(encoding="utf-8"), flags=re.S)
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    return re.sub(r"`\n\s*", " ", "\n".join(lines))


def ps_header(path: Path) -> str:
    match = re.search(r"<#(.*?)#>", path.read_text(encoding="utf-8"), re.S)
    assert match is not None
    return match.group(1)


def ps_statements(code: str, command: str) -> list[str]:
    return [line for line in code.splitlines() if re.search(rf"(?<![\w-]){command}\b", line)]


def balanced_group(text: str, start: int) -> str:
    """The text from the bracket at `start` to its matching close, outside quoted strings."""
    pairs = {"(": ")", "{": "}", "[": "]"}
    stack: list[str] = []
    quote: str | None = None
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            assert stack and stack.pop() == char
            if not stack:
                return text[start : index + 1]
    raise AssertionError(f"no close for the bracket at {start}")


def if_blocks(code: str, condition: str) -> list[tuple[int, str]]:
    """Where each `if (<condition>) { ... }` in `code` starts, and its body."""
    blocks = []
    for match in re.finditer(rf"if \({re.escape(condition)}\) ", code):
        blocks.append((match.start(), balanced_group(code, code.index("{", match.end() - 1))))
    return blocks


def watch_registration(code: str) -> str:
    calls = [
        call
        for call in ps_statements(code, "Register-ScheduledTask")
        if f"-TaskName '{WATCH_TASK}'" in call
    ]
    assert len(calls) == 1, calls
    return calls[0]


def test_register_script_takes_the_gaming_watch_switch():
    code = ps_code(REGISTER)
    param = balanced_group(code, code.index("param("))
    assert re.search(r"\[switch\]\s*\$GamingWatch\b", param)


def test_the_watch_task_is_registered_only_inside_the_switch_block():
    code = ps_code(REGISTER)
    call = watch_registration(code)
    assert len([body for _at, body in if_blocks(code, "$GamingWatch") if call in body]) == 1


def test_the_watch_task_runs_pythonw_gaming_watch_at_sign_in_with_the_same_settings():
    code = ps_code(REGISTER)
    call = watch_registration(code)
    assert "-Trigger $Trigger" in call
    assert "-Principal $Principal" in call
    assert "-Action $WatchAction" in call
    assert SETTINGS in call
    first = next(
        call
        for call in ps_statements(code, "Register-ScheduledTask")
        if f"-TaskName '{INSTANCE_TASK}'" in call
    )
    assert SETTINGS in first
    actions = [
        line
        for line in ps_statements(code, "New-ScheduledTaskAction")
        if line.lstrip().startswith("$WatchAction =")
    ]
    assert len(actions) == 1
    assert "-Execute $PythonwPath" in actions[0]
    assert "-Argument $WatchArgument" in actions[0]
    assert '\'"{0}/bin/instance.py" gaming-watch --root "{0}"\' -f $RootPath' in code


def test_pythonw_comes_from_the_python_directory_and_its_absence_refuses_before_any_task():
    code = ps_code(REGISTER)
    assert "$PythonDir = Split-Path -Path $PythonPath -Parent" in code
    assert "$PythonwPath = Join-Path -Path $PythonDir -ChildPath 'pythonw.exe'" in code
    checks = [
        (at, body)
        for at, body in if_blocks(code, "$GamingWatch")
        if "Test-Path -LiteralPath $PythonwPath -PathType Leaf" in body
    ]
    assert len(checks) == 1
    at, body = checks[0]
    assert "throw" in body
    # The refusal comes before the first task is registered, so a refusal changes nothing.
    assert at < code.index("Register-ScheduledTask")


def test_register_script_header_documents_the_switch():
    header = ps_header(REGISTER)
    assert "-GamingWatch" in header
    assert WATCH_TASK in header
    assert "pythonw.exe" in header


def test_unregister_script_removes_both_tasks_each_only_if_present():
    code = ps_code(UNREGISTER)
    removals = ps_statements(code, "Unregister-ScheduledTask")
    assert len(removals) == 2
    for name in (INSTANCE_TASK, WATCH_TASK):
        removal = f"Unregister-ScheduledTask -TaskName '{name}' -Confirm:$false"
        assert sum(removal in line for line in removals) == 1, name
        lookup = f"Get-ScheduledTask -TaskName '{name}' -ErrorAction SilentlyContinue"
        variable = re.search(rf"(\$\w+) = {re.escape(lookup)}", code)
        assert variable is not None, name
        # The removal is the else branch of the check that the lookup found nothing.
        check = f"if ($null -eq {variable.group(1)}) "
        check_at = code.index(check, variable.end())
        then_block = balanced_group(code, code.index("{", check_at))
        rest = code[code.index(then_block, check_at) + len(then_block) :]
        assert rest.lstrip().startswith("else"), name
        else_block = balanced_group(rest, rest.index("{"))
        assert removal in else_block, name
        assert removal not in then_block, name


def test_readme_says_how_to_register_the_watcher():
    section = readme_section("Gaming mode")
    for phrase in (
        "-GamingWatch",
        "register-signin-task.ps1",
        WATCH_TASK,
        "pythonw.exe",
        "unregister-signin-task.ps1",
    ):
        assert phrase in section, phrase


# --- gaming-scan -----------------------------------------------------------------------------

ENV_SENTINEL = "zq8envsentinel"
LAUNCHER = gamewatch.ProcessRecord("X:\\Library\\Example Game\\Launcher.exe")
EXCLUDED_JAVA = gamewatch.ProcessRecord(
    "X:\\Library\\Example Tool\\jre\\bin\\java.exe", cwd="X:\\Library\\Example Tool", args=()
)
STRAY_JAVA = gamewatch.ProcessRecord(JAVA, cwd="X:\\Projects\\tool", args=("-jar", "tool.jar"))
INSTALLER_JAVA = gamewatch.ProcessRecord(
    JAVA,
    cwd="Y:\\Servers\\Example Server",
    args=("-jar", "example-installer.jar", "--installServer"),
)


def scan_root(tmp_path: Path, present: Path, **kwargs) -> Path:
    env = (
        f"MIST_GAME_DIRS={LIBRARIES_LINE}\n"
        f"MIST_GAME_RUNTIME_DIRS={RUNTIME_LINE};{present}\n"
        "MIST_GAME_EXCLUDE=X:\\Library\\Example Tool;Backdrop*.exe\n"
        f"MIST_SCAN_TEST_SECRET={ENV_SENTINEL}\n"
    )
    return make_root(tmp_path / "daily", env, **kwargs)


def scan(root: Path, records, capsys) -> tuple[int, list[str], str]:
    h = Harness(root, {}, [records])
    code = instance.main(["gaming-scan", "--root", str(root)], h.ctx)
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.out + captured.err


def test_gaming_scan_prints_the_entries_the_verdict_and_why_in_order(tmp_path, capsys):
    present = tmp_path / "instances"
    present.mkdir()
    root = scan_root(tmp_path, present)
    # Interleaved, so the output's grouping is the scan's; within a group the listing's order
    # stands.
    records = [OTHER, STRAY_JAVA, LAUNCHER, GAME, INSTALLER_JAVA, EXCLUDED_JAVA, SERVER]
    code, lines, _text = scan(root, records, capsys)
    assert code == 0
    assert lines == [
        "MIST_GAME_DIRS[0] X:\\Library (missing)",
        "MIST_GAME_DIRS[1] X:\\Other Library (missing)",
        "MIST_GAME_RUNTIME_DIRS[0] Y:\\Servers (missing)",
        f"MIST_GAME_RUNTIME_DIRS[1] {present}",
        "MIST_GAME_EXCLUDE[0] X:\\Library\\Example Tool (missing)",
        "MIST_GAME_EXCLUDE[1] backdrop*.exe",
        "game: yes",
        f"counts MIST_GAME_DIRS[1]: {GAME.image}",
        f"counts MIST_GAME_RUNTIME_DIRS[0]: {JAVA} for Y:\\Servers\\Example Server",
        f"excluded shipped name *launcher*.exe: {LAUNCHER.image}",
        f"excluded MIST_GAME_EXCLUDE[0]: {EXCLUDED_JAVA.image}",
        f"not counted no runtime folder: {JAVA}",
        f"not counted installer: {JAVA}",
    ]


def test_gaming_scan_says_no_game_when_nothing_counts(tmp_path, capsys):
    root = scan_root(tmp_path, tmp_path)
    code, lines, _text = scan(root, [OTHER, LAUNCHER], capsys)
    assert code == 0
    assert "game: no" in lines
    assert not any(line.startswith("counts ") for line in lines)


def test_gaming_scan_prints_a_runtime_folder_taken_from_gamedir(tmp_path, capsys):
    client = gamewatch.ProcessRecord(
        JAVA, args=("--accessToken", SENTINEL, "--gameDir", "Y:\\Servers\\Example Pack")
    )
    code, lines, text = scan(scan_root(tmp_path, tmp_path), [client], capsys)
    assert code == 0
    assert f"counts MIST_GAME_RUNTIME_DIRS[0]: {JAVA} for Y:\\Servers\\Example Pack" in lines
    assert SENTINEL not in text


def test_gaming_scan_prints_no_argument_and_no_other_env_line(tmp_path, capsys):
    root = scan_root(tmp_path, tmp_path)
    records = [SERVER, INSTALLER_JAVA, STRAY_JAVA]
    _code, _lines, text = scan(root, records, capsys)
    for fragment in (SENTINEL, ENV_SENTINEL, "accessToken", "-Xmx4G", "server.jar", "tool.jar"):
        assert fragment not in text, fragment
    assert "--installServer" not in text and "example-installer.jar" not in text


def test_gaming_scan_writes_no_log(tmp_path, capsys):
    root = scan_root(tmp_path, tmp_path)
    before = sorted((root / "logs").iterdir())
    scan(root, [GAME, SERVER], capsys)
    assert sorted((root / "logs").iterdir()) == before == []


def test_gaming_scan_refuses_without_either_folder_variable(tmp_path, capsys):
    root = make_root(tmp_path / "daily", "MIST_GAME_EXCLUDE=X:\\Games\\Tool\n")
    code, _lines, text = scan(root, [GAME], capsys)
    assert code == 2
    assert "MIST_GAME_DIRS" in text and "MIST_GAME_RUNTIME_DIRS" in text


def test_gaming_scan_refuses_on_a_python_that_is_not_64_bit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(instance._load_gamewatch(), "is_64_bit", lambda: False)
    code, _lines, text = scan(scan_root(tmp_path, tmp_path), [GAME], capsys)
    assert code == 2
    assert "64-bit Python" in text


def test_gaming_scan_exits_1_when_gamewatch_cannot_be_loaded(tmp_path, monkeypatch, capsys):
    def missing():
        raise instance.InstanceError("cannot load gamewatch.py from bin")

    monkeypatch.setattr(instance, "_load_gamewatch", missing)
    code, _lines, text = scan(scan_root(tmp_path, tmp_path), [GAME], capsys)
    assert code == 1
    assert "gamewatch.py" in text


def test_gaming_scan_exits_1_when_listing_fails(tmp_path, capsys):
    root = scan_root(tmp_path, tmp_path)

    def broken() -> list[gamewatch.ProcessRecord]:
        raise instance._load_gamewatch().GameWatchError("listing needs Windows")

    ctx = dataclasses.replace(Harness(root, {}, [[]]).ctx, processes=broken)
    assert instance.main(["gaming-scan", "--root", str(root)], ctx) == 1
    assert "listing needs Windows" in capsys.readouterr().err
    assert list((root / "logs").iterdir()) == []


def test_gaming_scan_runs_before_a_release_is_switched_in_and_the_watch_does_not(tmp_path, capsys):
    root = scan_root(tmp_path, tmp_path, tag="<release-tag>", release=False)
    code, lines, _text = scan(root, [GAME], capsys)
    assert code == 0
    assert "game: yes" in lines
    h = Harness(root, {}, [[GAME]])
    assert instance.main(["gaming-watch", "--root", str(root)], h.ctx) == 2
    assert h.lister.calls == 0


def test_gaming_scan_is_a_listed_command():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(DAILY_SCRIPTS),
    )
    assert completed.returncode == 0, completed.stderr
    assert "gaming-scan" in completed.stdout


def test_readme_gaming_mode_section_describes_the_scan():
    section = readme_section("Gaming mode")
    for phrase in (
        "gaming-scan --root",
        "`MIST_GAME_DIRS`",
        "`MIST_GAME_RUNTIME_DIRS`",
        "`MIST_GAME_EXCLUDE`",
        "(missing)",
        "game: yes",
        "counts MIST_GAME_RUNTIME_DIRS[",
        "excluded",
        "not counted",
        "prints no argument",
        "writes no log",
    ):
        assert phrase in section, phrase
