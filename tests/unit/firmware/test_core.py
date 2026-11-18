# SPDX-License-Identifier: MIT
# Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
"""Host tests for the room endpoint's portable core, `mist_core`.

The C++ suites under `cpp/` hold the behaviour: each `MIST_TEST` case runs as its own pytest item,
named from the sources so collection needs no compiler. The contract driver prints every message
the core emits in scripted scenarios, and the tests here validate each one with the backend's own
models (`backend/realtime/protocol.py`, `backend/capabilities/events.py`), so a device message the
server would refuse fails here rather than on the device.
"""

import json
import re
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.capabilities.events import AlarmActionMessage, AlarmReportMessage
from backend.realtime.protocol import (
    CLIENT_MESSAGE_TYPES,
    ROOM_CLIENT_MESSAGE_TYPES,
    ROOM_SERVER_MESSAGE_TYPES,
    Direction,
    MsgType,
    decode_frame,
    parse_client_message,
)

from .conftest import (
    CORE_DIR,
    CPP_DIR,
    MISSING_COMPILER,
    SANITIZER_ENV,
    START_MARKER,
    STARTUP_ATTEMPTS,
    SUITES,
    CoreBuild,
    require_gxx,
    run_binary,
)

FIRMWARE_TESTS_DIR = Path(__file__).resolve().parent

# Every message the device sends (plan, Interfaces: device to backend); `device_stats` is the
# spike build's only, sent to the spike listener.
DEVICE_MESSAGE_TYPES = frozenset(
    {
        "voice_start",
        "wake",
        "barge_in",
        "mute",
        "interrupt",
        "playback_progress",
        "playback_done",
        "alarm_action",
        "alarm_report",
        "device_stats",
    }
)
DEVICE_STATS_FIELDS = frozenset(
    {
        "heap_free_min",
        "psram_free",
        "mic_overruns",
        "ring_overruns",
        "playback_underruns",
        "uptime_s",
        "reconnects",
    }
)

# The C++ standard headers the core may include; anything else (ESPHome, ESP-IDF, FreeRTOS,
# Arduino) would stop it building on the host.
STANDARD_HEADERS = frozenset(
    {
        "atomic",
        "cmath",
        "cstddef",
        "cstdint",
        "cstdio",
        "cstdlib",
        "cstring",
        "limits",
        "new",
        "type_traits",
    }
)
PLATFORM_HEADER = re.compile(r"esphome|esp_|freertos|sdkconfig|driver/|nvs|arduino|hal/", re.I)
INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
CASE = re.compile(r"^MIST_TEST\((\w+)\)", re.M)
INBOUND_TYPE = re.compile(r'\btype\.equals\("(\w+)"\)')
MIT_LINE = "SPDX-License-Identifier: MIT"


def _cases() -> list[object]:
    params: list[object] = []
    for suite in SUITES:
        source = (CPP_DIR / f"test_{suite}.cpp").read_text(encoding="utf-8")
        for case in CASE.findall(source):
            params.append(pytest.param(suite, case, id=f"{suite}::{case}"))
    return params


@pytest.mark.parametrize(("suite", "case"), _cases())
def test_cpp_case(core_build: CoreBuild, suite: str, case: str) -> None:
    result = core_build.run(suite, case)

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS {case}" in result.stdout


def test_every_suite_has_cases() -> None:
    counts = {suite: 0 for suite in SUITES}
    for param in _cases():
        counts[param.values[0]] += 1  # type: ignore[attr-defined]

    assert all(count > 0 for count in counts.values()), counts


# -- Contract: every message the core emits, validated by the backend ------------------------


@pytest.fixture(scope="module")
def contract_lines(core_build: CoreBuild) -> list[tuple[str, str, str]]:
    """The contract driver's output as (scenario, kind, rest) triples, kind TEXT or BINARY."""
    result = core_build.run("contract")
    assert result.returncode == 0, result.stdout + result.stderr
    lines: list[tuple[str, str, str]] = []
    scenario = ""
    for line in result.stdout.splitlines():
        kind, _, rest = line.partition(" ")
        if kind == START_MARKER:
            continue
        if kind == "SCENARIO":
            scenario = rest
        elif kind in ("TEXT", "BINARY"):
            lines.append((scenario, kind, rest))
        else:
            pytest.fail(f"unexpected contract output line: {line!r}")
    return lines


def _texts(lines: list[tuple[str, str, str]]) -> list[tuple[str, datetime, str]]:
    texts = []
    for scenario, kind, rest in lines:
        if kind == "TEXT":
            clock, _, text = rest.partition(" ")
            texts.append((scenario, datetime.fromtimestamp(int(clock) / 1000, UTC), text))
    return texts


def test_contract_covers_every_message_the_device_sends(
    contract_lines: list[tuple[str, str, str]],
) -> None:
    sent = {json.loads(text)["type"] for _, _, text in _texts(contract_lines)}

    assert sent == DEVICE_MESSAGE_TYPES
    assert any(kind == "BINARY" for _, kind, _ in contract_lines)


def test_a_room_session_accepts_every_type_the_device_sends_and_no_other():
    """A new device message fails here until the room session's allowlist takes it."""
    assert DEVICE_MESSAGE_TYPES - {"device_stats"} == ROOM_CLIENT_MESSAGE_TYPES


def _decoded_types() -> frozenset[str]:
    """The `type.equals("...")` strings in `read_inbound`'s body, read from `codec.cpp`'s source.

    The body runs from the brace after the signature to its matching brace; the function holds
    no brace inside a string or a comment, so counting braces finds it with no compiler.
    """
    source = (CORE_DIR / "codec.cpp").read_text(encoding="utf-8")
    opening = source.index("{", source.index("ReadError read_inbound("))
    depth = 0
    for end in range(opening, len(source)):
        depth += {"{": 1, "}": -1}.get(source[end], 0)
        if depth == 0:
            break
    return frozenset(INBOUND_TYPE.findall(source[opening:end]))


def test_a_room_session_is_sent_every_type_the_device_decodes_and_no_other():
    """A type the firmware starts to decode fails here until the room session's set takes it."""
    assert _decoded_types() == ROOM_SERVER_MESSAGE_TYPES


def test_every_protocol_message_parses_strictly_with_the_backend_codec(
    contract_lines: list[tuple[str, str, str]],
) -> None:
    checked = 0
    for _, _, text in _texts(contract_lines):
        if json.loads(text)["type"] in CLIENT_MESSAGE_TYPES:
            parse_client_message(text)
            checked += 1

    assert checked >= 8


def test_voice_start_carries_the_room_units_fields(
    contract_lines: list[tuple[str, str, str]],
) -> None:
    starts = [
        parse_client_message(text)
        for _, _, text in _texts(contract_lines)
        if json.loads(text)["type"] == "voice_start"
    ]

    assert [(start.req, start.muted) for start in starts] == [(1, False), (2, True)]
    for start in starts:
        assert (start.mode, start.sample_rate, start.frame_ms) == ("wake", 16000, 20)
        assert (start.aec, start.playback, start.endpoint, start.spotter) == (
            "device",
            True,
            "room",
            "endpoint",
        )
        assert start.reference is False


def test_wake_names_mist_with_its_backlog(contract_lines: list[tuple[str, str, str]]) -> None:
    wakes = [
        parse_client_message(text)
        for _, _, text in _texts(contract_lines)
        if json.loads(text)["type"] == "wake"
    ]

    assert len(wakes) == 1
    assert wakes[0].keyword == "mist"
    assert wakes[0].backlog_ms == 2400
    assert wakes[0].score == pytest.approx(0.82)


def test_every_alarm_report_validates_against_the_owner_clock(
    contract_lines: list[tuple[str, str, str]],
) -> None:
    outcomes = []
    for _, now, text in _texts(contract_lines):
        if json.loads(text)["type"] == "alarm_report":
            report = AlarmReportMessage.model_validate_json(text, context={"now": now})
            outcomes.append(report.outcome)

    assert set(outcomes) == {"rang", "stopped", "snoozed"}


def test_every_alarm_action_validates_and_carries_only_known_fields(
    contract_lines: list[tuple[str, str, str]],
) -> None:
    actions = []
    for _, _, text in _texts(contract_lines):
        data = json.loads(text)
        if data["type"] == "alarm_action":
            action = AlarmActionMessage.model_validate_json(text)
            assert set(data) <= set(AlarmActionMessage.model_fields), data
            assert action.request_id is not None
            actions.append(action.action)

    assert actions == ["snooze", "stop"]


def test_device_stats_carries_exactly_its_counters(
    contract_lines: list[tuple[str, str, str]],
) -> None:
    stats = [
        json.loads(text)
        for _, _, text in _texts(contract_lines)
        if json.loads(text)["type"] == "device_stats"
    ]

    assert len(stats) == 1
    assert set(stats[0]) == DEVICE_STATS_FIELDS | {"type"}
    for name in DEVICE_STATS_FIELDS:
        value = stats[0][name]
        assert isinstance(value, int) and not isinstance(value, bool) and value >= 0, name


def test_uplink_frames_decode_with_the_backend_codec_in_contiguous_seq(
    contract_lines: list[tuple[str, str, str]],
) -> None:
    frames = [
        decode_frame(bytes.fromhex(rest), direction=Direction.UP)
        for scenario, kind, rest in contract_lines
        if kind == "BINARY" and scenario == "uplink"
    ]

    # 120 frames of backlog (capped at 5000 ms, so all of them) and 3 live frames, then the
    # rejection: the 3 frames written after it are never sent.
    assert len(frames) == 123
    assert [frame.seq for frame in frames] == list(range(123))
    for frame in frames:
        assert frame.msg_type is MsgType.MIC_AUDIO
        assert frame.turn_tag == 0
        assert len(frame.payload) == 640


# -- The core's boundary: standard headers only, and the licence header ----------------------


def _core_sources() -> list[Path]:
    return sorted(path for path in CORE_DIR.iterdir() if path.suffix in (".h", ".cpp"))


def test_the_core_has_its_planned_files() -> None:
    names = {path.name for path in CORE_DIR.iterdir() if path.is_file()}

    assert names == {
        "__init__.py",
        "codec.h",
        "codec.cpp",
        "ring.h",
        "uplink.h",
        "uplink.cpp",
        "tag_queue.h",
        "tag_queue.cpp",
        "alarm_rules.h",
        "alarm_rules.cpp",
        "ui_state.h",
        "ui_state.cpp",
        "json_lite.h",
    }


def test_the_core_includes_only_the_standard_library_and_itself() -> None:
    own = {path.name for path in _core_sources()}
    offending = []
    for path in _core_sources():
        for style, header in INCLUDE.findall(path.read_text(encoding="utf-8")):
            allowed = header in STANDARD_HEADERS if style == "<" else header in own
            if not allowed:
                offending.append(f"{path.name}: {header}")

    assert offending == []


def test_no_core_file_names_an_esphome_or_idf_header() -> None:
    offending = [
        f"{path.name}: {header}"
        for path in _core_sources()
        for _, header in INCLUDE.findall(path.read_text(encoding="utf-8"))
        if PLATFORM_HEADER.search(header)
    ]

    assert offending == []


def test_every_zone_file_carries_the_mit_header() -> None:
    files = [
        *(path for path in CORE_DIR.iterdir() if path.is_file()),
        *(path for path in FIRMWARE_TESTS_DIR.iterdir() if path.suffix == ".py"),
        *(path for path in CPP_DIR.iterdir() if path.is_file()),
    ]
    missing = []
    for path in files:
        head = path.read_text(encoding="utf-8").splitlines()[:3]
        if not (any(MIT_LINE in line for line in head) and any("LICENSE" in line for line in head)):
            missing.append(path.name)

    assert files
    assert missing == []


# -- The build driver ------------------------------------------------------------------------


def test_require_gxx_returns_the_compiler_found() -> None:
    assert require_gxx(which=lambda name: f"/usr/bin/{name}", environ={}) == "/usr/bin/g++"


def test_require_gxx_skips_with_a_reason_without_a_compiler() -> None:
    with pytest.raises(pytest.skip.Exception, match=re.escape(MISSING_COMPILER)):
        require_gxx(which=lambda name: None, environ={})


def test_require_gxx_fails_in_ci_without_a_compiler() -> None:
    with pytest.raises(pytest.fail.Exception, match="CI is set"):
        require_gxx(which=lambda name: None, environ={"CI": "true"})


class _ScriptedLauncher:
    """Answers each launch with the next scripted (return code, stdout) and counts launches."""

    def __init__(self, outcomes: list[tuple[int, str]]) -> None:
        self.outcomes = outcomes
        self.launches = 0

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        returncode, stdout = self.outcomes[min(self.launches, len(self.outcomes) - 1)]
        self.launches += 1
        return subprocess.CompletedProcess(command, returncode, stdout, "")


STARTED = f"{START_MARKER}\n"


def test_a_fault_before_main_is_run_again() -> None:
    launcher = _ScriptedLauncher(
        [(-signal.SIGSEGV, ""), (-signal.SIGSEGV, ""), (0, f"{STARTED}PASS a\n")]
    )

    result = run_binary(["case"], {}, launch=launcher)

    assert result.returncode == 0
    assert launcher.launches == 3


def test_a_fault_before_main_on_every_start_still_fails_after_the_attempts() -> None:
    launcher = _ScriptedLauncher([(-signal.SIGSEGV, "")])

    result = run_binary(["case"], {}, launch=launcher)

    assert result.returncode == -signal.SIGSEGV
    assert launcher.launches == STARTUP_ATTEMPTS


def test_a_crash_after_the_start_marker_is_never_run_again() -> None:
    # A crash inside a test case, the threaded ring test's included, prints nothing after the
    # marker; it must fail on its first run.
    crashed_in_case = _ScriptedLauncher([(-signal.SIGSEGV, STARTED), (0, f"{STARTED}PASS a\n")])

    result = run_binary(["case"], {}, launch=crashed_in_case)

    assert result.returncode == -signal.SIGSEGV
    assert crashed_in_case.launches == 1


def test_a_failing_case_or_another_signal_is_never_run_again() -> None:
    failing = _ScriptedLauncher([(1, f"{STARTED}FAILED a\n")])
    aborted = _ScriptedLauncher([(-signal.SIGABRT, "")])

    assert run_binary(["case"], {}, launch=failing).returncode == 1
    assert run_binary(["case"], {}, launch=aborted).returncode == -signal.SIGABRT
    assert failing.launches == 1
    assert aborted.launches == 1


def test_every_test_binary_prints_the_start_marker_first(core_build: CoreBuild) -> None:
    for name in core_build.binaries:
        result = core_build.run(name, "--list") if name != "contract" else core_build.run(name)

        assert result.stdout.splitlines()[0] == START_MARKER, name


def test_sanitizers_never_install_their_own_segv_handler() -> None:
    assert "handle_segv=0" in SANITIZER_ENV["ASAN_OPTIONS"]
    assert "handle_segv=0" in SANITIZER_ENV["UBSAN_OPTIONS"]


def test_the_build_stays_in_its_temporary_directory(
    core_build: CoreBuild, tmp_path_factory: pytest.TempPathFactory
) -> None:
    base = tmp_path_factory.getbasetemp()

    assert core_build.directory.is_relative_to(base)
    assert all(binary.is_file() for binary in core_build.binaries.values())
    assert all(binary.parent == core_build.directory for binary in core_build.binaries.values())
    assert not any(path.suffix in (".o", ".out") for path in CORE_DIR.iterdir())
    assert not any(path.suffix in (".o", ".out") for path in CPP_DIR.iterdir())
