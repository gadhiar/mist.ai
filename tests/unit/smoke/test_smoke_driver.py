"""Tests for the live-path smoke instrument.

Scope: the pure halves of `scripts/smoke/drive_turns.py`,
`scripts/smoke/baseline.py` and `scripts/smoke/assert_artifacts.py` -- the
corpus loader, the outbound frame builder, the per-turn frame state machine,
the read-only SQLite opener, the cypher-shell output parser, the transcript
parser, and all four adjudicators.

WHY THESE TESTS EXIST
---------------------
The instrument is the only thing standing between the experiment and a wrong
conclusion. The failure it must not have is an assertion that prints FAIL
without printing why, or that prints FAIL where the honest answer is
INCONCLUSIVE -- the second sends people to debug code that is working. So the
adjudicators are tested at every branch, including the `error` and
`stream_cancelled` terminals, not only the happy path.

`drive_turns.py` is imported here for its pure logic. That import must not
require `websockets`, which is exactly why `resolve_websockets`
(`scripts/smoke/drive_turns.py`) does the import inside a function rather than
at module scope. `test_module_import_does_not_require_websockets` is the
standing check on that property.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from scripts.smoke import assert_artifacts as aa
from scripts.smoke import baseline as bl
from scripts.smoke import drive_turns as dt

REPO_ROOT = Path(__file__).resolve().parents[3]
TURNS_PATH = REPO_ROOT / "scripts" / "smoke" / "turns.json"


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def test_shipped_corpus_parses_into_five_turns():
    plan = dt.load_turn_plan(TURNS_PATH)

    assert plan.corpus_id == "SMOKE-7F3A"
    assert len(plan.turns) == 5
    assert [turn.index for turn in plan.turns] == [1, 2, 3, 4, 5]
    assert plan.turns[0].text.startswith("Logging this under tag SMOKE-7F3A.")
    assert plan.turns[4].purpose == "retrieval-probe"
    assert "Redpanda" in plan.corpus_facts
    assert plan.retrieval_probe_terms == ("Redpanda", "Kafka")


def test_every_shipped_turn_clears_the_three_word_extraction_floor():
    """`backend/knowledge/extraction/pipeline.py:640` skips fewer than 3 words."""
    plan = dt.load_turn_plan(TURNS_PATH)

    for turn in plan.turns:
        assert len(turn.text.split()) >= 3, turn


def test_load_turn_plan_rejects_an_empty_utterance(tmp_path):
    """An empty text would be skipped by `continue` at `backend/server.py:859`."""
    corpus = tmp_path / "turns.json"
    corpus.write_text(json.dumps({"turns": [{"index": 1, "text": "   "}]}), encoding="utf-8")

    with pytest.raises(dt.CorpusError, match="empty or non-string"):
        dt.load_turn_plan(corpus)


def test_load_turn_plan_rejects_a_file_with_no_turns(tmp_path):
    corpus = tmp_path / "turns.json"
    corpus.write_text(json.dumps({"turns": []}), encoding="utf-8")

    with pytest.raises(dt.CorpusError, match="no 'turns' list"):
        dt.load_turn_plan(corpus)


def test_load_turn_plan_reports_a_missing_file_by_name(tmp_path):
    with pytest.raises(dt.CorpusError, match="not found"):
        dt.load_turn_plan(tmp_path / "absent.json")


# ---------------------------------------------------------------------------
# Outbound frame
# ---------------------------------------------------------------------------


def test_text_frame_has_exactly_the_two_keys_the_server_reads():
    """`backend/server.py:856-858` reads `type` and `text` and nothing else."""
    frame = dt.build_text_frame("My name is Dana Whitfield.")

    assert frame == {"type": "text", "text": "My name is Dana Whitfield."}
    assert set(frame) == {"type", "text"}


def test_text_frame_refuses_an_empty_utterance():
    with pytest.raises(dt.CorpusError, match="empty text frame"):
        dt.build_text_frame("  ")


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------


def test_session_id_is_read_from_the_session_started_frame():
    frame = {
        "type": "session_started",
        "session_id": "7f3a0c2e-1111-4222-8333-444455556666",
        "protocol_version": "1.1.0",
        "mist_state": "idle",
        "capabilities": {"tts_enabled": False, "vad_enabled": True},
    }

    assert dt.session_id_from_frame(frame) == "7f3a0c2e-1111-4222-8333-444455556666"


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "stream_start", "turn_id": "x"},
        {"type": "session_started"},
        {"type": "session_started", "session_id": ""},
        {"type": "session_started", "session_id": None},
    ],
)
def test_session_id_is_none_for_anything_but_a_populated_handshake(frame):
    assert dt.session_id_from_frame(frame) is None


# ---------------------------------------------------------------------------
# Frame state machine, against recorded frame shapes
# ---------------------------------------------------------------------------


def _spec(index: int = 1, text: str = "Hello there friend.") -> dt.TurnSpec:
    return dt.TurnSpec(index=index, text=text, purpose="extraction-target")


#: Recorded from `backend/voice_processor.py:661-673`.
STREAM_COMPLETE = {
    "type": "stream_complete",
    "turn_id": "aaaa-1111",
    "full_text": "Noted, Dana. You work at Corvid Analytics.",
    "duration_ms": 4210,
    "tool_calls_used": 0,
}

#: Recorded from `backend/voice_processor.py:642-653`.
STREAM_CANCELLED = {
    "type": "stream_cancelled",
    "turn_id": "aaaa-1111",
    "partial_text": "Noted, Da",
}

#: Recorded from `backend/voice_processor.py:627-640`.
TURN_ERROR = {
    "type": "error",
    "kind": "bridge_timeout",
    "message": "knowledge bridge timed out",
    "retriable": True,
    "context": {"turn_id": "aaaa-1111"},
}

#: Recorded from `backend/server.py:808-816`. Connection-level, NOT a turn
#: terminal: its `context` is None.
PROTOCOL_ERROR = {
    "type": "error",
    "kind": "validation",
    "message": "Missing 'type' field",
    "retriable": False,
    "context": None,
}


def test_happy_path_turn_yields_stream_complete_with_its_fields():
    recorder = dt.TurnRecorder(_spec(), started_at=100.0)

    assert recorder.consume({"type": "stream_start", "turn_id": "aaaa-1111"}) is None
    assert recorder.consume({"type": "stream_token", "turn_id": "aaaa-1111", "token": "No"}) is None
    assert recorder.consume({"type": "stream_token", "turn_id": "aaaa-1111", "token": "ted"}) is None
    outcome = recorder.consume(STREAM_COMPLETE, now=104.5)

    assert outcome is not None
    assert outcome.terminal_type == "stream_complete"
    assert outcome.turn_id == "aaaa-1111"
    assert outcome.turn_id_matched is True
    assert outcome.text == "Noted, Dana. You work at Corvid Analytics."
    assert outcome.duration_ms == 4210
    assert outcome.tool_calls_used == 0
    assert outcome.token_count == 2
    assert outcome.elapsed_s == pytest.approx(4.5)
    assert recorder.streamed_text == "Noted"


def test_cancelled_turn_reports_partial_text_as_its_text():
    recorder = dt.TurnRecorder(_spec(), started_at=0.0)
    recorder.consume({"type": "stream_start", "turn_id": "aaaa-1111"})

    outcome = recorder.consume(STREAM_CANCELLED, now=1.0)

    assert outcome is not None
    assert outcome.terminal_type == "stream_cancelled"
    assert outcome.text == "Noted, Da"
    assert outcome.turn_id_matched is True
    assert outcome.duration_ms is None


def test_turn_error_reports_message_and_kind_and_matches_its_turn_id():
    recorder = dt.TurnRecorder(_spec(), started_at=0.0)
    recorder.consume({"type": "stream_start", "turn_id": "aaaa-1111"})

    outcome = recorder.consume(TURN_ERROR, now=2.0)

    assert outcome is not None
    assert outcome.terminal_type == "error"
    assert outcome.error_kind == "bridge_timeout"
    assert outcome.text == "knowledge bridge timed out"
    assert outcome.turn_id == "aaaa-1111"
    assert outcome.turn_id_matched is True


def test_protocol_error_terminates_the_turn_but_does_not_match_its_id():
    """`backend/server.py:808-816` builds this with `context: None`.

    It is not this turn's terminal, but it must still end the wait -- otherwise
    the driver sits on its per-turn timeout for a turn the server has already
    given up on.
    """
    recorder = dt.TurnRecorder(_spec(), started_at=0.0)
    recorder.consume({"type": "stream_start", "turn_id": "aaaa-1111"})

    outcome = recorder.consume(PROTOCOL_ERROR, now=0.5)

    assert outcome is not None
    assert outcome.terminal_type == "error"
    assert outcome.turn_id is None
    assert outcome.turn_id_matched is False


def test_non_terminal_frames_do_not_end_the_turn():
    """`state_cycle` idle is emitted with TTS off at `backend/voice_processor.py:696-698`."""
    recorder = dt.TurnRecorder(_spec(), started_at=0.0)

    for frame in (
        {"type": "state_cycle", "state": "think"},
        {"type": "heartbeat"},
        {"type": "transcription", "text": "x"},
        {"type": "tool_call_started", "name": "vault_search"},
        {"type": "state_cycle", "state": "idle"},
    ):
        assert recorder.consume(frame) is None


def test_feeding_a_recorder_after_its_terminal_raises():
    recorder = dt.TurnRecorder(_spec(), started_at=0.0)
    recorder.consume(STREAM_COMPLETE, now=1.0)

    with pytest.raises(RuntimeError, match="already terminated"):
        recorder.consume(STREAM_CANCELLED, now=2.0)


def test_terminal_turn_id_reads_the_nested_field_only_for_errors():
    assert dt.terminal_turn_id(STREAM_COMPLETE) == "aaaa-1111"
    assert dt.terminal_turn_id(STREAM_CANCELLED) == "aaaa-1111"
    assert dt.terminal_turn_id(TURN_ERROR) == "aaaa-1111"
    assert dt.terminal_turn_id(PROTOCOL_ERROR) is None


def test_summarise_outcomes_reports_zero_for_absent_terminal_types():
    outcomes = [
        dt.TurnRecorder(_spec(1), started_at=0.0).consume(STREAM_COMPLETE, now=1.0),
        dt.TurnRecorder(_spec(2), started_at=0.0).consume(TURN_ERROR, now=1.0),
    ]

    counts = dt.summarise_outcomes([o for o in outcomes if o is not None])

    assert counts == {
        "error": 1,
        "stream_cancelled": 0,
        "stream_complete": 1,
        "total": 2,
    }


def test_parse_frame_returns_none_for_binary_and_for_non_objects():
    assert dt.parse_frame(b"\x01\x02") is None
    assert dt.parse_frame("not json") is None
    assert dt.parse_frame("[1, 2]") is None
    assert dt.parse_frame('{"type": "heartbeat"}') == {"type": "heartbeat"}


def test_module_import_does_not_require_websockets():
    """The pure logic above must be testable on a host with no `websockets`.

    `resolve_websockets` imports it inside the function body for this reason.

    The real check is the module-level import scan below. An earlier revision
    also asserted `"websockets" not in sys.modules or dt.resolve_websockets is
    not None`, which was a tautology: `resolve_websockets` is a module-level
    function object and is never None, so the right disjunct always held and
    the assertion could not fail. Removed rather than left as a test that
    cannot fail -- it read exactly like the real one beside it.
    """
    assert callable(dt.resolve_websockets)
    source = (REPO_ROOT / "scripts" / "smoke" / "drive_turns.py").read_text(encoding="utf-8")
    module_level_imports = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "websockets" in line
    ]
    assert module_level_imports == []


def test_guard_message_names_the_variable_the_operator_must_set():
    message = dt._guard_message("ws://localhost:8003/ws", "not a recognized endpoint")

    assert "MIST_DEV_WS_HOSTS=localhost:8003,127.0.0.1:8003" in message
    assert "eval_isolation.py:308" in message


def test_guard_refuses_the_live_backend_and_admits_the_smoke_port(monkeypatch):
    """The denylist arm is at `backend/knowledge/eval_isolation.py:120-126`."""
    monkeypatch.setenv("MIST_DEV_WS_HOSTS", "localhost:8003,127.0.0.1:8003")

    with pytest.raises(SystemExit) as refusal:
        dt.guard_ws_target("ws://localhost:8001/ws")
    assert refusal.value.code == dt.EXIT_GUARD_REFUSED

    dt.guard_ws_target("ws://localhost:8003/ws")


def test_guard_refuses_the_smoke_port_when_the_allowlist_is_not_widened(monkeypatch):
    """`DEFAULT_DEV_WS_ENDPOINTS` (`eval_isolation.py:79`) names port 8002, not 8003."""
    monkeypatch.delenv("MIST_DEV_WS_HOSTS", raising=False)

    with pytest.raises(SystemExit) as refusal:
        dt.guard_ws_target("ws://localhost:8003/ws")
    assert refusal.value.code == dt.EXIT_GUARD_REFUSED


# ---------------------------------------------------------------------------
# baseline.py primitives
# ---------------------------------------------------------------------------


def _make_event_store(path: Path, *, sessions: int = 0, turns: int = 0) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE conversation_sessions (session_id TEXT, origin TEXT)")
    conn.execute("CREATE TABLE conversation_turn_events (session_id TEXT, turn_index INTEGER)")
    conn.execute(
        "CREATE TABLE curation_job_runs (run_id TEXT, job_name TEXT, trigger_source TEXT, "
        "started_at TEXT, outcome TEXT, examined INTEGER, produced INTEGER)"
    )
    for index in range(sessions):
        conn.execute("INSERT INTO conversation_sessions VALUES (?, 'real')", (f"s{index}",))
    for index in range(turns):
        conn.execute("INSERT INTO conversation_turn_events VALUES ('s0', ?)", (index,))
    conn.commit()
    conn.close()


def test_open_readonly_reads_real_rows_and_refuses_writes(tmp_path):
    db = tmp_path / "event_store.db"
    _make_event_store(db, sessions=2, turns=7)

    conn = bl.open_readonly(db)
    try:
        assert conn.execute("SELECT count(*) FROM conversation_sessions").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM conversation_turn_events").fetchone()[0] == 7
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO conversation_sessions VALUES ('x', 'real')")
    finally:
        conn.close()


def test_open_readonly_does_not_create_a_missing_database(tmp_path):
    missing = tmp_path / "absent.db"

    with pytest.raises(sqlite3.OperationalError):
        bl.open_readonly(missing)
    assert not missing.exists()


def test_table_counts_reports_a_missing_database_rather_than_zero(tmp_path):
    result = bl.table_counts(tmp_path / "absent.db")

    assert result["status"] == "unavailable"
    assert "no such file" in result["error"]


def test_table_counts_returns_real_counts(tmp_path):
    db = tmp_path / "event_store.db"
    _make_event_store(db, sessions=3, turns=11)

    result = bl.table_counts(db)

    assert result["status"] == "ok"
    assert result["counts"]["conversation_sessions"] == 3
    assert result["counts"]["conversation_turn_events"] == 11
    assert result["counts"]["curation_job_runs"] == 0


def test_parse_cypher_plain_parses_a_recorded_two_column_result():
    recorded = "entities, edges\n4, 9\n"

    assert bl.parse_cypher_plain(recorded) == [{"entities": "4", "edges": "9"}]


def test_parse_cypher_plain_keeps_commas_inside_quoted_values():
    recorded = 'id, display_name\n"corvid-analytics", "Corvid Analytics, Ltd"\n'

    assert bl.parse_cypher_plain(recorded) == [
        {"id": "corvid-analytics", "display_name": "Corvid Analytics, Ltd"}
    ]


@pytest.mark.parametrize("recorded", ["", "   \n", "entities\n"])
def test_parse_cypher_plain_returns_empty_for_output_with_no_data_rows(recorded):
    assert bl.parse_cypher_plain(recorded) == []


def test_file_listing_returns_real_sizes_and_skips_dot_git(tmp_path):
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "note.md").write_text("hello", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")

    result = bl.file_listing(tmp_path)

    assert result["status"] == "ok"
    assert set(result["files"]) == {"sessions/note.md"}
    assert result["files"]["sessions/note.md"]["size"] == 5
    assert result["files"]["sessions/note.md"]["mtime"] > 0


def test_file_listing_reports_a_missing_directory(tmp_path):
    result = bl.file_listing(tmp_path / "nope")

    assert result["status"] == "unavailable"
    assert "no such directory" in result["error"]


def test_git_status_reports_a_missing_directory_as_unavailable(tmp_path):
    result = bl.git_status(tmp_path / "definitely-absent")

    assert result["status"] == "unavailable"
    assert "no such directory" in result["error"]


def test_git_status_returns_real_porcelain_lines_for_a_repository():
    """Not asserted empty: the worktree this runs in may have staged work."""
    result = bl.git_status(REPO_ROOT)

    assert result["status"] == "ok"
    assert isinstance(result["porcelain"], list)
    assert all(isinstance(line, str) and line for line in result["porcelain"])


def test_newest_curation_rows_returns_real_rows_newest_first(tmp_path):
    db = tmp_path / "event_store.db"
    _make_event_store(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO curation_job_runs VALUES "
        "('old', 'self_reflection', 'scheduled', '2026-09-20T00:00:00', 'completed', 0, 0)"
    )
    conn.execute(
        "INSERT INTO curation_job_runs VALUES "
        "('new', 'confidence_decay', 'scheduled', '2026-09-22T00:00:00', 'completed', 7, 1)"
    )
    conn.commit()
    conn.close()

    result = bl.newest_curation_rows(db, limit=2)

    assert result["status"] == "ok"
    assert [row["run_id"] for row in result["rows"]] == ["new", "old"]
    assert result["rows"][0]["examined"] == 7


def test_run_command_reports_a_missing_executable_instead_of_raising():
    result = bl.run_command(["definitely-not-a-real-executable-9f3a"])

    assert result.ok is False
    assert "executable not found" in result.stderr


# ---------------------------------------------------------------------------
# baseline.py comparison
# ---------------------------------------------------------------------------


def _snapshot(
    *,
    sessions: int,
    turns: int,
    curation: int,
    curation_rows: list[dict] | None = None,
    nodes: int = 100,
) -> dict:
    return {
        "phase": "pre",
        "checks": {
            "B1_live_event_store": {
                "tables": {
                    "status": "ok",
                    "counts": {
                        "conversation_sessions": sessions,
                        "conversation_turn_events": turns,
                        "curation_job_runs": curation,
                    },
                },
                "newest_curation_rows": {"status": "ok", "rows": curation_rows or []},
            },
            "B2_live_vault": {
                "git": {"status": "ok", "porcelain": []},
                "listing": {"status": "ok", "files": {"a.md": {"size": 1, "mtime": 1.0}}},
            },
            "B3_dev_state": {
                "git": {"status": "ok", "porcelain": []},
                "listing": {"status": "ok", "files": {}},
                "tables": {
                    "status": "ok",
                    "counts": {
                        "conversation_sessions": 1,
                        "conversation_turn_events": 2,
                        "curation_job_runs": 3,
                    },
                },
            },
            "B4_live_neo4j": {
                "status": "ok",
                "counts": {"nodes": nodes, "entities": 10, "extracted_from_edges": 5},
            },
            "B8_docker_ps": {
                "status": "ok",
                "containers": {"mist-backend": "abc", "mist-neo4j": "def"},
            },
        },
    }


def test_compare_is_clean_when_nothing_moved():
    pre = _snapshot(sessions=4, turns=0, curation=206)
    post = _snapshot(sessions=4, turns=0, curation=206)

    report, code = bl.render_comparison(bl.compare(pre, post))

    assert code == bl.EXIT_CLEAN
    assert "VERDICT: CLEAN" in report


def test_compare_calls_a_new_live_turn_row_contamination():
    pre = _snapshot(sessions=4, turns=0, curation=206)
    post = _snapshot(sessions=4, turns=1, curation=206)

    deltas = bl.compare(pre, post)
    report, code = bl.render_comparison(deltas)

    assert code == bl.EXIT_CONTAMINATED
    contaminated = [d for d in deltas if d.verdict == bl.CONTAMINATION]
    assert [d.field for d in contaminated] == ["conversation_turn_events"]
    assert "must be exactly unchanged" in contaminated[0].reasoning
    assert "precise failure this whole isolation design exists to prevent" in (
        contaminated[0].reasoning
    )
    assert "VERDICT: CONTAMINATED" in report


def test_compare_calls_a_new_live_session_row_contamination_with_no_excuse():
    pre = _snapshot(sessions=4, turns=0, curation=206)
    post = _snapshot(sessions=5, turns=0, curation=206)

    deltas = bl.compare(pre, post)
    _, code = bl.render_comparison(deltas)

    assert code == bl.EXIT_CONTAMINATED
    contaminated = [d for d in deltas if d.verdict == bl.CONTAMINATION]
    assert [d.field for d in contaminated] == ["conversation_sessions"]
    assert "no benign explanation" in contaminated[0].reasoning


def test_compare_accepts_a_new_scheduled_curation_row_on_live_and_says_why():
    """The live scheduler runs on its own timer, `scheduler.py:293-327`."""
    new_row = {
        "run_id": "r2",
        "job_name": "confidence_decay",
        "trigger_source": "scheduled",
        "started_at": "2026-09-22T03:00:00+00:00",
        "outcome": "completed",
        "examined": 0,
        "produced": 0,
    }
    pre = _snapshot(sessions=4, turns=0, curation=206, curation_rows=[])
    post = _snapshot(sessions=4, turns=0, curation=207, curation_rows=[new_row])

    deltas = bl.compare(pre, post)
    report, code = bl.render_comparison(deltas)

    assert code == bl.EXIT_CLEAN
    adjudicated = [d for d in deltas if d.field.startswith("new curation row")]
    assert len(adjudicated) == 1
    assert adjudicated[0].verdict == bl.EXPECTED
    assert "live behaving normally, not contamination" in adjudicated[0].reasoning
    assert "examined=0 is the live symptom" in adjudicated[0].reasoning
    assert "EXPECTED" in report


def test_compare_flags_a_manual_curation_row_for_review():
    manual = {
        "run_id": "r3",
        "job_name": "self_reflection",
        "trigger_source": "manual",
        "started_at": "2026-09-22T03:00:00+00:00",
        "outcome": "completed",
        "examined": 0,
        "produced": 0,
    }
    pre = _snapshot(sessions=4, turns=0, curation=206, curation_rows=[])
    post = _snapshot(sessions=4, turns=0, curation=207, curation_rows=[manual])

    deltas = bl.compare(pre, post)

    reviewed = [d for d in deltas if d.verdict == bl.REVIEW]
    assert len(reviewed) == 1
    assert "run_all_once" in reviewed[0].reasoning


def test_compare_calls_a_live_graph_delta_contamination():
    pre = _snapshot(sessions=4, turns=0, curation=206, nodes=100)
    post = _snapshot(sessions=4, turns=0, curation=206, nodes=104)

    deltas = bl.compare(pre, post)
    _, code = bl.render_comparison(deltas)

    assert code == bl.EXIT_CONTAMINATED
    assert any(d.check == "B4_live_neo4j" and d.verdict == bl.CONTAMINATION for d in deltas)


def test_compare_calls_a_live_vault_file_change_contamination():
    pre = _snapshot(sessions=4, turns=0, curation=206)
    post = _snapshot(sessions=4, turns=0, curation=206)
    post["checks"]["B2_live_vault"]["listing"]["files"]["b.md"] = {"size": 2, "mtime": 9.0}

    deltas = bl.compare(pre, post)
    _, code = bl.render_comparison(deltas)

    assert code == bl.EXIT_CONTAMINATED
    listing = [d for d in deltas if d.check == "B2_live_vault" and d.field == "listing"]
    assert listing[0].post["added"] == ["b.md"]


def test_compare_is_undecided_rather_than_clean_when_a_check_was_unavailable():
    pre = _snapshot(sessions=4, turns=0, curation=206)
    post = _snapshot(sessions=4, turns=0, curation=206)
    post["checks"]["B4_live_neo4j"] = {"status": "unavailable", "error": "docker not running"}

    report, code = bl.render_comparison(bl.compare(pre, post))

    assert code == bl.EXIT_UNDECIDED
    assert "VERDICT: UNDECIDED" in report
    assert "not a check that passed" in report


def test_compare_expects_smoke_containers_and_reviews_others():
    pre = _snapshot(sessions=4, turns=0, curation=206)
    post = _snapshot(sessions=4, turns=0, curation=206)
    post["checks"]["B8_docker_ps"]["containers"]["mist-backend-smoke"] = "ghi"
    post["checks"]["B8_docker_ps"]["containers"]["something-else"] = "jkl"

    deltas = bl.compare(pre, post)

    by_field = {d.field: d for d in deltas if d.check == "B8_docker_ps"}
    assert by_field["mist-backend-smoke"].verdict == bl.EXPECTED
    assert by_field["something-else"].verdict == bl.REVIEW


def test_compare_calls_a_recreated_live_container_contamination():
    pre = _snapshot(sessions=4, turns=0, curation=206)
    post = _snapshot(sessions=4, turns=0, curation=206)
    post["checks"]["B8_docker_ps"]["containers"]["mist-backend"] = "different-id"

    deltas = bl.compare(pre, post)
    _, code = bl.render_comparison(deltas)

    assert code == bl.EXIT_CONTAMINATED
    recreated = [d for d in deltas if d.field == "mist-backend"]
    assert "recreated" in recreated[0].reasoning


# ---------------------------------------------------------------------------
# assert_artifacts.py -- transcript
# ---------------------------------------------------------------------------

SESSION_ID = "7f3a0c2e-1111-4222-8333-444455556666"


def _transcript_lines(terminals: list[str], *, closed: bool = True, summary: bool = True) -> str:
    lines = [
        {"record": "header", "schema_version": 1, "ws_url": "ws://localhost:8003/ws"},
        {"record": "session", "session_id": SESSION_ID},
    ]
    for index, terminal in enumerate(terminals, start=1):
        lines.append(
            {
                "record": "turn_outcome",
                "turn_index": index,
                "terminal_type": terminal,
                "turn_id": f"t{index}",
            }
        )
    if closed:
        lines.append({"record": "note", "event": "closed", "close_code": 1000})
    if summary:
        counts = {name: terminals.count(name) for name in sorted(set(dt.TERMINAL_FRAME_TYPES))}
        counts["total"] = len(terminals)
        lines.append(
            {
                "record": "summary",
                "schema_version": 1,
                "session_id": SESSION_ID,
                "disconnected_at": "2026-09-22T04:00:00+00:00",
                "terminal_counts": counts,
                "turns_planned": 5,
                "aborted": None,
            }
        )
    return "\n".join(json.dumps(line) for line in lines) + "\n"


def test_parse_transcript_reads_the_summary_record():
    facts = aa.parse_transcript(_transcript_lines(["stream_complete"] * 5))

    assert facts.present is True
    assert facts.session_id == SESSION_ID
    assert facts.terminal_counts["stream_complete"] == 5
    assert facts.disconnected_at == "2026-09-22T04:00:00+00:00"
    assert facts.closed is True
    assert facts.aborted is None


def test_parse_transcript_falls_back_to_turn_outcomes_when_the_driver_was_killed():
    text = _transcript_lines(["stream_complete", "stream_cancelled"], closed=False, summary=False)

    facts = aa.parse_transcript(text)

    assert facts.present is True
    assert facts.terminal_counts == {"stream_complete": 1, "stream_cancelled": 1, "total": 2}
    assert facts.closed is False
    assert "did not finish" in facts.aborted


def test_parse_transcript_does_not_call_a_failed_close_clean():
    """`closed` means CLEANLY closed; drive_turns.py records `close_error`."""
    lines = _transcript_lines(["stream_complete"] * 5, closed=False)
    lines = lines.replace(
        '{"record": "summary"',
        json.dumps({"record": "note", "event": "closed", "close_error": "ConnectionResetError"})
        + '\n{"record": "summary"',
    )

    facts = aa.parse_transcript(lines)

    assert facts.present is True
    assert facts.closed is False


def test_parse_transcript_reports_an_unusable_transcript():
    facts = aa.parse_transcript("not jsonl at all\n{}\n")

    assert facts.present is False
    assert "no summary and no turn_outcome" in facts.error


# ---------------------------------------------------------------------------
# assert_artifacts.py -- log parsing
# ---------------------------------------------------------------------------

#: Recorded shapes from `backend/knowledge/extraction/pipeline.py:642,650,676,687`.
SKIP_LOG = (
    "2026-09-22 04:00:01 - backend.knowledge.extraction.pipeline - INFO - "
    "Extraction skipped (too short) for 'ok'\n"
    "2026-09-22 04:00:02 - backend.knowledge.extraction.pipeline - INFO - "
    "Extraction skipped (rate-limited) for 'Corvid Analytics is based in Sheffield'\n"
    "2026-09-22 04:00:03 - backend.knowledge.extraction.pipeline - INFO - "
    "Extraction skipped (significance 0.118 < 0.300) for 'What do you know about my telem'\n"
    "2026-09-22 04:00:04 - backend.knowledge.extraction.pipeline - INFO - "
    "Extraction skipped (duplicate) for 'My name is Dana Whitfield'\n"
)


def test_parse_extraction_skips_identifies_all_four_gates_with_citations():
    skips = aa.parse_extraction_skips(SKIP_LOG)

    assert [skip.gate for skip in skips] == [
        "too short",
        "rate-limited",
        "significance",
        "duplicate",
    ]
    assert skips[2].reason == "significance 0.118 < 0.300"
    assert "pipeline.py:674-682" in skips[2].citation
    assert skips[1].utterance == "Corvid Analytics is based in Sheffield"


def test_parse_extraction_skips_returns_empty_for_a_log_with_none():
    assert aa.parse_extraction_skips("nothing interesting here\n") == []


def test_find_lines_matches_the_two_scheduler_markers():
    log = (
        "INFO - Curation scheduler started with 6 jobs\n"
        "INFO - Curation scheduler NOT started: MIST_CURATION_SCHEDULER_ENABLED is off.\n"
    )

    assert aa.find_lines(log, aa.SCHEDULER_STARTED) == [
        "INFO - Curation scheduler started with 6 jobs"
    ]
    assert len(aa.find_lines(log, aa.SCHEDULER_NOT_STARTED)) == 1


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------


def _turn_rows(count: int, session_id: str = SESSION_ID) -> list[dict]:
    return [
        {"session_id": session_id, "turn_index": index, "user_utterance": f"utterance {index}"}
        for index in range(count)
    ]


def _session_row(**overrides) -> dict:
    row = {
        "session_id": SESSION_ID,
        "origin": "real",
        "turn_count": 5,
        "input_modality": "text",
        "ended_at": None,
    }
    row.update(overrides)
    return row


def test_a1_passes_on_five_rows_and_a_matching_session_row():
    verdict = aa.adjudicate_a1(
        _turn_rows(5),
        [_session_row()],
        aa.parse_transcript(_transcript_lines(["stream_complete"] * 5)),
        SESSION_ID,
    )

    assert verdict.status == aa.PASS
    assert "5 turn rows" in verdict.headline
    assert any("ended_at is NOT asserted" in line for line in verdict.evidence)


def test_a1_fails_when_the_turns_completed_but_the_table_is_empty():
    verdict = aa.adjudicate_a1(
        [],
        [],
        aa.parse_transcript(_transcript_lines(["stream_complete"] * 5)),
        SESSION_ID,
    )

    assert verdict.status == aa.FAIL
    assert "conversation_handler.py:2279" in " ".join(verdict.evidence)
    assert "2306-2307" in " ".join(verdict.evidence)


def test_a1_is_inconclusive_when_fewer_than_five_turns_completed():
    transcript = aa.parse_transcript(
        _transcript_lines(["stream_complete", "stream_complete", "error"])
    )

    verdict = aa.adjudicate_a1([], [], transcript, SESSION_ID)

    assert verdict.status == aa.INCONCLUSIVE
    assert "2 of 5" in verdict.headline
    assert any("transcript is the discriminator" in line for line in verdict.evidence)


def test_a1_is_inconclusive_when_the_transcript_is_unusable():
    transcript = aa.parse_transcript("")

    verdict = aa.adjudicate_a1(_turn_rows(5), [_session_row()], transcript, SESSION_ID)

    assert verdict.status == aa.INCONCLUSIVE
    assert "no usable driver transcript" in verdict.headline


@pytest.mark.parametrize(
    ("overrides", "expected_fragment"),
    [
        ({"origin": "test"}, "origin is 'test'"),
        ({"input_modality": "voice"}, "input_modality is 'voice'"),
        ({"turn_count": 3}, "turn_count is 3"),
    ],
)
def test_a1_fails_on_each_wrong_session_column(overrides, expected_fragment):
    verdict = aa.adjudicate_a1(
        _turn_rows(5),
        [_session_row(**overrides)],
        aa.parse_transcript(_transcript_lines(["stream_complete"] * 5)),
        SESSION_ID,
    )

    assert verdict.status == aa.FAIL
    assert any(expected_fragment in line for line in verdict.evidence)


def test_a1_reports_a_null_turn_index_as_a_mismatch_rather_than_raising():
    rows = _turn_rows(5)
    rows[2]["turn_index"] = None

    verdict = aa.adjudicate_a1(
        rows,
        [_session_row()],
        aa.parse_transcript(_transcript_lines(["stream_complete"] * 5)),
        SESSION_ID,
    )

    assert verdict.status == aa.FAIL
    assert any("turn_index values are" in line for line in verdict.evidence)


def test_a1_ignores_rows_belonging_to_another_session():
    verdict = aa.adjudicate_a1(
        _turn_rows(5) + _turn_rows(3, session_id="other"),
        [_session_row(), _session_row(session_id="other", turn_count=3)],
        aa.parse_transcript(_transcript_lines(["stream_complete"] * 5)),
        SESSION_ID,
    )

    assert verdict.status == aa.PASS


# ---------------------------------------------------------------------------
# A2
# ---------------------------------------------------------------------------


def test_a2_passes_on_a_positive_entity_count():
    verdict = aa.adjudicate_a2(
        entity_count=3,
        edge_count=4,
        entity_rows=[{"id": "dana-whitfield", "entity_type": "Person", "provenance": "extraction"}],
        skips=[],
        a1_status=aa.PASS,
        log_available=True,
        cypher_error=None,
    )

    assert verdict.status == aa.PASS
    assert "dana-whitfield" in " ".join(verdict.evidence)


def test_a2_always_states_that_there_is_no_success_log_line():
    """`graph_writer.py:198` logs only for the DOCUMENT branch."""
    verdict = aa.adjudicate_a2(0, 0, [], [], aa.PASS, True, None)

    joined = " ".join(verdict.evidence)
    assert "NO success log line" in joined
    assert "graph_writer.py:198" in joined


def test_a2_is_inconclusive_with_the_gate_and_its_numbers_when_everything_was_skipped():
    skips = aa.parse_extraction_skips(SKIP_LOG)

    verdict = aa.adjudicate_a2(0, 0, [], skips, aa.PASS, True, None)

    assert verdict.status == aa.INCONCLUSIVE
    joined = " ".join(verdict.evidence)
    assert "significance 0.118 < 0.300" in joined
    assert "pipeline.py:674-682" in joined
    assert "normal behaviour, not a defect" in verdict.headline


def test_a2_fails_when_a1_passed_and_nothing_was_skipped_and_nothing_was_written():
    verdict = aa.adjudicate_a2(0, 0, [], [], aa.PASS, True, None)

    assert verdict.status == aa.FAIL
    assert "Extraction never ran" in verdict.headline
    assert "conversation_handler.py:1662-1674" in " ".join(verdict.evidence)


def test_a2_is_inconclusive_rather_than_failing_when_no_log_was_readable():
    verdict = aa.adjudicate_a2(0, 0, [], [], aa.PASS, False, None)

    assert verdict.status == aa.INCONCLUSIVE
    assert "no backend log was readable" in verdict.headline


def test_a2_is_inconclusive_when_a1_did_not_pass():
    verdict = aa.adjudicate_a2(0, 0, [], [], aa.INCONCLUSIVE, True, None)

    assert verdict.status == aa.INCONCLUSIVE
    assert "extraction had nothing to run on" in verdict.headline


def test_a2_is_inconclusive_when_the_cypher_query_failed():
    verdict = aa.adjudicate_a2(None, None, [], [], aa.PASS, True, "connection refused")

    assert verdict.status == aa.INCONCLUSIVE
    assert "connection refused" in " ".join(verdict.evidence)


# ---------------------------------------------------------------------------
# A3
# ---------------------------------------------------------------------------

NOTE_NAME = "2026-09-22-logging-this-under-tag-smoke-7f3a.md"
NOTE_TEXT = "---\nstatus: completed\n---\n\nDana Whitfield works at Corvid Analytics.\n"
FACTS = ["Dana Whitfield", "Corvid Analytics", "Redpanda"]
WRITTEN_LINE = (
    f"2026-09-22 04:00:10 - backend.chat.conversation_handler - DEBUG - "
    f"Session note written for {SESSION_ID} at /app/smoke-state/vault/sessions/{NOTE_NAME}\n"
)
THRESHOLD_LINE = (
    f"2026-09-22 04:00:10 - backend.chat.conversation_handler - DEBUG - "
    f"Session {SESSION_ID} below synthesis threshold; writing no vault note\n"
)


def test_a3_passes_and_attributes_the_note_to_the_disconnect_hook():
    verdict = aa.adjudicate_a3(
        before=set(),
        after={NOTE_NAME: 1_800_000_100.0},
        note_texts={NOTE_NAME: NOTE_TEXT},
        corpus_facts=FACTS,
        transcript=aa.parse_transcript(_transcript_lines(["stream_complete"] * 5)),
        log_text=WRITTEN_LINE,
        log_available=True,
        session_id=SESSION_ID,
    )

    assert verdict.status == aa.PASS
    joined = " ".join(verdict.evidence)
    assert "PROVEN" in joined
    assert "conversation_handler.py:1970" in joined
    assert "Dana Whitfield" in joined


def test_a3_attributes_the_note_to_catchup_when_the_end_session_line_is_absent():
    verdict = aa.adjudicate_a3(
        before=set(),
        after={NOTE_NAME: 1_800_000_900.0},
        note_texts={NOTE_NAME: NOTE_TEXT},
        corpus_facts=FACTS,
        transcript=aa.parse_transcript(_transcript_lines(["stream_complete"] * 5)),
        log_text="some other debug output\n",
        log_available=True,
        session_id=SESSION_ID,
    )

    assert verdict.status == aa.PASS
    joined = " ".join(verdict.evidence)
    assert "session_catchup.py:113" in joined
    assert "INFERRED" in joined


def test_a3_attribution_is_a_likelihood_only_when_no_log_was_readable():
    verdict = aa.adjudicate_a3(
        before=set(),
        after={NOTE_NAME: 1_800_000_000.0},
        note_texts={NOTE_NAME: NOTE_TEXT},
        corpus_facts=FACTS,
        transcript=aa.parse_transcript(_transcript_lines(["stream_complete"] * 5)),
        log_text="",
        log_available=False,
        session_id=SESSION_ID,
    )

    assert verdict.status == aa.PASS
    joined = " ".join(verdict.evidence)
    assert "LIKELIHOOD ONLY, not proof" in joined
    assert "logger.debug" in joined


def test_a3_is_inconclusive_when_the_note_names_no_corpus_fact():
    verdict = aa.adjudicate_a3(
        before=set(),
        after={NOTE_NAME: 1.0},
        note_texts={NOTE_NAME: "nothing relevant in here"},
        corpus_facts=FACTS,
        transcript=aa.parse_transcript(_transcript_lines(["stream_complete"] * 5)),
        log_text=WRITTEN_LINE,
        log_available=True,
        session_id=SESSION_ID,
    )

    assert verdict.status == aa.INCONCLUSIVE
    assert "names none of the corpus facts" in verdict.headline


def test_a3_is_inconclusive_flavour_one_when_the_synthesizer_declined():
    verdict = aa.adjudicate_a3(
        before=set(),
        after={},
        note_texts={},
        corpus_facts=FACTS,
        transcript=aa.parse_transcript(_transcript_lines(["stream_complete"] * 5)),
        log_text=THRESHOLD_LINE,
        log_available=True,
        session_id=SESSION_ID,
    )

    assert verdict.status == aa.INCONCLUSIVE
    assert "conversation_handler.py:1961-1963" in verdict.headline
    assert "not a pipeline failure" in verdict.headline
    assert "session_synthesizer.py:19" in " ".join(verdict.evidence)


def test_a3_fails_on_a_clean_close_with_no_note_and_no_declining_line():
    verdict = aa.adjudicate_a3(
        before=set(),
        after={},
        note_texts={},
        corpus_facts=FACTS,
        transcript=aa.parse_transcript(_transcript_lines(["stream_complete"] * 5)),
        log_text="INFO - something unrelated\n",
        log_available=True,
        session_id=SESSION_ID,
    )

    assert verdict.status == aa.FAIL
    joined = " ".join(verdict.evidence)
    assert "backend/server.py:1015" in joined
    assert "Session note write failed" in joined


def test_a3_is_inconclusive_when_the_driver_recorded_no_clean_close():
    transcript = aa.parse_transcript(
        _transcript_lines(["stream_complete"] * 5, closed=False, summary=False)
    )

    verdict = aa.adjudicate_a3(
        before=set(),
        after={},
        note_texts={},
        corpus_facts=FACTS,
        transcript=transcript,
        log_text="INFO - something unrelated\n",
        log_available=True,
        session_id=SESSION_ID,
    )

    assert verdict.status == aa.INCONCLUSIVE
    assert "clean close" in verdict.headline


def test_a3_ignores_a_note_that_was_already_there_before_the_conversation():
    verdict = aa.adjudicate_a3(
        before={NOTE_NAME},
        after={NOTE_NAME: 1.0},
        note_texts={},
        corpus_facts=FACTS,
        transcript=aa.parse_transcript(_transcript_lines(["stream_complete"] * 5)),
        log_text="INFO - something unrelated\n",
        log_available=True,
        session_id=SESSION_ID,
    )

    assert verdict.status == aa.FAIL
    assert "no note appeared" in verdict.headline


# ---------------------------------------------------------------------------
# A4
# ---------------------------------------------------------------------------


def _curation_row(**overrides) -> dict:
    row = {
        "run_id": "r1",
        "job_name": "self_reflection",
        "trigger_source": "scheduled",
        "started_at": "2026-09-22T04:10:00+00:00",
        "outcome": "completed",
        "examined": 5,
        "produced": 2,
        "error": None,
    }
    row.update(overrides)
    return row


STARTED_LOG = "INFO - Curation scheduler started with 6 jobs\n"
NOT_STARTED_LOG = (
    "INFO - Curation scheduler NOT started: MIST_CURATION_SCHEDULER_ENABLED is off.\n"
)


def test_a4_passes_on_a_scheduled_completed_run_that_examined_five():
    verdict = aa.adjudicate_a4([_curation_row()], STARTED_LOG, True)

    assert verdict.status == aa.PASS
    assert "examined=5" in verdict.headline


def test_a4_fails_when_the_run_completed_having_examined_nothing():
    verdict = aa.adjudicate_a4([_curation_row(examined=0)], STARTED_LOG, True)

    assert verdict.status == aa.FAIL
    assert "THIS IS THE BUG THE EXPERIMENT EXISTS TO FIND" in verdict.headline
    joined = " ".join(verdict.evidence)
    assert "run_record.py:83" in joined
    assert "self_reflection.py:60-61" in joined
    assert "store.py:315-322" in joined


def test_a4_fails_on_a_failed_outcome_and_points_at_the_error_column():
    verdict = aa.adjudicate_a4(
        [_curation_row(outcome="failed", examined=None, error="boom")], STARTED_LOG, True
    )

    assert verdict.status == aa.FAIL
    assert "outcome='failed'" in verdict.headline
    assert "error='boom'" in " ".join(verdict.evidence)


def test_a4_is_inconclusive_with_the_not_started_line_when_there_is_no_row():
    verdict = aa.adjudicate_a4([], NOT_STARTED_LOG, True)

    assert verdict.status == aa.INCONCLUSIVE
    assert "scheduler did not start" in verdict.headline
    joined = " ".join(verdict.evidence)
    assert "MIST_SMOKE_SCHEDULER=1" in joined
    assert NOT_STARTED_LOG.strip() in joined


def test_a4_is_inconclusive_and_rules_out_a_timer_when_the_scheduler_started_with_no_row():
    """`scheduler.py:305-311` makes every enabled job due on the first pass."""
    verdict = aa.adjudicate_a4([], STARTED_LOG, True)

    assert verdict.status == aa.INCONCLUSIVE
    joined = " ".join(verdict.evidence)
    assert "'it has not fired yet' is NOT an available answer" in joined
    assert "scheduler.py:305-311" in joined


def test_a4_is_inconclusive_when_neither_scheduler_line_was_found():
    verdict = aa.adjudicate_a4([], "", False)

    assert verdict.status == aa.INCONCLUSIVE
    joined = " ".join(verdict.evidence)
    assert aa.SCHEDULER_STARTED in joined
    assert aa.SCHEDULER_NOT_STARTED in joined


def test_a4_is_inconclusive_when_only_manual_rows_exist():
    verdict = aa.adjudicate_a4([_curation_row(trigger_source="manual")], STARTED_LOG, True)

    assert verdict.status == aa.INCONCLUSIVE
    assert "run_all_once" in verdict.headline


def test_a4_passes_but_notes_an_unexpected_examined_count():
    verdict = aa.adjudicate_a4([_curation_row(examined=9)], STARTED_LOG, True)

    assert verdict.status == aa.PASS
    assert any("expected 5" in line for line in verdict.evidence)


# ---------------------------------------------------------------------------
# assert_artifacts.py -- invocation guards and store reads
# ---------------------------------------------------------------------------


def test_session_id_must_be_a_uuid_because_the_server_mints_one():
    """`scripts/eval_harness/websocket_gauntlets_runbook.md` claims 'default'."""
    with pytest.raises(aa.InvocationError, match="not a uuid4"):
        aa._require_session_id("default")

    assert aa._require_session_id(SESSION_ID) == SESSION_ID


def test_missing_session_id_names_where_to_find_it():
    with pytest.raises(aa.InvocationError, match="SMOKE_SESSION_ID"):
        aa._require_session_id(None)


def test_read_event_store_returns_parsed_rows(tmp_path):
    db = tmp_path / "event_store.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE conversation_sessions (session_id TEXT, origin TEXT, turn_count INTEGER, "
        "input_modality TEXT, ended_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE conversation_turn_events (session_id TEXT, turn_index INTEGER, "
        "user_utterance TEXT)"
    )
    conn.execute(
        "CREATE TABLE curation_job_runs (run_id TEXT, job_name TEXT, trigger_source TEXT, "
        "started_at TEXT, outcome TEXT, examined INTEGER, produced INTEGER, error TEXT)"
    )
    conn.execute(
        "INSERT INTO conversation_sessions VALUES (?, 'real', 5, 'text', NULL)", (SESSION_ID,)
    )
    for index in range(5):
        conn.execute(
            "INSERT INTO conversation_turn_events VALUES (?, ?, ?)",
            (SESSION_ID, index, f"utterance number {index}"),
        )
    conn.execute(
        "INSERT INTO curation_job_runs VALUES "
        "('r1', 'self_reflection', 'scheduled', '2026-09-22T04:10:00', 'completed', 5, 2, NULL)"
    )
    conn.execute(
        "INSERT INTO curation_job_runs VALUES "
        "('r2', 'confidence_decay', 'scheduled', '2026-09-22T04:10:00', 'completed', 0, 0, NULL)"
    )
    conn.commit()
    conn.close()

    store = aa.read_event_store(db)

    assert store["status"] == "ok"
    assert len(store["turns"]) == 5
    assert store["turns"][0]["user_utterance"] == "utterance number 0"
    assert store["sessions"][0]["origin"] == "real"
    assert store["sessions"][0]["ended_at"] is None
    assert [row["job_name"] for row in store["curation"]] == ["self_reflection"]


def test_read_event_store_reports_a_missing_database(tmp_path):
    result = aa.read_event_store(tmp_path / "absent.db")

    assert result["status"] == "unavailable"


def test_list_session_notes_reads_real_files(tmp_path):
    sessions = tmp_path / "vault" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / NOTE_NAME).write_text(NOTE_TEXT, encoding="utf-8")
    (sessions / "not-a-note.txt").write_text("x", encoding="utf-8")

    notes = aa.list_session_notes(tmp_path / "vault")

    assert set(notes) == {NOTE_NAME}
    assert notes[NOTE_NAME] > 0


def test_list_session_notes_is_empty_when_the_vault_does_not_exist(tmp_path):
    assert aa.list_session_notes(tmp_path / "nope") == {}


def test_load_corpus_facts_reads_the_shipped_corpus():
    facts = aa.load_corpus_facts(TURNS_PATH)

    assert "Dana Whitfield" in facts
    assert "Emil Nakamura" in facts


def test_session_note_pattern_matches_the_writer_s_output_shape():
    """`backend/vault/writer.py:538` renders `<date>-<kebab-slug>.md`."""
    assert aa.SESSION_NOTE_RE.match("2026-09-22-logging-this-under-tag-smoke-7f3a.md")
    assert not aa.SESSION_NOTE_RE.match("2026-09-22-Logging.md")
    assert not aa.SESSION_NOTE_RE.match("notes.md")


def test_render_includes_the_status_the_assertion_and_every_evidence_line():
    verdict = aa.Verdict("A9", "example", aa.FAIL, "headline here", ("one", "two"))

    rendered = aa.render(verdict)

    assert "FAIL  A9 -- example" in rendered
    assert "headline here" in rendered
    assert "one" in rendered and "two" in rendered
