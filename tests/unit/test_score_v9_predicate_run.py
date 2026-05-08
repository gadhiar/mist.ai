"""Unit tests for scripts/eval_harness/score_v9_predicate_run.py.

Focused on session_id scoping — the V9 scorer's headline correctness
property post-fix. Extraction llm_call records have session_id=None, so
the scorer scopes via the time window (ts_iso range) of TURN records
matching the requested session_id, with an extraction-lag buffer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_harness.score_v9_predicate_run import (  # noqa: E402  -- after sys.path insert
    get_session_ts_window,
    main,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _turn(session_id: str, utterance: str, ts_iso: str) -> dict:
    return {
        "phase": "turn",
        "ts_iso": ts_iso,
        "session_id": session_id,
        "event_id": f"evt-{session_id}-{hash(utterance) & 0xFFFFFF:x}",
        "user_id": "User",
        "utterance": utterance,
        "retrieval": {"intent": "historical", "total_facts": 0},
        "llm_passes": [],
        "total_turn_ms": 100.0,
    }


def _extraction_llm_call(utterance: str, relationships: list[str], ts_iso: str) -> dict:
    """Build a phase=llm_call record mimicking the extraction.ontology call site."""
    response_payload = {
        "entities": [{"id": "e1", "name": "x", "type": "Concept"}],
        "relationships": [{"source": "e1", "target": "e1", "type": rt} for rt in relationships],
    }
    return {
        "phase": "llm_call",
        "ts_iso": ts_iso,
        "event_id": None,
        "session_id": None,  # the bug: extraction call site doesn't propagate
        "call_site": "extraction.ontology",
        "pass_num": 1,
        "model": "fake-model",
        "latency_ms": 1000.0,
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "Extract entities and relationships from the user message.",
                },
                {
                    "role": "user",
                    "content": (f'Utterance: "{utterance}"\n' f"Output:\n"),
                },
            ],
            "tools": None,
            "temperature": 0.0,
            "max_tokens": 2048,
        },
        "response": {
            "content": json.dumps(response_payload),
            "tool_calls": None,
            "partial": False,
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        },
    }


class TestGetSessionTsWindow:
    def test_returns_min_max_ts_for_session(self, tmp_path: Path) -> None:
        path = tmp_path / "debug.jsonl"
        _write_jsonl(
            path,
            [
                _turn("sess-A", "first", "2026-05-08T10:00:00+00:00"),
                _turn("sess-A", "second", "2026-05-08T10:01:30+00:00"),
                _turn("sess-A", "third", "2026-05-08T10:00:45+00:00"),
                _turn("sess-B", "other", "2026-05-08T11:00:00+00:00"),
            ],
        )

        with path.open() as fh:
            records = (json.loads(line) for line in fh if line.strip())
            window = get_session_ts_window(records, "sess-A")

        assert window == ("2026-05-08T10:00:00+00:00", "2026-05-08T10:01:30+00:00")

    def test_returns_none_when_no_matching_turns(self, tmp_path: Path) -> None:
        path = tmp_path / "debug.jsonl"
        _write_jsonl(path, [_turn("sess-A", "x", "2026-05-08T10:00:00+00:00")])

        with path.open() as fh:
            records = (json.loads(line) for line in fh if line.strip())
            window = get_session_ts_window(records, "sess-NONEXISTENT")

        assert window is None

    def test_ignores_non_turn_records(self, tmp_path: Path) -> None:
        """Non-turn records (llm_call, etc.) must not contribute to the window
        even if their session_id matches — they may carry different ts_iso
        semantics or null session_id.
        """
        path = tmp_path / "debug.jsonl"
        _write_jsonl(
            path,
            [
                _turn("sess-A", "x", "2026-05-08T10:00:00+00:00"),
                {
                    "phase": "llm_call",
                    "session_id": "sess-A",
                    "ts_iso": "2099-12-31T23:59:59+00:00",  # bogus future ts
                },
            ],
        )

        with path.open() as fh:
            records = (json.loads(line) for line in fh if line.strip())
            window = get_session_ts_window(records, "sess-A")

        assert window == ("2026-05-08T10:00:00+00:00", "2026-05-08T10:00:00+00:00")


class TestSessionScopedScoring:
    """The session_id arg must scope the extraction index to ONLY
    extractions that fell within the requested session's TURN ts_iso
    window (plus an extraction-lag buffer). Two replay runs of the same
    V9 inputs landing in the same JSONL file must NOT cross-pollinate.
    """

    def _write_v9_input(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "utterance": "Garbage collection is the mechanism for memory management.",
                    "tag": "v9-01-mechanism-gc",
                    "expected_behavior": {
                        "expected_edges": ["MECHANISM_OF"],
                        "expected_entities": ["Mechanism", "Concept"],
                        "rationale": "test",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_session_filter_excludes_other_sessions_extractions(
        self, tmp_path: Path, capsys
    ) -> None:
        """Two sequential replay runs of the SAME V9 input land in the
        same JSONL. Run A produced no MECHANISM_OF; Run B produced
        MECHANISM_OF. Scoping by --session-id must NOT cross-pollinate.
        """
        v9_input = tmp_path / "v9.jsonl"
        self._write_v9_input(v9_input)

        utt = "Garbage collection is the mechanism for memory management."
        debug = tmp_path / "debug.jsonl"
        _write_jsonl(
            debug,
            [
                # Run A at 10:00 — extraction yielded RELATED_TO (no MECHANISM_OF)
                _turn("sess-A", utt, "2026-05-08T10:00:00+00:00"),
                _extraction_llm_call(
                    utt,
                    relationships=["RELATED_TO"],
                    ts_iso="2026-05-08T10:00:05+00:00",
                ),
                # Run B at 11:00 — extraction yielded MECHANISM_OF
                _turn("sess-B", utt, "2026-05-08T11:00:00+00:00"),
                _extraction_llm_call(
                    utt,
                    relationships=["MECHANISM_OF"],
                    ts_iso="2026-05-08T11:00:05+00:00",
                ),
            ],
        )

        # Score scoped to sess-A: must NOT see B's MECHANISM_OF
        main(
            [
                "--input",
                str(v9_input),
                "--debug-jsonl",
                str(debug),
                "--session-id",
                "sess-A",
            ]
        )
        out_a = capsys.readouterr().out
        assert (
            "MECHANISM_OF | 1 | 0 | 0.00 | FAIL" in out_a
        ), f"sess-A scope must show MECHANISM_OF as 0/1 fired (FAIL); got:\n{out_a}"

        # Score scoped to sess-B: SHOULD see MECHANISM_OF
        main(
            [
                "--input",
                str(v9_input),
                "--debug-jsonl",
                str(debug),
                "--session-id",
                "sess-B",
            ]
        )
        out_b = capsys.readouterr().out
        assert (
            "MECHANISM_OF | 1 | 1 | 1.00 | PASS" in out_b
        ), f"sess-B scope must show MECHANISM_OF as 1/1 fired (PASS); got:\n{out_b}"

    def test_no_session_id_aggregates_across_runs(self, tmp_path: Path, capsys) -> None:
        """When --session-id is omitted, the scorer aggregates the entire
        JSONL (legacy behavior). Caller is responsible for ensuring
        single-run input.
        """
        v9_input = tmp_path / "v9.jsonl"
        self._write_v9_input(v9_input)

        utt = "Garbage collection is the mechanism for memory management."
        debug = tmp_path / "debug.jsonl"
        _write_jsonl(
            debug,
            [
                _turn("sess-A", utt, "2026-05-08T10:00:00+00:00"),
                _extraction_llm_call(
                    utt,
                    relationships=["RELATED_TO"],
                    ts_iso="2026-05-08T10:00:05+00:00",
                ),
                _turn("sess-B", utt, "2026-05-08T11:00:00+00:00"),
                _extraction_llm_call(
                    utt,
                    relationships=["MECHANISM_OF"],
                    ts_iso="2026-05-08T11:00:05+00:00",
                ),
            ],
        )

        main(["--input", str(v9_input), "--debug-jsonl", str(debug)])
        out = capsys.readouterr().out
        assert (
            "MECHANISM_OF | 1 | 1 | 1.00 | PASS" in out
        ), f"no --session-id must aggregate; got:\n{out}"

    def test_utterance_join_tolerates_whitespace_variation(self, tmp_path: Path, capsys) -> None:
        """The probe utterance and the extracted utterance must join even
        when they differ in surrounding/internal whitespace. Without
        whitespace normalization, any prompt-template change that adds or
        removes whitespace would drop every probe to MISSING.
        """
        v9_input = tmp_path / "v9.jsonl"
        v9_input.write_text(
            json.dumps(
                {
                    "utterance": "Garbage collection is the mechanism for memory management.",
                    "tag": "v9-01-mechanism-gc",
                    "expected_behavior": {
                        "expected_edges": ["MECHANISM_OF"],
                        "expected_entities": ["Mechanism", "Concept"],
                        "rationale": "test",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        # Extraction prompt rendered the utterance with extra surrounding
        # and internal whitespace -- shouldn't break the join.
        extracted_utt = "  Garbage collection  is the   mechanism for memory   management.  "
        debug = tmp_path / "debug.jsonl"
        _write_jsonl(
            debug,
            [
                _turn(
                    "sess-A",
                    "Garbage collection is the mechanism for memory management.",
                    "2026-05-08T10:00:00+00:00",
                ),
                _extraction_llm_call(
                    extracted_utt,
                    relationships=["MECHANISM_OF"],
                    ts_iso="2026-05-08T10:00:05+00:00",
                ),
            ],
        )

        main(["--input", str(v9_input), "--debug-jsonl", str(debug)])
        out = capsys.readouterr().out
        assert "MECHANISM_OF | 1 | 1 | 1.00 | PASS" in out, (
            f"whitespace-varied extraction must still join with the probe; " f"got:\n{out}"
        )

    def test_unknown_session_id_warns_and_aggregates(self, tmp_path: Path, capsys) -> None:
        """When --session-id is supplied but matches no TURN records, the
        scorer must emit a stderr warning and fall back to aggregating
        the full JSONL (rather than silently scoring zero).
        """
        v9_input = tmp_path / "v9.jsonl"
        self._write_v9_input(v9_input)

        utt = "Garbage collection is the mechanism for memory management."
        debug = tmp_path / "debug.jsonl"
        _write_jsonl(
            debug,
            [
                _turn("sess-A", utt, "2026-05-08T10:00:00+00:00"),
                _extraction_llm_call(
                    utt,
                    relationships=["MECHANISM_OF"],
                    ts_iso="2026-05-08T10:00:05+00:00",
                ),
            ],
        )

        main(
            [
                "--input",
                str(v9_input),
                "--debug-jsonl",
                str(debug),
                "--session-id",
                "sess-NONEXISTENT",
            ]
        )
        captured = capsys.readouterr()
        assert (
            "matched no TURN records" in captured.err
        ), f"unknown session_id must warn on stderr; got stderr:\n{captured.err!r}"
        assert "MECHANISM_OF | 1 | 1 | 1.00 | PASS" in captured.out, (
            "unknown session_id must fall back to aggregation; " f"got stdout:\n{captured.out}"
        )
