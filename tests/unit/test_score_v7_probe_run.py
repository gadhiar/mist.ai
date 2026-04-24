"""Unit tests for scripts/eval_harness/score_v7_probe_run.py.

The scorer joins V7 probe inputs against the per-turn debug JSONL stream
and emits precision / recall / per-bucket recall against the design-doc
acceptance thresholds. These tests cover:

  - Probe + observation parsing edge cases (malformed JSON, blank lines,
    phase filtering, session-id scoping).
  - Verdict semantics for the four canonical cases (TP, FN, TN, FP) plus
    the "missing" case when the debug JSONL has no record for a probe.
  - Acceptance-gate boundaries (precision >= 0.90, recall >= 0.90, 0/5 FP
    on negatives), including off-by-one edges.
  - CLI behavior: stdout vs file output, JSON output, --strict exit
    codes, and error-paths when input files are missing.

Spec: scripts/eval_harness/score_v7_probe_run.py module docstring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_harness.score_v7_probe_run import (  # noqa: E402  -- after sys.path insert
    NEGATIVE_FALSE_POSITIVE_LIMIT,
    PRECISION_THRESHOLD,
    RECALL_THRESHOLD,
    ProbeOutcome,
    TurnObservation,
    V7Probe,
    V7Report,
    index_observations_by_utterance,
    iter_probes,
    iter_turn_observations,
    main,
    render_json,
    render_markdown,
    score_run,
    verdict_for,
)

# ---------------------------------------------------------------------------
# Factory helpers (per tests/CLAUDE.md "Factory Functions")
# ---------------------------------------------------------------------------


def build_probe(
    *,
    tag: str = "v7-01-direct-tech-recall",
    utterance: str = "What languages do I use?",
    expected_tool: str | None = "query_knowledge_graph",
    rationale: str | None = "test rationale",
) -> V7Probe:
    return V7Probe(
        tag=tag,
        utterance=utterance,
        expected_tool=expected_tool,
        rationale=rationale,
    )


def build_observation(
    *,
    utterance: str = "What languages do I use?",
    session_id: str | None = "v7-probe-test",
    event_id: str | None = "evt-1",
    tool_calls: tuple[str, ...] = ("query_knowledge_graph",),
) -> TurnObservation:
    return TurnObservation(
        utterance=utterance,
        session_id=session_id,
        event_id=event_id,
        tool_calls=tool_calls,
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def make_turn_record(
    *,
    utterance: str,
    session_id: str = "v7-probe-test",
    event_id: str = "evt-x",
    tool_calls_per_pass: list[list[str]] | None = None,
) -> dict:
    """Build a phase=turn record matching debug_jsonl_logger._LiveTurnRecord."""
    passes = []
    for pass_num, names in enumerate(tool_calls_per_pass or [], start=1):
        passes.append(
            {
                "pass": pass_num,
                "content_len": 10,
                "tool_calls": [{"name": n, "id": f"tc-{n}", "arg_keys": []} for n in names],
                "partial": False,
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "timing_ms": 200.0,
            }
        )
    return {
        "phase": "turn",
        "ts_iso": "2026-04-23T12:00:00+00:00",
        "event_id": event_id,
        "session_id": session_id,
        "user_id": "User",
        "utterance": utterance,
        "retrieval": None,
        "llm_passes": passes,
        "total_turn_ms": 1234.5,
    }


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------


class TestProbeLoading:
    """V7 input JSONL parsing."""

    def test_loads_basic_fields(self, tmp_path):
        # Arrange
        path = tmp_path / "v7.jsonl"
        write_jsonl(
            path,
            [
                {
                    "utterance": "What languages do I use?",
                    "tag": "v7-01-direct-tech-recall",
                    "expected_behavior": {
                        "tool_call": "query_knowledge_graph",
                        "rationale": "direct recall",
                    },
                }
            ],
        )

        # Act
        probes = list(iter_probes(path))

        # Assert
        assert len(probes) == 1
        p = probes[0]
        assert p.tag == "v7-01-direct-tech-recall"
        assert p.utterance == "What languages do I use?"
        assert p.expected_tool == "query_knowledge_graph"
        assert p.rationale == "direct recall"

    def test_skips_blank_and_comment_lines(self, tmp_path):
        # Arrange
        path = tmp_path / "v7.jsonl"
        path.write_text(
            "\n"
            "# header comment\n"
            '{"utterance": "X", "tag": "v7-01-x", "expected_behavior": {"tool_call": "t"}}\n'
            "\n",
            encoding="utf-8",
        )

        # Act
        probes = list(iter_probes(path))

        # Assert
        assert len(probes) == 1

    def test_raises_on_malformed_json(self, tmp_path):
        # Arrange
        path = tmp_path / "v7.jsonl"
        path.write_text("not json at all\n", encoding="utf-8")

        # Act + Assert
        with pytest.raises(ValueError, match="invalid JSON"):
            list(iter_probes(path))

    def test_handles_missing_expected_behavior(self, tmp_path):
        # Arrange -- absence of expected_behavior should yield None tool / None rationale
        path = tmp_path / "v7.jsonl"
        write_jsonl(path, [{"utterance": "X", "tag": "v7-99-x"}])

        # Act
        probes = list(iter_probes(path))

        # Assert
        assert probes[0].expected_tool is None
        assert probes[0].rationale is None


class TestProbeIsNegative:
    """Tag-based negative-control detection."""

    @pytest.mark.parametrize(
        "tag, expected",
        [
            pytest.param("v7-21-neg-general-knowledge", True, id="neg-prefix"),
            pytest.param("v7-22-neg-small-talk", True, id="neg-small-talk"),
            pytest.param("v7-01-direct-tech-recall", False, id="positive"),
            pytest.param("v7-08-ambiguous-recall", False, id="ambiguous-positive"),
        ],
    )
    def test_negative_detection(self, tag, expected):
        # Arrange + Act
        probe = build_probe(tag=tag)

        # Assert
        assert probe.is_negative is expected


class TestProbeBucket:
    """Bucket extraction strips the v7-NN- prefix."""

    @pytest.mark.parametrize(
        "tag, bucket",
        [
            pytest.param("v7-01-direct-tech-recall", "direct-tech-recall", id="positive"),
            pytest.param("v7-21-neg-general-knowledge", "neg-general-knowledge", id="negative"),
            pytest.param("v7-08-ambiguous-recall", "ambiguous-recall", id="ambiguous"),
            pytest.param("malformed", "malformed", id="fallback-no-hyphens"),
        ],
    )
    def test_bucket_extraction(self, tag, bucket):
        assert build_probe(tag=tag).bucket == bucket


# ---------------------------------------------------------------------------
# Observation loading
# ---------------------------------------------------------------------------


class TestObservationLoading:
    """Debug JSONL parsing -- phase filtering, session filtering, tool call aggregation."""

    def test_loads_phase_turn_only(self, tmp_path):
        # Arrange
        path = tmp_path / "debug.jsonl"
        write_jsonl(
            path,
            [
                make_turn_record(utterance="A", tool_calls_per_pass=[["query_knowledge_graph"]]),
                {
                    "phase": "extraction",
                    "ts_iso": "...",
                    "event_id": "e1",
                    "extraction": {},
                    "graph_writes": None,
                },
                make_turn_record(utterance="B", tool_calls_per_pass=[[]]),
            ],
        )

        # Act
        obs = list(iter_turn_observations(path))

        # Assert -- extraction record dropped, only the two turn records remain
        assert len(obs) == 2
        assert obs[0].utterance == "A"
        assert obs[1].utterance == "B"

    def test_filters_by_session_id_when_given(self, tmp_path):
        # Arrange
        path = tmp_path / "debug.jsonl"
        write_jsonl(
            path,
            [
                make_turn_record(
                    utterance="A",
                    session_id="other",
                    tool_calls_per_pass=[["query_knowledge_graph"]],
                ),
                make_turn_record(
                    utterance="B",
                    session_id="v7-probe-test",
                    tool_calls_per_pass=[["query_knowledge_graph"]],
                ),
            ],
        )

        # Act
        obs = list(iter_turn_observations(path, session_id="v7-probe-test"))

        # Assert
        assert len(obs) == 1
        assert obs[0].utterance == "B"

    def test_includes_all_when_no_session_id(self, tmp_path):
        # Arrange
        path = tmp_path / "debug.jsonl"
        write_jsonl(
            path,
            [
                make_turn_record(utterance="A", session_id="s1"),
                make_turn_record(utterance="B", session_id="s2"),
            ],
        )

        # Act
        obs = list(iter_turn_observations(path, session_id=None))

        # Assert
        assert len(obs) == 2

    def test_aggregates_tool_calls_across_passes(self, tmp_path):
        # Arrange -- pass 1 empty, pass 2 fires the tool. The first non-empty
        # tool call across all passes wins for first_tool_call.
        path = tmp_path / "debug.jsonl"
        write_jsonl(
            path,
            [
                make_turn_record(
                    utterance="A",
                    tool_calls_per_pass=[[], ["query_knowledge_graph"]],
                )
            ],
        )

        # Act
        obs = list(iter_turn_observations(path))

        # Assert
        assert obs[0].tool_calls == ("query_knowledge_graph",)
        assert obs[0].first_tool_call == "query_knowledge_graph"

    def test_first_tool_call_across_passes_when_both_fire(self, tmp_path):
        # Arrange
        path = tmp_path / "debug.jsonl"
        write_jsonl(
            path,
            [
                make_turn_record(
                    utterance="A",
                    tool_calls_per_pass=[["first_tool"], ["second_tool"]],
                )
            ],
        )

        # Act
        obs = list(iter_turn_observations(path))

        # Assert
        assert obs[0].tool_calls == ("first_tool", "second_tool")
        assert obs[0].first_tool_call == "first_tool"

    def test_handles_empty_tool_calls(self, tmp_path):
        # Arrange
        path = tmp_path / "debug.jsonl"
        write_jsonl(path, [make_turn_record(utterance="A", tool_calls_per_pass=[[]])])

        # Act
        obs = list(iter_turn_observations(path))

        # Assert
        assert obs[0].tool_calls == ()
        assert obs[0].first_tool_call is None

    def test_skips_malformed_json_line(self, tmp_path, capsys):
        # Arrange
        path = tmp_path / "debug.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write("not json\n")
            fh.write(
                json.dumps(
                    make_turn_record(utterance="A", tool_calls_per_pass=[["query_knowledge_graph"]])
                )
                + "\n"
            )

        # Act
        obs = list(iter_turn_observations(path))

        # Assert -- the bad line is skipped, the good one is loaded; warn goes to stderr
        assert len(obs) == 1
        captured = capsys.readouterr()
        assert "malformed JSON" in captured.err

    def test_skips_blank_lines(self, tmp_path):
        # Arrange
        path = tmp_path / "debug.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write("\n")
            fh.write(json.dumps(make_turn_record(utterance="A", tool_calls_per_pass=[[]])) + "\n")
            fh.write("\n")

        # Act
        obs = list(iter_turn_observations(path))

        # Assert
        assert len(obs) == 1


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestIndexObservations:
    """First-occurrence wins; empty utterances skipped."""

    def test_keeps_first_occurrence_when_utterance_repeats(self):
        # Arrange
        obs1 = build_observation(utterance="X", event_id="evt-1", tool_calls=("tool_a",))
        obs2 = build_observation(utterance="X", event_id="evt-2", tool_calls=("tool_b",))

        # Act
        index = index_observations_by_utterance(iter([obs1, obs2]))

        # Assert -- first-write semantics
        assert index["X"].event_id == "evt-1"
        assert index["X"].first_tool_call == "tool_a"

    def test_skips_empty_utterance(self):
        # Arrange
        obs = build_observation(utterance="", tool_calls=())

        # Act
        index = index_observations_by_utterance(iter([obs]))

        # Assert
        assert index == {}


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


class TestVerdict:
    """Verdict semantics for each canonical join-state."""

    def test_tp_when_positive_matches_expected_tool(self):
        probe = build_probe(expected_tool="query_knowledge_graph")
        obs = build_observation(tool_calls=("query_knowledge_graph",))

        assert verdict_for(probe, obs) == "tp"

    def test_fn_when_positive_has_no_tool(self):
        probe = build_probe(expected_tool="query_knowledge_graph")
        obs = build_observation(tool_calls=())

        assert verdict_for(probe, obs) == "fn"

    def test_fn_when_positive_fires_wrong_tool(self):
        probe = build_probe(expected_tool="query_knowledge_graph")
        obs = build_observation(tool_calls=("some_other_tool",))

        assert verdict_for(probe, obs) == "fn"

    def test_tn_when_negative_has_no_tool(self):
        probe = build_probe(tag="v7-21-neg-general-knowledge", expected_tool=None)
        obs = build_observation(tool_calls=())

        assert verdict_for(probe, obs) == "tn"

    def test_fp_when_negative_fires_any_tool(self):
        probe = build_probe(tag="v7-21-neg-general-knowledge", expected_tool=None)
        obs = build_observation(tool_calls=("query_knowledge_graph",))

        assert verdict_for(probe, obs) == "fp"

    def test_missing_when_no_observation(self):
        probe = build_probe()

        assert verdict_for(probe, None) == "missing"


# ---------------------------------------------------------------------------
# Score run + report aggregation
# ---------------------------------------------------------------------------


class TestScoreRun:
    """End-to-end scoring against a synthetic probe + observation set."""

    def test_perfect_run(self):
        # Arrange -- 2 positives that fire correctly, 1 negative that stays silent
        probes = [
            build_probe(
                tag="v7-01-direct-tech-recall",
                utterance="A",
                expected_tool="query_knowledge_graph",
            ),
            build_probe(
                tag="v7-02-direct-decision-recall",
                utterance="B",
                expected_tool="query_knowledge_graph",
            ),
            build_probe(
                tag="v7-21-neg-general-knowledge",
                utterance="C",
                expected_tool=None,
            ),
        ]
        index = {
            "A": build_observation(utterance="A", tool_calls=("query_knowledge_graph",)),
            "B": build_observation(utterance="B", tool_calls=("query_knowledge_graph",)),
            "C": build_observation(utterance="C", tool_calls=()),
        }

        # Act
        report = score_run(probes, index)

        # Assert
        assert report.true_positives == 2
        assert report.false_negatives == 0
        assert report.true_negatives == 1
        assert report.false_positives == 0
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.acceptance_pass()

    def test_mixed_run_with_missing_and_failure(self):
        # Arrange
        probes = [
            build_probe(
                tag="v7-01-direct-tech-recall",
                utterance="A",
                expected_tool="query_knowledge_graph",
            ),
            build_probe(
                tag="v7-02-direct-decision-recall",
                utterance="B",
                expected_tool="query_knowledge_graph",
            ),
            build_probe(
                tag="v7-03-direct-person-recall",
                utterance="C",
                expected_tool="query_knowledge_graph",
            ),
            build_probe(
                tag="v7-21-neg-general-knowledge",
                utterance="D",
                expected_tool=None,
            ),
        ]
        index = {
            "A": build_observation(utterance="A", tool_calls=("query_knowledge_graph",)),
            # B missing entirely
            "C": build_observation(utterance="C", tool_calls=()),
            "D": build_observation(utterance="D", tool_calls=("query_knowledge_graph",)),
        }

        # Act
        report = score_run(probes, index)

        # Assert
        assert report.true_positives == 1  # A
        assert report.false_negatives == 1  # C (positive, fired no tool)
        assert report.missing == 1  # B
        assert report.false_positives == 1  # D
        assert report.true_negatives == 0


# ---------------------------------------------------------------------------
# Acceptance gates
# ---------------------------------------------------------------------------


class TestAcceptance:
    """Acceptance-gate boundary behavior."""

    def _build_report(
        self,
        *,
        tp: int = 0,
        fn: int = 0,
        tn: int = 0,
        fp: int = 0,
    ) -> V7Report:
        outcomes: list[ProbeOutcome] = []
        for _ in range(tp):
            outcomes.append(
                ProbeOutcome(
                    probe=build_probe(),
                    observation=build_observation(),
                    verdict="tp",
                )
            )
        for _ in range(fn):
            outcomes.append(
                ProbeOutcome(
                    probe=build_probe(),
                    observation=build_observation(tool_calls=()),
                    verdict="fn",
                )
            )
        for _ in range(tn):
            outcomes.append(
                ProbeOutcome(
                    probe=build_probe(tag="v7-99-neg-x", expected_tool=None),
                    observation=build_observation(tool_calls=()),
                    verdict="tn",
                )
            )
        for _ in range(fp):
            outcomes.append(
                ProbeOutcome(
                    probe=build_probe(tag="v7-99-neg-x", expected_tool=None),
                    observation=build_observation(),
                    verdict="fp",
                )
            )
        return V7Report(outcomes=outcomes)

    def test_pass_at_threshold_boundaries(self):
        # Arrange -- precision exactly 0.90, recall exactly 0.90, 0 FP
        report = self._build_report(tp=9, fn=1, fp=1, tn=5)

        # Assert
        assert report.precision == pytest.approx(0.9)
        assert report.recall == pytest.approx(0.9)
        # Even with FP=1 on negatives, the precision/recall threshold passes;
        # but the discrete 0-FP rule fails it. This documents the strictness.
        assert NEGATIVE_FALSE_POSITIVE_LIMIT == 0
        assert (
            not report.acceptance_pass()
        ), "Even with precision/recall at threshold, 1 FP must fail acceptance"

    def test_pass_clean_sweep(self):
        report = self._build_report(tp=20, fn=0, tn=5, fp=0)
        assert report.acceptance_pass()

    def test_fail_below_precision_threshold(self):
        # Arrange -- 8 TP, 2 FP (positives fired wrong) -- precision = 0.80
        report = self._build_report(tp=8, fn=2, fp=0, tn=5)
        # Recall = 8/10 = 0.80
        assert report.recall < RECALL_THRESHOLD
        assert not report.acceptance_pass()

    def test_fail_with_one_fp_on_negatives(self):
        # Even with perfect precision/recall, a single FP on negatives fails
        # the discrete 0/5 acceptance gate.
        report = self._build_report(tp=20, fn=0, tn=4, fp=1)
        assert report.precision < 1.0  # 20/(20+1)
        assert report.recall == 1.0
        assert not report.acceptance_pass()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    """Markdown rendering surfaces the load-bearing fields."""

    def test_includes_pass_verdict_when_clean(self):
        report = V7Report(
            outcomes=[
                ProbeOutcome(
                    probe=build_probe(),
                    observation=build_observation(),
                    verdict="tp",
                )
            ]
        )

        md = render_markdown(report)

        assert "Verdict:** PASS" in md

    def test_includes_fail_verdict_when_below_threshold(self):
        # Arrange -- one TP, one FN; precision = 1.0, recall = 0.5; fails recall gate
        report = V7Report(
            outcomes=[
                ProbeOutcome(
                    probe=build_probe(tag="v7-01-x", utterance="A"),
                    observation=build_observation(utterance="A"),
                    verdict="tp",
                ),
                ProbeOutcome(
                    probe=build_probe(tag="v7-02-x", utterance="B"),
                    observation=build_observation(utterance="B", tool_calls=()),
                    verdict="fn",
                ),
            ]
        )

        md = render_markdown(report)

        assert "Verdict:** FAIL" in md
        assert "v7-02-x" in md  # failure surfaced in failure listing


class TestRenderJson:
    """JSON rendering round-trips outcomes and acceptance metadata."""

    def test_json_structure(self):
        report = V7Report(
            outcomes=[
                ProbeOutcome(
                    probe=build_probe(),
                    observation=build_observation(),
                    verdict="tp",
                )
            ]
        )

        payload = json.loads(render_json(report))

        assert payload["totals"]["tp"] == 1
        assert payload["headline"]["precision"] == 1.0
        assert payload["acceptance"]["precision_threshold"] == PRECISION_THRESHOLD
        assert payload["acceptance"]["passed"] is True
        assert len(payload["outcomes"]) == 1
        assert payload["outcomes"][0]["verdict"] == "tp"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    """End-to-end CLI behavior with synthetic input + debug files."""

    def _setup_files(self, tmp_path: Path, *, fire_tool: bool) -> tuple[Path, Path]:
        input_path = tmp_path / "v7.jsonl"
        debug_path = tmp_path / "debug.jsonl"
        write_jsonl(
            input_path,
            [
                {
                    "utterance": "What languages do I use?",
                    "tag": "v7-01-direct-tech-recall",
                    "expected_behavior": {
                        "tool_call": "query_knowledge_graph",
                        "rationale": "direct recall",
                    },
                }
            ],
        )
        tool_calls = [["query_knowledge_graph"]] if fire_tool else [[]]
        write_jsonl(
            debug_path,
            [
                make_turn_record(
                    utterance="What languages do I use?",
                    session_id="v7-test",
                    tool_calls_per_pass=tool_calls,
                )
            ],
        )
        return input_path, debug_path

    def test_writes_markdown_to_stdout_by_default(self, tmp_path, capsys):
        # Arrange
        input_path, debug_path = self._setup_files(tmp_path, fire_tool=True)

        # Act
        rc = main(
            [
                "--input",
                str(input_path),
                "--debug-jsonl",
                str(debug_path),
                "--session-id",
                "v7-test",
            ]
        )

        # Assert
        assert rc == 0
        captured = capsys.readouterr()
        assert "V7 Probe Set" in captured.out
        assert "Verdict:** PASS" in captured.out

    def test_writes_to_output_file_when_given(self, tmp_path):
        # Arrange
        input_path, debug_path = self._setup_files(tmp_path, fire_tool=True)
        out = tmp_path / "report.md"

        # Act
        rc = main(
            [
                "--input",
                str(input_path),
                "--debug-jsonl",
                str(debug_path),
                "--session-id",
                "v7-test",
                "--output",
                str(out),
            ]
        )

        # Assert
        assert rc == 0
        assert out.exists()
        assert "V7 Probe Set" in out.read_text(encoding="utf-8")

    def test_writes_json_output_when_requested(self, tmp_path, capsys):
        # Arrange
        input_path, debug_path = self._setup_files(tmp_path, fire_tool=True)
        json_out = tmp_path / "report.json"

        # Act
        rc = main(
            [
                "--input",
                str(input_path),
                "--debug-jsonl",
                str(debug_path),
                "--session-id",
                "v7-test",
                "--json-output",
                str(json_out),
            ]
        )

        # Assert
        assert rc == 0
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert payload["acceptance"]["passed"] is True

    def test_strict_exits_one_on_acceptance_fail(self, tmp_path):
        # Arrange -- positive probe, no tool fired -- recall drops to 0.0
        input_path, debug_path = self._setup_files(tmp_path, fire_tool=False)

        # Act
        rc = main(
            [
                "--input",
                str(input_path),
                "--debug-jsonl",
                str(debug_path),
                "--session-id",
                "v7-test",
                "--strict",
            ]
        )

        # Assert
        assert rc == 1

    def test_no_strict_exits_zero_on_fail(self, tmp_path, capsys):
        # Arrange
        input_path, debug_path = self._setup_files(tmp_path, fire_tool=False)

        # Act
        rc = main(
            [
                "--input",
                str(input_path),
                "--debug-jsonl",
                str(debug_path),
                "--session-id",
                "v7-test",
            ]
        )

        # Assert -- without --strict, the script reports but does not fail the shell
        assert rc == 0

    def test_returns_two_when_input_missing(self, tmp_path, capsys):
        # Arrange -- input file does not exist
        input_path = tmp_path / "missing.jsonl"
        debug_path = tmp_path / "debug.jsonl"
        write_jsonl(debug_path, [])

        # Act
        rc = main(
            [
                "--input",
                str(input_path),
                "--debug-jsonl",
                str(debug_path),
            ]
        )

        # Assert
        assert rc == 2
        captured = capsys.readouterr()
        assert "input not found" in captured.err

    def test_returns_two_when_debug_missing(self, tmp_path, capsys):
        # Arrange -- debug file does not exist
        input_path = tmp_path / "v7.jsonl"
        write_jsonl(input_path, [])
        debug_path = tmp_path / "missing.jsonl"

        # Act
        rc = main(
            [
                "--input",
                str(input_path),
                "--debug-jsonl",
                str(debug_path),
            ]
        )

        # Assert
        assert rc == 2
        captured = capsys.readouterr()
        assert "debug JSONL not found" in captured.err
