"""Unit tests for scripts/eval_harness/score_v8_probe_run.py.

The scorer joins V8 probe inputs against the per-turn debug JSONL stream,
specifically `phase: llm_call` records with `call_site = "extraction.ontology"`,
and emits per-edge recall against the design-doc acceptance thresholds.

Tests cover:
  - Probe + debug-record parsing edge cases (malformed JSON, blank lines,
    phase filtering, call_site filtering, session-id scoping).
  - Lenient JSON parsing of the extraction LLM response content (clean,
    markdown-wrapped, malformed).
  - Indexing the two-record join (turn -> event_id, llm_call -> extraction).
  - Aggregation when an event_id has multiple extraction records (retries).
  - Verdict semantics: TP/PARTIAL/FN for positives, TN/FP for negatives,
    MISSING / PARSE_FAIL edge states.
  - Acceptance gates: per-bucket recall, overall recall, negative-FP rule.
  - CLI behavior: stdout vs file output, JSON output, --strict exit codes,
    error paths.
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

from eval_harness.score_v8_probe_run import (  # noqa: E402  -- after sys.path insert
    EXTRACTION_CALL_SITE,
    NEW_EDGE_TYPES,
    PER_BUCKET_RECALL_THRESHOLD,
    ExtractionRecord,
    ProbeOutcome,
    V8Probe,
    V8Report,
    build_indices,
    extract_utterance_from_request,
    iter_debug_records,
    iter_probes,
    main,
    parse_extraction_json,
    render_json,
    render_markdown,
    score_run,
)

# ---------------------------------------------------------------------------
# Factory helpers (per tests/CLAUDE.md)
# ---------------------------------------------------------------------------


def build_probe(
    *,
    tag: str = "v8-01-event-on-date",
    utterance: str = "I had a meeting on 2026-04-15",
    expected_edges: tuple[str, ...] = ("OCCURRED_ON",),
    expected_entities: tuple[str, ...] = ("Event", "Date"),
    rationale: str | None = "test rationale",
) -> V8Probe:
    return V8Probe(
        tag=tag,
        utterance=utterance,
        expected_edges=expected_edges,
        expected_entities=expected_entities,
        rationale=rationale,
    )


def build_extraction_record(
    *,
    event_id: str | None = "evt-1",
    session_id: str | None = "v8-test",
    entities: tuple[str, ...] = ("Event", "Date"),
    relationships: tuple[str, ...] = ("OCCURRED_ON",),
    parse_ok: bool = True,
    raw_response: str = "{}",
) -> ExtractionRecord:
    return ExtractionRecord(
        event_id=event_id,
        session_id=session_id,
        extracted_entity_types=frozenset(entities),
        extracted_relationship_types=frozenset(relationships),
        parse_ok=parse_ok,
        raw_response=raw_response,
    )


def make_extraction_user_message(utterance: str, scope: str = "user-scope") -> str:
    """Render an extraction user message exactly as EXTRACTION_USER_TEMPLATE does.

    Mirrors backend/knowledge/extraction/prompts.py:EXTRACTION_USER_TEMPLATE so
    extract_utterance_from_request can recover the utterance via the same
    pattern at runtime.
    """
    return (
        "Context:\n"
        "(no prior context)\n"
        f"Subject scope: {scope}\n"
        f'Utterance: "{utterance}"\n'
        "\n"
        "Output:"
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def make_turn_record(
    *,
    utterance: str,
    event_id: str = "evt-x",
    session_id: str = "v8-test",
) -> dict:
    return {
        "phase": "turn",
        "ts_iso": "2026-04-27T20:00:00+00:00",
        "event_id": event_id,
        "session_id": session_id,
        "user_id": "User",
        "utterance": utterance,
        "retrieval": None,
        "llm_passes": [],
        "total_turn_ms": 100.0,
    }


def make_llm_call_record(
    *,
    event_id: str | None = None,
    session_id: str | None = None,
    call_site: str = EXTRACTION_CALL_SITE,
    response_content: str = '{"entities": [], "relationships": []}',
    utterance: str = "test utterance",
) -> dict:
    """Build a phase=llm_call record with the utterance embedded in the user message.

    event_id and session_id default to None to mirror the current backend bug
    where the extraction call site does not propagate conversation context.
    Tests that exercise the propagated path can pass real values.
    """
    return {
        "phase": "llm_call",
        "ts_iso": "2026-04-27T20:00:01+00:00",
        "event_id": event_id,
        "session_id": session_id,
        "call_site": call_site,
        "pass_num": None,
        "model": "gemma-4-e4b",
        "latency_ms": 200.0,
        "request": {
            "messages": [
                {"role": "system", "content": "system prompt..."},
                {"role": "user", "content": make_extraction_user_message(utterance)},
            ],
            "tools": None,
        },
        "response": {
            "content": response_content,
            "tool_calls": None,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "partial": False,
        },
    }


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------


class TestProbeLoading:
    """V8 input JSONL parsing."""

    def test_loads_basic_fields(self, tmp_path):
        # Arrange
        path = tmp_path / "v8.jsonl"
        write_jsonl(
            path,
            [
                {
                    "utterance": "I had a meeting on 2026-04-15",
                    "tag": "v8-01-event-on-date",
                    "expected_behavior": {
                        "expected_edges": ["OCCURRED_ON"],
                        "expected_entities": ["Event", "Date"],
                        "rationale": "Event + Date",
                    },
                }
            ],
        )

        # Act
        probes = list(iter_probes(path))

        # Assert
        assert len(probes) == 1
        p = probes[0]
        assert p.tag == "v8-01-event-on-date"
        assert p.utterance == "I had a meeting on 2026-04-15"
        assert p.expected_edges == ("OCCURRED_ON",)
        assert p.expected_entities == ("Event", "Date")
        assert p.rationale == "Event + Date"
        assert p.is_negative is False

    def test_negative_probe_detection(self, tmp_path):
        # Arrange
        path = tmp_path / "v8.jsonl"
        write_jsonl(
            path,
            [
                {
                    "utterance": "Today is 2026-04-24",
                    "tag": "v8-17-neg-date-no-anchor",
                    "expected_behavior": {
                        "expected_edges": [],
                        "expected_entities": ["Date"],
                        "rationale": "no anchor",
                    },
                }
            ],
        )

        # Act
        probes = list(iter_probes(path))

        # Assert
        assert probes[0].is_negative is True
        assert probes[0].expected_edges == ()

    def test_handles_missing_expected_behavior(self, tmp_path):
        # Arrange
        path = tmp_path / "v8.jsonl"
        write_jsonl(path, [{"utterance": "X", "tag": "v8-99-x"}])

        # Act
        probes = list(iter_probes(path))

        # Assert
        assert probes[0].expected_edges == ()
        assert probes[0].expected_entities == ()
        assert probes[0].rationale is None

    def test_raises_on_malformed_json(self, tmp_path):
        # Arrange
        path = tmp_path / "v8.jsonl"
        path.write_text("not json\n", encoding="utf-8")

        # Act + Assert
        with pytest.raises(ValueError, match="invalid JSON"):
            list(iter_probes(path))


# ---------------------------------------------------------------------------
# Extraction JSON parsing
# ---------------------------------------------------------------------------


class TestParseExtractionJson:
    """Lenient JSON parsing of extraction LLM response content."""

    def test_parses_clean_json(self):
        content = json.dumps(
            {
                "entities": [
                    {"id": "u", "name": "User", "type": "User"},
                    {"id": "d", "name": "2026-04-15", "type": "Date"},
                ],
                "relationships": [{"source": "u", "target": "d", "type": "OCCURRED_ON"}],
            }
        )

        ok, ents, rels = parse_extraction_json(content)

        assert ok is True
        assert ents == frozenset({"User", "Date"})
        assert rels == frozenset({"OCCURRED_ON"})

    def test_parses_markdown_wrapped_json(self):
        # Models occasionally return JSON wrapped in markdown fences. The
        # regex fallback should still pick up the first {} block.
        content = (
            "Here you go:\n"
            '```json\n{"entities": [{"id": "x", "type": "Event"}], '
            '"relationships": []}\n```\n'
        )

        ok, ents, rels = parse_extraction_json(content)

        assert ok is True
        assert ents == frozenset({"Event"})
        assert rels == frozenset()

    def test_returns_failure_on_unparseable(self):
        ok, ents, rels = parse_extraction_json("not json at all")
        assert ok is False
        assert ents == frozenset()
        assert rels == frozenset()

    def test_returns_failure_on_empty(self):
        ok, ents, rels = parse_extraction_json("")
        assert ok is False
        assert ents == frozenset()
        assert rels == frozenset()

    def test_skips_non_dict_entities(self):
        content = json.dumps(
            {
                "entities": ["string", {"type": "Event"}, 42, {"type": "Date"}],
                "relationships": [{"type": "OCCURRED_ON"}],
            }
        )

        ok, ents, rels = parse_extraction_json(content)

        assert ok is True
        assert ents == frozenset({"Event", "Date"})
        assert rels == frozenset({"OCCURRED_ON"})

    def test_handles_missing_entities_or_relationships_keys(self):
        ok, ents, rels = parse_extraction_json('{"foo": "bar"}')
        assert ok is True
        assert ents == frozenset()
        assert rels == frozenset()


# ---------------------------------------------------------------------------
# Debug record loading + indexing
# ---------------------------------------------------------------------------


class TestDebugRecordLoading:
    """Debug JSONL parsing -- session filter, malformed line handling."""

    def test_filters_by_session_id(self, tmp_path):
        # Arrange
        path = tmp_path / "debug.jsonl"
        write_jsonl(
            path,
            [
                make_turn_record(utterance="A", session_id="other"),
                make_turn_record(utterance="B", session_id="v8-test"),
            ],
        )

        # Act
        recs = list(iter_debug_records(path, session_id="v8-test"))

        # Assert
        assert len(recs) == 1
        assert recs[0]["utterance"] == "B"

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
        recs = list(iter_debug_records(path, session_id=None))

        # Assert
        assert len(recs) == 2

    def test_skips_malformed_json_with_warn(self, tmp_path, capsys):
        # Arrange
        path = tmp_path / "debug.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write("not json\n")
            fh.write(json.dumps(make_turn_record(utterance="A")) + "\n")

        # Act
        recs = list(iter_debug_records(path))

        # Assert
        assert len(recs) == 1
        captured = capsys.readouterr()
        assert "malformed JSON" in captured.err


class TestExtractUtteranceFromRequest:
    """Recover the utterance from an extraction request user message."""

    def test_recovers_utterance_from_template(self):
        request = {
            "messages": [
                {"role": "system", "content": "..."},
                {
                    "role": "user",
                    "content": make_extraction_user_message("I had a meeting on 2026-04-15"),
                },
            ]
        }

        result = extract_utterance_from_request(request)

        assert result == "I had a meeting on 2026-04-15"

    def test_returns_none_when_pattern_missing(self):
        request = {"messages": [{"role": "user", "content": "no Utterance: marker here"}]}
        assert extract_utterance_from_request(request) is None

    def test_returns_none_when_messages_empty(self):
        assert extract_utterance_from_request({"messages": []}) is None

    def test_returns_none_when_request_not_dict(self):
        assert extract_utterance_from_request(None) is None
        assert extract_utterance_from_request("string") is None


class TestBuildIndices:
    """Single-pass indexing of extraction records keyed by utterance."""

    def test_indexes_extraction_records_by_utterance(self):
        # Arrange
        records = iter(
            [
                make_llm_call_record(
                    utterance="I had a meeting on 2026-04-15",
                    response_content='{"entities": [{"type": "Event"}], '
                    '"relationships": [{"type": "OCCURRED_ON"}]}',
                ),
            ]
        )

        # Act
        index = build_indices(records)

        # Assert
        assert "I had a meeting on 2026-04-15" in index
        ext = index["I had a meeting on 2026-04-15"][0]
        assert ext.extracted_entity_types == frozenset({"Event"})
        assert ext.extracted_relationship_types == frozenset({"OCCURRED_ON"})
        assert ext.parse_ok is True

    def test_filters_non_ontology_call_sites(self):
        # Arrange -- only extraction.ontology should index; chat.* and
        # extraction.scope_classifier should be skipped.
        records = iter(
            [
                make_llm_call_record(call_site="chat.initial", utterance="A"),
                make_llm_call_record(call_site="extraction.scope_classifier", utterance="B"),
                make_llm_call_record(call_site=EXTRACTION_CALL_SITE, utterance="C"),
            ]
        )

        # Act
        index = build_indices(records)

        # Assert
        assert "C" in index
        assert "A" not in index
        assert "B" not in index

    def test_aggregates_multiple_extraction_records_per_utterance(self):
        # Arrange -- two extraction calls for the same utterance (e.g. retry).
        records = iter(
            [
                make_llm_call_record(
                    utterance="X",
                    response_content='{"entities": [{"type": "Event"}], '
                    '"relationships": [{"type": "OCCURRED_ON"}]}',
                ),
                make_llm_call_record(
                    utterance="X",
                    response_content='{"entities": [{"type": "Date"}], '
                    '"relationships": [{"type": "PRECEDED_BY"}]}',
                ),
            ]
        )

        # Act
        index = build_indices(records)

        # Assert -- both records preserved; aggregation happens at score time.
        assert len(index["X"]) == 2

    def test_skips_extraction_with_unrecoverable_utterance(self):
        # Arrange -- malformed user message; extract_utterance returns None.
        record = make_llm_call_record(utterance="X")
        record["request"]["messages"][-1]["content"] = "garbage with no marker"
        records = iter([record])

        # Act
        index = build_indices(records)

        # Assert
        assert index == {}


# ---------------------------------------------------------------------------
# Score run + report aggregation
# ---------------------------------------------------------------------------


class TestScoreRun:
    """End-to-end scoring against the utterance-keyed extraction index."""

    def test_perfect_run(self):
        # Arrange -- one positive that produces the expected edge.
        probes = [build_probe(utterance="A", expected_edges=("OCCURRED_ON",))]
        index = {
            "A": [
                build_extraction_record(
                    relationships=("OCCURRED_ON",),
                    entities=("Event", "Date"),
                )
            ]
        }

        # Act
        report = score_run(probes, index)

        # Assert
        assert report.outcomes[0].matched is True
        assert "OCCURRED_ON" in report.outcomes[0].extracted_edges
        assert report.overall_recall == 1.0

    def test_aggregates_multiple_extractions_per_utterance(self):
        # Arrange -- two extraction records for the same utterance; expect union.
        probes = [build_probe(utterance="A", expected_edges=("OCCURRED_ON", "HAS_METRIC"))]
        index = {
            "A": [
                build_extraction_record(relationships=("OCCURRED_ON",), entities=("Event",)),
                build_extraction_record(relationships=("HAS_METRIC",), entities=("Metric",)),
            ]
        }

        # Act
        report = score_run(probes, index)

        # Assert -- both edges captured via union across retries.
        assert report.outcomes[0].extracted_edges == frozenset({"OCCURRED_ON", "HAS_METRIC"})

    def test_missing_when_no_extraction_matched(self):
        # Arrange -- probe for utterance with no extraction record.
        probes = [build_probe(utterance="A")]

        # Act
        report = score_run(probes, {})

        # Assert
        assert report.outcomes[0].matched is False
        assert report.missing == 1


class TestPerBucketStats:
    """Per-bucket recall computation."""

    def test_per_bucket_counts_expected_and_produced(self):
        # Arrange -- 2 probes expecting OCCURRED_ON; one produces it, one doesn't.
        outcomes = [
            ProbeOutcome(
                probe=build_probe(
                    tag="v8-01-x",
                    utterance="A",
                    expected_edges=("OCCURRED_ON",),
                ),
                extracted_edges=frozenset({"OCCURRED_ON"}),
                extracted_entities=frozenset(),
                matched=True,
                parse_ok=True,
            ),
            ProbeOutcome(
                probe=build_probe(
                    tag="v8-02-x",
                    utterance="B",
                    expected_edges=("OCCURRED_ON",),
                ),
                extracted_edges=frozenset(),
                extracted_entities=frozenset(),
                matched=True,
                parse_ok=True,
            ),
        ]
        report = V8Report(outcomes=outcomes)

        # Act
        stats = report.per_bucket_stats()

        # Assert
        assert stats["OCCURRED_ON"] == {"expected": 2, "produced": 1}
        assert report.per_bucket_recall()["OCCURRED_ON"] == 0.5


# ---------------------------------------------------------------------------
# Acceptance gates
# ---------------------------------------------------------------------------


class TestAcceptance:
    """Acceptance-gate boundary behavior."""

    def _build_report_from_pairs(
        self,
        positive_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]],
        negative_extractions: list[tuple[str, ...]] | None = None,
    ) -> V8Report:
        """Build a V8Report from (expected_edges, extracted_edges) pairs."""
        outcomes: list[ProbeOutcome] = []
        for i, (expected, extracted) in enumerate(positive_pairs):
            outcomes.append(
                ProbeOutcome(
                    probe=build_probe(
                        tag=f"v8-{i:02d}-test",
                        utterance=f"pos-{i}",
                        expected_edges=expected,
                    ),
                    extracted_edges=frozenset(extracted),
                    extracted_entities=frozenset(),
                    matched=True,
                    parse_ok=True,
                )
            )
        for j, extracted in enumerate(negative_extractions or []):
            outcomes.append(
                ProbeOutcome(
                    probe=build_probe(
                        tag=f"v8-{99 - j:02d}-neg-test",
                        utterance=f"neg-{j}",
                        expected_edges=(),
                    ),
                    extracted_edges=frozenset(extracted),
                    extracted_entities=frozenset(),
                    matched=True,
                    parse_ok=True,
                )
            )
        return V8Report(outcomes=outcomes)

    def test_pass_clean_sweep(self):
        # Arrange -- 4 positives (one per bucket) all producing expected edges
        report = self._build_report_from_pairs(
            positive_pairs=[
                (("OCCURRED_ON",), ("OCCURRED_ON",)),
                (("HAS_METRIC",), ("HAS_METRIC",)),
                (("REFERENCES_DOCUMENT",), ("REFERENCES_DOCUMENT",)),
                (("PRECEDED_BY",), ("PRECEDED_BY",)),
            ],
            negative_extractions=[(), ()],  # 2 clean negatives
        )

        # Assert
        assert report.overall_recall == 1.0
        assert all(r == 1.0 for r in report.per_bucket_recall().values())
        assert report.negative_false_positives == 0
        assert report.acceptance_pass()

    def test_fail_when_one_bucket_below_threshold(self):
        # Arrange -- 4 OCCURRED_ON probes; 2 produce, 2 don't (recall = 0.5)
        report = self._build_report_from_pairs(
            positive_pairs=[
                (("OCCURRED_ON",), ("OCCURRED_ON",)),
                (("OCCURRED_ON",), ("OCCURRED_ON",)),
                (("OCCURRED_ON",), ()),
                (("OCCURRED_ON",), ()),
                (("HAS_METRIC",), ("HAS_METRIC",)),
                (("REFERENCES_DOCUMENT",), ("REFERENCES_DOCUMENT",)),
                (("PRECEDED_BY",), ("PRECEDED_BY",)),
            ],
        )

        # Assert -- OCCURRED_ON recall = 0.5 < 0.75 threshold
        assert report.per_bucket_recall()["OCCURRED_ON"] == 0.5
        assert not report.acceptance_pass()

    def test_fail_when_negative_produces_new_edge(self):
        # Arrange -- all positives perfect, but one negative emits a new edge
        report = self._build_report_from_pairs(
            positive_pairs=[
                (("OCCURRED_ON",), ("OCCURRED_ON",)),
                (("HAS_METRIC",), ("HAS_METRIC",)),
                (("REFERENCES_DOCUMENT",), ("REFERENCES_DOCUMENT",)),
                (("PRECEDED_BY",), ("PRECEDED_BY",)),
            ],
            negative_extractions=[("HAS_METRIC",), ()],
        )

        # Assert
        assert report.overall_recall == 1.0
        assert report.negative_false_positives == 1
        assert not report.acceptance_pass()

    def test_negative_with_only_existing_edges_does_not_count_as_fp(self):
        # Arrange -- negative probe extracts USES (existing edge, not in
        # NEW_EDGE_TYPES). Should NOT count as a false positive.
        report = self._build_report_from_pairs(
            positive_pairs=[
                (("OCCURRED_ON",), ("OCCURRED_ON",)),
                (("HAS_METRIC",), ("HAS_METRIC",)),
                (("REFERENCES_DOCUMENT",), ("REFERENCES_DOCUMENT",)),
                (("PRECEDED_BY",), ("PRECEDED_BY",)),
            ],
            negative_extractions=[("USES",), ("KNOWS",)],
        )

        # Assert
        assert report.negative_false_positives == 0
        assert report.acceptance_pass()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    """Markdown rendering surfaces the load-bearing fields."""

    def test_includes_pass_verdict_when_clean(self):
        # Arrange -- minimal clean run
        outcomes = [
            ProbeOutcome(
                probe=build_probe(expected_edges=(e,)),
                extracted_edges=frozenset({e}),
                extracted_entities=frozenset(),
                matched=True,
                parse_ok=True,
            )
            for e in sorted(NEW_EDGE_TYPES)
        ]
        report = V8Report(outcomes=outcomes)

        md = render_markdown(report)

        assert "Verdict:** PASS" in md
        assert "OCCURRED_ON" in md

    def test_includes_failure_status_for_partial_extraction(self):
        # Arrange -- probe expects 2 edges, only 1 produced
        outcomes = [
            ProbeOutcome(
                probe=build_probe(
                    tag="v8-07-multi",
                    expected_edges=("HAS_GOAL", "HAS_METRIC"),
                ),
                extracted_edges=frozenset({"HAS_GOAL"}),
                extracted_entities=frozenset(),
                matched=True,
                parse_ok=True,
            )
        ]
        report = V8Report(outcomes=outcomes)

        md = render_markdown(report)

        assert "PARTIAL" in md
        assert "HAS_METRIC" in md  # missing edge surfaced


class TestRenderJson:
    """JSON rendering round-trips outcomes and per-bucket data."""

    def test_json_structure(self):
        outcomes = [
            ProbeOutcome(
                probe=build_probe(expected_edges=("OCCURRED_ON",)),
                extracted_edges=frozenset({"OCCURRED_ON"}),
                extracted_entities=frozenset({"Event"}),
                matched=True,
                parse_ok=True,
            )
        ]
        report = V8Report(outcomes=outcomes)

        payload = json.loads(render_json(report))

        assert payload["totals"]["probes"] == 1
        assert payload["headline"]["overall_recall"] == 1.0
        assert payload["per_bucket"]["OCCURRED_ON"]["produced"] == 1
        assert payload["acceptance"]["per_bucket_recall_threshold"] == (PER_BUCKET_RECALL_THRESHOLD)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    """End-to-end CLI behavior with synthetic input + debug files."""

    def _setup_files(
        self,
        tmp_path: Path,
        *,
        produce_edge: bool,
    ) -> tuple[Path, Path]:
        input_path = tmp_path / "v8.jsonl"
        debug_path = tmp_path / "debug.jsonl"
        write_jsonl(
            input_path,
            [
                {
                    "utterance": "I had a meeting on 2026-04-15",
                    "tag": "v8-01-event-on-date",
                    "expected_behavior": {
                        "expected_edges": ["OCCURRED_ON"],
                        "expected_entities": ["Event", "Date"],
                        "rationale": "test",
                    },
                },
                {
                    "utterance": "I had a meeting on 2026-04-15 (negative)",
                    "tag": "v8-99-neg-test",
                    "expected_behavior": {
                        "expected_edges": [],
                        "expected_entities": [],
                        "rationale": "negative",
                    },
                },
            ],
        )
        edges_pos = (
            '{"entities": [{"type": "Event"}, {"type": "Date"}], '
            '"relationships": [{"type": "OCCURRED_ON"}]}'
            if produce_edge
            else '{"entities": [], "relationships": []}'
        )
        write_jsonl(
            debug_path,
            [
                # Positive utterance -- extraction record carries the utterance
                # in its request user message so the scorer can recover it.
                make_llm_call_record(
                    session_id="v8-cli-test",
                    utterance="I had a meeting on 2026-04-15",
                    response_content=edges_pos,
                ),
                # Negative utterance (always extracts nothing).
                make_llm_call_record(
                    session_id="v8-cli-test",
                    utterance="I had a meeting on 2026-04-15 (negative)",
                    response_content='{"entities": [], "relationships": []}',
                ),
            ],
        )
        return input_path, debug_path

    def test_writes_markdown_to_stdout_by_default(self, tmp_path, capsys):
        # Arrange
        input_path, debug_path = self._setup_files(tmp_path, produce_edge=True)

        # Act
        rc = main(
            [
                "--input",
                str(input_path),
                "--debug-jsonl",
                str(debug_path),
                "--session-id",
                "v8-cli-test",
            ]
        )

        # Assert -- stdout contains report; rc=0 since --strict not passed
        assert rc == 0
        captured = capsys.readouterr()
        assert "V8 Probe Set" in captured.out

    def test_writes_to_output_file_when_given(self, tmp_path):
        # Arrange
        input_path, debug_path = self._setup_files(tmp_path, produce_edge=True)
        out = tmp_path / "report.md"

        # Act
        rc = main(
            [
                "--input",
                str(input_path),
                "--debug-jsonl",
                str(debug_path),
                "--session-id",
                "v8-cli-test",
                "--output",
                str(out),
            ]
        )

        # Assert
        assert rc == 0
        assert out.exists()
        assert "V8 Probe Set" in out.read_text(encoding="utf-8")

    def test_writes_json_output_when_requested(self, tmp_path):
        # Arrange
        input_path, debug_path = self._setup_files(tmp_path, produce_edge=True)
        json_out = tmp_path / "report.json"

        # Act
        rc = main(
            [
                "--input",
                str(input_path),
                "--debug-jsonl",
                str(debug_path),
                "--session-id",
                "v8-cli-test",
                "--json-output",
                str(json_out),
            ]
        )

        # Assert
        assert rc == 0
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert "headline" in payload

    def test_strict_exits_one_on_acceptance_fail(self, tmp_path, capsys):
        # Arrange -- positive probe doesn't produce expected edge
        input_path, debug_path = self._setup_files(tmp_path, produce_edge=False)

        # Act
        rc = main(
            [
                "--input",
                str(input_path),
                "--debug-jsonl",
                str(debug_path),
                "--session-id",
                "v8-cli-test",
                "--strict",
            ]
        )

        # Assert
        assert rc == 1

    def test_returns_two_when_input_missing(self, tmp_path, capsys):
        # Arrange
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
        # Arrange
        input_path = tmp_path / "v8.jsonl"
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
