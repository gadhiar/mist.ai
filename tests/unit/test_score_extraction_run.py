"""Unit tests for the F2 extraction-accuracy scorer."""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_harness.score_extraction_run import (  # noqa: E402
    GoldEntity,
    GoldRel,
    build_produced_index,
    canonical_id,
    iter_gold_probes,
    parse_produced,
)


class TestCanonicalId:
    def test_lowercases_and_hyphenates(self):
        assert canonical_id("Backend Work") == "backend-work"

    def test_strips_non_alnum_and_collapses_hyphens(self):
        assert canonical_id("C#  / .NET") == "c-net"

    def test_already_canonical_is_identity(self):
        assert canonical_id("rust") == "rust"


class TestIterGoldProbes:
    def test_parses_entities_and_relationships(self, tmp_path):
        p = tmp_path / "gold.jsonl"
        p.write_text(
            json.dumps(
                {
                    "utterance": "I use Rust",
                    "tag": "t1",
                    "expected_entities": [{"id": "rust", "type": "Technology"}],
                    "expected_relationships": [
                        {
                            "source": "user",
                            "source_type": "User",
                            "predicate": "USES",
                            "target": "Rust",
                            "target_type": "Technology",
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        probes = iter_gold_probes(p)
        assert len(probes) == 1
        assert probes[0].entities == (GoldEntity(id="rust", type="Technology"),)
        # target "Rust" is canonicalized to "rust"
        assert probes[0].relationships[0] == GoldRel(
            source="user",
            source_type="User",
            predicate="USES",
            target="rust",
            target_type="Technology",
        )


class TestParseProduced:
    def test_parses_entities_and_rels_and_canonicalizes(self):
        content = json.dumps(
            {
                "entities": [{"id": "Rust", "name": "Rust", "type": "Technology"}],
                "relationships": [
                    {
                        "source": "user",
                        "target": "Rust",
                        "type": "USES",
                        "properties": {"start_date": "2026-05-01"},
                    }
                ],
            }
        )
        ok, entities, type_by_id, rels = parse_produced(content)
        assert ok is True
        assert entities == (GoldEntity(id="rust", type="Technology"),)
        assert type_by_id == {"rust": "Technology"}
        assert rels[0]["source"] == "user" and rels[0]["target"] == "rust"
        assert rels[0]["predicate"] == "USES"
        assert rels[0]["properties"]["start_date"] == "2026-05-01"

    def test_returns_false_on_unparseable(self):
        ok, entities, type_by_id, rels = parse_produced("not json at all")
        assert ok is False and entities == () and rels == ()


class TestBuildProducedIndex:
    def test_indexes_extraction_ontology_records_by_utterance(self):
        content = json.dumps(
            {"entities": [{"id": "rust", "type": "Technology"}], "relationships": []}
        )
        records = [
            {
                "phase": "llm_call",
                "call_site": "extraction.ontology",
                "request": {
                    "messages": [{"role": "user", "content": 'Utterance: "I use Rust"\nOutput:'}]
                },
                "response": {"content": content},
            },
            # non-extraction record must be ignored
            {
                "phase": "llm_call",
                "call_site": "chat.final",
                "request": {"messages": []},
                "response": {},
            },
        ]
        index = build_produced_index(records)
        assert "I use Rust" in index
        assert index["I use Rust"].entities == (GoldEntity(id="rust", type="Technology"),)


from eval_harness.score_extraction_run import GoldProbe, Produced, score_run  # noqa: E402


def _produced(utterance, entities, rels, type_by_id):
    return Produced(
        utterance=utterance,
        parse_ok=True,
        entities=tuple(entities),
        entity_type_by_id=type_by_id,
        relationships=tuple(rels),
    )


class TestScoreRun:
    def test_perfect_match_scores_1(self):
        probe = GoldProbe(
            tag="t",
            utterance="I use Rust",
            entities=(GoldEntity("rust", "Technology"),),
            relationships=(GoldRel("user", "User", "USES", "rust", "Technology"),),
        )
        produced = {
            "I use Rust": _produced(
                "I use Rust",
                [GoldEntity("rust", "Technology")],
                [{"source": "user", "target": "rust", "predicate": "USES", "properties": {}}],
                {"rust": "Technology", "user": "User"},
            )
        }
        r = score_run([probe], produced)
        assert r.entity_precision == 1.0 and r.entity_recall == 1.0
        assert r.rel_precision == 1.0 and r.rel_recall == 1.0
        assert r.typing_accuracy == 1.0
        assert r.related_to_rate == 0.0

    def test_mistyped_relationship_fails_typing(self):
        # USES with target type Person is constraint-invalid (allowed target = Technology).
        probe = GoldProbe(
            tag="t",
            utterance="x",
            entities=(GoldEntity("sarah", "Person"),),
            relationships=(),
        )
        produced = {
            "x": _produced(
                "x",
                [GoldEntity("sarah", "Person")],
                [{"source": "user", "target": "sarah", "predicate": "USES", "properties": {}}],
                {"sarah": "Person", "user": "User"},
            )
        }
        r = score_run([probe], produced)
        assert r.typing_accuracy == 0.0  # 0 of 1 produced rels constraint-valid

    def test_related_to_rate(self):
        probe = GoldProbe(tag="t", utterance="x", entities=(), relationships=())
        produced = {
            "x": _produced(
                "x",
                [],
                [
                    {"source": "a", "target": "b", "predicate": "RELATED_TO", "properties": {}},
                    {"source": "a", "target": "c", "predicate": "USES", "properties": {}},
                ],
                {},
            )
        }
        r = score_run([probe], produced)
        assert r.related_to_rate == 0.5

    def test_valid_time_accuracy_prefix_match(self):
        probe = GoldProbe(
            tag="t",
            utterance="x",
            entities=(GoldEntity("rust", "Technology"),),
            relationships=(
                GoldRel("user", "User", "USES", "rust", "Technology", valid_from="2026-05"),
            ),
        )
        produced = {
            "x": _produced(
                "x",
                [GoldEntity("rust", "Technology")],
                [
                    {
                        "source": "user",
                        "target": "rust",
                        "predicate": "USES",
                        "properties": {"start_date": "2026-05-01"},
                    }
                ],
                {"rust": "Technology", "user": "User"},
            )
        }
        r = score_run([probe], produced)
        assert r.valid_time_accuracy == 1.0  # gold "2026-05" prefix-matches produced "2026-05-01"

    def test_unmatched_probe_counts_as_missing_recall(self):
        probe = GoldProbe(
            tag="t",
            utterance="never produced",
            entities=(GoldEntity("rust", "Technology"),),
            relationships=(GoldRel("user", "User", "USES", "rust", "Technology"),),
        )
        r = score_run([probe], {})
        assert r.entity_recall == 0.0 and r.rel_recall == 0.0
        assert r.matched_probes == 0

    def test_negative_probe_violation_counted(self):
        probe = GoldProbe(tag="neg", utterance="x", entities=(), relationships=())
        produced = {"x": _produced("x", [GoldEntity("a", "Topic")], [], {"a": "Topic"})}
        r = score_run([probe], produced)
        assert r.negative_probes == 1 and r.negative_violations == 1


from eval_harness.score_extraction_run import main, render_json, render_markdown  # noqa: E402


class TestRenderAndCli:
    def test_render_markdown_contains_metric_rows(self):
        from eval_harness.score_extraction_run import Report

        md = render_markdown(Report(total_probes=1, matched_probes=1))
        assert "Entity precision" in md
        assert "RELATED_TO rate" in md
        assert "SKIPPED" in md  # reconciliation hook

    def test_render_json_is_valid_json(self):
        from eval_harness.score_extraction_run import Report

        payload = json.loads(render_json(Report(total_probes=2, matched_probes=1)))
        assert payload["total_probes"] == 2
        assert "entity_precision" in payload

    def test_main_writes_report_and_returns_zero(self, tmp_path):
        gold = tmp_path / "gold.jsonl"
        gold.write_text(
            json.dumps(
                {
                    "utterance": "I use Rust",
                    "tag": "t1",
                    "expected_entities": [{"id": "rust", "type": "Technology"}],
                    "expected_relationships": [
                        {
                            "source": "user",
                            "source_type": "User",
                            "predicate": "USES",
                            "target": "rust",
                            "target_type": "Technology",
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        debug = tmp_path / "debug.jsonl"
        content = json.dumps(
            {
                "entities": [{"id": "rust", "type": "Technology"}],
                "relationships": [
                    {"source": "user", "target": "rust", "type": "USES", "properties": {}}
                ],
            }
        )
        debug.write_text(
            json.dumps(
                {
                    "phase": "llm_call",
                    "call_site": "extraction.ontology",
                    "request": {
                        "messages": [
                            {"role": "user", "content": 'Utterance: "I use Rust"\nOutput:'}
                        ]
                    },
                    "response": {"content": content},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "report.md"
        rc = main(["--gold", str(gold), "--debug-jsonl", str(debug), "--output", str(out)])
        assert rc == 0
        assert "Entity precision" in out.read_text(encoding="utf-8")

    def test_main_returns_2_on_missing_file(self, tmp_path):
        rc = main(
            ["--gold", str(tmp_path / "nope.jsonl"), "--debug-jsonl", str(tmp_path / "nope2.jsonl")]
        )
        assert rc == 2


class TestUtteranceJoinRoundTrip:
    """tests-quality-10: the scorers join debug records to gold probes by
    regexing the utterance out of the rendered prompt. If the pattern drifts
    from EXTRACTION_USER_TEMPLATE, every probe silently fails to join and
    runs score as empty -- pin the round-trip against the REAL render.
    """

    class _Defaults(dict):
        def __missing__(self, key):
            return ""

    def _render(self, utterance: str) -> str:
        from backend.knowledge.extraction.prompts import EXTRACTION_USER_TEMPLATE

        return EXTRACTION_USER_TEMPLATE.format_map(self._Defaults(utterance=utterance))

    def test_extraction_scorer_pattern_matches_real_template_render(self):
        from eval_harness.score_extraction_run import EXTRACTION_UTTERANCE_PATTERN

        utterance = "I use Rust for my backend work"
        m = EXTRACTION_UTTERANCE_PATTERN.search(self._render(utterance))
        assert m is not None, "score_extraction_run pattern no longer matches the template"
        assert m.group(1) == utterance

    def test_v8_scorer_pattern_matches_real_template_render(self):
        from eval_harness.score_v8_probe_run import (
            EXTRACTION_UTTERANCE_PATTERN as V8_PATTERN,
        )

        utterance = "We shipped the vault layer yesterday"
        m = V8_PATTERN.search(self._render(utterance))
        assert m is not None, "score_v8_probe_run pattern no longer matches the template"
        assert m.group(1) == utterance

    def test_v9_scorer_pattern_matches_real_template_render(self):
        from eval_harness.score_v9_predicate_run import (
            EXTRACTION_UTTERANCE_PATTERN as V9_PATTERN,
        )

        utterance = "The scheduler operates on the task queue"
        m = V9_PATTERN.search(self._render(utterance))
        assert m is not None, "score_v9_predicate_run pattern no longer matches the template"
        assert m.group(1) == utterance
