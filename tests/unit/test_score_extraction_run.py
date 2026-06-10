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
