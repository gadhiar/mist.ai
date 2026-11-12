"""Unit tests for the F2 extraction-accuracy scorer."""

import json
import sys
from pathlib import Path

import pytest

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

    def test_gold_loader_accepts_assertion_kind(self, tmp_path):
        line = (
            '{"utterance": "I stopped using X", "tag": "t-cease", '
            '"expected_entities": [{"id": "user", "type": "User"}], '
            '"expected_relationships": [{"source": "user", "source_type": "User", '
            '"predicate": "USES", "target": "x", "target_type": "Technology", '
            '"assertion_kind": "cease", "valid_to": "2025-03"}]}'
        )
        path = tmp_path / "gold.jsonl"
        path.write_text(line, encoding="utf-8")
        probes = iter_gold_probes(path)
        rel = probes[0].relationships[0]
        assert rel.assertion_kind == "cease"
        assert rel.valid_to == "2025-03"

    def test_gold_loader_defaults_assertion_kind_to_assert(self, tmp_path):
        # A relationship with no assertion_kind field defaults to "assert".
        p = tmp_path / "gold.jsonl"
        p.write_text(
            json.dumps(
                {
                    "utterance": "I use Rust",
                    "tag": "t-assert-default",
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
        probes = iter_gold_probes(p)
        assert probes[0].relationships[0].assertion_kind == "assert"

    def test_gold_loader_rejects_unknown_assertion_kind(self, tmp_path):
        # An invalid assertion_kind must fail loudly, naming the probe tag.
        p = tmp_path / "gold.jsonl"
        p.write_text(
            json.dumps(
                {
                    "utterance": "I use Rust",
                    "tag": "t-bad-kind",
                    "expected_entities": [{"id": "rust", "type": "Technology"}],
                    "expected_relationships": [
                        {
                            "source": "user",
                            "source_type": "User",
                            "predicate": "USES",
                            "target": "rust",
                            "target_type": "Technology",
                            "assertion_kind": "bogus",
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="t-bad-kind"):
            iter_gold_probes(p)


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

    def test_metric_entity_id_rebuilt_value_first_from_props(self):
        # ext-24 shape: model emits unit-first id, gold is value-first canonical.
        # The produced match id must collapse to canonical_metric_id(value, unit).
        content = json.dumps(
            {
                "entities": [
                    {
                        "id": "requests-per-second-12000",
                        "type": "Metric",
                        "properties": {"value": "12000", "unit": "requests-per-second"},
                    }
                ],
                "relationships": [],
            }
        )
        ok, entities, type_by_id, rels = parse_produced(content)
        assert ok is True
        assert entities == (GoldEntity(id="12000-requests-per-second", type="Metric"),)
        assert type_by_id == {"12000-requests-per-second": "Metric"}

    def test_metric_rel_endpoint_remapped_to_canonical_metric_id(self):
        # A relationship pointing at the Metric by its raw id must be rewritten to
        # the value-first canonical id, so the edge does not split FP+FN.
        content = json.dumps(
            {
                "entities": [
                    {"id": "api", "type": "Technology"},
                    {
                        "id": "requests-per-second-12000",
                        "type": "Metric",
                        "properties": {"value": "12000", "unit": "requests-per-second"},
                    },
                ],
                "relationships": [
                    {
                        "source": "api",
                        "target": "requests-per-second-12000",
                        "type": "HAS_METRIC",
                        "properties": {},
                    }
                ],
            }
        )
        ok, entities, type_by_id, rels = parse_produced(content)
        assert ok is True
        assert rels[0]["source"] == "api"
        assert rels[0]["target"] == "12000-requests-per-second"

    def test_metric_without_value_unit_falls_back_to_string_canonical(self):
        # No value/unit props -> plain string canonical_id, id unchanged.
        content = json.dumps(
            {
                "entities": [{"id": "some-metric", "type": "Metric"}],
                "relationships": [],
            }
        )
        ok, entities, type_by_id, rels = parse_produced(content)
        assert ok is True
        assert entities == (GoldEntity(id="some-metric", type="Metric"),)

    def test_metric_entity_id_rebuilt_with_quantity(self):
        # A p95 and a p99 latency of one value and unit score as two entities.
        content = json.dumps(
            {
                "entities": [
                    {
                        "id": "p95-latency",
                        "type": "Metric",
                        "properties": {"value": "120", "unit": "ms", "quantity": "p95 latency"},
                    },
                    {
                        "id": "p99-latency",
                        "type": "Metric",
                        "properties": {"value": "120", "unit": "ms", "quantity": "p99 latency"},
                    },
                ],
                "relationships": [],
            }
        )
        ok, entities, type_by_id, rels = parse_produced(content)
        assert ok is True
        assert entities == (
            GoldEntity(id="120-ms-p95-latency", type="Metric"),
            GoldEntity(id="120-ms-p99-latency", type="Metric"),
        )

    def test_metric_entity_id_rebuilt_from_value_and_quantity_without_unit(self):
        content = json.dumps(
            {
                "entities": [
                    {
                        "id": "max-queue-depth",
                        "type": "Metric",
                        "properties": {"value": 180, "quantity": "max queue depth"},
                    }
                ],
                "relationships": [],
            }
        )
        ok, entities, type_by_id, rels = parse_produced(content)
        assert ok is True
        assert entities == (GoldEntity(id="180-max-queue-depth", type="Metric"),)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "properties",
        [
            {"value": "120", "unit": "ms"},
            {"value": "120", "unit": "ms", "quantity": "p95 latency"},
            {"value": "87", "unit": "percent-coverage", "quantity": "coverage"},
            {"value": 180, "quantity": "max queue depth"},
            {"value": "42"},
            {},
        ],
    )
    async def test_metric_match_id_equals_normalizer_id(self, properties):
        # The scorer must score the id production stores, so both sides build it
        # from the same properties with the same shared builder.
        from backend.knowledge.extraction.normalizer import EntityNormalizer
        from backend.knowledge.extraction.ontology_extractor import ExtractionResult
        from tests.mocks.embeddings import FakeEmbeddingGenerator

        raw_id = "requests-per-second-12000"
        normalizer = EntityNormalizer(embedding_generator=FakeEmbeddingGenerator(), executor=None)
        entity = {"id": raw_id, "type": "Metric", "properties": dict(properties)}
        out = await normalizer.normalize(ExtractionResult(entities=[entity], relationships=[]))

        content = json.dumps(
            {
                "entities": [{"id": raw_id, "type": "Metric", "properties": dict(properties)}],
                "relationships": [],
            }
        )
        ok, entities, type_by_id, rels = parse_produced(content)
        assert ok is True
        assert entities == (GoldEntity(id=out.entities[0]["id"], type="Metric"),)

    @pytest.mark.asyncio
    async def test_parse_produced_equals_the_normalizer_on_the_whole_payload(self):
        # The scorer scores the ids, types and edges the graph stores: the same id rails
        # run over the same payload on both sides.
        from backend.knowledge.extraction.normalizer import EntityNormalizer
        from backend.knowledge.extraction.ontology_extractor import ExtractionResult
        from tests.mocks.embeddings import FakeEmbeddingGenerator

        payload = {
            "entities": [
                {"id": "user", "name": "user", "type": "User"},
                {"id": "python-3-11", "name": "Python 3.11", "type": "Technology"},
                {"id": "vs-code", "name": "VS Code", "type": "Technology"},
                {"id": "The Acme Corp", "name": "The Acme Corp", "type": "Organization"},
                {"id": "mist", "name": "MIST", "type": "Organization"},
                {"id": "c++", "name": "C++", "type": "Technology"},
                {"id": "tool-selection-0.94", "name": "tool selection", "type": "Metric"},
                {
                    "id": "cold-start",
                    "name": "cold start",
                    "type": "Metric",
                    "properties": {"value": 2.4, "unit": "s", "quantity": "cold start"},
                },
                {"id": "the-team-offsite", "name": "the team offsite", "type": "Milestone"},
                {"id": "2026-05-30", "name": "May 30", "type": "Date"},
                {"id": "my-sister", "name": "my sister", "type": "Person"},
                {"id": "python", "name": "Python", "type": "Technology"},
            ],
            "relationships": [
                {"source": "user", "target": "python-3-11", "type": "USES"},
                {"source": "user", "target": "vs-code", "type": "USES"},
                {"source": "user", "target": "The Acme Corp", "type": "WORKS_AT"},
                {"source": "mist", "target": "c++", "type": "IMPLEMENTED_WITH"},
                {"source": "the-acme-corp", "target": "tool-selection-0.94", "type": "HAS_METRIC"},
                {"source": "the-acme-corp", "target": "cold-start", "type": "HAS_METRIC"},
                {"source": "user", "target": "the-team-offsite", "type": "EXPERIENCED"},
                {"source": "the-team-offsite", "target": "2026-05-30", "type": "OCCURRED_ON"},
                {"source": "user", "target": "my-sister", "type": "KNOWS_PERSON"},
                {"source": "user", "target": "python", "type": "USES"},
            ],
        }
        normalizer = EntityNormalizer(embedding_generator=FakeEmbeddingGenerator(), executor=None)
        out = await normalizer.normalize(
            ExtractionResult(
                entities=json.loads(json.dumps(payload["entities"])),
                relationships=json.loads(json.dumps(payload["relationships"])),
            )
        )

        ok, entities, type_by_id, rels = parse_produced(json.dumps(payload))

        assert ok is True
        assert entities == tuple(GoldEntity(id=e["id"], type=e["type"]) for e in out.entities)
        assert type_by_id == {e["id"]: e["type"] for e in out.entities}
        stored_ids = {e["id"] for e in out.entities}
        assert [(r["source"], r["predicate"], r["target"]) for r in rels] == [
            (
                r["source"] if r["source"] in stored_ids else canonical_id(r["source"]),
                r["type"],
                r["target"] if r["target"] in stored_ids else canonical_id(r["target"]),
            )
            for r in out.relationships
        ]
        assert ("user", "USES", "python") in [
            (r["source"], r["predicate"], r["target"]) for r in rels
        ]
        assert ("team-offsite-2026-05-30", "OCCURRED_ON", "2026-05-30") in [
            (r["source"], r["predicate"], r["target"]) for r in rels
        ]

    @pytest.mark.asyncio
    async def test_parse_produced_reads_the_rows_batch_record_as_the_normalizer_does(self):
        # A list under the lead-in `Harbour weekend`, extracted as one batch: the model
        # wrote the lead-in's slug in front of each item's id. The apply side and the
        # scorer hand the same record to the same rails, so both strip it.
        from backend.extraction_contract.models import BatchRecordEntry, ExtractionPayload
        from backend.knowledge.extraction.normalizer import EntityNormalizer
        from backend.knowledge.extraction.ontology_extractor import ExtractionResult
        from tests.mocks.embeddings import FakeEmbeddingGenerator

        payload = {
            "entities": [
                {"id": "user", "name": "user", "type": "User"},
                {
                    "id": "harbour-weekend-ferry-crossing",
                    "name": "Ferry crossing",
                    "type": "Event",
                },
                {
                    "id": "harbour-weekend-the-lighthouse-walk",
                    "name": "the lighthouse walk",
                    "type": "Event",
                },
                {"id": "2026-05-30", "name": "2026-05-30", "type": "Date"},
            ],
            "relationships": [
                {
                    "source": "user",
                    "target": "harbour-weekend-ferry-crossing",
                    "type": "EXPERIENCED",
                },
                {
                    "source": "user",
                    "target": "harbour-weekend-the-lighthouse-walk",
                    "type": "EXPERIENCED",
                },
                {
                    "source": "harbour-weekend-ferry-crossing",
                    "target": "2026-05-30",
                    "type": "OCCURRED_ON",
                },
            ],
        }
        record = [
            BatchRecordEntry(
                units=[
                    {"text": text, "context": ["Harbour weekend"], "kind": "list_item"}
                    for text in ("ferry crossing on 30 May", "the lighthouse walk")
                ],
                context=[],
                payload=ExtractionPayload(**json.loads(json.dumps(payload))),
                prompt_tokens=120,
                completion_tokens=60,
                finish="stop",
            ).model_dump(mode="json")
        ]
        normalizer = EntityNormalizer(embedding_generator=FakeEmbeddingGenerator(), executor=None)
        out = await normalizer.normalize(
            ExtractionResult(
                entities=json.loads(json.dumps(payload["entities"])),
                relationships=json.loads(json.dumps(payload["relationships"])),
            ),
            batch_record=record,
        )

        ok, entities, type_by_id, rels = parse_produced(json.dumps(payload), batch_record=record)

        assert ok is True
        assert [e.id for e in entities] == [
            "user",
            "ferry-crossing-2026-05-30",
            "lighthouse-walk",
            "2026-05-30",
        ]
        assert entities == tuple(GoldEntity(id=e["id"], type=e["type"]) for e in out.entities)
        assert [(r["source"], r["predicate"], r["target"]) for r in rels] == [
            ("user", "EXPERIENCED", "ferry-crossing-2026-05-30"),
            ("user", "EXPERIENCED", "lighthouse-walk"),
            ("ferry-crossing-2026-05-30", "OCCURRED_ON", "2026-05-30"),
        ]
        assert [(r["source"], r["predicate"], r["target"]) for r in rels] == [
            (r["source"], r["type"], r["target"]) for r in out.relationships
        ]

    @pytest.mark.asyncio
    async def test_parse_produced_runs_both_engine_rails_as_the_normalizer_does(self):
        # One event of a list under `Harbour weekend`, written by two batches under two
        # spellings of its id and one name. The near-duplicate merge joins the second
        # into the first, and the own-context rail then drops the lead-in from the id
        # that is left. The apply side and the scorer run the same rails over the same
        # record, so the graph stores the id the scorer scores.
        from backend.extraction_contract.models import BatchRecordEntry, ExtractionPayload
        from backend.knowledge.extraction.normalizer import EntityNormalizer
        from backend.knowledge.extraction.ontology_extractor import ExtractionResult
        from tests.mocks.embeddings import FakeEmbeddingGenerator

        first = {"id": "harbour-weekend-ferry-crossing", "name": "Ferry crossing", "type": "Event"}
        second = {"id": "harbour-weekend-fery-crossing", "name": "Ferry crossing", "type": "Event"}
        user = {"id": "user", "name": "user", "type": "User"}
        day = {"id": "2026-05-30", "name": "2026-05-30", "type": "Date"}
        payload = {
            "entities": [user, first, second, day],
            "relationships": [
                {"source": "user", "target": first["id"], "type": "EXPERIENCED"},
                {"source": "user", "target": second["id"], "type": "EXPERIENCED"},
                {"source": second["id"], "target": "2026-05-30", "type": "OCCURRED_ON"},
            ],
        }
        record = [
            BatchRecordEntry(
                units=[{"text": text, "context": ["Harbour weekend"], "kind": "list_item"}],
                context=[],
                payload=ExtractionPayload(entities=held, relationships=[]),
                prompt_tokens=120,
                completion_tokens=60,
                finish="stop",
            ).model_dump(mode="json")
            for text, held in (
                ("ferry crossing", [user, first]),
                ("the fery crossing on 30 May", [second, day]),
            )
        ]
        normalizer = EntityNormalizer(embedding_generator=FakeEmbeddingGenerator(), executor=None)
        out = await normalizer.normalize(
            ExtractionResult(
                entities=json.loads(json.dumps(payload["entities"])),
                relationships=json.loads(json.dumps(payload["relationships"])),
            ),
            batch_record=record,
        )

        ok, entities, _, rels = parse_produced(json.dumps(payload), batch_record=record)

        assert ok is True
        assert [e.id for e in entities] == ["user", "ferry-crossing-2026-05-30", "2026-05-30"]
        assert entities == tuple(GoldEntity(id=e["id"], type=e["type"]) for e in out.entities)
        assert [(r["source"], r["predicate"], r["target"]) for r in rels] == [
            ("user", "EXPERIENCED", "ferry-crossing-2026-05-30"),
            ("ferry-crossing-2026-05-30", "OCCURRED_ON", "2026-05-30"),
        ]
        assert [(r["source"], r["predicate"], r["target"]) for r in rels] == [
            (r["source"], r["type"], r["target"]) for r in out.relationships
        ]

    def test_parse_produced_without_a_batch_record_strips_no_context(self):
        content = json.dumps(
            {
                "entities": [
                    {
                        "id": "harbour-weekend-ferry-crossing",
                        "name": "Ferry crossing",
                        "type": "Event",
                    }
                ],
                "relationships": [],
            }
        )

        _, entities, _, _ = parse_produced(content)

        assert [e.id for e in entities] == ["harbour-weekend-ferry-crossing"]

    def test_a_bare_endpoint_keeps_the_canonical_id_slug(self):
        # An endpoint that names no produced entity is slugged, as before the rails, and
        # takes no rail: the N1 rail runs on entities only.
        content = json.dumps(
            {
                "entities": [{"id": "user", "type": "User"}],
                "relationships": [
                    {"source": "user", "target": "The Unlisted Thing", "type": "USES"},
                    {"source": "User", "target": "Python 3.11", "type": "USES"},
                ],
            }
        )

        ok, entities, type_by_id, rels = parse_produced(content)

        assert [(r["source"], r["target"]) for r in rels] == [
            ("user", "the-unlisted-thing"),
            ("user", "python-311"),
        ]

    def test_the_aligner_reads_the_n1_rails_words(self):
        from eval_harness import entity_alignment

        from backend.knowledge.extraction import id_rails

        assert entity_alignment.N1_WORDS is id_rails.N1_WORDS

    def test_produced_entity_match_id_is_the_stored_id_and_idempotent(self):
        from eval_harness.score_extraction_run import _produced_entity_match_id

        from backend.knowledge.extraction.id_rails import stored_entity_id

        cases = [
            ("the-acme-corp", "Organization", {}),
            ("vs-code", "Technology", {}),
            ("tool-selection-0.94", "Metric", {}),
            ("p95", "Metric", {"value": "120", "unit": "msec", "quantity": "p95 latency"}),
        ]
        for raw_id, etype, props in cases:
            match_id = _produced_entity_match_id(raw_id, etype, props)
            assert match_id == stored_entity_id(raw_id, etype, props)
            assert _produced_entity_match_id(match_id, etype, {}) == match_id

    def test_non_metric_entity_not_affected_by_value_unit_props(self):
        # A non-Metric entity that happens to carry value/unit props keeps the
        # plain string canonical id (Metric-id rebuild is type-gated).
        content = json.dumps(
            {
                "entities": [
                    {
                        "id": "rust",
                        "type": "Technology",
                        "properties": {"value": "1", "unit": "thing"},
                    }
                ],
                "relationships": [],
            }
        )
        ok, entities, type_by_id, rels = parse_produced(content)
        assert ok is True
        assert entities == (GoldEntity(id="rust", type="Technology"),)


class TestTheRowTheScorerGivesTheRails:
    """Plan v15, T50 Acceptance 8: the rails read the raw record's scope, batch record and plan.

    A rail that reads any of the three gives the scorer the payload it gives the graph only
    when the scorer hands it what the service recorded for the turn.
    """

    _PAYLOAD = {
        "entities": [
            {"id": "user", "name": "user", "type": "User"},
            {"id": "rust", "name": "Rust", "type": "Technology"},
        ],
        "relationships": [{"source": "user", "target": "rust", "type": "USES"}],
    }
    _RECORD = [
        {
            "units": [{"text": "I use Rust", "context": [], "kind": "prose"}],
            "context": [],
            "payload": _PAYLOAD,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "finish": "stop",
        }
    ]
    _PLAN = {
        "payload": {"entities": [], "templates": [], "back_references": []},
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "finish_reason": "stop",
    }

    @pytest.fixture
    def rows(self, monkeypatch):
        import eval_harness.score_extraction_run as scorer

        seen = []
        real = scorer.rails.run_rails

        def spy(row, pack, **switches):
            seen.append(row)
            return real(row, pack, **switches)

        monkeypatch.setattr(scorer.rails, "run_rails", spy)
        return seen

    def _response(self, **extra):
        return {
            "scope": {"label": "user-scope", "confidence": 0.9},
            "payload": json.loads(json.dumps(self._PAYLOAD)),
            **extra,
        }

    def _produce(self, response):
        from eval_harness.score_extraction_run import produced_from_response

        return produced_from_response("I  use Rust", response)

    def test_the_row_carries_the_scope_label(self, rows):
        self._produce(self._response())
        assert rows and all(row.scope == "user-scope" for row in rows)

    def test_the_row_carries_the_batch_record(self, rows):
        self._produce(self._response(batch_record=json.loads(json.dumps(self._RECORD))))
        assert rows and all(row.batch_record == tuple(self._RECORD) for row in rows)

    def test_the_row_carries_the_turn_plan(self, rows):
        self._produce(self._response(turn_plan=json.loads(json.dumps(self._PLAN))))
        assert rows and all(row.turn_plan == self._PLAN["payload"] for row in rows)

    def test_a_record_without_them_is_a_row_of_one_batch_with_no_plan(self, rows):
        self._produce(self._response())
        assert rows and all((row.batch_record, row.turn_plan) == ((), None) for row in rows)

    def test_the_produced_payload_is_keyed_by_the_normalised_utterance(self, rows):
        produced = self._produce(self._response())
        assert produced.utterance == "I use Rust"
        assert produced.parse_ok is True
        assert produced.entity_type_by_id == {"user": "User", "rust": "Technology"}


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

    def test_per_probe_reports_fp_fn_detail(self):
        """per_probe entries must carry the exact FP/FN sets so failures are diagnosable."""
        probes = [
            GoldProbe(
                tag="t1",
                utterance="I use Rust",
                entities=(
                    GoldEntity(id="user", type="User"),
                    GoldEntity(id="rust", type="Technology"),
                ),
                relationships=(
                    GoldRel(
                        source="user",
                        source_type="User",
                        predicate="USES",
                        target="rust",
                        target_type="Technology",
                    ),
                ),
            )
        ]
        produced = {
            "I use Rust": Produced(
                utterance="I use Rust",
                parse_ok=True,
                entities=(
                    GoldEntity(id="user", type="User"),
                    GoldEntity(id="cargo", type="Technology"),
                ),
                entity_type_by_id={"user": "User", "cargo": "Technology"},
                relationships=(
                    {"source": "user", "target": "cargo", "predicate": "USES", "properties": {}},
                ),
            )
        }
        report = score_run(probes, produced)
        row = report.per_probe[0]
        assert row["matched"] is True
        assert row["entity_fps"] == [["cargo", "Technology"]]
        assert row["entity_fns"] == [["rust", "Technology"]]
        assert row["rel_fps"] == [["user", "USES", "cargo"]]
        assert row["rel_fns"] == [["user", "USES", "rust"]]

    def test_per_probe_unmatched_reports_no_fps_full_gold_fns(self):
        """An UNMATCHED probe must report empty FP lists and full-gold FN lists.

        Pins the unmatched branch (T2): a probe with no produced extraction
        cannot have false positives, and every gold entity/relationship is a
        false negative.
        """
        probe = GoldProbe(
            tag="t-unmatched",
            utterance="never produced",
            entities=(
                GoldEntity(id="user", type="User"),
                GoldEntity(id="rust", type="Technology"),
            ),
            relationships=(
                GoldRel(
                    source="user",
                    source_type="User",
                    predicate="USES",
                    target="rust",
                    target_type="Technology",
                ),
            ),
        )
        report = score_run([probe], {})
        row = report.per_probe[0]
        assert row["matched"] is False
        assert row["entity_fps"] == []
        assert row["rel_fps"] == []
        assert row["entity_fns"] == [["rust", "Technology"], ["user", "User"]]
        assert row["rel_fns"] == [["user", "USES", "rust"]]

    def test_assertion_kind_buckets_scored(self):
        """Buckets = {gold_total, found, correct} per kind; derivation is the
        engine's own derive_assertion_kind (explicit wins incl. explicit assert;
        absent + past + no end_date -> cease).

        `found` counts gold edges of that kind that were extracted at all (TP
        membership); `correct` counts found edges whose engine-derived kind
        matches gold. The retract probe's edge is NOT extracted, so found=0 --
        the count that catches a model which silently never emits cessations.
        """
        probes = [
            GoldProbe(
                tag="t-c",
                utterance="I stopped using X",
                entities=(
                    GoldEntity(id="user", type="User"),
                    GoldEntity(id="x", type="Technology"),
                ),
                relationships=(
                    GoldRel(
                        source="user",
                        source_type="User",
                        predicate="USES",
                        target="x",
                        target_type="Technology",
                        assertion_kind="cease",
                    ),
                ),
            ),
            GoldProbe(
                tag="t-r",
                utterance="I never used Z",
                entities=(
                    GoldEntity(id="user", type="User"),
                    GoldEntity(id="z", type="Technology"),
                ),
                relationships=(
                    GoldRel(
                        source="user",
                        source_type="User",
                        predicate="USES",
                        target="z",
                        target_type="Technology",
                        assertion_kind="retract",
                    ),
                ),
            ),
        ]
        produced = {
            # cease probe: edge found, no explicit kind, past + no end_date ->
            # engine derives cease (correct).
            "I stopped using X": _produced(
                "I stopped using X",
                [GoldEntity(id="user", type="User"), GoldEntity(id="x", type="Technology")],
                [
                    {
                        "source": "user",
                        "target": "x",
                        "predicate": "USES",
                        "properties": {"temporal_status": "past"},
                    }
                ],
                {"user": "User", "x": "Technology"},
            ),
            # retract probe: edge NOT extracted at all -> found=0.
            "I never used Z": _produced(
                "I never used Z",
                [GoldEntity(id="user", type="User")],
                [],
                {"user": "User"},
            ),
        }
        report = score_run(probes, produced)
        assert report.assertion_kind_buckets["cease"] == {
            "gold_total": 1,
            "found": 1,
            "correct": 1,
        }
        assert report.assertion_kind_buckets["retract"] == {
            "gold_total": 1,
            "found": 0,
            "correct": 0,
        }
        assert report.assertion_kind_buckets["assert"] == {
            "gold_total": 0,
            "found": 0,
            "correct": 0,
        }


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

    def test_every_summary_names_its_scorer_gold_and_pack(self, tmp_path):
        # Plan v14 T50 Acceptance 1: both reports carry the four fields of a reading.
        from backend.knowledge.ontologies import registry
        from scripts.eval_harness import governance

        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({"utterance": "I use Rust", "tag": "t1"}) + "\n")
        debug = tmp_path / "debug.jsonl"
        debug.write_text("", encoding="utf-8")
        md, js = tmp_path / "report.md", tmp_path / "report.json"
        argv = ["--gold", str(gold), "--debug-jsonl", str(debug), "--output", str(md)]
        assert main([*argv, "--json-output", str(js)]) == 0
        pack = registry.active_pack()
        expected = {
            "scorer_sha": governance.scorer_sha(_REPO_ROOT),
            "gold_version": governance.gold_version(gold, _REPO_ROOT, pack.version),
            "pack_prompt_hash": pack.pack_prompt_hash,
            "pack_rails_hash": pack.pack_rails_hash,
        }
        payload = json.loads(js.read_text(encoding="utf-8"))
        assert {name: payload.get(name) for name in expected} == expected
        governance.require_reading(payload)
        assert (
            f"- Reading: {governance.identity_text(expected)}"
            in md.read_text(encoding="utf-8").splitlines()
        )

    def test_main_returns_2_on_missing_file(self, tmp_path):
        rc = main(
            ["--gold", str(tmp_path / "nope.jsonl"), "--debug-jsonl", str(tmp_path / "nope2.jsonl")]
        )
        assert rc == 2


class TestTypesMatch:
    """Hierarchy-aware type matching (Task 9)."""

    def test_exact_match_passes(self):
        from eval_harness.score_extraction_run import types_match

        assert types_match(produced="Concept", gold="Concept") is True

    def test_parent_type_acceptable_for_child_gold(self):
        from eval_harness.score_extraction_run import types_match

        assert types_match(produced="Abstraction", gold="Concept") is True

    def test_wrong_sibling_fails(self):
        from eval_harness.score_extraction_run import types_match

        assert types_match(produced="Pattern", gold="Concept") is False

    def test_retired_gold_type_canonicalized_before_match(self):
        from eval_harness.score_extraction_run import types_match

        # gold "Topic" -> canonical "Concept"; produced "Concept" == canonical
        assert types_match(produced="Concept", gold="Topic") is True

    def test_retired_produced_type_canonicalized_before_match(self):
        from eval_harness.score_extraction_run import types_match

        # produced "Topic" -> canonical "Concept"; gold "Concept" == canonical
        assert types_match(produced="Topic", gold="Concept") is True

    def test_non_cluster_exact_pass(self):
        from eval_harness.score_extraction_run import types_match

        assert types_match(produced="Technology", gold="Technology") is True

    def test_non_cluster_type_mismatch_fails(self):
        from eval_harness.score_extraction_run import types_match

        assert types_match(produced="Technology", gold="Project") is False


class TestScoreRunSpecificity:
    """Integration: specificity flows through score_run into Report."""

    def test_specificity_in_report_when_leaf_matched(self):
        from eval_harness.score_extraction_run import GoldEntity, GoldProbe, Produced, score_run

        probe = GoldProbe(
            tag="t",
            utterance="x",
            entities=(GoldEntity(id="agile", type="Concept"),),
            relationships=(),
        )
        produced = {
            "x": Produced(
                utterance="x",
                parse_ok=True,
                entities=(GoldEntity(id="agile", type="Concept"),),
                entity_type_by_id={"agile": "Concept"},
                relationships=(),
            )
        }
        r = score_run([probe], produced)
        assert r.specificity == 1.0

    def test_specificity_in_report_when_parent_used(self):
        from eval_harness.score_extraction_run import GoldEntity, GoldProbe, Produced, score_run

        probe = GoldProbe(
            tag="t",
            utterance="x",
            entities=(GoldEntity(id="agile", type="Concept"),),
            relationships=(),
        )
        produced = {
            "x": Produced(
                utterance="x",
                parse_ok=True,
                entities=(GoldEntity(id="agile", type="Abstraction"),),
                entity_type_by_id={"agile": "Abstraction"},
                relationships=(),
            )
        }
        r = score_run([probe], produced)
        assert r.specificity == 0.0

    def test_specificity_vacuously_one_when_no_cluster_gold(self):
        from eval_harness.score_extraction_run import GoldEntity, GoldProbe, Produced, score_run

        probe = GoldProbe(
            tag="t",
            utterance="x",
            entities=(GoldEntity(id="rust", type="Technology"),),
            relationships=(),
        )
        produced = {
            "x": Produced(
                utterance="x",
                parse_ok=True,
                entities=(GoldEntity(id="rust", type="Technology"),),
                entity_type_by_id={"rust": "Technology"},
                relationships=(),
            )
        }
        r = score_run([probe], produced)
        assert r.specificity == 1.0


class TestGatesPassSpecificity:
    """_gates_pass enforces SPECIFICITY_FLOOR = 0.90."""

    def _base_report(self):
        """Return a Report that satisfies all gates except the ones under test.

        All entity/rel/typing metrics are set to perfect values so the only
        gate that can cause a failure is the one being exercised.
        """
        from eval_harness.score_extraction_run import Report

        r = Report(
            total_probes=1,
            matched_probes=1,
            entity_tp=1,
            entity_fp=0,
            entity_fn=0,
            rel_tp=0,
            rel_fp=0,
            rel_fn=0,
            typing_total=0,
            typing_ok=0,
            related_to_count=0,
            produced_rel_total=0,
            negative_probes=0,
            negative_violations=0,
        )
        for kind in ("assert", "cease", "retract"):
            r.assertion_kind_buckets[kind] = {"gold_total": 0, "found": 0, "correct": 0}
        return r

    def test_specificity_below_floor_fails_gate(self):
        from eval_harness.score_extraction_run import _gates_pass

        r = self._base_report()
        # specificity = 0/1 = 0.0 (below 0.90 floor)
        r.specificity_denominator = 1
        r.specificity_numerator = 0
        assert r.specificity == 0.0
        assert _gates_pass(r) is False

    def test_specificity_at_floor_passes_gate(self):
        from eval_harness.score_extraction_run import _gates_pass

        r = self._base_report()
        # specificity = 9/10 = 0.90 (exactly at floor)
        r.specificity_denominator = 10
        r.specificity_numerator = 9
        assert r.specificity == 0.90
        assert _gates_pass(r) is True

    def test_specificity_above_floor_passes_gate(self):
        from eval_harness.score_extraction_run import _gates_pass

        r = self._base_report()
        # specificity = 1.0 (all precise leaves, as in real passing runs)
        r.specificity_denominator = 5
        r.specificity_numerator = 5
        assert r.specificity == 1.0
        assert _gates_pass(r) is True

    def test_vacuous_specificity_passes_gate(self):
        from eval_harness.score_extraction_run import _gates_pass

        r = self._base_report()
        # No abstract-cluster gold entities: specificity_denominator == 0 -> 1.0
        assert r.specificity_denominator == 0
        assert r.specificity == 1.0
        assert _gates_pass(r) is True


class TestTypesMatchInEntityPR:
    """Hierarchy-aware entity P/R: parent type counts as TP, not FP+FN."""

    def test_parent_type_counts_as_tp_not_fp_fn(self):
        from eval_harness.score_extraction_run import GoldEntity, GoldProbe, Produced, score_run

        # Gold: Concept; Produced: Abstraction (parent). Should match as TP under
        # hierarchy-aware scoring rather than FP(Abstraction) + FN(Concept).
        probe = GoldProbe(
            tag="t",
            utterance="x",
            entities=(GoldEntity(id="agile", type="Concept"),),
            relationships=(),
        )
        produced = {
            "x": Produced(
                utterance="x",
                parse_ok=True,
                entities=(GoldEntity(id="agile", type="Abstraction"),),
                entity_type_by_id={"agile": "Abstraction"},
                relationships=(),
            )
        }
        r = score_run([probe], produced)
        assert r.entity_tp == 1
        assert r.entity_fp == 0
        assert r.entity_fn == 0


class TestMetricIdCanonicalizationInScoreRun:
    """ext-24/ext-25: produced-side Metric id rebuild collapses surface splits so
    the entity and its HAS_METRIC edge score as TP, not paired FP+FN.
    """

    def test_ext25_metric_no_props_string_fallback_resolves_to_tp(self):
        # Mirror of ext-24 but with NO value/unit props on the produced entity.
        # The string-based fallback (canonical_metric_id_from_id) must collapse
        # `requests-per-second-12000` -> `12000-requests-per-second` so both the
        # entity and the HAS_METRIC edge score as TP (0 FP, 0 FN).
        probe = GoldProbe(
            tag="ext-25-metric-no-props",
            utterance="Our API handles 12000 requests per second at peak",
            entities=(
                GoldEntity(id="api", type="Technology"),
                GoldEntity(id="12000-requests-per-second", type="Metric"),
            ),
            relationships=(
                GoldRel(
                    source="api",
                    source_type="Technology",
                    predicate="HAS_METRIC",
                    target="12000-requests-per-second",
                    target_type="Metric",
                ),
            ),
        )
        # Produced: Metric entity has NO value/unit props, raw id is unit-first.
        raw = json.dumps(
            {
                "entities": [
                    {"id": "api", "type": "Technology"},
                    {
                        "id": "requests-per-second-12000",
                        "type": "Metric",
                    },
                ],
                "relationships": [
                    {
                        "source": "api",
                        "target": "requests-per-second-12000",
                        "type": "HAS_METRIC",
                        "properties": {},
                    }
                ],
            }
        )
        ok, entities, type_by_id, rels = parse_produced(raw)
        produced = {
            "Our API handles 12000 requests per second at peak": Produced(
                utterance="Our API handles 12000 requests per second at peak",
                parse_ok=ok,
                entities=entities,
                entity_type_by_id=type_by_id,
                relationships=rels,
            )
        }
        r = score_run([probe], produced)
        # Both entities TP via string-fallback, no surface-split FP/FN.
        assert r.entity_tp == 2
        assert r.entity_fp == 0
        assert r.entity_fn == 0
        # HAS_METRIC edge resolves because the endpoint remap fired on the string path.
        assert r.rel_tp == 1
        assert r.rel_fp == 0
        assert r.rel_fn == 0

    def test_ext24_metric_surface_split_resolves_to_tp(self):
        # Gold is the value-first canonical (matches data/ingest gold ext-24).
        probe = GoldProbe(
            tag="ext-24-metric",
            utterance="Our API handles 12000 requests per second at peak",
            entities=(
                GoldEntity(id="api", type="Technology"),
                GoldEntity(id="12000-requests-per-second", type="Metric"),
            ),
            relationships=(
                GoldRel(
                    source="api",
                    source_type="Technology",
                    predicate="HAS_METRIC",
                    target="12000-requests-per-second",
                    target_type="Metric",
                ),
            ),
        )
        # Produced raw LLM output emits the unit-first id; parse_produced rebuilds it.
        raw = json.dumps(
            {
                "entities": [
                    {"id": "api", "type": "Technology"},
                    {
                        "id": "requests-per-second-12000",
                        "type": "Metric",
                        "properties": {"value": "12000", "unit": "requests-per-second"},
                    },
                ],
                "relationships": [
                    {
                        "source": "api",
                        "target": "requests-per-second-12000",
                        "type": "HAS_METRIC",
                        "properties": {},
                    }
                ],
            }
        )
        ok, entities, type_by_id, rels = parse_produced(raw)
        produced = {
            "Our API handles 12000 requests per second at peak": Produced(
                utterance="Our API handles 12000 requests per second at peak",
                parse_ok=ok,
                entities=entities,
                entity_type_by_id=type_by_id,
                relationships=rels,
            )
        }
        r = score_run([probe], produced)
        # Both entities TP, no surface-split FP/FN.
        assert r.entity_tp == 2
        assert r.entity_fp == 0
        assert r.entity_fn == 0
        # HAS_METRIC edge resolves because both endpoints are canonical.
        assert r.rel_tp == 1
        assert r.rel_fp == 0
        assert r.rel_fn == 0


class TestRenderSpecificity:
    """specificity appears in JSON and markdown output."""

    def test_render_json_includes_specificity(self):
        import json

        from eval_harness.score_extraction_run import Report, render_json

        payload = json.loads(render_json(Report(total_probes=1, matched_probes=1)))
        assert "specificity" in payload

    def test_render_markdown_includes_specificity(self):
        from eval_harness.score_extraction_run import Report, render_markdown

        md = render_markdown(Report(total_probes=1, matched_probes=1))
        assert "Specificity" in md or "specificity" in md


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


# ---------------------------------------------------------------------------
# Relations scored over aligned entity ids (plan v8, D8)
# ---------------------------------------------------------------------------


class TestGoldLoaderOptionalKeys:
    def _load(self, tmp_path, entity_extra, rel_extra):
        row = {
            "utterance": "I sit on the Kestrel Board",
            "tag": "t-opt",
            "expected_entities": [
                {"id": "kestrel-board", "type": "Organization", **entity_extra},
            ],
            "expected_relationships": [
                {
                    "source": "user",
                    "source_type": "User",
                    "predicate": "MEMBER_OF",
                    "target": "kestrel-board",
                    "target_type": "Organization",
                    **rel_extra,
                }
            ],
        }
        path = tmp_path / "gold.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return iter_gold_probes(path)[0]

    def test_absent_keys_take_their_defaults(self, tmp_path):
        probe = self._load(tmp_path, {}, {})
        assert probe.entities[0].alternatives == ()
        assert probe.relationships[0].predicate_alternatives == ()
        assert probe.relationships[0].beyond_prompt is False

    def test_alternatives_are_canonical_ids(self, tmp_path):
        probe = self._load(
            tmp_path, {"alternatives": ["The Kestrel Committee", "kestrel-group"]}, {}
        )
        # Alternatives take the id rails, so the leading article goes (N1 rail).
        assert probe.entities[0].alternatives == ("kestrel-committee", "kestrel-group")

    def test_predicate_alternatives_and_beyond_prompt(self, tmp_path):
        probe = self._load(
            tmp_path, {}, {"predicate_alternatives": ["WORKS_AT"], "beyond_prompt": True}
        )
        assert probe.relationships[0].predicate_alternatives == ("WORKS_AT",)
        assert probe.relationships[0].beyond_prompt is True


def _edge(source, predicate, target):
    return {"source": source, "target": target, "predicate": predicate, "properties": {}}


_ACME = GoldProbe(
    tag="t-acme",
    utterance="I work at Acme Corp",
    entities=(GoldEntity("user", "User"), GoldEntity("acme-corp", "Organization")),
    relationships=(GoldRel("user", "User", "WORKS_AT", "acme-corp", "Organization"),),
)


def _acme_index(produced_id, predicate="WORKS_AT"):
    entities = [GoldEntity("user", "User"), GoldEntity(produced_id, "Organization")]
    return {
        _ACME.utterance: _produced(
            _ACME.utterance,
            entities,
            [_edge("user", predicate, produced_id)],
            {e.id: e.type for e in entities},
        )
    }


_TYPO = GoldProbe(
    tag="t-typo",
    utterance="I work on the Kestrel board",
    entities=(GoldEntity("user", "User"), GoldEntity("kestrel-board", "Project")),
    relationships=(GoldRel("user", "User", "WORKS_ON", "kestrel-board", "Project"),),
)


def _typo_index():
    entities = [GoldEntity("user", "User"), GoldEntity("kestral-board", "Project")]
    return {
        _TYPO.utterance: _produced(
            _TYPO.utterance,
            entities,
            [_edge("user", "WORKS_ON", "kestral-board")],
            {e.id: e.type for e in entities},
        )
    }


class TestAlignedScoring:
    def test_an_n1_id_is_found_aligned_and_missed_exact(self):
        r = score_run([_ACME], _acme_index("the-acme-corp"))
        assert (r.rel_tp, r.rel_fp, r.rel_fn) == (0, 1, 1)
        assert (r.aligned_rel_tp, r.aligned_rel_fp, r.aligned_rel_fn) == (1, 0, 0)
        assert (r.aligned_rel_recall, r.aligned_rel_precision, r.aligned_rel_f1) == (1.0, 1.0, 1.0)
        probe = r.per_probe[0]
        assert probe["rel_fns"] == [["user", "WORKS_AT", "acme-corp"]]
        assert probe["aligned_rel_fns"] == []
        assert probe["aligned_rel_fps"] == []
        assert probe["aligned_pairs"] == [
            ["user", "user", "exact"],
            ["acme-corp", "the-acme-corp", "n1"],
        ]

    def test_kind_counts_cover_every_kind(self):
        from eval_harness.entity_alignment import ALIGNMENT_KINDS

        r = score_run([_ACME], _acme_index("the-acme-corp"))
        assert list(r.alignment_kinds) == list(ALIGNMENT_KINDS)
        assert r.alignment_kinds["exact"] == 1
        assert r.alignment_kinds["n1"] == 1
        assert sum(r.alignment_kinds.values()) == 2

    def test_id_agreement_is_the_rail_pairs_over_every_counted_pair(self):
        r = score_run([_ACME], _acme_index("the-acme-corp"))
        assert (r.id_rail_pairs, r.id_aligned_pairs) == (1, 2)
        assert r.id_agreement == 0.5
        assert score_run([_ACME], _acme_index("acme-corp")).id_agreement == 1.0

    def test_a_dated_event_id_agrees_as_an_exact_one(self):
        probe = GoldProbe(
            tag="t-event",
            utterance="The quill review is on 14 March",
            entities=(
                GoldEntity("user", "User"),
                GoldEntity("quill-review", "Event"),
                GoldEntity("2026-03-14", "Date"),
            ),
            relationships=(
                GoldRel("user", "User", "EXPERIENCED", "quill-review", "Event"),
                GoldRel("quill-review", "Event", "OCCURRED_ON", "2026-03-14", "Date"),
            ),
        )

        def index(event_id):
            entities = [
                GoldEntity("user", "User"),
                GoldEntity(event_id, "Event"),
                GoldEntity("2026-03-14", "Date"),
            ]
            edges = [
                _edge("user", "EXPERIENCED", event_id),
                _edge(event_id, "OCCURRED_ON", "2026-03-14"),
            ]
            return {
                probe.utterance: _produced(
                    probe.utterance, entities, edges, {e.id: e.type for e in entities}
                )
            }

        exact = score_run([probe], index("quill-review"))
        dated = score_run([probe], index("quill-review-2026-03-14"))
        assert dated.rel_tp == 0
        assert dated.aligned_rel_tp == exact.aligned_rel_tp == 2
        assert dated.id_agreement == exact.id_agreement == 1.0
        assert dated.alignment_kinds["event-date"] == 1

    def test_a_pending_pair_is_listed_and_scored_as_a_miss_until_accepted(self):
        r = score_run([_TYPO], _typo_index())
        assert (r.aligned_rel_tp, r.aligned_rel_fp, r.aligned_rel_fn) == (0, 1, 1)
        assert r.pending_pairs == [
            {
                "tag": "t-typo",
                "utterance": "I work on the Kestrel board",
                "gold_id": "kestrel-board",
                "gold_type": "Project",
                "produced_id": "kestral-board",
                "produced_type": "Project",
                "kind": "typo",
                "accepted": False,
            }
        ]

    def test_an_accepted_pending_pair_counts(self):
        r = score_run(
            [_TYPO], _typo_index(), accepted_pairs=[("t-typo", "kestrel-board", "kestral-board")]
        )
        assert (r.aligned_rel_tp, r.aligned_rel_fp, r.aligned_rel_fn) == (1, 0, 0)
        assert r.pending_pairs[0]["accepted"] is True
        assert r.alignment_kinds["typo"] == 1
        # The exact figures do not move.
        assert (r.rel_tp, r.rel_fp, r.rel_fn) == (0, 1, 1)

    def test_an_accepted_row_for_another_probe_does_not_count(self):
        r = score_run(
            [_TYPO], _typo_index(), accepted_pairs=[("t-other", "kestrel-board", "kestral-board")]
        )
        assert (r.aligned_rel_tp, r.aligned_rel_fp, r.aligned_rel_fn) == (0, 1, 1)
        assert r.pending_pairs[0]["accepted"] is False

    def test_a_gold_predicate_alternative_is_found(self):
        probe = GoldProbe(
            tag=_ACME.tag,
            utterance=_ACME.utterance,
            entities=_ACME.entities,
            relationships=(
                GoldRel(
                    "user",
                    "User",
                    "WORKS_AT",
                    "acme-corp",
                    "Organization",
                    predicate_alternatives=("MEMBER_OF",),
                ),
            ),
        )
        r = score_run([probe], _acme_index("acme-corp", predicate="MEMBER_OF"))
        assert r.rel_tp == 0
        assert r.aligned_rel_tp == 1

    def test_an_unmatched_probe_misses_every_gold_relation(self):
        r = score_run([_ACME], {})
        assert (r.aligned_rel_tp, r.aligned_rel_fp, r.aligned_rel_fn) == (0, 0, 1)
        assert r.per_probe[0]["aligned_rel_fns"] == [["user", "WORKS_AT", "acme-corp"]]
        assert r.per_probe[0]["aligned_pairs"] == []
        assert r.pending_pairs == []

    def test_an_empty_report_takes_the_vacuous_defaults(self):
        from eval_harness.score_extraction_run import Report

        r = Report()
        assert (r.aligned_rel_recall, r.aligned_rel_precision, r.aligned_rel_f1) == (1.0, 1.0, 1.0)
        assert r.id_agreement == 1.0
        assert r.alignment_kinds == {}
        assert r.pending_pairs == []


class TestTheHeadlineAndTheModelWrittenColumn:
    """Plan v15, T50 Acceptance 2: `[exact k1-4]`, and the model-written column beside it.

    The headline counts a relation found only through a pair of alignment kinds 1 to 4
    (exact, alternative, event-date, metric), the ids the graph stores as one node. The
    model-written column scores the same payload with the relation rails switched off.
    """

    def test_an_n1_pair_finds_nothing_on_the_headline(self):
        r = score_run([_ACME], _acme_index("the-acme-corp"))
        assert (r.headline_rel_tp, r.headline_rel_fp, r.headline_rel_fn) == (0, 1, 1)
        assert r.per_probe[0]["headline_rel_fns"] == [["user", "WORKS_AT", "acme-corp"]]
        assert r.per_probe[0]["headline_rel_fps"] == [["user", "WORKS_AT", "the-acme-corp"]]
        assert (r.aligned_rel_tp, r.aligned_rel_fn) == (1, 0)

    def test_an_exact_pair_finds_the_relation_on_the_headline(self):
        r = score_run([_ACME], _acme_index("acme-corp"))
        assert (r.headline_rel_tp, r.headline_rel_fp, r.headline_rel_fn) == (1, 0, 0)
        assert (r.headline_rel_recall, r.headline_rel_precision, r.headline_rel_f1) == (
            1.0,
            1.0,
            1.0,
        )

    def test_an_empty_report_takes_the_vacuous_headline_defaults(self):
        from eval_harness.score_extraction_run import Report

        r = Report()
        assert (r.headline_rel_recall, r.headline_rel_precision, r.headline_rel_f1) == (
            1.0,
            1.0,
            1.0,
        )

    @pytest.mark.parametrize("relation_rails", [True, False])
    def test_the_switch_reaches_the_rails(self, monkeypatch, relation_rails):
        import eval_harness.score_extraction_run as scorer

        seen = []
        real = scorer.rails.run_rails

        def spy(row, pack, **switches):
            seen.append(switches.get("relation_rails", True))
            return real(row, pack, **switches)

        monkeypatch.setattr(scorer.rails, "run_rails", spy)
        response = {"payload": {"entities": [], "relationships": []}}
        scorer.produced_from_response("I use Rust", response, relation_rails=relation_rails)
        assert seen == [relation_rails]

    def test_the_default_parse_runs_every_rail(self, monkeypatch):
        import eval_harness.score_extraction_run as scorer

        seen = []
        real = scorer.rails.run_rails

        def spy(row, pack, **switches):
            seen.append(switches.get("relation_rails", True))
            return real(row, pack, **switches)

        monkeypatch.setattr(scorer.rails, "run_rails", spy)
        scorer.parse_produced('{"entities": [], "relationships": []}')
        assert seen == [True]

    def test_the_three_counts_compare_the_columns(self):
        from eval_harness.score_extraction_run import relation_rails_counts

        def index(edges):
            entities = [
                GoldEntity("user", "User"),
                GoldEntity("acme-corp", "Organization"),
                GoldEntity("rust", "Technology"),
            ]
            return {
                _ACME.utterance: _produced(
                    _ACME.utterance,
                    entities,
                    [_edge(*edge) for edge in edges],
                    {e.id: e.type for e in entities},
                )
            }

        # A relation rail added the gold edge and removed the model's USES edge.
        headline = index([("user", "WORKS_AT", "acme-corp")])
        model_written = index([("user", "USES", "rust")])

        assert relation_rails_counts([_ACME], headline, model_written) == {
            "relation_rails_added": 1,
            "relation_rails_removed": 1,
            "found_only_by_relation_rails": 1,
        }
        assert relation_rails_counts([_ACME], headline, headline) == {
            "relation_rails_added": 0,
            "relation_rails_removed": 0,
            "found_only_by_relation_rails": 0,
        }


class TestEverySummaryPrintsBothColumns:
    """Plan v15, T50 Acceptance 2: each summary prints both columns and the three counts.

    The headline index finds both gold relations; the model-written index lacks the
    `t-acme` edge, as if a relation rail had added it.
    """

    _SEED, _B, _CONFIDENCE = 0, 2000, 0.95

    @staticmethod
    def _indexes():
        acme = [GoldEntity("user", "User"), GoldEntity("acme-corp", "Organization")]
        board = [GoldEntity("user", "User"), GoldEntity("kestrel-board", "Project")]

        def produced(probe, entities, edges):
            return _produced(probe.utterance, entities, edges, {e.id: e.type for e in entities})

        typo = produced(_TYPO, board, [_edge("user", "WORKS_ON", "kestrel-board")])
        headline = {
            _ACME.utterance: produced(_ACME, acme, [_edge("user", "WORKS_AT", "acme-corp")]),
            _TYPO.utterance: typo,
        }
        model_written = {_ACME.utterance: produced(_ACME, acme, []), _TYPO.utterance: typo}
        return headline, model_written

    def _summary(self):
        from eval_harness.score_extraction_run import column_summary

        headline, model_written = self._indexes()
        return column_summary(
            [_ACME, _TYPO],
            headline,
            model_written,
            seed=self._SEED,
            b=self._B,
            confidence=self._CONFIDENCE,
        )

    def _ci(self, counts, kind):
        from scripts.model_bench.probes.extraction import cluster_bootstrap_ratio_ci

        bounds = cluster_bootstrap_ratio_ci(
            counts, n_replicates=self._B, seed=self._SEED, confidence=self._CONFIDENCE, kind=kind
        )
        return None if bounds is None else list(bounds)

    def _diff_ci(self, first, second, kind):
        from scripts.model_bench.probes.extraction import cluster_bootstrap_paired_difference_ci

        bounds = cluster_bootstrap_paired_difference_ci(
            first,
            second,
            n_replicates=self._B,
            seed=self._SEED,
            confidence=self._CONFIDENCE,
            kind=kind,
        )
        return None if bounds is None else list(bounds)

    def _column(self, counts, recall, precision, f1):
        tp = sum(c[0] for c in counts.values())
        figures = {
            "recall": {"value": recall, "tp": tp, "n": 2, "ci95": self._ci(counts, "recall")},
            "precision": {
                "value": precision,
                "tp": tp,
                "n": tp,
                "ci95": self._ci(counts, "precision"),
            },
            "f1": {"value": f1, "ci95": self._ci(counts, "f1")},
        }
        # Every pair here is exact, so the aligned figures equal the headline's.
        return {**figures, "aligned": figures}

    def test_the_summary_holds_both_columns_the_rails_effect_and_the_counts(self):
        headline = {"t-acme": (1, 0, 0), "t-typo": (1, 0, 0)}
        model_written = {"t-acme": (0, 0, 1), "t-typo": (1, 0, 0)}

        assert self._summary() == {
            "columns": {
                "exact k1-4": self._column(headline, 1.0, 1.0, 1.0),
                "model-written": self._column(model_written, 0.5, 1.0, 2 / 3),
            },
            "rails_effect": {
                "recall_diff": {
                    "value": 0.5,
                    "ci95": self._diff_ci(model_written, headline, "recall"),
                },
                "precision_diff": {
                    "value": 0.0,
                    "ci95": self._diff_ci(model_written, headline, "precision"),
                },
                "f1_diff": {
                    "value": 1 - 2 / 3,
                    "ci95": self._diff_ci(model_written, headline, "f1"),
                },
            },
            "relation_rails_added": 1,
            "relation_rails_removed": 0,
            "found_only_by_relation_rails": 1,
        }

    def test_render_json_carries_the_columns_after_the_identity(self):
        columns = self._summary()
        identity = {"scorer_sha": "s", "gold_version": "g"}
        out = json.loads(render_json(score_run([_ACME, _TYPO], {}), identity, columns))
        assert list(out)[:2] == ["scorer_sha", "gold_version"]
        assert {key: out[key] for key in columns} == json.loads(json.dumps(columns))

    def test_render_markdown_prints_both_columns_the_rails_effect_and_the_counts(self):
        columns = self._summary()
        md = render_markdown(score_run([_ACME, _TYPO], {}), None, columns)

        def ratio(figure):
            return f"{figure['value']:.3f} ({figure['tp']}/{figure['n']}) {ci(figure['ci95'])}"

        def ci(bounds, sign=""):
            return f"ci95 [{bounds[0]:{sign}.3f}, {bounds[1]:{sign}.3f}]"

        def row(name):
            c = columns["columns"][name]
            a = c["aligned"]
            return (
                f"| {name} | {ratio(c['recall'])} | {ratio(c['precision'])} | "
                f"{c['f1']['value']:.3f} {ci(c['f1']['ci95'])} | {ratio(a['recall'])} | "
                f"{ratio(a['precision'])} | {a['f1']['value']:.3f} {ci(a['f1']['ci95'])} |"
            )

        effect = columns["rails_effect"]
        assert (
            "| Column | Recall | Precision | F1 | Aligned recall | Aligned precision | "
            "Aligned F1 |"
        ) in md
        assert row("exact k1-4") in md
        assert row("model-written") in md
        assert (
            "Rails effect (exact k1-4 minus model-written): "
            f"recall {effect['recall_diff']['value']:+.3f} "
            f"{ci(effect['recall_diff']['ci95'], '+')}; "
            f"precision {effect['precision_diff']['value']:+.3f} "
            f"{ci(effect['precision_diff']['ci95'], '+')}; "
            f"f1 {effect['f1_diff']['value']:+.3f} {ci(effect['f1_diff']['ci95'], '+')}"
        ) in md
        assert (
            "relation_rails_added=1 relation_rails_removed=0 found_only_by_relation_rails=1"
        ) in md

    def test_the_cli_writes_both_columns_into_both_reports(self, tmp_path):
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({"utterance": "I use Rust", "tag": "t1"}) + "\n")
        debug = tmp_path / "debug.jsonl"
        debug.write_text("", encoding="utf-8")
        md, js = tmp_path / "report.md", tmp_path / "report.json"
        argv = ["--gold", str(gold), "--debug-jsonl", str(debug), "--output", str(md)]
        assert main([*argv, "--json-output", str(js)]) == 0
        payload = json.loads(js.read_text(encoding="utf-8"))
        assert list(payload["columns"]) == ["exact k1-4", "model-written"]
        assert {
            key: payload[key] for key in ("relation_rails_added", "relation_rails_removed")
        } == {
            "relation_rails_added": 0,
            "relation_rails_removed": 0,
        }
        text = md.read_text(encoding="utf-8")
        assert "| exact k1-4 | " in text
        assert "| model-written | " in text

    def test_no_columns_prints_no_column_section(self):
        md = render_markdown(score_run([_ACME], {}))
        assert "| Column |" not in md
        assert "columns" not in json.loads(render_json(score_run([_ACME], {})))
