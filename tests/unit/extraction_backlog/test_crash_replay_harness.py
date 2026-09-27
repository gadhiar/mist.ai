"""The eval-Neo4j crash-replay harness keeps its entities apart under curation dedup.

`tests/integration/extraction_backlog/test_crash_replay_canonical.py` runs the real
curation pipeline on a real Neo4j, whose Tier-3 resolver merges two entities of
compatible types when Neo4j's `vector.similarity.cosine` score reaches
`SIMILARITY_THRESHOLD`. The unit tier's fake executor never evaluates that
Cypher, so a harness whose embeddings collide passes here and fails only on the
eval instance. The first eval run (2026-09-27) failed exactly that way: with
`FakeEmbeddingGenerator`, "t2acrash-zig" and "t2acrash-rust" scored 0.9203, zig
merged into rust, and no zig node was written. These tests check the harness's
actual embedding provider against the thresholds, in Neo4j's score scale.
"""

from __future__ import annotations

import itertools

from backend.knowledge.config import ExtractionConfig
from backend.knowledge.curation.deduplication import SIMILARITY_THRESHOLD
from tests.integration.extraction_backlog.harness import (
    TURNS,
    entity_names,
    make_embeddings,
    neo4j_cosine_score,
    raw_cosine,
)
from tests.mocks.embeddings import FakeEmbeddingGenerator


def test_no_two_harness_entities_merge_under_neo4j_cosine_dedup() -> None:
    embed = make_embeddings().generate_embedding
    names = entity_names()
    assert len(names) >= 3  # dev, rust, zig: guards against a vacuous pass
    for a, b in itertools.combinations(names, 2):
        score = neo4j_cosine_score(embed(a), embed(b))
        assert score < SIMILARITY_THRESHOLD, (
            f"{a!r} vs {b!r}: Neo4j cosine score {score:.4f} >= {SIMILARITY_THRESHOLD}; "
            "curation would merge them on the eval instance"
        )


def test_no_two_harness_utterances_trip_the_input_dedup_gate() -> None:
    embed = make_embeddings().generate_embedding
    threshold = ExtractionConfig().dedup_similarity_threshold
    for (_, _, _, a), (_, _, _, b) in itertools.combinations(TURNS, 2):
        assert raw_cosine(embed(a), embed(b)) < threshold


def test_harness_embeddings_are_deterministic_and_the_right_dimension() -> None:
    first = make_embeddings().generate_embedding("t2acrash-zig")
    second = make_embeddings().generate_embedding("t2acrash-zig")
    assert first == second
    assert len(first) == 384


def test_the_old_fake_embeddings_would_merge_zig_into_rust() -> None:
    """Pins the root cause of the 2026-09-27 failure, so a revert is recognisable."""
    fake = FakeEmbeddingGenerator()
    score = neo4j_cosine_score(
        fake.generate_embedding("t2acrash-zig"), fake.generate_embedding("t2acrash-rust")
    )
    assert score >= SIMILARITY_THRESHOLD
