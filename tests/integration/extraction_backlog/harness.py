"""Fixture data and embeddings for the eval-Neo4j crash-replay tests.

Kept apart from `test_crash_replay_canonical.py` so the unit tier can import it
without that module's import-time probe of the eval Neo4j endpoint, and check
the one property the eval run depends on and the unit tier's fake executor
cannot evaluate: that the test's own entities do NOT collide under curation's
embedding dedup.

Why that needs checking. Curation's Tier-3 resolver merges an incoming entity
into an existing one of a compatible type when Neo4j's
`vector.similarity.cosine(stored, probe) >= SIMILARITY_THRESHOLD`
(`backend/knowledge/curation/deduplication.py`, `_find_existing`). Neo4j's
cosine function returns `(1 + cos) / 2`, not the raw cosine, so the 0.92
threshold is a raw cosine of 0.84. `tests.mocks.embeddings.FakeEmbeddingGenerator`
fills 32 dimensions with SHA-256 bytes in [0, 1]: every vector is all-positive,
so unrelated texts sit near raw cosine 0.75 and some pairs clear 0.84. Its
docstring says it "should not be used for similarity threshold testing". The
first eval run (2026-09-27) used it: "t2acrash-zig" and "t2acrash-rust" scored
0.9203, turn 1's `zig` merged into `rust`, and the run wrote no `zig` node at all.

Tier 3 also applies a name veto (`backend/knowledge/curation/name_veto.py`): a
candidate above the threshold is still rejected when the two names carry
different numeric tokens or different month tokens. It does not keep this
harness's entities apart: every name starts with `t2acrash-`, so each carries
the same numeric token {2} and no month token, and the veto passes every pair.
The score check in `tests/unit/extraction_backlog/test_crash_replay_harness.py`
remains the guard.
"""

from __future__ import annotations

import hashlib
import math

PREFIX = "t2acrash-"
EMBEDDING_DIMENSION = 384
SESSION = f"{PREFIX}session"
TURNS = [
    (f"{PREFIX}evt-0", 0, "2026-09-01T10:00:00+00:00", "I really use rust"),
    (f"{PREFIX}evt-1", 1, "2026-09-01T10:05:00+00:00", "I really use zig"),
]


def payload(req):
    """Turn 0 -> dev USES rust; turn 1 -> dev USES zig. All ids prefixed."""
    word = req.utterance.split()[-1]
    dev = {"id": f"{PREFIX}dev", "type": "Person", "name": f"{PREFIX}Dev"}
    tech = {"id": f"{PREFIX}{word}", "type": "Technology", "name": f"{PREFIX}{word}"}
    rel = {
        "source": dev["id"],
        "target": tech["id"],
        "type": "USES",
        "properties": {"confidence": 0.9},
    }
    return [dev, tech], [rel]


def entity_names() -> list[str]:
    """Every display name the payload writes, for the collision check."""
    names: set[str] = set()
    for _, _, _, utterance in TURNS:
        entities, _ = payload(type("Req", (), {"utterance": utterance})())
        names.update(e["name"] for e in entities)
    return sorted(names)


class SignedHashEmbeddingGenerator:
    """Deterministic embeddings that behave like a real model's for unrelated text.

    Each dimension is a signed value in [-1, 1] drawn from a SHA-256 stream over
    the text, so vectors of different texts are near-orthogonal (raw cosine
    around 0, spread about 1/sqrt(384)), far below any dedup threshold, while
    the same text always maps to the same vector. Satisfies the
    `EmbeddingProvider` protocol used by `build_curation_pipeline`.
    """

    def __init__(self, *, dimension: int = EMBEDDING_DIMENSION) -> None:
        self._dimension = dimension

    def generate_embedding(self, text: str) -> list[float]:
        """Return a deterministic, zero-mean vector for `text`."""
        values: list[float] = []
        counter = 0
        while len(values) < self._dimension:
            block = hashlib.sha256(f"{counter}\x1f{text}".encode()).digest()
            values.extend(b / 127.5 - 1.0 for b in block)
            counter += 1
        return values[: self._dimension]

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts."""
        return [self.generate_embedding(t) for t in texts]


def make_embeddings() -> SignedHashEmbeddingGenerator:
    """The embedding provider the eval-Neo4j crash-replay tests curate with.

    One factory, so the unit-tier collision check tests exactly what the eval
    run uses (`tests/unit/extraction_backlog/test_crash_replay_harness.py`).
    """
    return SignedHashEmbeddingGenerator()


def raw_cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity of two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def neo4j_cosine_score(a: list[float], b: list[float]) -> float:
    """What Neo4j's `vector.similarity.cosine` returns: `(1 + cos) / 2`."""
    return (1.0 + raw_cosine(a, b)) / 2.0
