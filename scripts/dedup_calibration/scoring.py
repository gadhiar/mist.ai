"""Embed labelled pairs the way the graph does and score them.

Which text is embedded, and where that was established:

- Probe side (`a`): the entity's display name. `EntityDeduplicator._find_existing`
  embeds `display_name` for the tier-3 cosine probe
  (`backend/knowledge/curation/deduplication.py:149-154`).
- Stored side, `extracted_vs_extracted`: the display name. The extraction
  write path stores `generate_embedding(display_name)` in
  `CurationGraphWriter` (`backend/knowledge/curation/graph_writer.py:384-388`),
  with `display_name = entity.get("name", entity_id)` at `graph_writer.py:375`.
- Stored side, `extracted_vs_seed`: `embedding_text_for(display_name,
  description, node_id)`, the builder the seed embedding backfill uses
  (`backend/knowledge/admin.py:319` and `:387`, defined in
  `backend/knowledge/embeddings/embedding_text.py:39-72`). With no
  description that is exactly the display name.

Embedding goes through the `EmbeddingProvider` interface
(`backend/interfaces.py:29-33`), one `generate_embedding` call per distinct
text, as the graph does.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.interfaces import EmbeddingProvider
from backend.knowledge.embeddings.embedding_text import embedding_text_for

from .dataset import SLICE_EXTRACTED_VS_SEED, LabelledPair
from .metrics import cosine_similarity, neo4j_score_from_cosine


@dataclass(frozen=True, slots=True)
class ScoredPair:
    """A labelled pair with its raw cosine and Neo4j score."""

    pair: LabelledPair
    cosine: float
    neo4j_score: float


def probe_text(pair: LabelledPair) -> str:
    """Text the dedup probe embeds for side `a`."""
    return pair.a


def stored_text(pair: LabelledPair) -> str:
    """Text embedded for the stored node on side `b`."""
    if pair.slice == SLICE_EXTRACTED_VS_SEED:
        return embedding_text_for(pair.b, pair.b_description, pair.b)
    return pair.b


def score_pairs(pairs: list[LabelledPair], embedder: EmbeddingProvider) -> list[ScoredPair]:
    """Embed both sides of every pair and compute cosine and Neo4j score.

    Raises:
        ValueError: An embedding is a zero vector or the dimensions disagree.
    """
    cache: dict[str, list[float]] = {}

    def embed(text: str) -> list[float]:
        if text not in cache:
            cache[text] = embedder.generate_embedding(text)
        return cache[text]

    scored: list[ScoredPair] = []
    for pair in pairs:
        cosine = cosine_similarity(embed(probe_text(pair)), embed(stored_text(pair)))
        scored.append(
            ScoredPair(pair=pair, cosine=cosine, neo4j_score=neo4j_score_from_cosine(cosine))
        )
    return scored
