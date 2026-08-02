"""Single authority for the rebuild-determinism version stamps.

MIST stamps every entity, fact edge, and provenance edge with an
(`ontology_version`, `extraction_version`, `model_hash`) triple. The stamps are
descriptive -- no pipeline code branches on them; the active ontology is chosen
by Python import, and `canonical_serialize` deliberately excludes the triple so
the determinism proof reads "same log + same epoch => same facts". One mechanism
makes their CONSISTENCY load-bearing anyway: `extraction_cache.cache_key` hashes
`event_id|ontology_version|extraction_version|model_hash`, so two sites
disagreeing about a stamp is not a mislabel but a hard cache miss that makes a
deterministic rebuild impossible.

The split this module enforces:

- A version that describes CODE lives in code and is DERIVED, never restated.
  `ONTOLOGY_VERSION` reads the active ontology object itself, so the stamp
  cannot disagree with the ontology actually in use.
- `EXTRACTION_VERSION` describes the extraction prompt, which has no derivable
  source, so it is the one literal here -- mechanically paired to the prompt
  content by `TestExtractionVersionDriftGuard`.
- `model_hash` names a DEPLOYED MODEL FILE rather than code, so it stays on
  `KnowledgeConfig` and remains env-configurable via `MIST_MODEL_HASH`.

`EXTRACTION_VERSION` would naturally live beside the prompt it describes in
`backend/knowledge/extraction/prompts.py`, but `backend/knowledge/config.py`
cannot import it from there: the `backend.knowledge.extraction` package
`__init__` eagerly imports `ontology_extractor`, which imports
`KnowledgeConfig`, so the import cycles whenever `config` is the entry module.
This module is a leaf -- it imports the ontology and nothing else -- so every
consumer can read it regardless of import order.
"""

from __future__ import annotations

from backend.knowledge.ontologies.v1_0_0 import ONTOLOGY_V1_0_0

# Derived, never restated. `TestOntologyVersionHasOneAuthority` fails if any
# module under `backend/` reintroduces a literal ontology-version stamp.
ONTOLOGY_VERSION: str = ONTOLOGY_V1_0_0.version

# Bump whenever EXTRACTION_SYSTEM_PROMPT / EXTRACTION_USER_TEMPLATE or the
# ontology contract they encode changes, then re-pin PINNED_SHA256 in
# tests/unit/knowledge/extraction/test_prompts.py.
#
# ALSO on a bump: the R1.4.5 golden log's authored extraction cache goes cold,
# because `extraction_cache.cache_key` hashes this value. It is not a migration
# -- the cache is regenerated, not converted -- and it is not a manual step
# either: `scripts/golden_log/generate.py` derives the triple from here and
# materializes a fresh cache per test run, so nothing needs re-authoring. The
# checked-in artifact holds only payloads, never keys. Regenerate it only when
# the SCHEDULE or the gold corpus changes:
#     python -m scripts.golden_log.generate            # rewrite
#     python -m scripts.golden_log.generate --check    # verify current
# 2026-06-12-r1: deep-review prompt fix (direction rules for
# USES/DEPENDS_ON/WORKS_WITH source sets, undirected WORKS_WITH).
# 2026-06-12-r2: emit assertion_kind signal (assert|cease|retract) per
# relationship (C3 spec 6.2 -- cessation/retraction reconciliation).
# 2026-06-12-r3: RECOMMENDS / HAS_HABIT predicates (ontology v1.3.0) plus
# date-entity discrimination (Rules 16-17, Examples 23-24).
# 2026-06-12-r4: precision rules -- HAS_HABIT recurrence-cadence tightening
# (Rule 17) + no prepositional over-extraction (Rule 18).
# 2026-06-14-r5: MECE taxonomy -- Abstraction fallback type, abstract-type
# tests block (Rules 19-20), third-party facts rule, retire Topic/Milestone
# from entity list (22 -> 21), retype Example 9 Milestone -> Event, add
# Examples 25-26 (third-party shape, Abstraction fallback).
EXTRACTION_VERSION: str = "2026-06-14-r5"


def compose_model_hash(config: object) -> str:
    """Return the `model_hash` stamp: the LLM hash folded with the embedding model.

    **Both the LLM and the embedding model belong in this stamp**, and the second
    one is not obvious. Task R1.1e established why: the deterministic identity
    resolver compares STORED embedding vectors by cosine, so a different embedding
    model produces different vectors, and a near-threshold merge (~0.92) can flip.
    The same log then yields a different graph. Folding the embedding model in
    makes an embedding swap a NEW EPOCH rather than a silent cross-epoch
    determinism break -- pinned by
    `test_rebuild_stamps_model_hash_includes_embedding_model_identity`.

    This exists as a shared function because the two sites that need the value
    disagreed. Review finding L4 (2026-08-02): `build_curation_pipeline` composed
    it while `EventStore.ensure_initial_epoch` wrote the bare `config.model_hash`,
    so the epoch triple and the writer triple differed on 2 of 3 fields.

    That is not cosmetic. `extraction_cache.cache_key` hashes
    `event_id|ontology_version|extraction_version|model_hash` and `LogRegenerator`
    builds its lookup from the EPOCH row, so a disagreement is a total, permanent
    `ColdCacheError` on every turn of every rebuild -- the failure mode this
    module's docstring names.

    The collapse deliberately goes toward the COMPOSED form, not the bare one: the
    composed value is the determinism-correct one, and dropping it to make the two
    sides agree would have deleted an R1.1e guard. Callers must not re-compose
    this inline.

    Args:
        config: A `KnowledgeConfig`. Typed loosely to keep this module a leaf --
            importing `KnowledgeConfig` here would reintroduce the cycle the
            module docstring describes.

    Returns:
        The composite stamp, e.g. `"gemma-...-v1|emb:all-MiniLM-L6-v2"`.
    """
    return f"{config.model_hash}|emb:{config.embedding.model_name}"  # type: ignore[attr-defined]
