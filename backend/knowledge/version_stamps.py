"""Single authority for the rebuild-determinism version stamps.

MIST stamps every entity, fact edge, and provenance edge with an
(`ontology_version`, `extraction_version`, `model_hash`) triple. The stamps are
descriptive -- no pipeline code branches on them; the active ontology is chosen
by Python import, and `canonical_serialize` deliberately excludes the triple so
the determinism proof reads "same log + same epoch => same facts". One mechanism
makes `extraction_version` and `model_hash`'s CONSISTENCY load-bearing anyway:
`extraction_cache.cache_key` hashes `event_id|extraction_version|model_hash`
(verified via `grep -n 'raw = "|".join' backend/knowledge/extraction_cache.py`),
so two sites disagreeing about EITHER of those two stamps is not a mislabel but
a hard cache miss that makes a deterministic rebuild impossible. `ontology_version`
is the exception, as of `extraction-cache-phase-1` spec D3: it is still stamped
on every edge for audit, but deliberately excluded from the cache key, so two
sites disagreeing about it now IS a mislabel a rebuild will not catch on its
own -- not a cache miss.

The split this module enforces:

- A version that describes CODE lives in code and is DERIVED, never restated.
  `ONTOLOGY_VERSION` reads the active ontology object itself, so the stamp
  cannot disagree with the ontology actually in use.
- `EXTRACTION_VERSION` describes the extraction prompt and the code that shapes
  extraction output, neither of which has a derivable version, so it is a
  literal. It moves for a prompt change -- mechanically paired to the prompt
  content by `TestExtractionVersionDriftGuard` -- and also for a code change
  meant to re-extract history at the next cutover. `CODE_REVISIONS`, below in
  this module, records every such code change with the version it shipped
  under; a code change for new turns only adds an entry there and leaves the
  version alone. The only other version literals here are the history in
  `CODE_REVISIONS`, which records past values and is never read as the
  current one.
- `model_hash` names a DEPLOYED MODEL FILE rather than code, so it stays on
  `KnowledgeConfig` and remains env-configurable via `MIST_MODEL_HASH`.

`EXTRACTION_VERSION` would naturally live beside the prompt it describes in
`backend/knowledge/extraction/prompts.py`, but `backend/knowledge/config.py`
cannot import it from there: the `backend.knowledge.extraction` package
`__init__` eagerly imports `ontology_extractor`, which imports
`KnowledgeConfig`, so the import cycles whenever `config` is the entry module.
This module is a leaf -- its one backend import is the ontology, beside the
standard library's `dataclasses` -- so every consumer can read it regardless
of import order.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.knowledge.ontologies.v1_0_0 import ONTOLOGY_V1_0_0

# Derived, never restated. `TestNoBackendModuleRestatesAStamp` (in
# tests/unit/knowledge/test_version_stamps.py) fails if any module under
# `backend/` reintroduces a literal ontology-version stamp. It catches the
# binding shapes listed on `_named_bindings` there; a literal that needs
# constant propagation to see is out of its reach.
ONTOLOGY_VERSION: str = ONTOLOGY_V1_0_0.version

# Bump whenever EXTRACTION_SYSTEM_PROMPT / EXTRACTION_USER_TEMPLATE or the
# ontology contract they encode changes, then re-pin PINNED_SHA256 in
# tests/unit/knowledge/extraction/test_prompts.py. Bump it too for a change to
# the code that shapes extraction output when history should be re-extracted
# under it, and record that change in CODE_REVISIONS below.
#
# ALSO on a bump: the R1.4.6 hydration artifact goes stale. Hydration drives an
# authored corpus through the LIVE path into the isolated `dev` stack
# (docker-compose.dev-hydration.yml) so the dev graph carries reconciliation-
# written edges instead of the two-property seed shape. It costs 87 LLM turns,
# so it is a RARE REGENERATION on exactly three triggers -- an ontology bump,
# a prompt bump (this constant), or a corpus change -- and the rest of the time
# developers restore the snapshot:
#     python -m scripts.hydration.snapshot list      # which artifacts are fresh
#     python -m scripts.hydration.snapshot restore --artifact <dir>
#
# Nothing needs doing here on a bump. `scripts/hydration/manifest.py` READS
# this value rather than restating it, so the next `restore` recomputes the
# triple, sees the drift, and REFUSES the artifact by name -- the same
# derived-not-restated discipline that keeps the extraction cache honest. The
# refusal is the notification; regenerate when it fires.
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
# 2026-11-06-r6: Stage 2 list pass (MIS-194), a code change with no prompt
# change -- a structured turn is split into units that keep their parent
# context and extracted four units a call, each later call seeing up to eight
# earlier units; a batch cut short at the output limit is split and retried.
# Bumped so history is re-extracted under it at the next cutover.
# 2026-11-10-r8: the stored id is built from the payload's `id` field through
# shared id rails, the prompt states its id, list and predicate conventions,
# and medium is the deployed effort, each LLM call logging its token counts.
# r7 named the withdrawn list frame path's builds, so this skips to r8 to keep
# the stamps unambiguous. Re-extracts history at the next cutover.
EXTRACTION_VERSION: str = "2026-11-10-r8"


@dataclass(frozen=True, slots=True)
class CodeRevision:
    """One deliberate change to the code that shapes extraction output.

    Attributes:
        extraction_version: The `EXTRACTION_VERSION` the change shipped under.
        code_identity: `backend.knowledge.extraction.run_identity.code_identity`
            of the service as it ships, after the change.
        date: The day the entry was added, `YYYY-MM-DD`.
        reason: What changed and why.
    """

    extraction_version: str
    code_identity: str
    date: str
    reason: str


# Every code identity the extraction service has shipped, oldest first. The
# drift guard in tests/unit/knowledge/extraction/test_run_identity.py fails
# until the newest entry names the current EXTRACTION_VERSION and the current
# code identity, so each edit to a prompt, a per-stage token limit, the
# ontology version or a module in `RUN_IDENTITY_CODE` lands with a dated line
# here. Add the line under the same `extraction_version` when the change is for
# new turns only; bump EXTRACTION_VERSION first when it should re-extract
# history at the next cutover.
#
# Kept here rather than beside the run identity code because each entry binds
# an extraction-version literal, which `TestNoBackendModuleRestatesAStamp`
# allows in this module alone.
CODE_REVISIONS: tuple[CodeRevision, ...] = (
    CodeRevision(
        extraction_version="2026-06-14-r5",
        code_identity="ci1-b030743cefc8b571347edb718211297adce19e0beae2490d2a639ea7df9522e5",
        date="2026-11-06",
        reason=(
            "First recorded code identity (MIS-188): the r5 prompt and the service code "
            "as they stood when the run identity was added."
        ),
    ),
    CodeRevision(
        extraction_version="2026-11-06-r6",
        code_identity="ci1-4894d508798bcaa72ff1d24421c85db43e14b640ea8f7d3de2433ce15a281505",
        date="2026-11-06",
        reason=(
            "Stage 2 list pass (MIS-194): structure_splitter joins the covered code, and "
            "a structured turn is extracted in batches of LIST_BATCH_MAX_UNITS = 4 units "
            "with LIST_CONTEXT_MAX_UNITS = 8 earlier units as context; a batch cut short "
            "at EXTRACT_MAX_TOKENS is split in two and retried. Re-extracts history."
        ),
    ),
    CodeRevision(
        extraction_version="2026-11-06-r6",
        code_identity="ci1-29646c2d855e3c09f3147ea73eaf8ca32c9f43c949ccf1d8529aea8f2152d328",
        date="2026-11-09",
        reason=(
            "Serving identity (MIS-196): each stage's request is built by its own method, "
            "which ExtractionEngine.request_profile also calls, and run() stamps the "
            "serving identity it is given. Every request is sent exactly as before. New "
            "turns only."
        ),
    ),
    CodeRevision(
        extraction_version="2026-11-06-r6",
        code_identity="ci1-6ed7f68d0599961f548ea162e1282b43367786e813638813f2343539878ab8c6",
        date="2026-11-09",
        reason=(
            "Stage caps by reasoning effort (T32): STAGE_CAPS gives each stage the cap of "
            "the effort the adapter sends it, low (96, 2048, 400) and medium and high "
            "(1024, 32768, 400), with derive sent at low effort under every setting; "
            "_invoke logs an `extraction length stop:` line with counts and a repetition "
            "flag; LlamaServerProvider takes the client timeout. At low effort every "
            "request is sent exactly as before. New turns only."
        ),
    ),
    CodeRevision(
        extraction_version="2026-11-10-r8",
        code_identity="ci1-c1e2f66b661adb398251bbe32d764f7513c5b85d867b09fdf08305c4cdaf1f56",
        date="2026-11-10",
        reason=(
            "v11 (r8): the stored id comes from the `id` field through shared id rails, the "
            "prompt states its conventions, and medium is the deployed effort. Every later "
            "change under r8 updates this entry's code_identity, since only the final "
            "identity ships. So far: _invoke logs one `extraction call:` line per LLM call "
            "with its cap, token counts and finish reason; PROMPT_RESERVE_TOKENS is 14336, "
            "so medium needs a context of 47,104 and low still fits 16,384; prompt group 1 "
            "(identity) states N1 in Rule 2, N2, N4, N14, N15 and N16, the Event rule and "
            "the Metric rule as Rules 21 to 27, corrects Examples 6, 7, 10, 12, 15, 20 and "
            "24, and adds Examples 27 to 29; prompt group 2 (lists) states how a structured "
            "turn reads, the list rules and itinerary items, N3, N5, N7 and N13 as Rules 28 "
            "to 33, adds the stated place to Example 29, and adds Example 30; prompt group 3 "
            "(glosses) quotes the ontology's description after each user-centric predicate "
            "and the Organization and Event types, glosses WORKS_AT as employment only, states "
            "N17 named teams, N8, N10, N11 and N12 as Rules 34 to 38, amends Rule 6's "
            "\"don't like X anymore\" to N12's pair, and adds Example 31; T40 round 1 "
            "revises groups 2 and 3 to fit the 13,312-token prompt bound: Rule 28 states "
            "that a list item's id is its own words, with no cadence, scope, owner, purpose "
            "or quantity added, Rule 38 folds into Rule 6, Rule 34 and Rules 29 to 37 are "
            "tightened, the glosses keep Organization, WORKS_AT and MEMBER_OF, and Example "
            "29 returns to its group 1 text; T40 round 2 restores the KNOWS, INTERESTED_IN, "
            "EXPERT_IN and LEARNING glosses, Rule 34's clause that each team member WORKS_AT "
            "the named employer with no employer in the team's id, a stated place as a "
            "Location in Rule 31, and Rule 29's literal user EXPERIENCED <event>. "
            "Re-extracts history."
        ),
    ),
)


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
    `event_id|extraction_version|model_hash`
    (verified via `grep -n 'raw = "|".join' backend/knowledge/extraction_cache.py`)
    and `LogRegenerator` builds its lookup from the EPOCH row, so a
    `model_hash` disagreement -- the field L4 differed on -- is still a total,
    permanent `ColdCacheError` on every turn of every rebuild, the failure mode
    this module's docstring names. `ontology_version` no longer carries that
    risk (spec D3): a disagreement there is a mislabel now, not a cache miss.

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
