# Extraction Accuracy Gold Corpus -- Design

**Date authored:** 2026-06-10
**Scope:** `data/ingest/extraction-gold-2026-06-10.jsonl`
**Consumer:** `scripts/mist_admin.py replay` (emits `MIST_DEBUG_JSONL`) + `scripts/eval_harness/score_extraction_run.py`
**Purpose:** Ground-truth measurement of extraction accuracy for sub-project A (F2). Establishes the baseline the C1/C2/C3 gates improve on.

## Record schema

One JSON object per line:

```json
{
  "utterance": "I started using Rust in May 2026",
  "tag": "ext-09-temporal-absolute",
  "expected_entities": [{"id": "rust", "type": "Technology"}],
  "expected_relationships": [
    {"source": "user", "source_type": "User", "predicate": "USES",
     "target": "rust", "target_type": "Technology", "valid_from": "2026-05"}
  ]
}
```

- `id`/`source`/`target` are **canonical** (lowercase-hyphenated); the scorer canonicalizes produced ids the same way before matching.
- `source_type`/`target_type` use ontology entity-type names; `predicate` uses an extractable relationship type. Typing is validated against `RELATIONSHIP_CONSTRAINTS`.
- `valid_from`/`valid_to` (optional) are ISO date prefixes (`"2026"`, `"2026-05"`, `"2026-05-20"`); scored precision-aware against produced `properties.start_date`/`end_date`. **Use absolute dates** in gold temporal probes -- relative expressions ("last month") resolve against the run's reference_date and are not deterministic.
- Negative probes (directives, small talk) carry empty `expected_entities`/`expected_relationships`.

## Categories (seed set)

| Tag | Tests |
|---|---|
| ext-01..03 | core user facts: USES, USED_FOR, DISLIKES, WORKS_AT |
| ext-04 | event anchored to a date (OCCURRED_ON, direction Event->Date) |
| ext-05 | quantified fact (HAS_METRIC) |
| ext-06..07 | skill state (LEARNING, EXPERT_IN) |
| ext-08 | third-party person (KNOWS_PERSON) |
| ext-09 | valid-time (absolute date) |
| ext-10..11 | negative controls: directive (Bug K), small talk |
| ext-12 | structural (IS_A) |

## Corpus rules (gold labeling policy)

### Date-entity policy (locked 2026-06-12, C3 T1)

Date entities appear in gold ONLY when an edge targets them:
- OCCURRED_ON anchor (e.g. ext-04): `Event -[OCCURRED_ON]-> Date` -- Date entity in gold.
- PRECEDED_BY date-target (Rule 13): `X -[PRECEDED_BY]-> Date` -- Date entity in gold.

A date that merely scopes a stative fact (e.g. "I started using Rust in May
2026") belongs in `valid_from`/`valid_to` on the relationship, not as a
standalone Date entity (see ext-09). Do not add a Date node to gold for
temporal qualifiers of non-event facts.

### Anchor-entity / anchor-edge policy

Every user-scope predicate (USES, DISLIKES, WORKS_AT, LEARNING, EXPERT_IN,
KNOWS_PERSON, EXPERIENCED, etc.) requires the `user` entity in
`expected_entities` AND the corresponding outgoing edge from `user` in
`expected_relationships`. This follows Extraction Rule 1. The deep-review
correction (Batch G, 2026-06-10) added anchor entities; C3 T1 (2026-06-12)
added the missing anchor edge on ext-04.

## Metrics + provisional gates

Tiered: core user-centric predicates gate; the v1.1.0 long-tail is tracked-not-gated until C3 tuning. All thresholds are provisional until the baseline (below); a ~12-record seed is a regression signal, not statistical certification.

- Entity precision >= 0.90, recall >= 0.80
- Relationship precision >= 0.90, recall >= 0.80
- Relationship-typing accuracy >= 0.90
- RELATED_TO rate <= 0.10
- Valid-time accuracy (tracked)
- Reconciliation-action accuracy: SKIPPED (requires C2 telemetry)

## How to run (isolated via F1's throwaway-quad)

```bash
# 1. Eval Neo4j up (F1). Isolate ALL stores on seed too: seed's vault bootstrap
#    writes identity/user notes, so without the trio overrides it touches the
#    live vault (Inv-A8). Match the replay's isolation.
docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml --profile eval up -d mist-neo4j-eval
MSYS_NO_PATHCONV=1 docker compose exec -T \
  -e MIST_EVAL_ISOLATION=1 -e NEO4J_URI=bolt://mist-neo4j-eval:7687 \
  -e MIST_SIDECAR_DB_PATH=/app/data/eval-run/vault_sidecar.db \
  -e EVENT_STORE_DB_PATH=/app/data/eval-run/event_store.db \
  -e MIST_VAULT_ROOT=/app/data/eval-run/vault \
  mist-backend python scripts/mist_admin.py seed

# 2. Replay the gold corpus, emitting the extraction debug log
MSYS_NO_PATHCONV=1 docker compose exec -T \
  -e MIST_EVAL_ISOLATION=1 -e NEO4J_URI=bolt://mist-neo4j-eval:7687 \
  -e MIST_SIDECAR_DB_PATH=/app/data/eval-run/vault_sidecar.db \
  -e EVENT_STORE_DB_PATH=/app/data/eval-run/event_store.db \
  -e MIST_VAULT_ROOT=/app/data/eval-run/vault \
  -e MIST_DEBUG_JSONL=/app/data/runtime/extraction-baseline.jsonl -e MIST_DEBUG_LLM_JSONL=1 \
  mist-backend python scripts/mist_admin.py replay \
  data/ingest/extraction-gold-2026-06-10.jsonl --session-id ext-baseline

# 3. Score (--strict: fail loudly on a broken probe join or any
# negative-control violation instead of silently scoring a partial run)
MSYS_NO_PATHCONV=1 docker compose exec -T mist-backend python scripts/eval_harness/score_extraction_run.py \
  --gold data/ingest/extraction-gold-2026-06-10.jsonl \
  --debug-jsonl data/runtime/extraction-baseline.jsonl \
  --output data/runtime/extraction-baseline-report.md \
  --strict

# 4. Teardown (F1)
docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml --profile eval rm -sfv mist-neo4j-eval
```

## Baseline results (2026-06-12, corrected corpus)

Re-baseline executed 2026-06-12 at C3 kickoff per the deep-review correction
(foundation-f123-3): the original gold labels omitted the prompt-mandated
`user` anchor entity (Extraction Rule 1) on the 8 user-scope probes
(ext-01/02/03/04/06/07/08/09), so the 2026-06-10 run's entity precision 0.652
was a harness artifact (15 TP + 8 guaranteed anchor FPs). The ext-04 anchor
edge was labeled on 2026-06-12 (C3 T1) -- the gold contract now matches
Extraction Rule 1 for both entities and edges. All pre-correction numbers are
NON-COMPARABLE; this table is the C3 no-regression reference.

Gemma 4 E4B Q5_K_M, ontology v1.2.1, main @ de5e566, seed graph, isolated
quad (eval Neo4j + throwaway trio). All 12 probes matched in the debug log;
every probe produced an extraction record. Run artifacts:
`data/runtime/extraction-rebaseline-2026-06-12.jsonl` + `-report.md`.

| Metric | Value | Gate | Pass |
|---|---|---|---|
| Entity precision | 0.957 | >= 0.90 | PASS |
| Entity recall | 0.957 | >= 0.80 | PASS |
| Relationship precision | 0.917 | >= 0.90 | PASS |
| Relationship recall | 0.917 | >= 0.80 | PASS |
| Typing accuracy | 1.000 | >= 0.90 | PASS |
| RELATED_TO rate | 0.083 | <= 0.10 | PASS |
| Valid-time accuracy | 1.000 | (tracked) | - |
| Negative-control violations | 0 | == 0 | PASS |

Reconciliation-action accuracy: SKIPPED (requires C2 telemetry).

Reading (corrected, post-C3-T1): all 8 gates pass. The artifact correction
confirmed entity precision was never the problem (0.957), and typing accuracy
moved 0.833 -> 1.000 between the 2026-06-10 run (ontology v1.1.0, hardcoded
constraints) and this run (v1.2.1, ontology-derived RELATIONSHIP_CONSTRAINTS
via C1). Relationship precision moved from the gold-artifact 0.833 to 0.917
once ext-04's anchor edge was labeled (C3 T1); relationship recall likewise
corrected from 0.909 to 0.917. RELATED_TO rate is 0.083 (one default-predicate
edge on this run); below gate but worth watching at corpus expansion.
All gates pass at this corrected baseline -- C3's remaining tasks target
further precision improvements, not gate recovery (12-record seed; expand
toward 60-100 before treating gates as commitments).

### Superseded baseline (2026-06-10, pre-correction corpus -- NON-COMPARABLE)

Entity P 0.652 / R 1.000, rel P 0.750 / R 0.818, typing 0.833, RELATED_TO
0.000, valid-time 1.000, 0 negative violations. Ontology v1.1.0. Retained for
history only; the entity-precision FAIL and typing FAIL were measured against
anchor-less gold labels and the pre-C1 hardcoded constraint table.

## Known limitations
- ~12-record seed: regression signal, not statistical certification of 0.90. Expand toward 60-100 (per-category) before treating gates as commitments.
- Reconciliation-action accuracy unmeasurable until C2 emits telemetry.
- Valid-time accuracy only meaningful for absolute-date probes.
