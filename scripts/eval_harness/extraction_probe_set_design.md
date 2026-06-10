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
# 1. Eval Neo4j up (F1)
docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml --profile eval up -d mist-neo4j-eval
MSYS_NO_PATHCONV=1 docker compose exec -T -e MIST_EVAL_ISOLATION=1 -e NEO4J_URI=bolt://mist-neo4j-eval:7687 \
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

# 3. Score
MSYS_NO_PATHCONV=1 docker compose exec -T mist-backend python scripts/eval_harness/score_extraction_run.py \
  --gold data/ingest/extraction-gold-2026-06-10.jsonl \
  --debug-jsonl data/runtime/extraction-baseline.jsonl \
  --output data/runtime/extraction-baseline-report.md

# 4. Teardown (F1)
docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml --profile eval rm -sfv mist-neo4j-eval
```

## Baseline results (YYYY-MM-DD)

_Filled by Task 5._

## Known limitations
- ~12-record seed: regression signal, not statistical certification of 0.90. Expand toward 60-100 (per-category) before treating gates as commitments.
- Reconciliation-action accuracy unmeasurable until C2 emits telemetry.
- Valid-time accuracy only meaningful for absolute-date probes.
