# V9 v1.1.0 Predicate-Coverage Probe Set Design

**Date authored:** 2026-05-08
**Scope:** `data/ingest/v9-v1-1-0-predicate-coverage.jsonl` (12 probes)
**Consumer:** `scripts/mist_admin.py replay` + `scripts/eval_harness/score_v9_predicate_run.py`
**Purpose:** Validate that the three v1.1.0 predicates that did NOT fire on the V6 conversational sample (MECHANISM_OF, INPUT_TO, COMPRISES) CAN fire when given clear, structurally-engineered triggers.

---

## Why this set exists

The 2026-05-07 ontology v1.1.0 expansion added 8 new predicates and 5 new entity types. Post-expansion V6 (30 turns conversational) showed:

- 5 of 8 new predicates fired spontaneously: OPERATES_ON (7x), IMPROVES (3x), APPLICABLE_TO (3x), NAMING_CONVENTION_OF (3x), STRATEGY_FOR (2x).
- 3 of 8 did NOT fire: **MECHANISM_OF**, **INPUT_TO**, **COMPRISES**.

V6 is conversational architecture discussion — it doesn't engineer for these specific predicates. V9 isolates the question: do the unfired predicates fire when an utterance carries the source/target entity types and the predicate's semantic shape unambiguously?

If V9 produces zero fires for any unfired predicate, that predicate is functionally absent from production extraction — root cause is either prompt few-shot blindness, validator drop, or genuine extractor limitation. Targeted prompt iteration follows.

If V9 produces ≥1 fire per predicate, the predicate works in principle and the V6 zero-count is an input-distribution gap, not a capability gap.

## Design principles

1. **One predicate-per-bucket.** Three buckets, four probes each — per-predicate recall is independently computable.
2. **Make the predicate structurally inevitable.** Each utterance carries source/target entity types the predicate expects (Mechanism + Concept for MECHANISM_OF, DataStructure/Concept + Mechanism/Technology for INPUT_TO, Project/Mechanism + Technology/DataStructure for COMPRISES).
3. **Acceptance is per-predicate recall, not aggregate.** A single aggregate number obscures which predicate is failing.
4. **Lower threshold than V8.** Each predicate is "alive" if ≥1 of 4 probes fires (recall ≥ 0.25). This is intentionally lenient — V9's purpose is presence-of-life, not production quality.
5. **No negative controls.** V9 confirms the predicates CAN fire. Over-eagerness (false positives on unrelated content) is V6's job to surface and is out of scope here.

## Breakdown

### Positive — 12 probes, 4 per predicate

| Bucket | Probes | Tags | Expected edge | Source -> Target type |
|---|---|---|---|---|
| MECHANISM_OF | 4 | v9-01 .. v9-04 | `MECHANISM_OF` | Mechanism\|Pattern -> Concept\|Technology\|Topic\|Strategy |
| INPUT_TO | 4 | v9-05 .. v9-08 | `INPUT_TO` | DataStructure\|Concept\|Document -> Mechanism\|Strategy\|Technology\|Pattern |
| COMPRISES | 4 | v9-09 .. v9-12 | `COMPRISES` | Technology\|Project\|DataStructure\|Mechanism\|Strategy -> DataStructure\|Mechanism\|Concept\|Technology\|Pattern |

## Acceptance criteria

- **Per-predicate recall ≥ 0.50** (each predicate fires on ≥2 of 4 probes).
- **No aggregate recall threshold** — aggregate would obscure per-predicate failure.

**Threshold history.** Initial floor was 0.25 (presence-of-life check, >=1 of 4 probes). Raised to 0.50 on 2026-05-08 after Example 20 (`MECHANISM_OF -- phrasal variations`) lifted MECHANISM_OF from 0.25 to 0.50. The 0.50 floor better reflects "the predicate is robust enough to fire on multiple natural phrasings, not just the canonical one."

If a predicate fails its 0.50 floor, the recommended remediation is targeted prompt iteration: add a few-shot example to `backend/knowledge/extraction/prompts.py` mirroring the failed probe's structure (anchored entity types + the missing predicate as the obvious extraction). Then re-run V9 to confirm recall recovery.

## How to run

Single-shot probes — extraction is per-utterance and does not depend on conversation history. Run via `mist_admin replay` (chat-path) for clean session isolation:

```bash
MSYS_NO_PATHCONV=1 docker compose exec -T \
  -e MIST_DEBUG_JSONL=/app/data/runtime/v9-<run-tag>.jsonl \
  -e MIST_DEBUG_LLM_JSONL=1 \
  mist-backend python scripts/mist_admin.py replay \
  data/ingest/v9-v1-1-0-predicate-coverage.jsonl \
  --output data/ingest/v9-<run-tag>-output.jsonl \
  --session-id v9-replay-<tag>
```

Then score:

```bash
MSYS_NO_PATHCONV=1 docker compose exec -T mist-backend \
  python scripts/eval_harness/score_v9_predicate_run.py \
  --input data/ingest/v9-v1-1-0-predicate-coverage.jsonl \
  --debug-jsonl /app/data/runtime/v9-<run-tag>.jsonl \
  --session-id v9-replay-<tag>
```

## References

- 2026-05-07 ontology v1.1.0 ship session (commit 3bdd010 + transfer doc)
- `backend/knowledge/ontologies/v1_0_0.py` — ONTOLOGY_V1_0_0 (now version 1.1.0)
- `backend/knowledge/extraction/prompts.py` — Rules 14-15 + examples 15-19
- `mist-ai-voice-chat-path-unification` workstream — origin of the unfired-predicate flag
