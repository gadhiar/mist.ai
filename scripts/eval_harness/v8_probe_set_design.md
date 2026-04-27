# V8 Edge-Production Probe Set Design

**Date authored:** 2026-04-24
**Scope:** `data/ingest/v8-edge-production-inputs.jsonl` (20 queries)
**Consumer:** `scripts/mist_admin.py replay` + future scorer at `scripts/eval_harness/score_v8_probe_run.py`
**Purpose:** Validate end-to-end production of the four post-MVP edge types -- `OCCURRED_ON`, `HAS_METRIC`, `REFERENCES_DOCUMENT`, `PRECEDED_BY` -- on the producer side.

---

## Why this set exists

The 2026-04-22 ontology expansion added 4 new node types (`Date`, `Milestone`, `Metric`, `Document`) and 4 new edges to v1.0.0 of the ontology. The post-expansion V6 gauntlet rerun confirmed three of four new node types are produced spontaneously (`Document` x2, `Date` x1, `Metric` x1 across 30 turns), but **no instances of the four new edges landed end-to-end**.

The 2026-04-23 session note flagged this as producer-side, but did not isolate the cause. Three diagnostic possibilities:

1. **Few-shot blind spot.** The model never emits the new edge types because it has no anchor -- only Example 9 (Milestone+OCCURRED_ON), Example 10 (Metric+HAS_METRIC), Example 11 (Document+REFERENCES_DOCUMENT), and the post-2026-04-24 Example 12 (Event+OCCURRED_ON) cover these edges.
2. **Validator drop.** The model emits the edges but they fail validation. Caught by `TestValidatorOntologyConsistency` at unit-test time, so unlikely to be silent.
3. **V6 input gap.** The model emits the edges but on inputs we haven't tested -- V6 is a 30-turn conversational gauntlet that doesn't engineer for the new edges specifically.

V8 isolates (3): each probe is engineered to make the new edge type the OBVIOUS extraction. If V8 still produces zero new edges, (1) is confirmed and the next step is targeted few-shot expansion (one example per absent bucket).

## Design principles

1. **One edge-type per bucket.** Four positive buckets, one per new edge type, so per-edge recall is independently computable.
2. **Make the new edge structurally inevitable.** Each utterance carries the source/target entity types the edge requires (Event+Date for `OCCURRED_ON`, Project+Metric for `HAS_METRIC`, etc.) in unambiguous form.
3. **Negative controls test over-eagerness.** Probes where a Date / number / Document mention is present but does NOT motivate the edge.
4. **Acceptance is per-edge recall, not aggregate.** The new edges have asymmetric difficulty -- `HAS_METRIC` requires recognizing a quantified property; `REFERENCES_DOCUMENT` requires recognizing a specific named artifact. A single aggregate number obscures which one is failing.
5. **Initial thresholds are looser than V7's.** V7 hit 0.90/0.90 against `query_knowledge_graph` because tool-call selection has 6+ months of prompt iteration behind it. The new edges have only the 4 few-shots above; expect a lower starting baseline and ratchet up.

## Breakdown

### Positive (edge expected) -- 16 probes

| Bucket | Probes | Tags | Expected edges | Notes |
|---|---|---|---|---|
| OCCURRED_ON | 4 | v8-01..v8-04 | `OCCURRED_ON` | Event/Milestone anchored to Date. Mixes ISO dates, relative ('yesterday'), and life events. |
| HAS_METRIC | 4 | v8-05..v8-08 | `HAS_METRIC` | Project/Technology/Goal HAS_METRIC Metric. Mixes counts, throughput, targets, benchmarks. |
| REFERENCES_DOCUMENT | 4 | v8-09..v8-12 | `REFERENCES_DOCUMENT` | User REFERENCES_DOCUMENT Document. Covers ADR / paper / book / RFC. |
| PRECEDED_BY | 4 | v8-13..v8-16 | `PRECEDED_BY` | Event/Milestone PRECEDED_BY Event/Milestone/Date. Mixes target types. |

Note on v8-07 (`Goal HAS_METRIC`): the probe is expected to produce TWO edges -- `User HAS_GOAL Goal` (existing) and `Goal HAS_METRIC Metric` (new). The expected_edges list captures both so the scorer doesn't penalize the model for surfacing both.

### Negative controls (no new edge) -- 4 probes

| Bucket | Tag | Trap |
|---|---|---|
| Date no anchor | v8-17 | Date entity present, no Event/Milestone to anchor -- `OCCURRED_ON` would be a false positive |
| Number not metric | v8-18 | "3 hours" is a temporal expression about a personal moment, not a measurement of a graph entity |
| Document mention | v8-19 | "documents" is generic intent, not a named Document -- `REFERENCES_DOCUMENT` would be a false positive |
| Temporal no precedence | v8-20 | Time-of-day pattern, not Events with explicit ordering -- `PRECEDED_BY` would be a false positive |

## Acceptance criteria

A production-ready MIST build, running this probe set with each line treated as a standalone session turn, should produce:

- **Per-edge recall**: >= 0.75 per bucket. Lower than V7's 0.90 because the new-type edges have only 4 supporting few-shots; the design assumes initial recall around 0.50-0.75 and ratchets up as few-shots are added.
- **Cross-bucket overall positive recall**: >= 0.70 across all 16 positives.
- **False-positive rate on negatives**: 0/4 (discrete check; mirrors V7's 0/5 rule). Negative-control violations are individually diagnostic and any non-zero count is a real bug.

These thresholds are deliberately looser than V7's. If the run lands above them, the next step is to push toward 0.90/0.90 by adding one few-shot per absent bucket; if below, the prompt needs additional examples and possibly a rule clarification.

## How to run

**Today (manual, single-run):**

```bash
docker compose exec -T mist-backend python scripts/mist_admin.py replay \
    data/ingest/v8-edge-production-inputs.jsonl \
    --session-id v8-probe-$(date +%Y%m%d) \
    --output data/ingest/v8-report.jsonl
```

The `replay` command writes per-utterance results, but does NOT capture the extracted entity/edge types in its `--output` file (only utterance / response / duration / ok / error). To score the run, the future scorer needs one of:

1. **`MIST_DEBUG_LLM_JSONL=1`** (preferred) -- captures the full LLM response content via `phase: llm_call` records in `MIST_DEBUG_JSONL`. The scorer parses the response JSON to extract entity types + relationship types per turn, then joins against this file by utterance.
2. **Direct Neo4j query** -- after the replay completes, query for entities/edges created in the session_id. Higher fidelity (validates post-storage state) but requires graph access from the scorer.
3. **New extraction-detail debug emission** -- add per-extraction entity/relationship type breakdown to `phase: extraction` records in `debug_jsonl_logger._LiveTurnRecord.record_extraction`. Cleanest long-term but requires a backend code change.

Path (1) is the recommended route for the V8 scorer commit -- it requires only an env-var flip and parsing already-emitted JSON.

## Known limitations

- **No source/target type validation.** V8 scores the edge TYPE (was `OCCURRED_ON` produced?), not source/target correctness (was it `Event -> Date` or accidentally `Date -> Event`?). A future V8.1 layer can add structural assertions.
- **Single-turn framing.** Each probe is independent. Real conversations build context; V8 is a discrete classifier check on isolated utterances.
- **No ambiguity probes.** V7 had ambiguity buckets ("the framework I said I wanted to try again"). V8 prioritizes coverage breadth over depth -- ambiguity probes are V8.1 candidates.
- **Asymmetric counts.** 4 negatives vs 5 in V7. The four buckets each get one negative; adding a fifth blanket negative ("how are you?") is V8.1.
- **Threshold calibration is provisional.** The 0.75/0.70 numbers are pre-baseline guesses. The first V8 run becomes the calibration anchor.

## Followups

- `scripts/eval_harness/score_v8_probe_run.py` -- one-shot scorer. Joins V8 expected edges against `phase: llm_call` debug records (parses response JSON for entity/relationship types). Mirrors `score_v7_probe_run.py` structure: per-bucket recall, confusion matrix, acceptance verdict.
- `v8-multi-turn` -- same intents but spread across 5-25 turns of conversation context.
- `v8-edge-quality` -- structural validation of source/target type correctness.
- Post-V8 prompt iteration: if recall is below 0.75 per bucket, add one few-shot per absent edge type.
- Closure on the producer-side gap: re-run V6 gauntlet after V8 + few-shot additions to confirm the new edges land in unstructured conversation, not just engineered probes.
