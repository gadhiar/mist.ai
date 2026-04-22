# V7 Tool-Heavy Probe Set Design

**Date authored:** 2026-04-22
**Scope:** `data/ingest/v7-tool-heavy-inputs.jsonl` (25 queries)
**Consumer:** `scripts/mist_admin.py replay` + future tool-calling evaluation harness
**Purpose:** Unblock the `mist-ai-tool-calling-production-rigor` workstream with a reusable, pre-classified probe set that exercises `query_knowledge_graph` tool-selection under production conditions.

---

## Why this set exists

The existing gauntlet (`data/ingest/v6-inputs.jsonl`) is a 30-turn *conversation* designed around ontology extraction. It is inadequate for tool-calling evaluation for three reasons:

1. Its turns are statements, not questions -- they trigger extraction, not retrieval.
2. It assumes a single session-id, so coreference dominates. Tool-selection should be robust across stateless probes.
3. There is no gold label for "tool should have fired here." Success today is measured by downstream metric leakage, not the decision itself.

V7 is engineered so each line carries an expected-behavior label (`tool_call`, `rationale`) directly in the JSONL. The scoring harness can then compute a single confusion-matrix per run: did MIST call `query_knowledge_graph` when it should have, and stay silent when it shouldn't.

## Design principles

1. **Cover the positive and negative space.** 20 positive probes (tool expected) + 5 negative controls (tool use would be a false positive).
2. **Each negative control is a high-value trap.** Capital-city trivia, small talk, technical explainer, creative prose, acknowledgement turn. If the model is over-eager, these catch it.
3. **Mix edge types.** Positive probes span `USES`, `WORKS_AT`, `LEARNING`, `DECIDED`, `INTERESTED_IN`, `HAS_GOAL`, `PREFERS`, `KNOWS_PERSON`, `OCCURRED_ON` (post-MVP). Coverage of extractable edges prevents the set from baking in bias toward a narrow slice of the graph.
4. **Include phrasings that should trip naive classifiers.** "Have I mentioned..." and "Did I decide..." are syntactically hedged but semantically require recall. "Hey, good morning!" is a cheerful opener that looks like it could preface a personal question. "Can you explain how HTTP/2 differs..." sounds technical but is answerable from pretraining.
5. **One ambiguity probe per 5 positive cases.** "Hmm, what was that framework I said I wanted to try again?" -- a well-calibrated MIST reads this as a recall signal, not a general question.

## Breakdown

### Positive (tool expected) — 20 probes

Counts below are verified against the JSONL (probes listed by tag). Buckets sum to 20.

| Bucket | Probes | Tags | Edge / entity focus |
|---|---|---|---|
| Tech recall | 1 | v7-01 | `USES` |
| Person recall | 1 | v7-03 | `KNOWS_PERSON` with project-context filter |
| Decision recall | 3 | v7-02, v7-11, v7-17 | `DECIDED` (one present, one existence check, one temporal-recent) |
| Skill recall | 2 | v7-04, v7-13 | `EXPERT_IN`, `RELATED_TO` (multi-hop on v7-13) |
| Preference / interest | 3 | v7-05, v7-07, v7-16 | `INTERESTED_IN`, `PREFERS` |
| Project / work | 2 | v7-06, v7-15 | `WORKS_ON` (list form + multi-fact form) |
| Goal | 2 | v7-09, v7-14 | `HAS_GOAL` (refresh + progress) |
| Ambiguous / coref | 2 | v7-08, v7-18 | coreference via "again"; recall is implied, not direct |
| Learning | 1 | v7-12 | `LEARNING` with current-temporal filter |
| Cross-entity | 1 | v7-10 | `KNOWS_PERSON` + `WORKS_AT` composition |
| Employment | 1 | v7-19 | `WORKS_AT` existence |
| Milestone | 1 | v7-20 | `Milestone` node (post-MVP additive type) |

### Negative controls (no tool) — 5 probes

| Bucket | Example | Trap |
|---|---|---|
| General knowledge | "What is the capital of Australia?" | Over-eager classifier might invoke the tool on any question |
| Small talk | "Hey, good morning!" | Greeting regex must not trigger retrieval |
| Technical explainer | "Can you explain how HTTP/2 differs from HTTP/1.1?" | Answerable from pretraining; tool use is slop |
| Creative | "Write me a short haiku about autumn." | Generative task; tool use is a clear false positive |
| Acknowledgement | "Thanks, that helps." | One of the cheapest false-positive surfaces |

## Acceptance criteria

A production-ready MIST build, running this probe set with each line treated as a standalone session turn, should produce:

- **Tool-selection precision** (of tool calls, fraction that matched the expected positive): >= 0.90.
- **Tool-selection recall** (of expected positives, fraction that fired the tool): >= 0.90.
- **False-positive rate on negative controls**: 0/5.

These thresholds mirror the `phase3_orchestrator.sh` baseline of `tool_selection >= 0.90` used elsewhere in the harness. The stricter "0/5 on negatives" is a discrete check rather than a rate, to flag single over-eager calls immediately.

## How to run

**Today (manual, single-run):**

```bash
docker compose exec -T mist-backend python scripts/mist_admin.py replay \
    data/ingest/v7-tool-heavy-inputs.jsonl \
    --session-id v7-probe-$(date +%Y%m%d) \
    --output data/ingest/v7-report.jsonl
```

Each line is replayed as an independent turn (the replay command does not enforce a shared conversational context by default). Tool calls land in the `ConversationHandler` log stream and can be extracted from the backend's per-turn debug JSONL (when `MIST_DEBUG_JSONL=1`).

**Scoring (manual today, tool-calling harness integration is out of scope for this commit):**

Until a dedicated scorer lands, the report is evaluated by:

1. Joining the per-turn debug JSONL against this input file on `tag`.
2. For each turn, comparing `turn.tool_calls[0].name == expected_behavior.tool_call` (with `None` matching "no tool was called").
3. Reporting the confusion matrix + precision/recall.

A one-liner python aggregator at `scripts/eval_harness/score_v7_probe_run.py` is a morning followup.

## Known limitations

- **No ground truth for tool arguments.** V7 scores the *decision* to call, not the argument quality. Argument quality belongs to a future expansion (V7.1 or the tool-calling workstream proper).
- **Single-turn framing.** The probe deliberately does not establish prior context. A production workload might see "what's my favorite DB again?" only after several turns establishing preferences. That is a scenario for a V8 multi-turn tool probe set.
- **Graph state assumptions.** Positive probes presume the graph has been populated with a minimally plausible user profile. Empty-graph runs will still exercise the decision correctly (the tool should still fire; it just returns no results). Pair with `seed` + a short pre-probe conversation for realistic hit rates.

## Followups

- `scripts/eval_harness/score_v7_probe_run.py` -- one-shot scorer against the debug-JSONL stream.
- `v7-multi-turn` -- same intents but phrased as turns 5-25 of a running conversation.
- `v7-negative-expansion` -- lift negative control count to 10 covering more slop categories (apologies, meta questions, "are you there?", etc).
