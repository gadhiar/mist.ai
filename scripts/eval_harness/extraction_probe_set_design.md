# Extraction Accuracy Gold Corpus -- Design

**Date authored:** 2026-06-10
**Scope:** `data/ingest/extraction-gold-2026-06-10.jsonl`
**Consumer:** `scripts/mist_admin.py replay --extraction-only` (authoritative; emits `MIST_DEBUG_JSONL`) + `scripts/eval_harness/score_extraction_run.py`
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
- `assertion_kind` (optional, added C3 T3) records the temporal-assertion semantics of a gold edge: `assert` (default, omit the field), `cease` (the fact stopped holding -- optionally bounded by `valid_to`), `retract` (the fact was never true). The loader (`iter_gold_probes`) defaults absent fields to `assert` and raises a `ValueError` naming the probe tag on any other value. Bucket scoring on this label lands in a later C3 task; T3 only adds the labels + the loader contract.
- Negative probes (directives, small talk) carry empty `expected_entities`/`expected_relationships`.

## Categories (60-probe corpus)

The corpus expanded 12 -> 60 on 2026-06-12 (C3 T3) to give per-category
statistical mass and to add the assertion-kind, same-turn-pair, habit,
recommendation, and date-discrimination shapes that later C3 tasks tune
against. Each `expected_relationships` entry MAY carry an `assertion_kind`
(`assert` default, encoded as the loader default; only `cease`/`retract`
need explicit labels). Cease probes with a stated stop date also carry
`valid_to`.

| Category | Tags | Total | What it tests |
|---|---|---|---|
| Core user facts | ext-01..03, ext-13..19 | 10 | USES, USED_FOR, DISLIKES (incl. negation-form "I don't like YAML"), WORKS_AT (3), PREFERS |
| Events + dates | ext-04, ext-20..23 | 5 | mixed Event/Milestone, all date-anchored (OCCURRED_ON; Event probes also carry the user EXPERIENCED anchor) |
| Quantified | ext-05, ext-24..25 | 3 | HAS_METRIC (Technology/Project source) |
| Skill state | ext-06..07, ext-26..28 | 5 | LEARNING, EXPERT_IN, STRUGGLES_WITH |
| Third-party | ext-08, ext-29..32 | 5 | KNOWS_PERSON scope discipline; ext-31/32 also carry RECOMMENDS gold (predicate lands at T10 -- baseline FN by design) |
| Valid-time | ext-09, ext-33..36 | 5 | absolute + textual since/from/until/between; resolver exercise |
| Negative controls | ext-10..11, ext-37..40 | 6 | directive, small talk, question-form, hypothetical -- empty extraction |
| Structural | ext-12, ext-41..42 | 3 | IS_A, PART_OF |
| Cessation (assertion_kind=cease) | ext-43..47 | 5 | "stopped using" (+stop date -> valid_to), "quit job", "gave up learning", "don't use anymore", "dropped" |
| Retraction (assertion_kind=retract) | ext-48..52 | 5 | distinct surfaces: "never actually", "I misspoke", "scratch that", "I was wrong about", "correction:" |
| Same-turn SINGLE pair | ext-53..54 | 2 | cease+assert in one turn; assert target sorts alphabetically BEFORE cease target (constructs the bad ordering a later engine task fixes) |
| Habit / recurrence (HAS_HABIT) | ext-55..57 | 3 | "journal every night", "review PRs every Friday", "long run every Sunday" (predicate lands at T10 -- baseline FN by design) |
| Date-entity discrimination | ext-58..60 | 3 | 2 stative "since <date>" (valid_from, NO Date entity) vs 1 event "on <date>" (Date entity + OCCURRED_ON) |

Counts sum to 48 new / 60 total. Per-category author counts (NEW probes
only): core 7, events 4, quantified 2, skill 3, third-party 4, valid-time 4,
negative 4, structural 2, cease 5, retract 5, same-turn 2, habit 3,
date-discrimination 3 = 48.

### Same-turn pair ordering (ext-53 / ext-54)

Both same-turn pairs are deliberately constructed so the ASSERT edge's target
sorts alphabetically BEFORE the CEASE edge's target -- the ordering a later
engine task (same-turn pair handling) must correct. ext-53 "I left Zeta for
Acme": assert target `acme` < cease target `zeta`. ext-54 was authored as "I
switched from Redis to Memcached" (NOT Postgres->SQLite): assert target
`memcached` < cease target `redis`. (Postgres->SQLite would put the cease
target `postgres` first, which is NOT the bad ordering, so the techs were
chosen to preserve the assert-sorts-first property in both pairs.)

### RECOMMENDS / HAS_HABIT gold (deliberate baseline FNs)

ext-31, ext-32 (RECOMMENDS) and ext-55, ext-56, ext-57 (HAS_HABIT) carry gold
edges for predicates the ontology does not yet have. They score as relationship
false-negatives at the C3 baseline on purpose: the baseline measures the size
of that gap so T10 (ontology gains RECOMMENDS + HAS_HABIT) can show the
recovery. The recommendation probes keep their user KNOWS_PERSON anchor edge
(that one scores normally); the RECOMMENDS edge is third-party-sourced
(Person -> Technology).

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
`expected_entities` AND the corresponding outgoing user-sourced ANCHOR edge in
`expected_relationships`. This follows Extraction Rule 1. The rule governs the
anchor edge only -- it does not require every edge to source from `user`: a
record may also carry non-anchor edges between other entities (e.g. ext-01's
`Technology -[USED_FOR]-> Topic`, ext-04's `Event -[OCCURRED_ON]-> Date`, or
ext-31's `Person -[RECOMMENDS]-> Technology`). The deep-review correction
(Batch G, 2026-06-10) added anchor entities; C3 T1 (2026-06-12) added the
missing anchor edge on ext-04.

## Metrics + provisional gates

Tiered: core user-centric predicates gate; the v1.1.0 long-tail is tracked-not-gated until C3 tuning. All thresholds are provisional until the baseline (below); a ~12-record seed is a regression signal, not statistical certification.

- Entity precision >= 0.90, recall >= 0.80
- Relationship precision >= 0.90, recall >= 0.80
- Relationship-typing accuracy >= 0.90
- RELATED_TO rate <= 0.10
- Valid-time accuracy (tracked)
- Reconciliation-action accuracy: SKIPPED (requires C2 telemetry)

## How to run (isolated via F1's throwaway-quad)

### Authoritative path: extraction-only (`replay --extraction-only`)

F2 is measured EXTRACTION-ONLY. Pass `--extraction-only` to the `replay`
subcommand: each probe runs the production extraction pipeline with NO chat-reply
generation. This is byte-reproducible (two fresh-quad runs produce byte-identical
score JSONs), so a single run is a valid gate. The commands in this section use
`--extraction-only`; the `MIST_FIXED_CLOCK` pin keeps the extraction prompt's
`reference_date` stable across runs (and is harmless even though no chat reply is
generated). Use `--session-id ext-only-d1` / `ext-only-d2` for a determinism pair.

The env pin is identical to the full-chat path below (`LLM_CONVERSATION_TEMPERATURE`
is irrelevant on the extraction-only path -- no chat call -- but harmless to set):

```
-e LLM_TEMPERATURE=0.0 -e LLM_CONVERSATION_TEMPERATURE=0.0 -e PYTHONHASHSEED=0 \
-e MIST_FIXED_CLOCK=2026-06-13T00:00:00+00:00
```

To confirm determinism: run the seed + `replay --extraction-only` + score TWICE on
fresh quads (distinct `--session-id`, fresh `mist-neo4j-eval` instance each), and
diff the two score JSONs -- they must be byte-identical on aggregate metrics AND
`per_probe` FP/FN arrays. (Certified 2026-06-13; see "Baseline results
(60-probe corpus, extraction-only, deterministic)".)

### Determinism env for the SUPERSEDED full-chat path

The full-chat `replay` path (without `--extraction-only`) drives the FULL
`handle_message` path, not extraction in isolation. That path makes three LLM
calls per turn at two different temperatures: `extraction.scope_classifier` and
`extraction.ontology` at `LLM_TEMPERATURE` (0.0, greedy), and `chat.initial` at
`LLM_CONVERSATION_TEMPERATURE` (production default 0.7, stochastic). The
extraction prompt's `Context:` block embeds the assistant's chat reply, so a
stochastic chat turn makes the extraction INPUT differ run-to-run -- which makes
the (otherwise greedy) extraction OUTPUT differ too. Pinning only
`LLM_TEMPERATURE=0.0` is therefore insufficient: the 2026-06-12 C3 T3 determinism
check caught two replays diverging on 20/60 probes for exactly this reason (root
cause: conversational-temperature contamination of the extraction input,
compounded by RAG over the per-turn-accumulating graph). Even with the whole chain
pinned greedy + fixed clock, the full-chat path still diverges on 18/60 probes due
to flash-attn FP nondeterminism on the long chat generation -- which is why F2
moved to extraction-only.

To make F2 replays reproducible, pin the WHOLE chain greedy plus a stable hash
seed AND a fixed clock on BOTH the seed and the replay exec:

```
-e LLM_TEMPERATURE=0.0 -e LLM_CONVERSATION_TEMPERATURE=0.0 -e PYTHONHASHSEED=0 \
-e MIST_FIXED_CLOCK=2026-06-13T00:00:00+00:00
```

`PYTHONHASHSEED=0` neutralizes a secondary set-ordering nondeterminism observed
in the chat tool-schema enum serialization (a `display_hint` enum order flipped
between runs). NOTE: that enum-ordering flip is a latent PRODUCTION bug in the
tool-schema serialization (non-deterministic iteration over a set/dict feeding
the schema), tracked separately -- `PYTHONHASHSEED=0` masks it for reproducible
evals but does not fix the underlying ordering instability.

`MIST_FIXED_CLOCK` (added 2026-06-13, commit `3ca09e4`) pins the wall-clock
`rendered_at` that the seed bootstrap stamps into `users/<id>.md`. That
Provenance timestamp is read into the chat system prompt (always-inject curated
profile); an unpinned wall-clock value made the turn-1 chat prompt differ
between runs, perturbing the greedy reply and diverging the conversation history
from turn 2 on. Set it to any single ISO-8601 instant -- the only requirement is
that BOTH replays in a determinism pair use the SAME value. It MUST be passed on
the `seed` exec (the seed writes the prompt-facing note) as well as the `replay`
exec (the handler's C-pattern snapshot writeback and currency-filter `$now`).
Unset in production -> live wall-clock (behavior unchanged). The clock is
injected via DI (`ConversationHandler.now_fn`, wired in
`backend.factories.build_now_fn`); it is not a monkeypatch.

Caveat: production chat runs at conversation temp 0.7. This procedure measures
extraction accuracy over a DETERMINISTIC (temperature-0 / mode) assistant
context. That is the standard reproducibility tradeoff for an LLM-in-the-loop
eval -- the baseline is a fixed, comparable reference point, not a sample of
production's stochastic conversational distribution.

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
  -e LLM_TEMPERATURE=0.0 -e LLM_CONVERSATION_TEMPERATURE=0.0 -e PYTHONHASHSEED=0 \
  -e MIST_FIXED_CLOCK=2026-06-13T00:00:00+00:00 \
  mist-backend python scripts/mist_admin.py seed

# 2. Replay the gold corpus EXTRACTION-ONLY (authoritative path), emitting the
#    extraction debug log. --extraction-only runs the production extraction
#    pipeline with NO chat reply (deterministic). Drop the flag for the
#    superseded full-chat path.
MSYS_NO_PATHCONV=1 docker compose exec -T \
  -e MIST_EVAL_ISOLATION=1 -e NEO4J_URI=bolt://mist-neo4j-eval:7687 \
  -e MIST_SIDECAR_DB_PATH=/app/data/eval-run/vault_sidecar.db \
  -e EVENT_STORE_DB_PATH=/app/data/eval-run/event_store.db \
  -e MIST_VAULT_ROOT=/app/data/eval-run/vault \
  -e LLM_TEMPERATURE=0.0 -e LLM_CONVERSATION_TEMPERATURE=0.0 -e PYTHONHASHSEED=0 \
  -e MIST_FIXED_CLOCK=2026-06-13T00:00:00+00:00 \
  -e MIST_DEBUG_JSONL=/app/data/runtime/extraction-baseline.jsonl -e MIST_DEBUG_LLM_JSONL=1 \
  mist-backend python scripts/mist_admin.py replay \
  data/ingest/extraction-gold-2026-06-10.jsonl --session-id ext-baseline --extraction-only

# 3. Score. The extraction-only run is byte-reproducible, so --strict is safe
# (it fails loudly on a broken probe join or any negative-control violation
# instead of silently scoring a partial run). Pass --session-id to filter the
# debug log to this run.
MSYS_NO_PATHCONV=1 docker compose exec -T mist-backend python scripts/eval_harness/score_extraction_run.py \
  --gold data/ingest/extraction-gold-2026-06-10.jsonl \
  --debug-jsonl data/runtime/extraction-baseline.jsonl \
  --session-id ext-baseline \
  --output data/runtime/extraction-baseline-report.md \
  --json-output data/runtime/extraction-baseline.json \
  --strict

# 4. Teardown (F1)
docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml --profile eval rm -sfv mist-neo4j-eval
```

To empirically confirm determinism (recommended after any procedure change),
run steps 1-3 twice with `--extraction-only` and the env above (distinct
`--session-id` and `--json-output` paths, fresh `mist-neo4j-eval` instance each),
and diff the two score JSON reports on the aggregate metrics AND the `per_probe`
FP/FN arrays -- on the extraction-only path they are byte-identical (certified
2026-06-13). The full-chat path (no flag) instead diverges on 18/60 probes.

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

### Baseline results (60-probe corpus, extraction-only, deterministic) -- AUTHORITATIVE C3 FLOOR

**This is the authoritative C3 regression floor.** F2 is measured EXTRACTION-ONLY:
the production extraction pipeline (subject-scope classifier -> ontology
extraction -> validation -> curation/graph write) runs per probe utterance with
NO chat-reply generation and NO same-turn assistant reply in the extraction
context. The chat reply is conversational noise the gold does not encode and is
the sole source of F2 nondeterminism (flash-attn FP noise on long greedy
generations -- see the superseded full-chat band below for the localized
evidence); the extraction calls themselves are deterministic at temperature 0.
Two extraction-only replays on fresh isolated quads are byte-identical, so a
single run is now a valid gate -- no band needed.

**Methodology.** Per probe, `mist_admin.py replay --extraction-only` invokes the
ConversationHandler's production extraction entry point (`_extract_knowledge_async`
-> `ExtractionPipeline.extract_from_utterance`) directly, after the same Step 0
vault-path pre-allocation and event-store turn record `handle_message` performs --
but with `conversation_history=[]`, `assistant_message=""`, and NO `chat.initial`
/ `chat.final` call. The same `extraction.ontology` / `extraction.scope_classifier`
`llm_call` debug records are emitted (via the instrumented provider), so
`score_extraction_run.py` consumes them unchanged. The 60 gold probes are
single-utterance and self-contained, so an empty conversation history is the
faithful extraction input.

**Determinism certification.** Two full 60-probe extraction-only replays on FRESH
isolated quads (sessions `ext-only-d1` / `ext-only-d2`, distinct throwaway
trios + a fresh `mist-neo4j-eval` instance each; `LLM_TEMPERATURE=0.0
LLM_CONVERSATION_TEMPERATURE=0.0 PYTHONHASHSEED=0
MIST_FIXED_CLOCK=2026-06-13T00:00:00+00:00`), scored WITHOUT `--strict`. The two
score JSONs are BYTE-IDENTICAL (SHA-256 `ee59d59e...` on both) on aggregate
metrics AND on every `per_probe` FP/FN array (0 diffs across all 60 probes). The
debug logs carry 60 `extraction.ontology` + 60 `extraction.scope_classifier`
records and ZERO `chat.*` records, confirming no reply was generated. This is the
determinism certification the full-chat path could not achieve (it diverged on
18/60 probes). Artifacts: `data/runtime/extraction-ext-only-d{1,2}.{jsonl,json}`
+ `-report.md`.

Gemma 4 E4B Q5_K_M, ontology v1.2.1, branch `feat/subproject-a-c3-extraction-accuracy`
@ `f1b6bcb`, seed graph, isolated quad. All 60 probes matched in the debug log.

| Metric | Value | k/n | Gate | Pass |
|---|---|---|---|---|
| Matched probes | 60/60 | 60/60 | == total | PASS |
| Entity precision | 0.769 | 103/134 | >= 0.90 | FAIL |
| Entity recall | 0.873 | 103/118 | >= 0.80 | PASS |
| Relationship precision | 0.672 | 45/67 | >= 0.90 | FAIL |
| Relationship recall | 0.714 | 45/63 | >= 0.80 | FAIL |
| Typing accuracy | 0.761 | 51/67 | >= 0.90 | FAIL |
| RELATED_TO rate | 0.000 | 0/67 | <= 0.10 | PASS |
| Valid-time accuracy | 0.875 | 7/8 | (tracked) | - |
| Negative-control violations | 0 | 0 | == 0 | PASS |

Reconciliation-action accuracy: SKIPPED (requires C2 telemetry).

(k/n: entity P denominator = produced entities, recall denominator = gold
entities; rel analogous; typing = constraint-valid of produced rels; RELATED_TO
= default-predicate edges of produced rels; valid-time = correct of date-bearing
gold edges on matched probes.)

Reading: below-gate on entity/rel precision, rel recall, and typing -- this is
the FLOOR the remaining C3 tasks improve, not a gate-passing result (gates are
provisional until C3 tuning; see "Metrics + provisional gates"). Because the run
is byte-reproducible, every gap direction is exact -- there is no measurement-noise
band to reason around. Negative-control violations are 0 here (vs 1 on the
full-chat band): removing the chat reply removes the contamination that made
Gemma mine entities from a directive/question/hypothetical control via the reply
text, so the negative controls now pass cleanly.

**Per-category FP/FN breakdown (byte-identical across d1/d2).** From the `per_probe`
diagnostics; counts are absolute FP/FN tallies summed within the category.

| Category | n | entity FP | entity FN | rel FP | rel FN |
|---|---|---|---|---|---|
| Core user facts | 10 | 2 | 0 | 1 | 2 |
| Events + dates | 5 | 3 | 3 | 4 | 4 |
| Quantified | 3 | 3 | 2 | 1 | 1 |
| Skill state | 5 | 3 | 2 | 0 | 0 |
| Third-party | 5 | 3 | 2 | 9 | 7 |
| Valid-time | 5 | 4 | 0 | 0 | 0 |
| Negative controls | 6 | 0 | 0 | 0 | 0 |
| Structural | 3 | 3 | 3 | 0 | 0 |
| Cessation | 5 | 1 | 0 | 0 | 0 |
| Retraction | 5 | 0 | 0 | 0 | 0 |
| Same-turn pair | 2 | 1 | 0 | 1 | 0 |
| Habit / recurrence | 3 | 5 | 3 | 4 | 3 |
| Date-discrimination | 3 | 3 | 0 | 2 | 1 |
| TOTAL | 60 | 31 | 15 | 22 | 18 |

The load-bearing categories are Third-party (rel FP 9 / rel FN 7), Habit/recurrence
(entity FP 5, rel FP 4 / FN 3), Events+dates, and Structural (entity FP/FN 3 each).
Because the run is deterministic, these integers are exact, not directional.

Deliberate-FN note (RECOMMENDS / HAS_HABIT): the RECOMMENDS gold edges (ext-31,
ext-32) and HAS_HABIT gold edges (ext-55, ext-56, ext-57) are EXPECTED relationship
false-negatives until the ontology gains those predicates at T10. They sit inside
the Third-party rel-FN and Habit/recurrence rows above. Measured on purpose; T10
recovers them. Do NOT read those FNs as an extraction-quality regression.

**Deliberate deviation from production (known measurement choice).** Production
runs extraction with the just-generated assistant reply embedded in the extraction
"Context:" block; the extraction-only harness omits it (empty
`conversation_history` + empty `assistant_message`). This is intentional: the reply
is conversational noise the gold does not encode AND is the sole source of
nondeterminism. The trade-off is that the baseline measures extraction over a
no-reply context rather than production's with-reply context. For the 60
single-utterance gold probes this is faithful (the gold labels are authored from
the utterance alone, never the reply), and it buys byte-reproducible single-run
gating. The full-chat path remains available (`replay` without `--extraction-only`)
for any future measurement that wants the with-reply context.

**Assertion-kind buckets (T4 -- engine-shared derivation).** Scored on the SAME
certified extraction-only run (`ext-only-d1`; byte-identical on `ext-only-d2` and
across re-scores, so the bucket table inherits the floor's reproducibility). The
kind is whatever the engine derives: the scorer imports
`reconciliation.derive_assertion_kind` (anti-drift -- no parallel
reimplementation, same pattern as `RELATIONSHIP_CONSTRAINTS`). Per kind:
`gold_total` = gold edges of that kind; `found` = gold edges of that kind
extracted at all (relationship-TP membership); `correct` = found edges whose
engine-derived kind matches gold. `found` is tracked explicitly because the
~12 cease+retract gold edges in 63 could be omitted wholesale and still clear the
0.80 overall rel-recall gate -- accuracy-over-found alone would be blind to that.

| Kind | correct/found | found/gold_total |
|---|---|---|
| assert | 28/33 | 33/51 |
| cease | 6/7 | 7/7 |
| retract | 0/5 | 5/5 |

Reading (the gap T5 closes): this is the PRE-prompt-change baseline -- no produced
edge carries an explicit `assertion_kind`, so every kind is whatever the engine
INFERS from `temporal_status`. `cease` already lands 6/7: the model marks these
"stopped using X" utterances `temporal_status=past`, and the engine's interim
past + no-end_date -> CEASE mapping recovers them. `retract` is the headline gap:
all 5 retraction edges are EXTRACTED (found 5/5 -- "I never used Z" produces the
USES edge structurally, so it is a rel-TP), but NONE derives as retract
(correct 0/5) -- without the prompt emitting `assertion_kind=retract` there is no
past-tense signal and `derive_assertion_kind` returns ASSERT for all five. The
edges are present; only the kind label is missing. `assert` is 28/33 correct: of
33 found assert-gold edges, 5 are mis-derived as CEASE (the model tagged a
present-tense fact `temporal_status=past`, tripping the interim coercion) -- a
second class of label error T5's explicit-kind emission removes. The 33/51 found
reflects the 18 assert-gold rel-FNs already in the floor's rel-recall, not a
bucket-specific miss. T5 (prompt emits explicit `assertion_kind`) is measured
against exactly these three rows.

### Phase B checkpoint (prompt r2) -- assertion_kind buckets on the r2 prompt

Phase B (the `assertion_kind` signal: extraction prompt r2 emitting the field
explicitly + engine gate + bucket scoring + same-turn arbitration) is committed
(branch `feat/subproject-a-c3-extraction-accuracy` @ `3162927`, the r2 prompt
commit region). Re-measurement of the SAME deterministic extraction-only 60-probe
F2 instrument on the r2 prompt. Two replays on FRESH isolated quads (sessions
`ext-c3-phb-d1` / `ext-c3-phb-d2`, distinct throwaway trios + a fresh
`mist-neo4j-eval` instance each; full pin `LLM_TEMPERATURE=0.0
LLM_CONVERSATION_TEMPERATURE=0.0 PYTHONHASHSEED=0
MIST_FIXED_CLOCK=2026-06-13T00:00:00+00:00` + `--extraction-only`), scored
WITHOUT `--strict`.

**Determinism still holds on r2 (kernel property unchanged by the prompt edit).**
The two score JSONs are BYTE-IDENTICAL (SHA-256
`8b257442222bf16362eec82e75ab3a4e7b80b3d02c645fbdde9ef5d6d473eea2` on both) on
aggregate metrics AND the full 60-entry `per_probe` FP/FN arrays AND
`assertion_kind_buckets`. The debug logs carry 60 `extraction.ontology` + 60
`extraction.scope_classifier` records and ZERO `chat.*` records. Artifacts:
`data/runtime/extraction-ext-c3-phb-d{1,2}.{jsonl,json}` + `-report.md`.

Gemma 4 E4B Q5_K_M, ontology v1.2.1, seed graph, isolated quad. All 60 probes
matched.

| Metric | r2 | k/n (r2) | baseline | delta |
|---|---|---|---|---|
| Matched probes | 60/60 | 60/60 | 60/60 | 0 |
| Entity precision | 0.795 | 101/127 | 0.769 | +0.026 |
| Entity recall | 0.856 | 101/118 | 0.873 | -0.017 |
| Relationship precision | 0.734 | 47/64 | 0.672 | +0.062 |
| Relationship recall | 0.746 | 47/63 | 0.714 | +0.032 |
| Typing accuracy | 0.828 | 53/64 | 0.761 | +0.067 |
| RELATED_TO rate | 0.000 | 0/64 | 0.000 | 0 |
| Valid-time accuracy | 0.875 | 7/8 | 0.875 | 0 |
| Negative-control violations | 0 | 0 | 0 | 0 |

(k/n: entity P denominator = produced entities, recall denominator = gold
entities; rel analogous; typing = constraint-valid of produced rels; RELATED_TO
= default-predicate edges of produced rels; valid-time = correct of date-bearing
gold edges on matched probes.) No regressions: relationship precision and recall
both up, no neg-control increase, no new RELATED_TO.

**Assertion-kind buckets (r2).** Scored on the same certified extraction-only run
(`ext-c3-phb-d1`; byte-identical on `d2`). Kind is whatever the engine derives
via the shared `derive_assertion_kind` (anti-drift). On r2 the model emits
`assertion_kind` explicitly inside `properties` -- all 12 found cease/retract
edges carry the correct explicit label, so the kind is driven by the r2 signal,
not the interim `temporal_status=past -> CEASE` mapping.

| Kind | correct/found (r2) | found/gold_total (r2) | correct/found (baseline) | found/gold_total (baseline) |
|---|---|---|---|---|
| assert | 34/35 | 35/51 | 28/33 | 33/51 |
| cease | 7/7 | 7/7 | 6/7 | 7/7 |
| retract | 5/5 | 5/5 | 0/5 | 5/5 |

Reading: `retract` moved OFF 0 to perfect (0/5 -> 5/5 correct) -- the r2 prompt
labels "never actually / misspoke / scratch that / was wrong about / correction:"
as `retract`, which the explicit-field branch of `derive_assertion_kind` then
honors. `cease` strengthened to perfect (6/7 -> 7/7). `assert` improved (28/33
-> 34/35 correct): the 5 pre-r2 present-tense-fact-mis-coerced-to-CEASE errors
are gone; the single remaining mis-derivation is ext-35-validtime-until
(`user -LEARNING-> go`), where the model emitted `assertion_kind=cease` +
`temporal_status=past` + `end_date=2025-12-01` on a bounded "until" probe (read
the stated end as a cessation). Every cease/retract/same-turn-pair gold edge is
both extracted (rel-TP) and correctly kinded -- 0 NOT-FOUND, 0 MIS-KIND in those
buckets. Determinism confirmed: both runs byte-identical (sha256 above).

### Phase C checkpoint (prompt r3) -- rel precision + RECOMMENDS/HAS_HABIT + date-entity

Phase C (the precision levers: ontology v1.3.0 adding RECOMMENDS + HAS_HABIT and
retiring started/ended/duration props at T10; extraction prompt r3 adding the
date-entity discrimination rule + recommendation/habit extraction rules +
Examples 23/24 at T11; the scorer-parity sync) is committed (branch
`feat/subproject-a-c3-extraction-accuracy`, HEAD region `c49bec1` prompt r3 +
`d17e9bb` scorer parity). Re-measurement of the SAME deterministic
extraction-only 60-probe F2 instrument on the r3 prompt + v1.3.0 ontology. Two
replays on FRESH isolated quads (sessions `ext-c3-phc-d1` / `ext-c3-phc-d2`,
distinct throwaway trios + a FRESH `mist-neo4j-eval` instance each -- the eval
Neo4j was torn down and recreated between d1 and d2 so both replays start from an
identical 138-write / 32-embedding seed; full pin `LLM_TEMPERATURE=0.0
LLM_CONVERSATION_TEMPERATURE=0.0 PYTHONHASHSEED=0
MIST_FIXED_CLOCK=2026-06-13T00:00:00+00:00` + `--extraction-only`). d1 scored
WITHOUT `--strict`; d2 scored WITH `--strict` (exit 1 -- the EXPECTED gate-miss
signal, NOT an error: rel precision 0.831 < 0.90 headline gate; the report is
captured regardless).

**Determinism still holds on r3 (kernel property unchanged by the prompt +
ontology edits).** The two score JSONs are BYTE-IDENTICAL (SHA-256
`a7c00a370634c4bc5edf6b200c7de145778b8c05eef4572dba51ce44656d50a1` on both) on
aggregate metrics AND the full 60-entry `per_probe` FP/FN arrays AND
`assertion_kind_buckets` (`diff -q` empty; single byte-identical JSON covers all
three). The debug logs carry 60 `extraction.ontology` + 60
`extraction.scope_classifier` records and ZERO `chat.*` records. Artifacts:
`data/runtime/extraction-ext-c3-phc-d{1,2}.{jsonl,json}` + `-report.md`.

Gemma 4 E4B Q5_K_M, ontology v1.3.0, seed graph, isolated quad. All 60 probes
matched.

| Metric | r3 | k/n (r3) | r2 | delta r2 | r1 baseline | delta r1 | Gate | Pass |
|---|---|---|---|---|---|---|---|---|
| Matched probes | 60/60 | 60/60 | 60/60 | 0 | 60/60 | 0 | == total | PASS |
| Entity precision | 0.855 | 106/124 | 0.795 | +0.060 | 0.769 | +0.086 | >= 0.90 | FAIL |
| Entity recall | 0.898 | 106/118 | 0.856 | +0.042 | 0.873 | +0.025 | >= 0.80 | PASS |
| Relationship precision | 0.831 | 54/65 | 0.734 | +0.097 | 0.672 | +0.159 | >= 0.90 | FAIL |
| Relationship recall | 0.844 | 54/64 | 0.746 | +0.098 | 0.714 | +0.130 | >= 0.80 | PASS |
| Typing accuracy | 0.892 | 58/65 | 0.828 | +0.064 | 0.761 | +0.131 | >= 0.90 | FAIL |
| RELATED_TO rate | 0.000 | 0/65 | 0.000 | 0 | 0.000 | 0 | <= 0.10 | PASS |
| Valid-time accuracy | 0.875 | 7/8 | 0.875 | 0 | 0.875 | 0 | (tracked) | - |
| Negative-control violations | 0 | 0 | 0 | 0 | 0 | 0 | == 0 | PASS |

Reconciliation-action accuracy: SKIPPED (requires C2 telemetry).

(k/n: entity P denominator = produced entities, recall denominator = gold
entities; rel analogous; typing = constraint-valid of produced rels; RELATED_TO
= default-predicate edges of produced rels; valid-time = correct of date-bearing
gold edges on matched probes.) r1 = the extraction-only authoritative floor
(prompt r1); r2 = the Phase B checkpoint (prompt r2). The rel-recall denominator
moved 63 -> 64 (and assert bucket gold_total 51 -> 52) because the gold corpus
gained ext-08's `priya -[RECOMMENDS]-> duckdb` edge at commit `82dd50b`
(ontology v1.3.0 / T10) -- the RECOMMENDS predicate did not exist in gold at the
r2 baseline (deliberate-FN placeholder), so this is the intended gold update that
pairs with the T10 ontology change, not a scorer or determinism drift. The F2
scorer (`score_extraction_run.py`) is byte-unchanged since r2; the `d17e9bb`
parity sync only touched the gauntlet `scorers.py` mirror.

No regressions vs r2: all four precision/recall/typing metrics up (rel precision
+0.097 and rel recall +0.098 are the largest movers), no neg-control increase, no
new RELATED_TO. Three gates remain FAIL (entity P 0.855, rel P 0.831, typing
0.892 -- all below the 0.90 bar); the four passing gates (entity R, rel R,
RELATED_TO, neg-control) hold.

**Per-category FP/FN breakdown (byte-identical across d1/d2).** From the
`per_probe` diagnostics; counts are absolute FP/FN tallies summed within the
category. r1 floor counts in parentheses for trajectory.

| Category | n | entity FP | entity FN | rel FP | rel FN |
|---|---|---|---|---|---|
| Core user facts | 10 | 3 | 1 | 0 | 2 |
| Events + dates | 5 | 3 | 3 | 3 | 3 |
| Quantified | 3 | 5 | 4 | 3 | 2 |
| Skill state | 5 | 0 | 0 | 0 | 0 |
| Third-party | 5 | 2 | 1 | 2 | 1 |
| Valid-time | 5 | 1 | 0 | 0 | 0 |
| Negative controls | 6 | 0 | 0 | 0 | 0 |
| Structural | 3 | 1 | 1 | 0 | 0 |
| Cessation | 5 | 0 | 0 | 0 | 0 |
| Retraction | 5 | 0 | 0 | 0 | 0 |
| Same-turn pair | 2 | 1 | 0 | 1 | 0 |
| Habit / recurrence | 3 | 1 | 1 | 1 | 1 |
| Date-discrimination | 3 | 1 | 1 | 1 | 1 |
| TOTAL | 60 | 18 | 12 | 11 | 10 |

Trajectory vs the r1 floor (rel FP 22 / FN 18; entity FP 31 / FN 15): Third-party
collapsed from rel FP 9 / FN 7 to FP 2 / FN 1 (RECOMMENDS landed), Habit from rel
FP 4 / FN 3 to FP 1 / FN 1 (HAS_HABIT landed), Skill state / Cessation /
Retraction are now fully clean. The residual rel load is in Events+dates and
Quantified, where the gap is entity-surface/typing mismatch driving the matched
edge off its gold tuple rather than a missing predicate.

**Per-probe rel FP/FN diagnostics (all 11 FP + 10 FN; cause class).** The rel-key
match is exact on `(source, predicate, target)`, so an entity-surface or
entity-type mismatch on either endpoint splits one logical edge into a paired
FP+FN. Cause classes: `entity-surface` = right predicate, endpoint canonical id
differs from gold (e.g. `12000-requests-per-second` vs `requests-per-second-12000`);
`entity-typing` = right id, wrong entity type (Milestone vs Project etc.) shifts
the typing but the tuple key still matches unless the id also moved; `over-extract`
= an extra edge with no gold counterpart; `anchor-miss` = the user-sourced anchor
edge absent; `mis-predicate` = wrong predicate chosen.

| Probe | FP/FN | Edge (source -[pred]-> target) | Cause class |
|---|---|---|---|
| ext-01-uses | FN | rust -[USED_FOR]-> backend-work | non-anchor USED_FOR miss (2nd edge dropped) |
| ext-14-uses-usedfor | FN | terraform -[USED_FOR]-> cloud-infrastructure | entity-typing (target Concept vs gold Topic shifts key) |
| ext-20-event-on-date | FN | user -[EXPERIENCED]-> birthday-party | anchor-miss (user entity FN; EXPERIENCED anchor dropped) |
| ext-21-milestone-on-date | FP | mobile-app -[OCCURRED_ON]-> 2026-06-01 | entity-surface (Project surface `mobile-app` vs gold Milestone `mobile-app-launch`) |
| ext-21-milestone-on-date | FN | mobile-app-launch -[OCCURRED_ON]-> 2026-06-01 | entity-surface (paired with FP above) |
| ext-22-milestone-on-date | FP | closing-on-house -[OCCURRED_ON]-> 2026-03-12 | entity-surface (Event surface vs gold Milestone `house-closing`) |
| ext-22-milestone-on-date | FP | user -[EXPERIENCED]-> closing-on-house | over-extract (EXPERIENCED on the mis-typed Event surface) |
| ext-22-milestone-on-date | FN | house-closing -[OCCURRED_ON]-> 2026-03-12 | entity-surface (paired with FP above) |
| ext-24-metric | FP | api -[HAS_METRIC]-> requests-per-second-12000 | entity-surface (metric id ordering vs gold `12000-requests-per-second`) |
| ext-24-metric | FN | api -[HAS_METRIC]-> 12000-requests-per-second | entity-surface (paired with FP above) |
| ext-25-metric | FP | coverage-87-percent -[APPLICABLE_TO]-> codebase | over-extract (spurious APPLICABLE_TO) |
| ext-25-metric | FP | test-suite -[HAS_METRIC]-> coverage-87-percent | entity-surface (metric id `coverage-87-percent` vs gold `87-percent-coverage`) |
| ext-25-metric | FN | test-suite -[HAS_METRIC]-> 87-percent-coverage | entity-surface (paired with FP above) |
| ext-29-third-party-knows | FP | anjali -[WORKS_AT]-> biotech | over-extract (WORKS_AT not in gold; scope creep on third party) |
| ext-29-third-party-knows | FN | user -[KNOWS_PERSON]-> anjali | anchor-miss (user entity FN; KNOWS_PERSON anchor dropped) |
| ext-31-third-party-recommends | FP | graphql -[STRATEGY_FOR]-> new-service | over-extract (RECOMMENDS + KNOWS_PERSON gold edges BOTH correct; this is a spurious 3rd edge) |
| ext-54-same-turn-single-pair | FP | user -[WORKS_ON]-> project | over-extract (generic `project` entity; not in gold) |
| ext-56-habit | FP | user -[HAS_HABIT]-> pull-requests | entity-surface (HAS_HABIT fires; target `pull-requests` Topic vs gold `pr-review` Concept) |
| ext-56-habit | FN | user -[HAS_HABIT]-> pr-review | entity-surface (paired with FP above) |
| ext-58-dateentity-stative-since | FP | user -[HAS_HABIT]-> mentoring-junior-engineers | mis-predicate (emitted HAS_HABIT; gold is EXPERT_IN mentoring) |
| ext-58-dateentity-stative-since | FN | user -[EXPERT_IN]-> mentoring | mis-predicate (paired with FP above) |

Dominant cause class: entity-surface mismatch (metric id ordering, Milestone-vs-
Project/Event surface) -- 7 of the 11 rel FPs and 7 of the 10 rel FNs are
surface/typing-driven paired FP+FN, where the predicate is correct but an endpoint
id/type differs from gold and splits the key. Genuine over-extractions (no gold
counterpart): ext-22 EXPERIENCED, ext-25 APPLICABLE_TO, ext-29 WORKS_AT, ext-31
STRATEGY_FOR, ext-54 WORKS_ON -- 5 FPs. Anchor-edge misses: ext-20, ext-29 -- 2
FNs. One genuine mis-predicate: ext-58 (HAS_HABIT chosen over EXPERT_IN on a
"since <date>" stative -- the new HAS_HABIT predicate over-firing on a
non-recurrence stative). The ext-35 "until <date>" cease-vs-assert case from the
r2 buckets is NOT a rel FP/FN here (ext-35 is rel-CLEAN); it only ever surfaced as
a bucket mis-kind, and the valid-time miss (7/8) is a date-precision mismatch on a
single date-bearing edge, not a rel-key FP/FN.

**Assertion-kind buckets (r3).** Scored on the same certified extraction-only run
(`ext-c3-phc-d1`; byte-identical on `d2`).

| Kind | correct/found (r3) | found/gold_total (r3) | correct/found (r2) | found/gold_total (r2) | correct/found (r1) | found/gold_total (r1) |
|---|---|---|---|---|---|---|
| assert | 42/42 | 42/52 | 34/35 | 35/51 | 28/33 | 33/51 |
| cease | 7/7 | 7/7 | 7/7 | 7/7 | 6/7 | 7/7 |
| retract | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 5/5 |

Reading: no bucket regresses below Phase B. `cease` and `retract` hold perfect
(7/7, 5/5 found AND correct). `assert` correct/found is now PERFECT (42/42 vs r2's
34/35) -- the single r2 mis-derivation (ext-35-validtime-until kinded CEASE) is
gone; every found assert-gold edge derives as assert. assert `found/gold_total`
rose 35/51 -> 42/52: the gold denominator gained one edge (ext-08 RECOMMENDS, see
above) and `found` rose by 7 (the RECOMMENDS/HAS_HABIT edges that were structural
FNs at r2 are now extracted, lifting rel recall). The 10 still-unfound assert-gold
edges (52 - 42) are the rel-FN set above (entity-surface/typing splits + anchor
misses + the ext-58 mis-predicate), not a bucket-kind failure.

**The 6 RECOMMENDS / HAS_HABIT probes (per-probe).** All 6 now emit the new
predicate (none defaults to RELATED_TO; KNOWS_PERSON anchor present on all
recommendation probes). 4 of 6 are fully CLEAN (every gold rel extracted, no FP);
the 2 imperfect ones emit the CORRECT predicate and differ only on an endpoint or
an extra edge:

| Probe | Utterance | Gold edges | Result |
|---|---|---|---|
| ext-08-third-party-person | "My mentor Priya suggested I try DuckDB" | user-[KNOWS_PERSON]->priya; priya-[RECOMMENDS]->duckdb | CLEAN (both edges TP) |
| ext-31-third-party-recommends | "My architect Mateo recommended GraphQL for the new service" | user-[KNOWS_PERSON]->mateo; mateo-[RECOMMENDS]->graphql | RECOMMENDS + anchor BOTH correct; 1 FP = spurious graphql-[STRATEGY_FOR]->new-service |
| ext-32-third-party-recommends | "A teammate named Lena pushed me toward trying Bun" | user-[KNOWS_PERSON]->lena; lena-[RECOMMENDS]->bun | CLEAN (both edges TP) |
| ext-55-habit | "I journal every night before bed" | user-[HAS_HABIT]->journaling | CLEAN (TP) |
| ext-56-habit | "I review PRs every Friday" | user-[HAS_HABIT]->pr-review | HAS_HABIT fires; target `pull-requests` (Topic) vs gold `pr-review` (Concept) -- 1 FP + 1 FN (entity-surface) |
| ext-57-habit | "I do a long run every Sunday" | user-[HAS_HABIT]->long-run | CLEAN (TP) |

The deliberate baseline FNs (RECOMMENDS ext-31/32, HAS_HABIT ext-55/56/57, plus
ext-08 RECOMMENDS) are recovered at the predicate level: T10 + r3 turned all six
from structural rel-FNs into extracted edges. The 2 residual gaps (ext-31 extra
STRATEGY_FOR edge, ext-56 target-surface) are over-extraction / entity-surface
classes, not predicate misses.

### Phase C iteration (prompt r4) -- precision-rule re-measure

Path-A disciplined iteration on the r3 gate miss. r4 adds two GENERALIZABLE
precision rules (no probe names in the prompt): Edit A tightens the HAS_HABIT
clause (Rule 17) so a continuous "since <date>" stative routes to the matching
stative predicate (EXPERT_IN / LEARNING / INTERESTED_IN) with valid_from instead
of over-firing HAS_HABIT; Edit B adds Rule 18 (extract the asserted fact, not
incidental trailing-prepositional scope -- do NOT manufacture STRATEGY_FOR /
APPLICABLE_TO / WORKS_ON / PART_OF / USED_FOR for a scoping phrase; the
"Rust USED_FOR backend-work IS the fact" contrast protects legitimate USED_FOR
recall). Committed on `feat/subproject-a-c3-extraction-accuracy` (r4 prompt commit
`0b6d5b7`; extraction_version `2026-06-12-r4`, prompt sha256 re-pinned to
`38f379f757f40f69e8da583cb6b5342fb8bf5015e512c98ddc38e50f853092a8`). Re-measured
on the SAME deterministic extraction-only 60-probe F2 instrument.

**Determinism still holds on r4.** Two replays on FRESH isolated quads (sessions
`ext-c3-r4-d1` / `ext-c3-r4-d2`, distinct throwaway trios + a FRESH `mist-neo4j-eval`
instance each -- eval Neo4j torn down and recreated between d1 and d2, both from
an identical 138-write / 32-embedding seed; full pin `LLM_TEMPERATURE=0.0
LLM_CONVERSATION_TEMPERATURE=0.0 PYTHONHASHSEED=0
MIST_FIXED_CLOCK=2026-06-13T00:00:00+00:00` + `--extraction-only`). d1 scored
WITHOUT `--strict`; d2 scored WITH `--strict` (exit 0 -- `--strict` enforces
probe-join integrity + negative-control violations, both clean; it does NOT
hard-fail on the headline precision gates, which remain reported FAIL). The two
score JSONs are BYTE-IDENTICAL (SHA-256
`e368a42076ad6576d96010f680ca2356a26acbe6638d127b7bc8e7e87e37bc4d` on both;
`diff -q` empty) on aggregate metrics AND the full 60-entry `per_probe` FP/FN
arrays AND `assertion_kind_buckets`. The debug logs carry 60 `extraction.ontology`
+ 60 `extraction.scope_classifier` records and ZERO `chat.*` records. Artifacts:
`data/runtime/extraction-ext-c3-r4-d{1,2}.{jsonl,json}` + `-report.md`.

Gemma 4 E4B Q5_K_M, ontology v1.3.0, seed graph, isolated quad. All 60 probes
matched.

| Metric | r4 | k/n (r4) | r3 | delta r3 | Gate | Pass |
|---|---|---|---|---|---|---|
| Matched probes | 60/60 | 60/60 | 60/60 | 0 | == total | PASS |
| Entity precision | 0.831 | 103/124 | 0.855 | -0.024 | >= 0.90 | FAIL |
| Entity recall | 0.873 | 103/118 | 0.898 | -0.025 | >= 0.80 | PASS |
| Relationship precision | 0.812 | 52/64 | 0.831 | -0.019 | >= 0.90 | FAIL |
| Relationship recall | 0.812 | 52/64 | 0.844 | -0.032 | >= 0.80 | PASS |
| Typing accuracy | 0.875 | 56/64 | 0.892 | -0.017 | >= 0.90 | FAIL |
| RELATED_TO rate | 0.000 | 0/64 | 0.000 | 0 | <= 0.10 | PASS |
| Valid-time accuracy | 0.875 | 7/8 | 0.875 | 0 | (tracked) | - |
| Negative-control violations | 0 | 0 | 0 | 0 | == 0 | PASS |

Reconciliation-action accuracy: SKIPPED (requires C2 telemetry).

(k/n: entity P denominator = produced entities, recall denominator = gold
entities; rel analogous; typing = constraint-valid of produced rels; RELATED_TO
= default-predicate edges of produced rels; valid-time = correct of date-bearing
gold edges on matched probes.)

Reading: ALL FOUR targeted genuine errors are eliminated at the structural-edge
level -- ext-25 APPLICABLE_TO, ext-31 STRATEGY_FOR, ext-54 WORKS_ON (Rule 18) and
ext-58 HAS_HABIT (Rule A, now correctly EXPERT_IN) are all gone from the FP set.
But every aggregate metric REGRESSED slightly vs r3 (rel P -0.019, rel R -0.032,
entity P -0.024). The regression does NOT come from the two new rules: it is
prompt-hash-perturbation drift on THREE probes that were rel-CLEAN at r3 and are
NOT logical cases for either rule -- ext-23 ("wedding on 2026-05-30": the model
chose entity id `wedding-event` vs gold `wedding-2026-05-30`, splitting the
EXPERIENCED + OCCURRED_ON pair into 2 FP + 2 FN, pure entity-surface), ext-30
("had coffee with Devin": new over-extraction of a `coffee-meeting` Event +
EXPERIENCED edge), and ext-35 ("learning Go until Dec 2025": new spurious
`go OCCURRED_ON 2025-12-01` + Date entity, a Rule-16 date-discrimination miss).
Changing the prompt text shifts the greedy argmax on these near-tie probes -- the
same flash-attn-FP-near-tie sensitivity documented for the full-chat band, here
triggered by the prompt-string delta rather than generation length. The net trade
is 5 fixed FPs (4 targeted spurious edges + 1 ext-25 metric-surface variant) for
4 new drift edges (ext-23 ×2, ext-30, ext-35) plus 2 dropped assert-found edges
(ext-23 surface split), netting -2 rel TP / +1 rel FP / +2 rel FN.

**RECALL-SAFETY (Rule 18): NO new structural-edge false-negative.** ext-01
(`rust USED_FOR backend-work`) was ALREADY a rel-FN at r3 (the model emits only
the `user USES rust` anchor and drops the 2nd USED_FOR edge); it remains FN at r4
with byte-identical behavior -- Rule 18's protective contrast did not suppress it,
and it was never extracted to begin with. ext-14 (`terraform USED_FOR
cloud-infrastructure`) is FN in BOTH r3 and r4 (entity-typing split, target
Concept vs gold Topic). No USED_FOR / PART_OF / structural gold edge that r3
correctly extracted was dropped by r4. The only NEW rel-FNs are the ext-23
entity-surface pair (drift, not a Rule-18 suppression).

**Per-probe verification of the 4 targeted probes (r4):**

| Probe | r3 error | r4 result | Verdict |
|---|---|---|---|
| ext-58 | `user HAS_HABIT mentoring-junior-engineers` (mis-predicate) | HAS_HABIT GONE; emits `user EXPERT_IN mentoring-junior-engineers`; gold `user EXPERT_IN mentoring` -> entity-surface split (`mentoring-junior-engineers` Concept vs `mentoring` Skill) | PARTIAL (predicate fixed; residual is canonicalization) |
| ext-25 | spurious `coverage-87-percent APPLICABLE_TO codebase` | APPLICABLE_TO GONE; residual `test-suite HAS_METRIC 87-percent` vs gold `87-percent-coverage` is entity-surface only | RESOLVED (targeted spurious edge gone) |
| ext-31 | spurious `graphql STRATEGY_FOR new-service` | STRATEGY_FOR GONE; ext-31 has NO rel FP (KNOWS_PERSON + RECOMMENDS both correct) -> fully CLEAN | RESOLVED |
| ext-54 | spurious `user WORKS_ON project` | WORKS_ON GONE; ext-54 has NO rel FP (only a stray `project` entity FP remains) | RESOLVED (targeted structural edge gone) |

**Assertion-kind buckets (r4).** Scored on the same certified run (`ext-c3-r4-d1`;
byte-identical on `d2`).

| Kind | correct/found (r4) | found/gold_total (r4) | correct/found (r3) | found/gold_total (r3) |
|---|---|---|---|---|
| assert | 39/40 | 40/52 | 42/42 | 42/52 |
| cease | 7/7 | 7/7 | 7/7 | 7/7 |
| retract | 5/5 | 5/5 | 5/5 | 5/5 |

Reading: `cease` and `retract` hold perfect (7/7, 5/5) -- no regression below
Phase B. `assert` correct/found dipped 42/42 -> 39/40: `found` dropped by 2 (the
ext-23 surface-split pair is no longer rel-TP, so 2 assert-gold edges fell out of
found), and 1 mis-derivation appeared (ext-35: r4 emitted `assertion_kind=cease`
+ `temporal_status=past` on the assert-gold "until Dec 2025" LEARNING edge, the
same "until -> read as cease" class seen in the r2 buckets). Both are the drift
probes above, not a Rule-A/18 consequence.

**Per-category FP/FN breakdown (byte-identical across d1/d2).** r3 counts in
parentheses for trajectory.

| Category | n | entity FP | entity FN | rel FP | rel FN |
|---|---|---|---|---|---|
| Core user facts | 10 | 3 (3) | 2 (1) | 0 (0) | 2 (2) |
| Events + dates | 5 | 4 (3) | 4 (3) | 5 (3) | 5 (3) |
| Quantified | 3 | 3 (5) | 2 (4) | 2 (3) | 2 (2) |
| Skill state | 5 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| Third-party | 5 | 2 (2) | 1 (1) | 2 (2) | 1 (1) |
| Valid-time | 5 | 1 (1) | 0 (0) | 1 (0) | 0 (0) |
| Negative controls | 6 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| Structural | 3 | 3 (1) | 3 (1) | 0 (0) | 0 (0) |
| Cessation | 5 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| Retraction | 5 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| Same-turn pair | 2 | 1 (1) | 0 (0) | 0 (1) | 0 (0) |
| Habit / recurrence | 3 | 1 (1) | 1 (1) | 1 (1) | 1 (1) |
| Date-discrimination | 3 | 1 (1) | 1 (1) | 1 (1) | 1 (1) |
| TOTAL | 60 | 19 (18) | 14 (12) | 12 (11) | 12 (10) |

The two targeted categories MOVED in the intended direction at the predicate
level: Same-turn pair rel FP 1 -> 0 (ext-54 WORKS_ON gone), and the ext-58
date-discrimination + ext-25 quantified spurious edges are gone (Quantified rel FP
3 -> 2, entity FP 5 -> 3). The regression is isolated to Events+dates (rel FP/FN
3 -> 5 each: the ext-23 surface split + ext-30 over-extract + ext-35 spurious
OCCURRED_ON all land here) and Structural entity FP/FN 1 -> 3 (ext-41/ext-42
entity-typing drift). None of these is a Rule-A/18 case.

**Remaining FP/FN classified (pure-canonicalization vs other).** Of the 12 rel FPs:
8 are entity-surface paired splits (PURE canonicalization -- ext-21, ext-22
closing-on-house, ext-23 ×2, ext-24, ext-25, ext-56, ext-58); 4 are genuine
over-extractions (ext-22 EXPERIENCED on mis-typed Event, ext-29 WORKS_AT
third-party scope creep, ext-30 EXPERIENCED incidental meeting, ext-35 OCCURRED_ON
date-discrimination miss). Of the 4 over-extractions, 2 are NEW drift (ext-30,
ext-35) and 2 are r3 carryovers (ext-22, ext-29). The 12 rel FNs are the paired
splits of the 8 surface FPs plus the structural under-extractions (ext-01, ext-14
USED_FOR drops; ext-20, ext-29 anchor misses) -- all either canonicalization
(surface/typing) or pre-existing recall gaps. The ~70% canonicalization residual
the controller scoped out of this iteration is confirmed: 8 of 12 rel FP and the
bulk of rel FN are pure entity-surface/typing, unaddressable by precision rules
alone.

### Full-chat reference band (60-probe corpus, 2026-06-13) -- NON-DETERMINISTIC, SUPERSEDED

Superseded by the extraction-only baseline above (the authoritative C3 floor).
Retained for provenance: this is the full-chat replay path (chat reply generated
per turn, then extraction over a context containing that reply). It is NOT
byte-reproducible -- two runs diverge on 18/60 probes -- and the localized evidence
below is what motivated moving F2 to extraction-only.

Status: the MIST chain is REPRODUCIBLE up to the LLM sampler (the clock fix
landed). Byte-identical full-chat extraction scores remain gated by ONE
irreducible source outside MIST's code -- llama-server flash-attention
floating-point nondeterminism at greedy decoding. The band below is a tight
reference (two runs, deltas <= 0.032 on every metric), not a single byte-stable
point.

**What the clock fix achieved (commit `3ca09e4`).** The 2026-06-12 determinism
check diverged on 20/60 probes because a wall-clock `rendered_at` stamped into
the seeded `users/<id>.md` Provenance block reached the chat system prompt (the
always-inject curated profile), differing run-to-run and perturbing the turn-1
greedy reply, which then propagated through conversation history. The fix injects
a clock via DI (`ConversationHandler.now_fn` + `VaultWriter.upsert_user/
upsert_identity/upsert_user_snapshot` accepting `rendered_at`, threaded from the
seed bootstrap; env seam `MIST_FIXED_CLOCK`, see "How to run"). Verified:
- Cheap seed check: two pinned seeds produce BYTE-IDENTICAL `users/user.md` and
  `identity/mist.md`.
- Turn-1 isolation: with the clock pinned, the turn-1 `chat.initial` REQUEST is
  byte-identical across runs (the `rendered_at` line now reads the fixed
  `2026-06-13T00:00:00+00:00` in both). The prompt-facing nondeterminism is gone.

**Residual source (localized, IS the LLM sampler -- not MIST code).** Two full
60-probe replays on fresh isolated quads (sessions `ext-c3-d1` / `ext-c3-d2`,
`LLM_TEMPERATURE=0.0 LLM_CONVERSATION_TEMPERATURE=0.0 PYTHONHASHSEED=0
MIST_FIXED_CLOCK=2026-06-13T00:00:00+00:00`), scored WITHOUT `--strict`. Still
diverge on 18/60 per-probe rows. Per-call_site request/response byte comparison:

| call_site | n | request identical | response identical |
|---|---|---|---|
| extraction.scope_classifier | 60 | 60/60 | 60/60 |
| chat.initial | 60 | 1/60 | 0/60 |
| extraction.ontology | 60 | 0/60 | 0/60 |
| chat.final | 6 | 0/6 | 0/6 |

The turn-1 `chat.initial` REQUEST is now identical (the 1/60) -- the clock fix
worked. But its RESPONSE differs even at temperature 0.0: identical 2738-token
prompt, different greedy continuation (d1 "high-concurrency services / systems-
level tooling", 203 tokens; d2 "API services / embedded systems / CLI tools",
256 tokens). From turn 2 on, the diverged reply rides in conversation history, so
later `chat.initial` requests differ too. Confirmed by diffing the turn-1
`extraction.ontology` request: it differs in EXACTLY the two lines carrying the
embedded `[assistant]:` reply -- every downstream divergence traces back to the
LLM's non-deterministic greedy output, nothing else.

Direct confirmation it is the inference engine, not MIST: sending the byte-
identical turn-1 prompt to `mist-llm` 3x at `temperature 0.0, seed 3407` produced
2 distinct outputs of 3. The seed is irrelevant at greedy (argmax does not
sample), so this is logit-level FP nondeterminism, not RNG. Root cause:
`LLAMA_ARG_FLASH_ATTN=on` -- the CUDA flash-attention kernel uses non-
deterministic floating-point reduction order, so logits vary at the bit level and
a near-tie argmax occasionally flips, cascading into a different continuation. It
is INTERMITTENT (most calls are stable; a 4x repeat may or may not catch it) and
scales with sequence length, which is why the short `extraction.scope_classifier`
calls are 60/60 stable while the long 2700+-token `chat.initial` calls are not.

**Reference band (the C3 regression floor).** Artifacts:
`data/runtime/extraction-c3-d{1,2}.{jsonl,json}` + `-report.md`. The d1 column is
the recorded reference; d2 shows the band. k/n shown for d1.

| Metric | d1 | d2 | delta | k/n (d1) | Gate | d1 Pass |
|---|---|---|---|---|---|---|
| Matched probes | 60/60 | 60/60 | - | 60/60 | == total | PASS |
| Entity precision | 0.741 | 0.750 | 0.009 | 100/135 | >= 0.90 | FAIL |
| Entity recall | 0.847 | 0.864 | 0.017 | 100/118 | >= 0.80 | PASS |
| Relationship precision | 0.687 | 0.662 | 0.025 | 46/67 | >= 0.90 | FAIL |
| Relationship recall | 0.730 | 0.714 | 0.016 | 46/63 | >= 0.80 | FAIL |
| Typing accuracy | 0.806 | 0.838 | 0.032 | 54/67 | >= 0.90 | FAIL |
| RELATED_TO rate | 0.000 | 0.000 | 0.000 | 0/67 | <= 0.10 | PASS |
| Valid-time accuracy | 0.875 | 0.875 | 0.000 | 14/16 | (tracked) | - |
| Negative-control violations | 1 | 1 | 0 | 1 | == 0 | FAIL |

Reconciliation-action accuracy: SKIPPED (requires C2 telemetry).

(k/n: entity P denominator = produced entities, recall denominator = gold
entities; rel analogous; typing = correctly-typed of produced-and-matched rels;
RELATED_TO = default-predicate edges of produced rels; valid-time = correct of
date-bearing probes.)

Reading: these are below-gate on entity/rel precision, rel recall, and typing --
expected. This is the FLOOR the remaining C3 tasks improve, not a gate-passing
result (the gates are provisional until C3 tuning + corpus expansion; see "Metrics
+ provisional gates"). The band (max delta 0.032) is small enough that the
direction of each gap is unambiguous despite the LLM noise.

**Per-category FP/FN breakdown (d1).** From the `per_probe` diagnostics. Counts
are absolute FP/FN tallies summed within the category.

| Category | n | entity FP | entity FN | rel FP | rel FN |
|---|---|---|---|---|---|
| Core user facts | 10 | 2 | 0 | 2 | 2 |
| Events + dates | 5 | 3 | 2 | 4 | 3 |
| Quantified | 3 | 4 | 3 | 1 | 1 |
| Skill state | 5 | 1 | 1 | 0 | 0 |
| Third-party | 5 | 4 | 2 | 7 | 5 |
| Valid-time | 5 | 4 | 0 | 0 | 0 |
| Negative controls | 6 | 3 | 0 | 1 | 0 |
| Structural | 3 | 4 | 4 | 1 | 1 |
| Cessation | 5 | 1 | 0 | 0 | 0 |
| Retraction | 5 | 0 | 2 | 0 | 1 |
| Same-turn pair | 2 | 1 | 0 | 0 | 0 |
| Habit / recurrence | 3 | 5 | 3 | 4 | 3 |
| Date-discrimination | 3 | 3 | 1 | 1 | 1 |

The category tallies sit ON the noise band -- a probe whose chat reply flipped
between d1 and d2 can shift its own category's FP/FN by a count or two. Treat the
shape (which categories carry the load: Third-party rel FP/FN, Habit/recurrence,
Structural entity FN) as directional, not the exact integers.

Deliberate-gap note: the RECOMMENDS gold edges (ext-31, ext-32) and HAS_HABIT
gold edges (ext-55, ext-56, ext-57) are EXPECTED relationship false-negatives
until the ontology gains those predicates at T10 (they sit inside the Third-party
rel-FN and Habit/recurrence rows above). Measured on purpose; T10 recovers them.
Do not read those FNs as an extraction-quality regression.

Negative-control note: d1 shows 1 negative-control violation (a real Rule-8/
Rule-10 model miss where Gemma mined entities from a directive/question/
hypothetical control). This is a C3 PROMPT-TUNING target, not a determinism
issue -- both runs are at 1, and the violation is the negative controls doing
their job. (The prior 2026-06-12 pinned run showed 2; the candidates were ext-38
question-form and ext-39 hypothetical.)

Secondary finding (tool-schema enum ordering): `PYTHONHASHSEED=0` neutralizes a
set-ordering nondeterminism in the chat tool-schema enum serialization (a
`display_hint` enum order flipped between unseeded runs). With the seed pinned,
`extraction.scope_classifier` is 60/60 byte-identical. The underlying unordered-
iteration in the tool-schema serialization is a latent PRODUCTION reproducibility
bug tracked separately -- masked here, not fixed.

**Path to a byte-stable single-run baseline (infra, out of scope for the clock
fix).** The clock fix is the complete CODE-side fix; the residual is an inference-
engine property. To get det1 == det2 byte-for-byte, one of:
1. Restart `mist-llm` with `LLAMA_ARG_FLASH_ATTN=off` for eval runs. Trade-off:
   the baseline would then measure extraction over a flash-attn-OFF assistant
   context, which is NOT production (production runs flash-attn ON), and CUDA non-
   flash attention is itself not guaranteed bit-reproducible. Needs validation.
2. Run extraction-eval LLM calls on a CPU/deterministic-kernel build for the
   baseline (slow, but bit-stable).
3. Accept the band: gate on the d1/d2 envelope (e.g. require both runs above the
   floor) rather than a single byte-identical number. The band is tight (<= 0.032)
   and the gap directions are unambiguous, so this is defensible for a regression
   floor.
None of these is a DI/code change, so they are deferred to an infra decision. The
clock fix removed the only MIST-code nondeterminism source; the full-chat chain is
reproducible up to the LLM sampler.

RESOLVED a different way (2026-06-13): the extraction-only path achieves
byte-stable single-run scoring WITHOUT any of the three infra options above. It
removes the long `chat.initial` generation entirely -- the only call long enough
for the flash-attn FP noise to flip an argmax -- leaving only the short, robust
`extraction.scope_classifier` + `extraction.ontology` calls (60/60 byte-identical
historically, and the full score JSON is byte-identical here). flash-attn stays ON
(production-faithful for the extraction calls). See "Baseline results (60-probe
corpus, extraction-only, deterministic)".

### Superseded baseline (2026-06-10, pre-correction corpus -- NON-COMPARABLE)

Entity P 0.652 / R 1.000, rel P 0.750 / R 0.818, typing 0.833, RELATED_TO
0.000, valid-time 1.000, 0 negative violations. Ontology v1.1.0. Retained for
history only; the entity-precision FAIL and typing FAIL were measured against
anchor-less gold labels and the pre-C1 hardcoded constraint table.

## Known limitations
- F2 replay reproducibility is DONE and byte-stable via the EXTRACTION-ONLY path
  (`replay --extraction-only`, 2026-06-13): two full 60-probe runs on fresh
  isolated quads produce BYTE-IDENTICAL score JSONs (aggregate + per_probe). F2 no
  longer needs a band -- a single run gates. This is the authoritative C3 floor.
  The extraction-only path removes the long `chat.initial` generation (the sole
  flash-attn-sensitive call) and keeps flash-attn ON for the short extraction
  calls. Deliberate deviation: the production with-reply extraction context is
  omitted (the reply is noise the gold does not encode); documented as a known
  measurement choice under the baseline section.
- The SUPERSEDED full-chat path (`replay` without the flag) is NOT byte-stable:
  the clock fix (commit `3ca09e4`) made the chat PROMPT byte-reproducible
  (turn-1 `chat.initial` request identical; seeded `users/user.md` identical), but
  llama-server flash-attention FP nondeterminism makes the long greedy chat
  generation non-bit-reproducible, so two full-chat replays diverge on 18/60
  probes -- a TIGHT band (deltas <= 0.032). Retained for provenance only; a
  byte-stable full-chat run would need `LLAMA_ARG_FLASH_ATTN=off` or a
  deterministic-kernel eval build (see "Path to a byte-stable single-run
  baseline"). Extraction-only sidesteps this entirely.
- 60-record corpus (C3 T3): per-category mass is now meaningful (5-10 probes in
  most categories), but several categories are still small (same-turn pair = 2,
  quantified/structural/habit/date-discrimination = 3). Treat per-category
  numbers as directional, not certified.
- RECOMMENDS / HAS_HABIT gold edges are deliberate FNs until the ontology gains
  those predicates (T10); do not read those FNs as an extraction-quality
  regression.
- assertion_kind labels are present in gold but NOT yet bucket-scored: cease /
  retract edges currently match as plain source+predicate+target tuples.
  Assertion-kind bucket scoring is a later C3 task.
- Reconciliation-action accuracy unmeasurable until C2 emits telemetry.
- Valid-time accuracy only meaningful for absolute-date probes.
- Pre-T7 rebuild baselines are NON-COMPARABLE for same-turn-pair turns (a cease
  and an assert of the same predicate in one turn, e.g. "I left Zeta for Acme").
  T7 changed the intra-turn processing order to retract/cease BEFORE assert
  (`_sort_key` in `reconciliation.py`); pre-T7 runs processed those edges in
  plain `(type, source, target)` order, so the close reason, valid_to, and
  cease/supersession flags on the superseded edge differ. Final open-belief
  state is unchanged across the two orderings -- only provenance/version-chain
  on the closed copy differs -- but any per-probe diff against a pre-T7 rebuild
  on a same-turn-pair probe is expected, not a regression.
