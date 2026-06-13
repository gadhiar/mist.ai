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

### Determinism env (MANDATORY for reproducible F2 replays)

F2 replay drives the FULL `handle_message` path, not extraction in isolation.
That path makes three LLM calls per turn at two different temperatures:
`extraction.scope_classifier` and `extraction.ontology` at `LLM_TEMPERATURE`
(0.0, greedy), and `chat.initial` at `LLM_CONVERSATION_TEMPERATURE` (production
default 0.7, stochastic). The extraction prompt's `Context:` block embeds the
assistant's chat reply, so a stochastic chat turn makes the extraction INPUT
differ run-to-run -- which makes the (otherwise greedy) extraction OUTPUT differ
too. Pinning only `LLM_TEMPERATURE=0.0` is therefore insufficient: the
2026-06-12 C3 T3 determinism check caught two replays diverging on 20/60 probes
for exactly this reason (root cause: conversational-temperature contamination of
the extraction input, compounded by RAG over the per-turn-accumulating graph).

To make F2 replays reproducible, pin the WHOLE chain greedy plus a stable hash
seed on BOTH the seed and the replay exec:

```
-e LLM_TEMPERATURE=0.0 -e LLM_CONVERSATION_TEMPERATURE=0.0 -e PYTHONHASHSEED=0
```

`PYTHONHASHSEED=0` neutralizes a secondary set-ordering nondeterminism observed
in the chat tool-schema enum serialization (a `display_hint` enum order flipped
between runs). NOTE: that enum-ordering flip is a latent PRODUCTION bug in the
tool-schema serialization (non-deterministic iteration over a set/dict feeding
the schema), tracked separately -- `PYTHONHASHSEED=0` masks it for reproducible
evals but does not fix the underlying ordering instability.

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
  mist-backend python scripts/mist_admin.py seed

# 2. Replay the gold corpus, emitting the extraction debug log
MSYS_NO_PATHCONV=1 docker compose exec -T \
  -e MIST_EVAL_ISOLATION=1 -e NEO4J_URI=bolt://mist-neo4j-eval:7687 \
  -e MIST_SIDECAR_DB_PATH=/app/data/eval-run/vault_sidecar.db \
  -e EVENT_STORE_DB_PATH=/app/data/eval-run/event_store.db \
  -e MIST_VAULT_ROOT=/app/data/eval-run/vault \
  -e LLM_TEMPERATURE=0.0 -e LLM_CONVERSATION_TEMPERATURE=0.0 -e PYTHONHASHSEED=0 \
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

To empirically confirm determinism (recommended after any procedure change),
run steps 1-3 twice with the env above (distinct `--session-id` and
`--json-output` paths, fresh quad each), and diff the two score JSON reports on
the aggregate metrics AND the `per_probe` FP/FN arrays -- they must be identical.

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

### Baseline results (60-probe corpus) -- BLOCKED: residual replay nondeterminism

C3 T3 Step 6 ran the pre-any-prompt-change replay twice on the expanded
60-probe corpus to pin replay determinism (the precondition that justifies
every later single-run C3 gate). The determinism check ran in two rounds; it is
NOT yet green.

**Round 1 (LLM_TEMPERATURE=0.0 only):** two replays diverged on 20/60 probes.
Root-causing by call_site (comparing request/response bytes) showed
`extraction.scope_classifier` was already deterministic, but `chat.initial`
(production conversation temperature 0.7, stochastic) varied, and since the
extraction prompt's `Context:` block embeds the assistant reply, the (greedy)
extraction input varied with it. Root cause: conversational-temperature
contamination of the extraction input. Fix: pin the whole chain greedy --
`LLM_TEMPERATURE=0.0 LLM_CONVERSATION_TEMPERATURE=0.0 PYTHONHASHSEED=0` on both
seed and replay (now mandated in "How to run" above).

**Round 2 (full greedy chain pinned):** re-ran the two replays on fresh
isolated quads each (sessions `ext-c3-det1` / `ext-c3-det2`, both
`LLM_TEMPERATURE=0.0 LLM_CONVERSATION_TEMPERATURE=0.0 PYTHONHASHSEED=0`), scored
WITHOUT `--strict`, and diffed the score JSONs on aggregate metrics AND
`per_probe` FP/FN arrays. STILL NOT IDENTICAL -- 20/60 per-probe rows differ.
Artifacts: `data/runtime/extraction-c3-det{1,2}.jsonl` + `-report.md` + `.json`.

| Metric | det1 | det2 | identical? |
|---|---|---|---|
| Matched probes | 60/60 | 60/60 | yes |
| Entity precision | 0.713 | 0.761 | NO |
| Entity recall | 0.864 | 0.890 | NO |
| Relationship precision | 0.605 | 0.696 | NO |
| Relationship recall | 0.730 | 0.762 | NO |
| Typing accuracy | 0.776 | 0.870 | NO |
| RELATED_TO rate | 0.013 | 0.015 | NO |
| Valid-time accuracy | 0.875 | 0.875 | yes |
| Negative-control violations | 2 | 2 | yes |

Reconciliation-action accuracy: SKIPPED (requires C2 telemetry).

**Residual root cause (localized, NOT the LLM sampler).** Per-call_site
request/response byte comparison across det1/det2:

| call_site | n | request identical | response identical |
|---|---|---|---|
| extraction.scope_classifier | 60 | 60/60 | 60/60 |
| chat.initial | 60 | 0/60 | 3/60 |
| extraction.ontology | 60 | 0/60 | 27/60 |
| chat.final | 4 | 0/4 | 0/4 |

`chat.initial` requests differ 0/60 -- the chat INPUT is non-reproducible, so
greedy decoding cannot save it. Diffing the turn-1 `chat.initial` request (no
graph accumulation yet) isolates the SOLE differing line to a wall-clock
timestamp embedded in the chat system prompt's user-snapshot block:

```
## Provenance
- rendered_at: 2026-06-13T03:20:11.109426+00:00   (det1)
- rendered_at: 2026-06-13T03:46:09.679254+00:00   (det2)
```

Source: `backend/chat/conversation_handler.py:2079-2080` supplies
`rendered_at = datetime.now(UTC).isoformat()` to `query_user_snapshot`, which
renders it into the prompt (`backend/vault/user_snapshot.py:115`). That one
differing token perturbs turn-1's greedy chat reply (det1 emits an extra
sentence; det2 truncates earlier); the differing reply enters turn-2's
conversation history, so from turn 2 on the histories have permanently diverged
(the turn-2+ "residual" differences are that turn-1 reply sentence propagating
through history). Confirmed: stripping `rendered_at` from turn 1 makes it
byte-identical; the 59 later-turn residuals are all downstream of the single
turn-1 perturbation. Temperature was verified `0.0` on `chat.initial` in both
runs, so the env fix worked -- it is necessary but not sufficient.

`user_snapshot.py:41` even documents `rendered_at` as "caller-supplied for
determinism in tests", but the production caller passes live wall-clock time.

**Fix required before this baseline can be certified (NOT done in T3 -- out of
scope, it touches the chat/vault production path):** make the chat-path
`rendered_at` reproducible for replays -- e.g. derive it from a fixed
replay reference_date / event timestamp rather than `datetime.now()`, or omit
the provenance timestamp from the prompt-facing snapshot. Once that lands,
re-run det1/det2 and confirm the score JSONs are byte-identical, THEN record the
certified 60-probe baseline + per-category breakdown here.

Secondary finding (tool-schema enum ordering): `PYTHONHASHSEED=0` was added to
neutralize a set-ordering nondeterminism in the chat tool-schema enum
serialization (a `display_hint` enum order flipped between unseeded runs). With
the seed pinned, `extraction.scope_classifier` is now 60/60 byte-identical, so
the seed did its job. The underlying unordered-iteration in the tool-schema
serialization is a latent PRODUCTION reproducibility bug tracked separately --
NOT fixed in T3.

Deliberate-gap note (valid in both rounds): the RECOMMENDS gold edges (ext-31,
ext-32) and HAS_HABIT gold edges (ext-55, ext-56, ext-57) are EXPECTED
relationship false-negatives until the ontology gains those predicates at T10;
both runs show them as FNs (model substitutes wrong predicates). Measured on
purpose; T10 recovers it. Negative-control violations in the round-2 runs were
ext-38 (question-form "What's the best way to learn a new language?" -- Gemma
mined comprehensible-input / language-acquisition entities) and ext-39
(hypothetical Berlin/German) -- real Rule-8/Rule-10 model misses the negative
controls exist to catch, candidates for C3 prompt tuning.

Per-category FP/FN breakdown is omitted because det1/det2 disagree on the
per-probe diagnostics; recording one run's table would imply a stability the
data does not have. Add it once replay determinism is fixed and a single
canonical run is reproducible.

### Superseded baseline (2026-06-10, pre-correction corpus -- NON-COMPARABLE)

Entity P 0.652 / R 1.000, rel P 0.750 / R 0.818, typing 0.833, RELATED_TO
0.000, valid-time 1.000, 0 negative violations. Ontology v1.1.0. Retained for
history only; the entity-precision FAIL and typing FAIL were measured against
anchor-less gold labels and the pre-C1 hardcoded constraint table.

## Known limitations
- BLOCKER (C3 T3): replay is not yet reproducible. Pinning the full greedy chain
  (`LLM_TEMPERATURE=0.0 LLM_CONVERSATION_TEMPERATURE=0.0 PYTHONHASHSEED=0`) fixed
  the conversational-temperature contamination, but a RESIDUAL blocker remains: a
  wall-clock `rendered_at` timestamp injected into the chat system prompt
  (`backend/chat/conversation_handler.py:2079`) perturbs the turn-1 greedy chat
  reply, which propagates through conversation history into the extraction
  inputs -- two replays still diverge on 20/60 probes (see "Baseline results").
  No single run is a certified regression floor until the chat-path timestamp is
  made replay-reproducible (out of T3 scope; touches the chat/vault production
  path). Every later single-run C3 gate depends on fixing it first.
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
