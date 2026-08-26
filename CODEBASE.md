# MIST.AI Codebase Context

**Last Updated:** 2026-08-25 (**extraction-cache Phase 1 (spec D1, D2, D3, D10) COMPLETE,
MERGED to `main` at `c2cc748` (`--no-ff`), and PUSHED -- `main` and `origin/main` are both at
`227ad8f`. (An earlier revision of this entry said "NOT pushed; `origin/main` is still `330ff1c`",
which was true when written and was falsified by the push later the same day. Verify with `git
rev-parse HEAD origin/main` rather than trusting this line.)
`--no-ff` deliberately, not a rebase: the review documents cite this branch's commit hashes and a
rebase would invalidate every one (the same reasoning as the 2026-08-19 Q2-1 merge). Branch
`feat/extraction-cache-production-writer` merged at its tip `1171423`; commit count on the merged
branch: `git log --oneline 330ff1c..1171423 | wc -l` -- run it rather than hardcoding, since a
hardcoded count goes stale the instant it is committed. The first 28 landed the seven
implementation tasks; every commit after that closes one or more whole-branch-review fix waves --
see this file's PRIOR ENTRY history below and `.superpowers/sdd/2026-08-18-extraction-cache-phase-1/`
for which waves and how many.

Post-merge verification on `main` @ `c2cc748`, measured not inherited: suite **3070 passed / 6
skipped / 3 xfailed / 0 failed**; live graph **32 nodes / 30 relationships**; tree clean (the
`data/extraction_cache.db` a factory-built test now creates is gitignored as of this phase, so
"tree clean" is a usable signal again). Same preconditions as below. The production extraction path now writes one extraction-cache row
per turn --
outcome, skip reason, the C1 bitemporal `recorded_at`, and the Stage 1.5 subject-scope
classification -- and `graph-rebuild-from-log` replays Stages 3-6 as PURE CODE against that cached
Stage-2 decision, re-deriving everything ontology-dependent rather than trusting the cached
payload for anything past Stage 2 (spec D2). D4-D9 are later phases, not this one.

The whole-branch review returned APPROVE WITH FINDINGS (0 Critical, 4 Important, 11 Minor); this
fix wave closed all four Important findings plus one documentation record: (I1) the cache's
`initialize()` inside `build_extraction_pipeline` is now failure-isolated the same way the vector
store beside it already was -- an unwritable or absent data root degrades to
`extraction_cache=None` / `rebuild_stamps=None` (the exact both-None pairing this optionality was
designed for; `_record_skip`/`_record_extraction` already no-op on it) with a WARNING naming the
degraded consequence, not an uncaught exception that silently disabled the whole knowledge layer
via `KnowledgeIntegration`'s outer `except Exception`. (I2) `scripts/mist_admin.py`'s admin CLI
now imports and calls the shared `production_cache_path(config)` instead of re-deriving the same
expression inline -- closing the THIRD instance of the same ":memory:"-sentinel defect Tasks 5 and
7 had already fixed once each elsewhere. (I3) `ExtractionCache.get()`'s docstring is corrected to
describe Task 6's actual replay behaviour -- `_assert_cache_coverage` still tests `is None` only
(correctly: a recorded decision IS coverage), while the `rebuild()` replay loop branches on
`outcome` and treats a `skipped` row as a no-op -- rather than the pre-Task-6 state it had frozen
in place. (I4) this entry: the header and Current Focus below were 6 days and 7 tasks stale before
this update. Also logged in `KNOWN_ISSUES.md` (I5): the pre-existing `_SOURCE_THRESHOLDS`
significance-threshold shadowing bug on the dominant `"conversation"` extraction source --
pre-existing and out of this phase's scope, but escalated because the significance decision it
makes is now written into the cache and inherited verbatim by every rebuild.

Suite 2996 -> 3069, 0 failed (2996 is this file's own recorded number at `330ff1c`, not a fresh
re-measurement; 3068 was the CONTROLLER's measurement at `2a79689`, which the whole-branch review
cited rather than independently re-ran -- the review says so explicitly ("I did not re-run the
unit suite. The controller measured 3068/7/3/0 and instructed me not to"). 3069 is this fix
wave's own re-measurement, +1 from this wave's new degraded-path test proving I1's fix, and was
independently re-measured a second time by the scoped re-review after this wave landed (also
3069/7/3/0). Live graph 32 nodes / 30 relationships,
reverified live after this fix wave, unchanged throughout every commit on this branch. Suite warnings are
DELIBERATELY not quoted as a number here: measured 7, 8, 9, and 10 across separate runs at
identical commits on this same branch -- not reproducible, unlike pass/skip/xfail, which are
stable and are what is reported.

Preconditions for the numbers above, stated because omitting one is what made the 2026-08-05
record unreproducible: Docker stack up and healthy (`docker compose ps` shows mist-backend /
mist-llm / mist-neo4j all `Up (healthy)`); suite via `MSYS_NO_PATHCONV=1 docker compose exec -T
mist-backend python -m pytest tests/unit/ -q` (`MSYS_NO_PATHCONV=1` stops Git-Bash-on-Windows from
rewriting `/app`; `-T` is required non-interactively); graph counts via `MSYS_NO_PATHCONV=1 docker
compose exec -T mist-neo4j cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n)"` (and the
`()-[r]->()` form for relationships).

Next: R1.6's `live == rebuilt` closure gate, and extraction-cache Phase 2/3 (D4, then D5/D6/D7 --
spec section 8: Phase 2 and 3 are coupled and must land together), which couple to it. Phase 4
(D8, `diff`/`promote`) is INDEPENDENT of that pair and ships any time after Phase 1; D9 accrues
automatically with no separate work -- corrected here after this file's first draft folded both
into a single misleading "(D4-D9)" range.**)

- **FOLLOW-UP --** 2026-08-25 (same day, post-merge): three carried items closed before opening
  R1.6. (1) **The eval-gated integration tests were run** -- they had skipped throughout Phase 1
  because they need the EVAL instance, not the staging one Phase 1 stood up. Brought up
  `docker-compose.eval-neo4j.yml --profile eval` (bolt 7688) and ran all five files: **12 passed**
  (`test_deterministic_resolver`, `test_bitemporal_currency_live`, `test_selfmodel_partition` gate
  on a socket probe and run automatically once the instance is up; `test_seed_label_split` is
  fail-closed and additionally needs `MIST_EVAL_ISOLATION=1` +
  `NEO4J_URI=bolt://mist-neo4j-eval:7687`). Live graph verified 32/30 after -- `test_seed_label_split`
  performs a full `reset_graph(include_derived=True)`, so that check is the point, not a formality.
  Eval instance torn down (tmpfs, wiped). (2) **`_SOURCE_THRESHOLDS` shadowing FIXED** (KNOWN_ISSUES
  I5) -- see that file for the full record. The diagnostic inverted the issue's own framing: the
  logged bug was "a dead config knob", but `orchestrator_summary` and `agent_tool_output` have ZERO
  production callers while the real production sources (`admin-cli`, `v2-ingest`,
  `v2-ingest-commit`) have no table entries at all -- so the table's only live entry was the one
  breaking the config, and the abstraction, not just the knob, was the defect. Fixed by removing the
  `"conversation"` key; the `.get(source, config)` lookup was already correct. **A no-op on
  production data** (all three threshold authorities read 0.3, exactly the removed hardcode), which
  is why it was done NOW: that free window closes the moment the env knob is tuned, after which the
  same fix splits cached extraction history into pre- and post-fix decisions. Suite **3070 -> 3075**,
  0 failed; live graph 32/30 unchanged. A second finding fell out of the diagnostic and is recorded
  because it is the same defect class Phase 1's review kept surfacing: **eight test sites set
  `significance_threshold=0.0` intending to disable the significance gate, and for the
  `conversation` default source that never worked** -- they ran against a 0.3 gate while their code
  said 0.0, passing only because their utterances scored above it. The fix makes them truthful; they
  were deliberately not otherwise touched. (3) This file's own "NOT pushed" claim was corrected --
  it was falsified by the push later the same day.

- **PRIOR ENTRY --** 2026-08-19: Audit finding Q2-1 CLOSED AND MERGED to `main` (`a81c474`,
  `--no-ff`). Four review waves; each found something the previous one missed. Wave 2's "Minor 6"
  was misclassified and was the real find -- a `#` in the store path became a URI fragment, so
  `?mode=ro` was never applied and the "read-only" open was read-write-CREATE, in the guard added
  to close Q2-1. Wave 3 killed a mutant guard that performed 13 committed writes to the replay
  sources while all ten tests passed. Wave 4 found eight docstring citations stale by exactly +178
  lines, broken by the wave that had just fixed a citation -- so every line-number reference in
  `scripts/mist_admin.py` is now enclosing-symbol-plus-grep, and line shifts can no longer generate
  findings there. Suite 2975 -> 2996, 0 failed. Live graph 32 nodes and the live event store
  (212992 bytes, mtime 2026-08-02) untouched throughout.
- **PRIOR ENTRY --** 2026-08-05: Audit finding Q2-1 closed on a branch (not yet merged):
  `graph-rebuild-from-log --dry-run` no longer writes to the live SQLite event store. Proved by
  execution -- pre-fix code wrote 2 job rows per invocation, post-fix 0, both against a COPY of
  the live store. Suite 2975 -> 2983 at that point.
- **PRIOR ENTRY --** 2026-08-05: Eval-harness scorer audit + the non-vacuity fix for `scorers.py`
  COMPLETE and ff-merged. R1.4.6 T0 landed earlier the same day.

---

## The rebuild's dry-run is now actually dry (audit Q2-1)

**The defect.** `mist_admin graph-rebuild-from-log --dry-run`, whose entire advertised contract is
"proof-first, dry-run only", wrote to the LIVE SQLite event store on every run: `initialize()` on
both stores (`mkdir` + `executescript` + a conditional `ALTER TABLE`), then a
`rebuild-<epoch>-<uuid>` job row, a checkpoint per turn, and a finalize -- **doubled**, because
`_build_once` runs twice for the determinism gate.

**Why no guard caught it, and why no guard ever would have.** `assert_rebuild_target_not_live` and
`assert_neo4j_isolated` both reason about bolt URIs; a SQLite path is invisible to them. The
isolation model equated "live state" with "the live Neo4j graph". `LogRegenerator` held THREE
dependencies and only the Neo4j leg was guarded -- while that same class had already solved the
identical problem on its Neo4j leg, where `source_conn` reads live and `staging_conn` takes the
writes.

**The fix is structural, not another guard.** The event store was doing two unrelated jobs: it was
the replay SOURCE (which must be live) and the sink for the rebuild's own progress rows (which must
not be). `backend/knowledge/regeneration/rebuild_journal.py` splits them --
`EventStoreRebuildJournal` (durable, used by the golden-log replay and integration tests against
their own disposable stores) and `NullRebuildJournal` (records nothing, wired by the CLI).
`journal` is a REQUIRED constructor argument: a default is exactly what the bug looked like, since
any implicit "journal into the store you were given" sends a proof run's rows to the live ledger.
`rebuild()` now also refuses `resume_from` against a non-durable journal rather than silently
restarting from the top and reporting success.

**Both `initialize()` calls are gone**, replaced by `_assert_replay_source_exists`. Calling
`initialize()` also MANUFACTURED the absence it was meant to tolerate: on a machine with no event
store it created an empty one, and the run then reported "No epochs found" -- indistinguishable
from a store that exists and is empty. A rebuild replays an existing log; it does not bring one
into being.

**Proved by execution, not by reading the diff.** Against a COPY of the live store (live never
touched), with staging Neo4j up:

    pre-fix   -> re_extraction_jobs: 2   ('rebuild-1-1b369bed', 'rebuild-1-07ad9814')
    post-fix  -> re_extraction_jobs: 0

The two rows from one invocation are the doubling made visible. Live `re_extraction_jobs` was 0
before this work and is 0 after; the defect was latent and never fired against live.

**The precondition that makes those two numbers comparable, stated because omitting it made the
record unreproducible.** Both runs were handed a MANUFACTURED `extraction_cache.db` in the copy
directory, created by a separate harness command, because no production code path creates one
(see the open item below). Without it the post-fix run refuses at `_assert_replay_source_exists`
before `LogRegenerator` is constructed and returns `0` for an entirely different reason -- so a
reader reproducing this from the original wording would have measured the refusal and read it as
the fix. The whole-branch review raised precisely that (HIGH-2/F1). Re-verified 2026-08-05 with
the cache present: the post-fix run prints `[rebuild] determinism gate PASSED (rebuild-twice
byte-identical)` and exits 0, so the journal path WAS exercised and the `0` measures it. **The
proof stands; the record of it was incomplete** -- the same defect class as the code bug, in the
documentation.

**Regression mechanisms, each mutation-proved in both directions.**
`tests/unit/scripts/test_rebuild_cli_is_read_only.py` asserts BEHAVIOURALLY that
`_build_log_regenerator` initializes neither replay source, refuses a missing/unschema'd/truncated
one, and constructs a journal whose runtime `type` is `NullRebuildJournal`. Its first version
checked those properties with `ast` and the review defeated all three with a single mutant that
fully restored the live write -- by moving `initialize()` one frame out into a module-level helper
(an AST rule scoped to one function stops at the frame boundary) and by
`import EventStoreRebuildJournal as NullRebuildJournal` (the spelling the rule matched stayed
identical while the object became durable and live-bound). Both mutants now fail. The file also
carried a FALSE justification for going static -- that a behavioural test needed a model load and
a live Neo4j -- which was never checked before being written; patching the function-local imports
reaches it with neither. `RecordingEventStore` in `test_rebuild_scoping.py` also lost its
job-write methods, so a regression that reaches for one fails with `AttributeError`.
**"Each mutation-proved" was itself an over-claim until 2026-08-18.** The builder calls
`_assert_replay_source_exists` TWICE (event store, then extraction cache), every refusal test
drove the event store, and because that is the guard checked first, deleting the extraction-cache
call left the whole file green. Closed by
`test_a_missing_extraction_cache_is_refused_even_with_a_valid_event_store`, which seeds a valid
event store so only the second guard can refuse; mutation-proved in both directions (delete the
call -> 1 failed / 5 passed with `DID NOT RAISE`; restore -> 6 passed).

**STILL NOT closed, named so the above is not read as more than it is:**

- **Nothing in production ever writes an extraction cache, so `graph-rebuild-from-log --dry-run`
  is now UNCONDITIONALLY non-functional against any live-derived config.** `grep -rn
  "ExtractionCache\|extraction_cache" backend/` returns zero hits; the only non-test
  constructions are `mist_admin.py` (this read-only replay path) and
  `scripts/golden_log/generate.py:330`, which writes `extraction-cache.db` at its own disposable
  root -- a DIFFERENT filename from the `extraction_cache.db` the CLI derives. Every invocation
  therefore hits `ColdCacheError` and exits 2. **An earlier draft of this entry said "a rebuild of
  a non-empty log would ColdCacheError"; that condition was wrong in both directions** and the
  review falsified it (F3). The guard tests schema, not coverage, so it fires on an empty log too.
  **A SECOND false claim was written into this entry while correcting the first, and is corrected
  here 2026-08-18 rather than removed:** it said that on a machine where the pre-fix `initialize()`
  had already created an empty cache the old existence-only check did NOT fire at all, and that
  this was why the guard now checks for the required table rather than the path. Both halves are
  wrong. `ExtractionCache.initialize()` executescripts a DDL whose sole statement is
  `CREATE TABLE IF NOT EXISTS extraction_cache`, so such a machine HAS the table and the NEW check
  passes there too -- the schema check buys nothing on the machines it was said to be for.
  (Verified by execution: `initialize()` into a temp dir, then
  `SELECT name FROM sqlite_master WHERE type='table'` over a `mode=ro` connection, returns
  `['extraction_cache']`; the event store's equivalent returns a list containing `epoch_ledger`,
  from `schema.sql:134`.) What the schema check DOES buy over an existence check is the truncated
  file and the 0-byte file that `touch` or a bare `sqlite3.connect()` on the path leaves -- on
  both, the first read raises out of `sqlite3` and escapes as a traceback. The refusal is correct
  and the guard is worth keeping; only its stated reason was invented. **The R1.6 live==rebuilt
  closure gate has no runnable path until a production writer for the extraction cache exists.**
  That prerequisite is now the blocking item, and nothing else in the repo tracks it.
- **The determinism gate compares two empty graphs whenever it can run at all.** The live log
  holds 0 turns (`conversation_turn_events`: 0, `conversation_sessions`: 0), so "rebuild-twice
  byte-identical" is vacuous over live data. Stated in the past/conditional tense deliberately:
  the review falsified an earlier present-tense phrasing (F2), since after this branch the gate
  is unreachable on a live config at all (previous bullet). Same vacuity class the 2026-08-05
  scorer audit closed for F2 and V7; not closed here.
- **`--dry-run` is still `required=True` and still never read** (audit Q1-3). It is no longer
  certifying something false, but it remains a flag that cannot alter control flow. A durable-journal
  branch was deliberately NOT added behind it -- that would be a dead branch justified by a future
  caller.

---

## Eval-harness scorers: what is closed and what is NOT

**Stated precisely, because the temptation to write "scorer vacuity closed" is exactly the defect this work removed.**

**Closed** for the seven scorers in `scripts/eval_harness/scorers.py` (`cf48440..842bb90`, suite 2926 -> 2952):

- Each returns a `ScoreOutcome` carrying `examined: int | None`, and `enforce_non_vacuity()` forces a fail at `examined == 0`. `None` (cannot count) stays distinct from `0` (looked, found nothing) end to end -- guard, aggregation, markdown, and JSON `null`.
- **All five dispatch sites are guarded**, not one. The whole-branch gate found the guard applied only in `_ingest_record` while four sites read `.score` raw -- including the **D2 kill-switch** that decides whether phases D3-D5 run, where a candidate extracting nothing scored 1.0 and the switch did not trip. Fixed, with a test that extracts the live heredoc from `phase3_orchestrator.sh` and proves the behaviour by subprocess.
- `SCORER_EXAMINES` declares what each count counts, guarded by a test that fails when a scorer is added without one -- mutation-proved in both directions.
- Reports and `dump_run_scores_json` print the count, so a score and its denominator travel together.

**Also closed 2026-08-05** (`609dbdf..0636743`, suite 2952 -> 2975), for **F2 and V7 only**:

- **F2 fails closed on a corpus with zero probes.** Verified by execution beforehand: `score_run([], {})` returned entity precision 1.0, rel precision 1.0, typing 1.0, and `_gates_pass` True. The join-integrity check that exists to stop this was what let it through -- `matched_probes == total_probes` is `0 == 0`. A probe-less gold file now exits 1 under `--strict` and prints a VACUOUS RUN banner explaining the rows reflect an empty corpus. **The trigger was never a mistyped path** (`main()` guards that and exits 2) but a gold file that exists and parses to nothing -- truncated, emptied, all-comment, over-filtered.
- **Every F2 gate row carries its denominator.** Two rows reading an identical 0.833 now render `25/30` and `5/6`; a gate with no meaningful count renders `n/a`, not `0/0`. All nine pairs were independently traced to the computations that produce them.
- **V7 fails closed when zero negative controls were examined.** Confirmed by execution first: `acceptance_pass()` returned True at precision 1.0 / recall 1.0 / FP 0 while all five negatives were `verdict="missing"`.
- **The five-site `enforce_non_vacuity` list is now a mechanism**, not a discipline -- an AST + heredoc static check that fires when an unguarded `.score` read is added, mutation-proved in both directions.

**STILL NOT closed, and named so nobody reads the above as more than it is:**

- **V7 still passes over POSITIVE probes it never examined.** Structurally identical to the negative-control defect just fixed, and untouched by it. A positive probe that never joins gets `verdict="missing"` -- counted in neither TP nor FN -- so it drops out of the recall denominator rather than counting against it, and `acceptance_pass()` has no missing-positives term. Verified 2026-08-05: 10 of 20 positives never joining still yields precision 1.000, recall 1.000, PASS, `--strict` exit 0. Recorded rather than fixed because it needs its own mutation proof and review. **Second-order risk from our own fix:** the report now renders a "Negative controls examined" row which, sitting beside symmetric-looking Precision and Recall rows, can imply examination is verified generally. It is verified only for negatives.
- **`score_v8_probe_run.py` and `score_v9_predicate_run.py` are untouched.** V9 was independently re-verified to fail closed already; V8's audited problem was the misquoted `0.938`, not a scorer defect.
- **The candidate-drop finding is untouched** (audit section 2.3): a candidate whose llama-server never becomes healthy is silently dropped from every scored table while still counted as a candidate.
- **Past denominators remain unrecoverable.** Counts are printed going forward, not backward.
- **Historical numbers are not invalidated, and their denominators are not recoverable.** The recorded F2 figures were computed from real data (none equal the vacuous defaults). But gate rows never printed counts, so whether the 0.83 relationship-precision near-miss rested on 25 comparisons or 5 cannot be recovered from the report as generated.
- **`V8 0.938` should not be cited as evidence of model capability again.** It is a recall sub-metric from iter2, whose acceptance verdict was FAIL on over-extraction -- the exact defect class it was used to rule out. All four recorded V8 baselines show FAIL (`scripts/eval_harness/v8_probe_set_design.md:67-72`).

**VERIFIED STATE (2026-08-04, every row re-read from source at branch close, not carried forward):**

| Fact | Value | How verified |
|---|---|---|
| Local `main` HEAD | `1598412` + this docs commit | `git log` after ff-merge |
| Commits ahead of origin | **0** after push -- `main` and `origin/main` MATCH | `git rev-list --count origin/main..HEAD` |
| Unit suite | **2926 passed / 7 skipped / 3 xfailed / 0 failed** | full run in container, clean tree, run by the coordinator directly -- NOT self-reported by an implementer |
| Suite warnings | 9, all from files this branch never touched | warnings summary inspected source-by-source |
| Ontology version | **1.4.0** | `backend.knowledge.version_stamps.ONTOLOGY_VERSION` at runtime |
| Extraction version | **2026-06-14-r5** | same module, runtime |
| Live graph | 32 nodes / 30 relationships, UNTOUCHED this session | no Neo4j write of any kind was made |
| `epoch_ledger` | 2 rows; `epoch_id 2` current | unchanged since the 2026-08-03 repair |
| Stack | `mist-backend` / `mist-llm` / `mist-neo4j` all Up (healthy) | `docker compose ps` |

**What R1.4.6 T0 shipped** (4 commits, `036ae3b..1598412`, suite 2918 -> 2926):

- **`4f4ec7b` + `264587c` -- the turn-result side-channel moved off the singleton.**
  `KnowledgeIntegration.last_complete` / `last_error` were per-TURN values on a process-wide
  object, safe only because of a `generation_lock` invariant enforced in a different file.
  They are now ContextVars in `backend/request_context.py`. A consumer-side reset was built
  and then REMOVED in `264587c` as unreachable: `KnowledgeIntegration.enabled` is assigned
  only in `__init__` and never flips, so the producer either runs on every turn of the process
  or on none, and its reset is the first statement of the generator body. What remains is a
  test pinning that ordering, mutation-proved to fail when the reset moves below a yield.
- **`02f9c7c` + `1598412` -- a closed connection's residue can no longer reach the next
  session.** Nothing in the disconnect path reset any `VoiceProcessor` state, so a parked
  utterance outlived the connection that produced it and the next turn's `finally` drained and
  respawned it -- replaying an already-ended session's words into a live one, the same
  misfiling class as `feb472a` by a different route. `reset_connection_state()` clears
  `latest_user_input` under `input_lock`, called only when `active_connections` is empty.

**`reset_connection_state` deliberately does almost nothing, and that is the finding.** It
shipped clearing four things; the whole-branch review proved three of them unreachable and the
fourth actively harmful, and `1598412` reduced it to one. `audio_queue` has no writer anywhere
in `backend/` (`KNOWN_ISSUES.md:114-115` already recorded it as unused, and the test that
"proved" the drain worked supplied its own precondition). VAD state is fed only by
`process_audio_chunk`, which has no caller (`KNOWN_ISSUES.md:117-118`). And
`interrupt_flag.clear()` was a **regression**: every in-turn reader sits downstream of that
turn's own unconditional clear, so it could never help -- but a client that interrupts and then
disconnects would have had the cancelled turn run to completion and persist into the closing
session. Each exclusion now carries its verified reason in the method's docstring.

**Five rationales were falsified on this branch, and the pattern is worth more than the fixes.**
Four were in the implementation plan (a consumer reset justified by a branch that cannot vary;
a `TYPE_CHECKING` import justified by a cycle that does not exist; the `audio_queue` and VAD
resets) and the fifth was in the fix for the other four (a docstring claiming `audio_queue`'s
"only feeder, `process_audio_chunk`" when that function feeds `vad_processor` and `audio_queue`
has no feeder at all). In every case **the conclusion was right and the stated reason was
invented rather than checked**, and every one took seconds of `grep` to falsify. That is the
same defect class the 2026-08-03 review named -- a documented claim contradicted by observable
state -- occurring inside the documents describing its removal. The scoped per-task reviews
could not catch most of them, because the deadness lives OUTSIDE the diff, in the absence of a
caller and in `KNOWN_ISSUES.md`. The whole-branch gate is what caught them.

**Also filed:** `docs/superpowers/specs/2026-08-04-connection-scoped-transport.md` (local,
untracked per `.gitignore:69`) -- two transport-layer defects found during the sweep and
deliberately NOT fixed here: `broadcast_messages` (`server.py:105-125`) sends every message to
every connection, and `interrupt_flag` is one shared `Event` so any client's interrupt cancels
every in-flight turn. Neither is fixable at the attribute layer. Its open question must be
settled first: whether broadcast is partly INTENDED, since MIST is single-user and mirroring
across two windows is plausible design. ADR-017 calls multi-client "out of scope for v1.0" and
the broadcaster "single-channel", which leans toward artifact rather than intent.

**What landed 2026-08-03** (19 commits, `a5a7499..987651f`, ff-merged to local `main`, unit suite
2552 -> 2679, live graph untouched throughout except one deliberate epoch repair):

- **I7 seed embedding gate** (`63d4707`..`e367ae4`) -- recomputes each seeded node's embedding
  from the AUTHORED source and compares by cosine. It catches the mode nothing else can see:
  `canonical_serialize` excludes `embedding`, so every determinism and equality check is
  byte-identical whether vectors are present, absent, or all-zero. Wired into `cmd_seed` AFTER
  the backfill, which is what closes it -- the backfill commits after the graph writes, so its
  own count proves nothing. Live-verified PASS.
- **`ensure_initial_epoch` wired into production** (`f4aeadd`) -- it had five correct unit tests
  and zero production callers while the live `epoch_ledger` sat empty. Mutation-proved in BOTH
  directions: removing the call fails 5 new tests, and all 7 pre-existing epoch tests in
  `test_store.py` still pass without it. That second half is the defect class stated concretely.
- **Version-stamp collapse** (`11dd23c`) -- eleven authorities reduced to one each. Three were
  live Cypher write paths stamping `'1.0.0'` on every skill-derived, capability-derived and
  self-model entity while the curation path stamped `1.4.0`.
- **R1.4.5 golden log** (`938568c`..`e11caa2`) -- 87-turn authored corpus from all 60 gold
  records plus purpose-authored staleness turns; replay proven to execute. The first rebuild in
  this project that is not vacuous.
- **`model_hash` single authority** (`67d4bf5`) plus a live epoch repair (`epoch_id 2` supersedes
  the drifted row, `prev_epoch_id=1`, history preserved).
- **P6 session-id propagation** (`8d6c161`) and `end_session` scoping (`eb7a8b4`), then
  Priority-1 remediation (`987651f`) closing two green mutations in guards shipped the same day.

**The systemic finding, which governs the current work queue.** A seven-arc whole-codebase
adversarial review produced ~60 findings; roughly 20 of them are ONE defect: **a feature built,
tested, and wired to nothing.** Every instance had correct unit tests; not one asserted
reachability. Six shapes, each invisible to the check that catches the previous one: no
production caller / called with a null optional dependency / called reading the wrong instance of
shared state / always-None argument on a live path / implemented parameter never passed / wiring
connecting two different module objects. Nine live production bugs fall out of it -- see
"Known live bugs" below. **Root enabler: emptiness is a universal alibi.** "Job ran, found no
data, returned zeros" and "job ran with a null dependency, returned zeros before looking" are
indistinguishable in the logs, which is why hydration (R1.4.6) is now sequenced BEFORE R1.5.
Full register: `docs/superpowers/specs/2026-08-02-review-findings-register.md` (gitignored,
local only).

**POST-MERGE, 2026-08-04 -- three more pieces landed on `main` (`b1be0e3`, `e7ade99`, `4ae31fc`).**

- **`b1be0e3` -- A LIVE-GRAPH LOSS HAZARD, found in the surface the review never searched.**
  `assert_rebuild_target_not_live` was a DENYLIST OF ONE HOSTNAME SPELLING. Measured against
  `live = bolt://mist-neo4j:7687`: `bolt://mist-neo4j:7687` refused, but **`bolt://localhost:7687`
  and `bolt://127.0.0.1:7687` PASSED** -- the same database, since the live bolt port is
  host-published. `cmd_graph_rebuild_from_log` calls that guard and then runs
  `MATCH (n) DETACH DELETE n` unconditionally (`grep -n "assert_rebuild_target_not_live\|DETACH
  DELETE" scripts/mist_admin.py`), so `--staging-uri bolt://localhost:7687` -- the
  natural spelling from the host -- would have wiped the canonical graph. Now an allowlist, and
  deliberately narrower than first specified: staging ONLY, excluding eval (the test DB) and dev
  (because R1.6 treats the DEV graph as the "live" side, so admitting it as a write target would
  let a rebuild delete an 87-turn hydrated fixture).
- **`e7ade99` -- D3.** Curation job runs are now persisted to `curation_job_runs`, with
  `examined` and `produced` as SEPARATE columns so "ran, looked at nothing" is a different row
  from "ran, looked at N, changed nothing". `graph_health_events` is written too and kept
  distinct -- it is a metric time series for one job; the new table is the audit fact for all of
  them. `run_once()` and `_loop` now share one execution path so they cannot drift.
- **`4ae31fc` -- R1.4.6 T1 + T4.** Dev compose profile, mechanical isolation guard (refuses a
  root that IS, SITS UNDER, or **CONTAINS** live state -- the third arm matters because `restore`
  clears its target), and snapshot/restore whose manifest READS the stamp triple rather than
  restating it, so a stale artifact refuses itself by name. T2 (the hydrator) and T3 are NOT
  built: the corpus must be authored with Raj.

**Still open from the `scripts/` audit** (`docs/superpowers/specs/2026-08-04-scripts-audit.md`,
gitignored): `graph-rebuild-from-log` writes job rows and a schema script to the LIVE SQLite
event store at `~/.mist` despite advertising dry-run only -- both isolation guards are Neo4j-only
and neither has any notion of the event store. Severity bounded (append-only rows, not a wipe),
but the contract claim and the writes cannot both stand. The audit also flagged its own biggest
gap: ~3900 lines of eval-harness scorer logic, which is exactly where "passed=True over zero
examined" lives.

**PRIOR -- REMEDIATION, 2026-08-03 -- 12 commits, MERGED to local `main` @ `9b363ba`, branch deleted.**
Live graph untouched at 32/30 throughout; no write of any kind was made to Neo4j, the event
store, or the vault.

**HOW THE MERGE GATE ACTUALLY RAN, stated plainly because it is not the usual shape.** FOUR
successive read-only review agents (a `scripts/` auditor, a whole-branch reviewer, and two narrow
single-question gates) completed their work and then went idle WITHOUT returning any report.
Every implementation agent reported normally; only the review agents failed, which points at how
they were dispatched rather than at the agents. **No independent review verdict exists for this
branch.** The gate was therefore run by the coordinator directly, and covered:

- Every commit's central claim mutation-proved independently of its authoring agent, production
  file restored byte-identically each time.
- **The composition-root mutation matrix, both directions.** Reverting the `server.py` call site
  fails the AST guard (1 test) while the wiring tests stay GREEN (12 pass); making the resolver
  return `None` leaves the AST guard GREEN (7 pass) while 5 wiring tests FAIL. Neither mechanism
  subsumes the other and neither is dead.
- **Exemption anti-rot**, verified length-sensitive rather than always-raising: a reasonless
  entry raises, an empty callee raises, a >=60-char justification is accepted.
- **Scope creep on the rebuild change, cleared by enumeration.** Exactly ONE production caller of
  `get_all_turns_for_reextraction` (`log_regenerator.py:247`), which passes `origins` explicitly.
  The store defaults to `None` (neutral read); only the REBUILD defaults to `('real',)`
  (fail-closed policy). No existing caller's behaviour changed.
- **The composition risk, cleared empirically.** `CurationScheduler._loop` initializes
  `last_run.get(name, 0.0)`, so every enabled job fires IMMEDIATELY at start. A real container
  restart ran all three newly-wired jobs against the live graph for the first time; graph before
  and after 32/30, nothing written.
- The import change verified at RUNTIME (container restarted, single `Started server process`,
  zero first-party duplicates scanned inside the running container), not by tests alone.

**NOT covered by any party:** a systematic fact-check of the commit messages' claims, and a dead-fix
sweep across ALL new guards rather than the highest-risk one. Those two categories are exactly
where the previous session's own deliverables failed THREE times, so treat them as open risk on
this branch rather than as cleared.

| Commit | Substance |
|---|---|
| `7c1fbc1` | This file -- ~11 false/stale claims corrected, each re-read from source |
| `c48729f` | Module-identity guard (Mechanism B) + mirror mutants surviving in two guards shipped 2026-08-03 |
| `feb472a` | **LIVE BUG** -- a queued turn inherited the previous turn's `session_id` |
| `538f848` | Mechanism A composition-root guard + `server.py:444`, the one line behind three dead features |
| `2658aba` | **LIVE BUGS x2** -- self-model score pinned at 0; coverage capped at 83.3% |
| `0a75e34` | **LIVE BUGS x2** -- the rebuild never scoped turns by epoch or by origin |
| `4657305` | Comments describing wiring that does not exist (`atexit`, `source_event_id`, stamp authorities) |
| `d4bf16c` | The module-identity guard was silently examining less than it did the day before |

**The most important find was not on the work list.** `VoiceProcessor.latest_user_input` held a
parked utterance's TEXT and nothing else. The drain that replays it runs on the FINISHING turn's
thread, and `spawn_with_context` snapshots the CALLING thread's context -- so a queued turn
inherited `current_session_id` from whichever turn happened to finish. That id selects the
conversation history, the EventStore session, the vault note path and the graph provenance, so
**one connection's utterance was filed into another connection's memory.** Inert until 2026-08-03
because every session id was the literal `"default"`; making session ids real converted it from
latent to live. It is the THIRD member of that class after `end_session` ending every session --
expect more, and see the R1.4.6 T0 sweep.

**Two consequences of this branch's own fixes, stated because they are easy to miss:**

1. **D3 got worse.** `scheduler.py:135` discards every `JobResult`. Until `538f848` both affected
   jobs returned all-zero results, so discarding them lost nothing; they now produce real counts
   that `_loop` throws away. Graph WRITES still land -- only the counts are lost.
2. **The self-model health score now pins at 100.** `min(100, count/5*100)` against 21 seeded
   nodes. The fix moved it from permanently-0 to permanently-100; both are near-zero signal, and
   the second is less alarming, which arguably makes it worse. The threshold of 5 predates the
   21-node seed. Scoring-policy decision, deliberately left open.

**Method note worth carrying.** The single most useful result of the night came from mutation-
testing with the ACTUAL historical bug rather than a synthetic sentinel. A `:__Nonexistent__`
label mutation probes only the label axis and passed on the first try; restoring the real
original query -- which carried a PREDICATE as well as a label -- exposed a second axis where the
test fake silently dropped any WHERE clause it had no model for. A synthetic mutation proves a
test can fail; only the real bug proves the test covers the axis the code was wrong on.

**This header previously carried ~11 false or stale claims** and is read into every session by
`/mist-status`, so the errors propagated. They are corrected in this revision and named here so
the correction is auditable rather than silent: a deleted branch presented as current; "125
commits ahead" (was 9 when written, is 19 now); **three mutually inconsistent unit-test counts in
one file**; ontology `v1.3.0` (is 1.4.0); a four-authorities paragraph describing a state that one
later commit had already fixed; "five unit tests" for `ensure_initial_epoch` (the number was also
copied into `conversation_handler.py:838` and the epoch test file); Phase 10 re-certified as
accurate when `emit_seed_vault_provenance` has zero production references; `regeneration/`
described as a tombstone when it holds the live R1.2 rebuild machinery; `DebugJSONLLogger` "5
record phases each gated by its own env var"; `scripts/seed_data.yaml` "not on `main`"; and a
standing imperative not to re-run `mist_admin.py seed` that R1.4 T14 closed.

**One register claim was itself wrong and is corrected rather than copied.** The review recorded
`DebugJSONLLogger` as "7 phases, 3 gated". It is **7 phases** (`turn`, `extraction`, `llm_call`,
`retrieval_candidates`, `llm_request_raw`, `reconciliation`, `vault`) of which **5** carry their
own env gate (`llm_call`, `retrieval_candidates`, `llm_request_dump`, `vault`, `reconciliation`);
`turn` and `extraction` are gated by the master `MIST_DEBUG_JSONL` only. Verifying a correction
against source rather than against the document reporting it is the whole point of this protocol.

PRIOR ENTRY -- 2026-08-02 (**Epoch wiring COMPLETE + ff-merged to local `main` @ `f4aeadd`; version-stamp authority collapse ACCEPTED and folded into R1.4.5.** `main` was 7 ahead of origin, NOT pushed.

**`ensure_initial_epoch` is now called from production.** `ConversationHandler.__init__`, immediately after `initialize()` -- the one path production always opens the event store on (`mist_admin.py` also constructs one, but only when a subcommand runs). `now_iso` from the injected clock. Placed inside the existing `try` because every realistic failure mode already fails `initialize()` one line up, so a separate handler would guard nothing while adding a bare `except Exception` the convention forbids.

**The tests are the point, not the one-liner.** `tests/unit/chat/test_conversation_handler_epoch.py` (6 tests). `TestProductionCallerExists` asserts that constructing a handler THE WAY PRODUCTION DOES leaves an epoch in the ledger -- the assertion every one of this branch's dead-wired features lacked. **Mutation-proved BOTH directions: commenting out the call fails 5 of the 6 new tests, and all 7 PRE-EXISTING epoch tests in `test_store.py` still pass without it.** That second half is the defect class demonstrated concretely -- those tests were correct and thorough about the method's behaviour and completely blind to whether anything invoked it. Also covers clock discipline (`activated_at` equals an injected fixed instant), idempotency across two handlers on one file db, and a disabled-store side-effect boundary. Unit **2588 passed / 6 skipped / 3 xfailed**.

**LIVE VERIFIED by restarting `mist-backend`:** log reads `Event store: wrote provisional initial epoch 1 (ontology=1.4.0, extraction=2026-06-12-r1)`; `epoch_ledger` went 0 -> 1 row, `provisional=1`. Verifying THIS fix by unit tests alone would have been self-defeating.

**The restart surfaced a second drift, and challenging it produced a better answer than the one first proposed.** `.env:48` pins `EXTRACTION_VERSION=2026-06-12-r1`, overriding the code's own default of `2026-06-14-r5` (`config.py:606` and `:662`), while `vault/writer.py:60` hardcodes r5 and `graph_writer.py:259` falls back to `"1.2.1"` under a comment reading "no hardcoded version literal" -- **four authorities.** The first instinct was "decide which value is right." Raj challenged the premise and was correct: **version stamps here are PURELY DESCRIPTIVE and nothing branches on them** -- the active ontology is chosen by Python import, not by the env var; `extraction/prompts.py` has zero references to `extraction_version`; every consumer only writes the value; and `canonical_serialize.py:39` deliberately EXCLUDES the stamp triple so the determinism proof reads "same log + same epoch => same facts". While data is seeded and regenerable, a bump is a scripted regeneration, not a migration. **What survives is that the VALUE does not matter but CONSISTENCY does:** `cache_key = sha256(event_id|ontology_version|extraction_version|model_hash)` is the sole mechanism where a label becomes a behaviour, and disagreement there is a hard `ColdCacheError` -- in the exact mechanism R1.4.5 depends on. Accepted fix, folded into R1.4.5: collapse to one authority, have the golden log's cache generator derive its stamps from the same place the rebuild reads them so they cannot drift by construction, drop the `.env` pins so env cannot silently override a code default with a STALER value, and remove the `graph_writer` fallback. Timing is fortunate -- **no extraction-cache database exists yet**, so there are zero mislabelled entries and the fix currently costs nothing; after R1.4.5 authors its cache it would invalidate every entry. Note `epoch_id 1` captured the drifted value and `ensure_initial_epoch` is idempotent, so collapsing the authorities must also supersede that row (it is `provisional=1`, explicitly R1.6's to redefine). ****[STATE CORRECTION 2026-08-03: the paragraph above describes a FOUR-authority problem that was already understated and is now RESOLVED. The collapse commit `11dd23c` found ELEVEN authorities, not four -- three of them live Cypher write paths stamping a stale `'1.0.0'` on every skill-derived, capability-derived and self-model entity. All eleven are collapsed to one each; `backend/knowledge/version_stamps.py` is the sole authority, `ONTOLOGY_VERSION` is DERIVED from the active ontology object rather than restated, and the `.env` pins were removed. `model_hash` was missed by that pass and collapsed separately in `67d4bf5`. The paragraph is retained because its REASONING -- that the values are descriptive but their CONSISTENCY is load-bearing through `cache_key` -- is still the correct frame; only its inventory was wrong. It was left standing as current for a day after the fix landed one commit later.]**

This downgrades the `ONTOLOGY_VERSION` drift R1.6 inherited from a correctness problem to a consistency-of-authority problem.**

PRIOR ENTRY -- 2026-08-01 (I7 -- seed embedding gate -- **COMPLETE, ff-MERGED to local `main` @ `e367ae4`, LIVE-VERIFIED READ-ONLY.** 5 commits (`63d4707` T1, `b830bbe` T2, `4148245`+`a5c70cb` T3, `e367ae4` T4), branch deleted, `main` now **5 ahead of origin** and NOT pushed. Closes the last deferred R1.4 review finding.

**What it does that nothing else could.** `check_embeddings` recomputes each seeded node's embedding from the AUTHORED seed source and compares by cosine (>= 0.999). That catches the mode no presence, count, dimension or norm check can see: a node whose source text changed while its stored vector did not. `_WIPE_NODES` only deletes seed-stamped nodes matching `NOT (n)--()`, so a seeded node that has acquired a conversation edge survives a wipe with its old vector; reapply refreshes `display_name`/`description` via `ON MATCH SET n += $properties` but not `embedding`, and the backfill then skips it because its `WHERE n.embedding IS NULL` guard is false. The vector stays stale indefinitely. **The blindness was structural, not an oversight:** `canonical_serialize.py:45` excludes `embedding`, so `assert_rebuild_twice_identical` and `live_vs_rebuilt_report` are byte-identical whether every vector is present, absent or all-zero.

**Shape of the gate.** Conditions are first-problem-wins per node: absent -> null -> non-list -> dimension -> L2 norm < 1e-6 -> recomputed-length disagreement -> cosine. `_CHECK_EMBEDDING_QUERY` returns `n.embedding` and NOTHING else -- the graph's `display_name`/`description` are not merely unused but unavailable, so a future edit cannot make the comparison self-satisfying by accident (a test pins `"display_name" not in query`). `GateResult.examined` added and the gate FAILS CLOSED at `examined == 0`, because a gate reporting `passed=True` having examined nothing is a defect this exact codebase shipped once (`check_negation_proximity`, C1, fixed in `608c1dc`). T1 extracted `embedding_text_for` so the em-dash join is answered in ONE place rather than the three it was heading for -- the direct C1 lesson -- with the separator written as `chr(0x2014)` so a non-UTF-8 round-trip cannot silently substitute a hyphen, and a test asserting each backfill's emitted text EQUALS the builder's output.

**T4 is what actually closes the finding.** The gate is registered as the fifth `seed-verify` gate AND run inside `cmd_seed` immediately after the backfill. The backfill runs AFTER the graph writes have committed, so a failure in it (model load, cache miss, OOM) leaves a fully-seeded, fully-unembedded graph that every other gate passes -- the shape of both historical live losses. Printing the backfilled count proves nothing: it counts rows the backfill BELIEVED it wrote, from the same code that failed to write them. `seed --no-embeddings` now makes `seed-verify` fail, intentionally.

**Verification.** MUTATION PROOF: weakening the cosine comparison to a presence check failed exactly 3 tests (`test_fails_when_the_stored_embedding_was_computed_from_different_text`, `test_reports_one_failure_per_bad_node_and_leaves_good_ones_alone`, `test_a_display_name_edit_that_left_the_vector_behind_fires_on_that_node_alone`) and nothing else moved. Real-source tests run against the actual `mist-memory/seed/*.md`, `examined >= 32`, covering BOTH text-builder branches (`user.md` has ZERO `description:` fields across its 11 nodes, `mist.md` has 14 across 21 -- so entity-partition nodes embed on `display_name` alone and that is legitimate, not a defect). LIVE `seed-verify` read-only: `embeddings` **PASS**; `containment` still FAILs the same 5 pre-existing content-drift facts, unchanged. Graph unchanged throughout: 32 nodes / 30 rels, 32/32 on `embedding`/`display_name`/`entity_type`. Unit **2582 passed / 6 skipped / 3 xfailed** (from 2552 at branch point, +30). No live write of any kind; `mist_admin.py seed` was never run.

**One plan overstatement corrected by the implementer:** the applier CAN write an embedding -- `SeedNode` is `extra="allow"` and `embedding` is not in `_APPLIER_OWNED_NODE_PROPERTIES`, so an authored `embedding:` key would flow through `$properties`. No seed node does this today, and condition 4 would catch a bad hand-authored vector anyway.

**ALSO FOUND 2026-08-01, verified, NOT yet fixed -- `ensure_initial_epoch` has ZERO production callers.** `backend/event_store/store.py:457` is correct and has 5 unit tests; a repo-wide grep returns only `store.py` itself and `tests/unit/event_store/test_store.py`. Live confirmation: `epoch_ledger` = 0 rows. R1.4's record states a provisional epoch was written; it was not. R1.6's `live == rebuilt` closure depends on it, and a golden-log replay has nothing to rebuild against without it. Call site resolved: `ConversationHandler.__init__`, immediately after `self.event_store.initialize()` -- the one path production always opens the store on (`mist_admin.py`'s construction is CLI-only). `self._now_fn` is already in scope by then, so no new plumbing. **This is the sixth instance of this branch's recurring defect class (a feature that passes every test while doing nothing); the fix must add a test asserting a production caller exists, or the next refactor silently re-orphans it.** A seventh instance was found in the same pass: `StalenessDetector.confirmation_list` is built, tested and scheduled weekly (`factories.py:492-493`) and consumed by nothing.

**The ~1-in-6 intermittent unit failure is CLOSED as not reproducible.** 14 consecutive green runs on clean `main` @ `a5c70cb`'s parent with varying `PYTHONHASHSEED`; under a true 1-in-6 rate that is a 7.8% outcome. Most plausibly killed by R1.3.1's follow-on filewatcher debounce fix (`40c8c35`), which took `test_filewatcher.py` from 5 failures in 7 runs to 5/5 green. Stopped at 14 rather than 20 because concurrent branch edits would have contaminated later runs.

PRIOR ENTRY -- 2026-07-31 (R1.4 -- seed-utterance migration + Phase-1 data gate -- **NODE-DEFINITION GAP CLOSED, LIVE-VERIFIED TWICE INCLUDING A FULL WIPE-AND-RECREATE CYCLE, NOT YET MERGED.** Branch `feat/r1.4-seed-source-and-data-gate`, 14 tasks landed (T1-T14). T10 found and reverted a live data-loss defect (see PRIOR ENTRY below for the full incident); the user's decision was to complete the design rather than revert or narrow scope. T11 added `SeedNode` (id + ontology type + open-ended descriptive properties, deliberately `extra="allow"` vs `SeedFact`'s `extra="forbid"`) and load-time referential integrity (every fact's subject/object must have a matching node definition -- the exact shape of T10's defect, now caught before any graph write). T12 made the applier write the ontology type label and every descriptive property (`admin.py`'s established `MERGE ... SET n:{label}` shape), round-trip-proven (every written property asserted equal to source, not merely "a write occurred") and mutation-proved against the specific case a reviewer independently re-checked before approving (dropping the `n += $properties` merge clause while params stayed intact -- Task 4's params-vs-query-text hole in a new costume). T13 re-authored the real seed source (`mist-memory/seed/{mist,user}.md`) with all 32 node definitions, scripted (not hand-transcribed) from the retired `scripts/seed_data.yaml` and cross-checked against both that YAML and the live graph node-by-node -- one genuine, deliberate exclusion: `mist-identity`'s live `personality_summary` originates from `GraphStore.ensure_mist_identity()`'s hardcoded bootstrap default, not the seed source, and was ruled dead/vestigial (nothing reads it; `ON CREATE SET`-only means it cannot survive a wipe regardless). T14 added `check_node_definitions` (the gate that would have caught T10's defect: every seeded node must carry its ontology type label and a non-null `display_name` in the live graph, tested on a graph state where it must actually fire) and fixed Gate 3's containment check to match on `SeedNode.display_name` directly instead of the raw kebab id.

**Live verification (T14), the real proof:** backed up fresh (`data/graph_snapshots/pre-r1.4-task14-full-backup-2026-08-01.json`, gitignored, local only), ran `mist_admin.py seed` twice. Run 1 (nothing pre-stamped) MERGE-matched the restored nodes and added the new properties -- not yet a real test. **Run 2 wiped 30 edges and 32 nodes and recreated every one of them from scratch** -- the exact scenario that destroyed the graph in T10 -- and the full post-run state was byte-identical to run 1 on every dimension checked: 32 nodes / 30 rels, 21 `:__SelfModel__` / 11 `:__Entity__`, 0 dual-labeled, 0 `VaultNote`, 32/32 stamped, 32/32 carrying `display_name` and `entity_type`, 32/32 embedded, every ontology label present at the correct count (`MistIdentity` 1, `MistTrait` 9, `MistCapability` 5, `MistPreference` 6, `User` 1, `Organization` 1, `Technology` 6, `Project` 1, `Skill` 1, `Concept` 1 = 32), and `ensure_mist_identity()`'s exact production query matched cleanly with `ON CREATE` correctly not firing. `seed-verify` (all four gates, `node-definitions` exercised live for the first time): `facts-present` PASS, `node-definitions` PASS, `negation-proximity` PASS, `containment` FAIL on 5 of 30 facts (below). `tests/integration/` (including `test_seed_label_split.py`'s own independent live reset+reseed, now safe to run and explicitly re-authorized): 34 passed / 23 skipped / 1 failed, the failure being the same pre-baselined `test_cluster_5_reproducers.py::test_retrieval_candidates_record_carries_session_id` by exact name. Full unit suite stable at 2537 passed / 6 skipped / 3 xfailed / 0 failed across every check in this pass.

**Two known, non-blocking items surfaced by this task, neither hit a STOP condition:**
1. **Gate 3 (containment) fails 5 of 30 real facts, genuinely, not a gate bug**: `llama-cpp` (display name "llama.cpp") vs the body's "llama-server" (different name, not a rendering mismatch); `cognitive-architecture` (display name "Cognitive Architecture") vs the body's lowercase hyphenated "cognitive-architecture"; and three hardware facts (`rtx-4070-super`, `ryzen-7-7800x3d`, `ddr5-32gb`) whose display names carry a vendor prefix or "RAM" suffix ("NVIDIA RTX 4070 SUPER", "AMD Ryzen 7 7800X3D", "32 GB DDR5 RAM") the body's natural-language mention omits. Not loosened to force green, per instruction -- these are real, minor content drifts between `seed_data.yaml`'s formal naming and `user.md`'s prose, for a human to reconcile (either direction) in a follow-up.
2. **A full wipe-and-recreate silently drops legacy bookkeeping properties that were never part of the `SeedFact`/`SeedNode` model at all**: `provenance`, `confidence`, `temporal_status`, `event_id`, `first_seen_at`, `last_seen_at`, `mutable`, `ontology_version` -- artifacts of the retired `apply_seed`/`_seed_metadata` system, present on every node after T10's restore (since MERGE preserved them across every merge-only run) and gone after run 2's genuine delete-and-recreate, because nothing in the new applier ever sets them. Distinct from the T10/T11-14 finding (which was about ontology labels and `display_name`/`entity_type`, all of which ARE now correctly preserved) -- this is the same architectural shape one property-family later. At least one concrete consequence identified: `admin.count_non_seed_entities()` (the safety guard behind `reset_graph`'s `--include-derived` refusal) reads `n.provenance == 'seed'`, which a freshly-recreated node no longer carries, so the guard would misclassify seed-derived nodes as non-seed after a second re-seed. Not fixed -- out of Task 14's stated scope (which was display_name/entity_type/labels), flagged for a decision on whether these fields still matter or are themselves vestigial like `personality_summary`.

**Whole-branch review (coordinator, 2026-07-31), two fixes landed after T14's report, both committed:** (1) Embeddings incident: the coordinator ran `_backfill_embeddings_for_seed` live directly (0->32/32) while T14's report was in flight, independently disclosed rather than left for me to discover as an unexplained write -- verified 32/32 at 384 dimensions, real distinct vectors, not zero-vectors. `tests/integration/knowledge/test_seed_label_split.py`'s `real_neo4j_connection` fixture (a pre-existing, T10-documented-not-introduced hazard: full `reset_graph(include_derived=True)` wipe against production by inherited `NEO4J_URI` configuration, unguarded by the existing `backend/knowledge/eval_isolation.py` mechanism) now fails closed -- `pytest.skip` unless `MIST_EVAL_ISOLATION` is active, `assert_neo4j_isolated` validates the target even then. Mutation-tested both directions (skips by default; raises `EvalIsolationError` before connecting when isolation is declared but `NEO4J_URI` still resolves live). This test no longer runs in the default integration pass -- a real, accepted coverage loss; pointing it at the disposable `mist-neo4j-eval` instance (`docker-compose.eval-neo4j.yml`, currently not running) is follow-up work. Committed `4bc0b10`. (2) **C1, a fifth instance of this branch's recurring defect class**: `check_negation_proximity` (Gate 4) searched the raw `fact.object` kebab id, never fixed to match T14's `check_containment` fix onto `SeedNode.display_name` -- real prose never contains the raw id, so the scan loop never ran and the gate reported `passed=True` having examined nothing (silent, unlike containment's loud 29/30 fail before its own fix). Live measurement: 4/30 facts scannable before the fix, 0/20 in `seed/mist.md` -- the entire persona layer outside the gate's reach. Fixed by extracting a shared `_search_term_for(fact_object, node_by_id)` used by both Gate 3 and Gate 4, per the review's suggestion that answering "how does this object appear in prose" independently in two places is what let this happen. Mutation-tested via a live revert of just the gate's resolution logic (helpers left intact): fails exactly the two tests built to catch it, nothing else moves. Re-verified live, read-only: node/rel counts and all properties unchanged (32/30, 32/32 throughout); `seed-verify` now: `facts-present`/`node-definitions`/`negation-proximity` PASS, `containment` still FAILs the same 5 pre-existing content-drift facts, unchanged. Unit suite 2543 passed (2537 + 6 new tests) / 6 skipped / 3 xfailed; integration re-confirmed 33/24/1 (one test moved skip, same pre-baselined failure). Committed `608c1dc`. **Still not merged** -- coordinator is adjudicating four further Important findings from the same review separately.

**Whole-branch review, round 2 (coordinator, 2026-07-31): I4/I1/I5 fixed, four more deferred with rulings, not merged yet.** Committed `a5a7499`. **I4 (must-fix, defeats wipe scoping):** `SeedNode.extra="allow"` let an authored property share a name with the applier's own bookkeeping stamps (`entity_type`/`seed_version`/`updated_at`/`created_at`); the `properties` dict spread the authored extras LAST, so an authored `seed_version` silently won -- un-wipeable graph litter, since `wipe_seed_version` scopes on an exact match. Two independent layers, each separately mutation-tested: `SeedNode._no_applier_owned_extras` (a `model_validator` rejecting the four reserved names as extras at construction time, covering the loader and any direct caller) and `applier.py`'s `properties` dict now spreading authored extras FIRST, applier-owned stamps LAST, with `created_at` excluded from the spread entirely (it must never be a `properties` key at all -- it is set only by `_MERGE_NODE`'s own `ON CREATE` clause; letting it in would corrupt the create-only guarantee via `n += $properties` firing on every future `ON MATCH`). **I1 (misleading R1.6 hand-off):** `log_regenerator.py`'s `copy_self_model_partition` docstring and call-site comment both still said `apply_seed_documents` "hardcodes `:__Entity__` with no routing to `:__SelfModel__`" -- false since this branch's own Task 4 partition rework (`5bbaac1`). Corrected: partition routing exists, the real remaining gap is `rebuild()` having no seed-apply step at all. **I5 (cheap, keeps two lists one authority):** the applier writes nodes driven by fact references, `check_node_definitions` iterates `doc.nodes` directly -- they agreed today (32/32) but nothing enforced it by construction. Added `_validate_no_unreferenced_node_definitions` (loader.py, the reverse of Task 11's referential-integrity check). **Deferred, ruled, not touched:** I2 (a future `seed_version` bump strands old content -- owned by whoever first bumps it, note in seed-source docs), I3 (`reseed()` non-atomic / `cmd_seed` takes no pre-wipe snapshot -- document the manual-backup requirement every live run in this sub-project already followed), I6 (a third seed document breaks `bootstrap_vault_from_seed` silently -- one doc line), I7 (no gate covers embeddings, the failure that recurred twice live -- a fifth gate is a genuine follow-up, not a same-pass addition). Live-verified read-only throughout (32/30, 32/32 unchanged); `seed-verify` unchanged (still the same 5 known content-drift `containment` failures). Unit 2552 passed (2543 + 9 new) / 6 skipped / 3 xfailed; integration 33/24/1, same pre-baselined failure. **Coordinator now owns final verification and the merge.**

**Not merged.** Coordinator owns the whole-branch review and the fast-forward. PRIOR ENTRY -- 2026-07-31 (R1.4, T10: **NOT SHIPPABLE AS DESIGNED, BLOCKED** -- the live-data-loss incident that led to the work above. Branch `feat/r1.4-seed-source-and-data-gate`, 9 tasks landed (T1-T9), T10 (retirements + live verification) found a live-data-loss defect during its own verification and stopped per design rather than merging over it. T1-T9 built a versioned seed source (`mist-memory/seed/{mist,user}.md`, `SeedFact`/`SeedDocument` models, `apply_seed_documents`/`reseed`, three `seed-verify` gates) to replace `scripts/seed_data.yaml`. **T10's live run of `reseed()`'s wipe-then-recreate cycle proved the new seed source cannot express NODE DEFINITIONS -- only FACTS (edges).** `SeedFact` carries subject/predicate/object/valid_from/valid_to and nothing else; the applier's `_MERGE_NODE` sets only `id`/`created_at`/`updated_at`/`seed_version`. The old `apply_seed` carried `entity_type`/`display_name`/description/pronouns/self_concept via structured per-entity YAML dicts and applied ontology-specific labels (`SET n:{label}`); the new one has no equivalent. This was invisible through T1-T9 because MERGE preserves untouched properties on a match, and every seed run before T10 matched richly-labelled nodes left over from the old `apply_seed` path. Only a genuine delete-and-recreate exposes it -- which is exactly what `reseed()`'s wipe does once a node carries the `seed_version` stamp from a prior run, and T10's live verification (run seed, run it again for idempotency) is what triggered that for the first time. **Live consequence, confirmed and then reverted:** all 32 nodes stripped to a bare partition label (`__Entity__`/`__SelfModel__`) plus four properties, zero ontology labels (`MistIdentity`/`MistTrait`/`MistCapability`/`MistPreference`/`User`/`Organization`/`Technology`/etc. all gone), zero `display_name`/`entity_type`/`pronouns`/`self_concept`. `GraphStore.ensure_mist_identity()` (called at backend startup) started raising `ConstraintValidationFailed` instead of no-op MERGE-ing, meaning the next backend restart would have hard-failed. Persona injection (`get_mist_identity_context()`) was degraded live. **Restored** from a pre-task full-graph JSON backup (`data/graph_snapshots/pre-r1.4-task10-full-backup-2026-07-31.json`, all labels + properties, taken before T10's first live write) -- verified node-for-node, `ensure_mist_identity()`'s exact production query re-tested clean post-restore. Graph is back to its R1.3.1 state; **do not re-run `mist_admin.py seed` or `tests/integration/knowledge/test_seed_label_split.py` until the model gap is fixed -- both wipe-and-recreate through the same defective applier and will immediately re-strip the graph.** **[SUPERSEDED 2026-08-01 -- READ THIS BEFORE ACTING ON THE SENTENCE ABOVE: R1.4 T11-T14 closed the node-definition gap and re-seeding is SAFE, proven twice live including one genuine full wipe-and-recreate. The prohibition is preserved only as history. It survived as a standing imperative for two days after it stopped being true, which is precisely the failure mode this file's maintenance protocol exists to prevent.]** Fix requires a T1/T4-level model change (a node-definition concept on `SeedFact`/`SeedDocument` carrying type + descriptive properties) -- explicitly out of scope for a same-task patch; not attempted. `scripts/seed_data.yaml` (`git rm`'d, UNCOMMITTED) is now understood to be the only remaining full node-definition source outside git history and the JSON backup -- do not let its staged deletion land without first extracting what the model-change work needs, or resolve it via `git restore --staged --worktree scripts/seed_data.yaml`. T10 additionally landed two independently-shippable, non-applier-touching fixes not yet committed: (1) `VaultWriter.upsert_identity_body` + `bootstrap_vault_from_seed` repointed onto `documents: list[SeedDocument]`, verbatim-body rendering for both `identity/mist.md` and `users/<id>.md` (old `upsert_identity`/structured-dict path kept, unused, undeleted -- ~10 pre-existing tests depend on it); (2) the T3 `origin` session-provenance discriminator's write-AND-read wiring closed end to end (`MIST_SESSION_ORIGIN` env -> `EventStoreConfig.session_origin` -> `ConversationHandler` -> `EventStore.start_session`; `ConversationSession.origin` field added -- the column existed since T3 but had no read path at all). A live-only Gate 3 (`seed-verify`'s containment check) defect was also found and ruled fixable independently (kebab-id vs display-name substring mismatch, 29/30 facts false-fail) but the fix was not implemented -- work stopped when the node-definition defect was found first. Full detail, all live numbers, and the complete file list: `.superpowers/sdd/2026-07-31-r1.4-seed-source-and-data-gate/task-10-report.md` (gitignored, local only). PRIOR ENTRY -- 2026-07-31 (R1.3.1 -- vault write-policy correction -- COMPLETE, ff-merged to local `main` @ `9e48a92` from `feat/r1.3.1-vault-write-policy` (17 commits, 8 tasks). Removed per-turn vault appending entirely: R1.3 deleted the `DERIVED_FROM->VaultNote` contract that was the ONLY justification for it -- the appended note was the anchor that edge pointed at -- so Bucket 2 is now synthesis-only. A session note is written ONCE at session end by the new `SessionSynthesizer` (`backend/chat/session_synthesizer.py`) from the session's turns READ FROM THE EVENT-STORE LOG, not live memory; the same code path therefore also serves `SessionNoteCatchup` (`backend/vault/session_catchup.py`, `run_forever()` wired into `server.py` lifespan) for sessions whose process died before session end. Four cost gates: skip sessions with no graph state (one Cypher), defer to live traffic, bounded per-pass retry persisted as `status: skipped` stubs, production event store only. Deleted `append_turn_to_session` / `update_entities_extracted` / `append_session_synthesis` / `mark_session_completed` / `_append_turn_sync` / `peek_turn_count` and the `SENTINEL` / `VaultConfig.append_sentinel` / `MIST_VAULT_APPEND_SENTINEL` leftovers; `MistSessionFrontmatter` dropped `turn_count` / `participants` / `append_sentinel_offset`, gained `title` + the `skipped` status. `write_session_note` derives `date:` from the path stem (`YYYY-MM-DD-<slug>`) and RAISES on a non-canonical path -- the old `datetime.now()` fallback drifted across UTC midnight, and the single session note MIST had ever written was itself a specimen of that bug (path allocated 06-09 at 23:59, note stamped 06-10 after midnight; corrected). **Also collapsed the two session-id namespaces:** `EventStore.start_session` now TAKES the chat-layer session id instead of minting its own `uuid4`, and `ConversationHandler._es_session_ids` is deleted outright -- which fixed a PRE-EXISTING bug where the live extraction path wrote the external id and the replay path wrote the internal one into `ConversationContext.conversation_id`, independently re-traced and confirmed. `MIST.md` (injected into MIST's context every turn) corrected and ADR-011 amended (amendment, not rewrite; `knowledge-vault` @ `3a94278`, committed local, NOT pushed). Process note worth keeping: TWO features on this branch passed their own tests while doing NOTHING -- catch-up synthesized zero notes because three id spaces were conflated, and `run_forever` ran at most once per boot because `is_conversation_active` was bound to `handler.sessions`, a map that only ever GROWS (its sole remover `clear_session` has zero production callers). Both were plan defects, not implementation defects, and both were found by adversarial live tracing, not by tests. Six separate guards were caught green with live regressions wired in. Final verification: unit **2404 collected / 2396 passed / 5 skipped / 3 xfailed**; integration 58 collected / 33 passed / 2 failed (BOTH still the same pre-existing baselined pair from R1.3, exact file:line match) / 23 skipped. **SMOKE-TESTED LIVE 2026-07-31** (first time this feature has ever executed -- the container had been up 26h, booted before the wiring existed): backend restarted clean, catch-up scheduled, ONE pass ran, 11 candidates found, LLM probed once, graph queried, all 11 skipped, ZERO synthesis calls, vault byte-identical, no exceptions, no spin. FOLLOW-ON 2026-07-31 (`40c8c35`, on `main`): the `test_filewatcher.py` debounce flake is FIXED -- all 13 fixed `asyncio.sleep(debounce_ms/1000 + margin)` waits replaced with a bounded `_wait_for(predicate, settle=, timeout=)` poll. The margin was never the issue (measured fire latency 100.7-103.0ms against a 250ms budget, ~147ms headroom), so failures required a >147ms event-loop stall that no fixed sleep can absorb. `_wait_for` returns rather than asserting, so callers' existing assertions remain the sole failure diagnostic; `settle` waits past the debounce window after the predicate first holds, used wherever the assertion is exactly-N or negative (polling for `>= 1` would otherwise return before a late duplicate arrived and pass vacuously -- which is exactly what the collapse tests exist to catch). Four pure-negative tests deliberately KEEP their fixed sleeps: with no positive condition to poll, the sleep only gives the watcher a chance to misbehave, and a stall makes a spurious call less likely, not more. Suite now 5/5 green across both fixed and random ordering (was 5 failures in 7 runs); `test_filewatcher.py` runtime 5.56s -> 4.31s. KNOWN OPEN: (1) **R1.4 GAINED A PRECONDITION:** the 11 legacy event-store sessions carry PRE-COLLAPSE uuids, so once R1.4 re-seeds `ConversationContext` on event-store ids they all become matching candidates, find no note at their derived path, and write a SECOND note for a session that already has one -- including the real `-37a8` note, which is `authored_by: user-edit`. Drain or age-bound those rows BEFORE re-seeding. (3) Minor, follow-up: a REFUSED session-note write is reported as success (`_handle_write_session_note` returns `str(path)` unconditionally), so the vault-op debug stream records `ok=True` for a write that did not happen. PRIOR ENTRY -- 2026-07-30 (R1.3 -- vault->graph fact retirement -- COMPLETE, ff-merged to `main` @ `cdcae55` from `feat/r1.3-vault-graph-fact-retirement` (24 commits, 10 tasks). Retired every path by which editing a vault markdown file wrote a FACT into Neo4j: deleted `GraphStoreProtocol.upsert_user` / `GraphStore.upsert_user`, `mark_orphaned_by_provenance_path` / `get_orphaned_provenance_paths`, `ExtractionPipeline.extract_from_file`, `GraphRegeneratorConfig`, and the whole `GraphRegenerator` class (`backend/knowledge/curation/graph_regenerator.py`) -- its bus-event responsibility moved onto `VaultFilewatcher._do_reindex`. Dropped `vault_note_path` threading from the curation path; retired the `--scope` / `--retry-orphaned` vault-rebuild CLI modes in `mist_admin` (both were already call-time dead by the time T8 removed them). Entity provenance re-anchored from `DERIVED_FROM->VaultNote` onto `EXTRACTED_FROM->ConversationContext`; `RebuildStamps` now ride that edge family across every extraction but remain WRITE-ONLY -- no consumer reads them, a real decision deferred to R1.4/R1.6. `apply_seed`'s `DERIVED_FROM->VaultNote` seeding path is DELIBERATELY RETAINED (R1.4's scope, not this branch's). Two rewritten integration tests (`test_phase3_production_wiring_smoke.py`, `test_vault_edit_read_path.py`, renamed from `test_adr010_invariant5.py`) assert the retirement by graph node/edge COUNT DELTA, not by shape -- nine mutations proved shape-based guards permeable before the delta form landed; the unit-level twin (`tests/unit/vault/test_filewatcher_graph_noop.py`) went through 4 mutation-hardening rounds to close 10/10 escape routes, including patching `GraphStore.__init__` on the class object (not a module binding) after two rounds of import-laundering escapes via `backend.factories` re-exports. Final verification (T10): unit 2358 passed / 5 skipped / 3 xfailed, 2366 collected, 0 errors (was 2426 at branch start; net -68 is the sum of nine independently-reconciled per-task deltas). Integration 58 collected / 0 errors / 33 passed / 23 skipped / 2 failed -- BOTH BASELINED, pre-existing, not this branch's: `test_cluster_5_reproducers.py::test_retrieval_candidates_record_carries_session_id` (`session_id` not reaching retrieval-candidate records) and `knowledge/test_seed_label_split.py::test_seed_yields_only_entity_nodes` (expects 31 seeded `:__Entity__` nodes, live graph has 11 -- state drift, plausibly the 2026-06-29 self-model dedup 41->21; asserts against the live DB, so the real fix is likely moving it onto the F1 isolated eval Neo4j). Two grep retirement proofs both clean against their expected-survivor lists (writer.py's graph->vault `upsert_user`/`upsert_user_snapshot`, the quarantined `backend/knowledge/regeneration/` package, `apply_seed`, the ontology's `VaultNote` node type, and the sidecar's `subject="VaultNote"` prompt-assembly pseudo-facts). One orphaned observation: `backend/knowledge/curation/bucket1_reader.py` (file->graph-edge parser built for the now-deleted `GraphRegenerator`) has no production caller left, only its own unit test; it is inert (returns a dataclass, performs no graph write) and not a retirement failure, but worth pruning or repurposing when R1.4 lands. NEXT: **R1.4 (seed-utterance migration) is LOAD-BEARING** -- the Phase-1 curated-profile facts written by the now-deleted `upsert_user` will not survive a graph rebuild until migrated to seed-utterances -- then R1.5 (LEARNING staleness) -> R1.6 (`live == rebuilt` GREEN closure + the new ADR formalizing graph-wins-for-facts, superseding ADR-010 Inv-5/Inv-6; R1.3 deliberately leaves ADR-010 formally unamended). PRIOR ENTRY -- 2026-07-29 (doc refresh after a ~1-month gap; content reflects HEAD @ `6a05cbd` on `main`, tree clean, 61 ahead of origin; last code work 2026-06-29. SINCE MIS-124, Sub-project A advanced into R1 -- the utterance->graph regenerator: R1.0 (`:__SelfModel__` partition) + R1.1 (deterministic identity resolver + canonical content-equality) landed 2026-06-15; the self-model partition migration was applied to live mist-neo4j and the self-model dual-seed dedup designed 2026-06-23; the dedup SHIPPED + APPLIED TO LIVE (self-model 41 -> 21 nodes, persona de-doubled, `identity/mist.md` now a graph no-op) and R1.2 (proof-first `log_regenerator`: cache-driven log->graph rebuild into isolated staging + rebuild-twice byte-identical determinism gate GREEN + `graph-rebuild-from-log --dry-run` CLI) landed 2026-06-29. extraction_version + ontology UNCHANGED since MIS-124 (`2026-06-14-r5` / v1.4.0 -- R1 is regeneration + determinism, not extraction). Tests ~2399 -> 2426 unit green. NEXT: R1.3 (vault->graph fact retirement) -> R1.4 (seed-utterance migration + Phase-1 data gate, LOAD-BEARING) -> R1.5 (LEARNING staleness) -> R1.6 (`live == rebuilt` GREEN closure + new ADR). PRIOR ENTRY -- 2026-06-14 (MIS-124 -- ontology v1.4.0 MECE taxonomy + entity canonicalization -- COMPLETE on `feat/mis-124-ontology-v1.4.0-mece-taxonomy`. `extraction_version = "2026-06-14-r5"`, ontology v1.4.0. Retired Topic (->Concept) + Milestone (->Event, event_type=milestone); added the `Abstraction` supertype with a generic `parent_type` mechanism + "emit parent when no child clearly fits" fallback over the 7 abstract leaves. Overlap-handling architecture: a canonical (id,type) resolver generalizing the normalizer `RESERVED_NAMES` (retired-type coercion, bounded curated registry, Metric value/unit + string numeric-first id canonicalization, parent fallback) + hierarchy-aware validator/scorer/dedup (accepts-parent-iff-accepts-Concept, `types_match`, cluster-widened Tier-3 dedup) + a gated specificity floor against fallback gaming. Third-party facts now tracked (Person/Org-sourced) with `CONFIDENCE_EXTERNAL.third_party_penalty` applied. F2 re-baseline (extraction-only, re-adjudicated gold 2026-06-14): **TYPING ACCURACY CLEARED 0.875 -> 0.909 (PASS)**; rel precision 0.812 -> 0.833 (DOCUMENTED NEAR-MISS of the 0.90 gate); rel recall 0.846, entity precision 0.854, RELATED_TO 0.000, neg-controls 0, specificity 1.000. The remaining rel-precision residual is model entity-extraction QUALITY (event-naming consistency, metric structured-field emission, occasional predicate choice) + small-model prompt sensitivity / flash-attn near-tie drift -- NOT a canonicalization gap; the canonicalization lever is spent (it closed typing and reduced live-graph fragmentation). Follow-up: constrained/grammar-guided decoding or a larger model (extraction-quality frontier, separate decision). Tests ~2309 -> 2399. PRIOR ENTRY -- 2026-06-13 (C3 -- Sub-project A extraction accuracy -- COMPLETE; `extraction_version = "2026-06-12-r4"`, ontology v1.3.0. assertion_kind signal landed end-to-end (shared `derive_assertion_kind` + explicit-kind gate in the engine, bucket scoring in the F2 scorer, prompt emits it); same-turn cease/assert arbitration fixed with live-Cypher proof (cease 7/7, retract 5/5, assert perfect). Ontology v1.3.0 adds RECOMMENDS + HAS_HABIT and retires the universal `started`/`ended`/`duration` relationship props (superseded by the bitemporal interval). F2 relationship precision 0.672 -> ~0.83 (r3 0.831 / r4 0.812, flash-attn near-tie band) -- a DOCUMENTED NEAR-MISS of the 0.90 gate, residual is ~70% entity-id/type canonicalization + flash-attn drift, not extraction quality (closed by follow-up B). V7 tool-decision PASS: recall 0.950 (from 0.650), precision 1.000, FP 0/5, deterministic 25/25. F2 now measured via a byte-reproducible extraction-only harness; full-chat non-reproducibility root-caused to chat-path stochasticity + flash-attention (not MIST code). Tests 2252 -> ~2309. Prior entry (2026-06-12): deep review of the 109-commit unpushed span -> 71 confirmed findings fixed across 10 batches (ontology v1.2.1 semantics, currency-filter gaps, production filewatcher writer wiring, temporal_status->CEASE, confidence forwarding, event-loop offloading, F2 gold-corpus user-anchor correction); tests 2158 -> 2252.)))
**Branch (CURRENT):** `fix/register-remediation-2026-08-03`, branched from local `main` @ `987651f`. `main` is 19 commits ahead of origin and has never been pushed. **`feat/r1.4-seed-source-and-data-gate` named below was DELETED after its ff-merge on 2026-08-01** -- it was still presented as the current branch by this file until 2026-08-03, which is the single most misleading kind of staleness here, since `/mist-status` reads this line into every session.

**Branch (HISTORICAL -- R1.4, merged and deleted):** `feat/r1.4-seed-source-and-data-gate` (branched from `main` @ `40c8c35`), 17 commits ahead (T1-T14 plus three whole-branch-review fixes: ...`b966938` T9, `84a5bd9` T10, `849ac3c` T11, `4fca9d1` T12, `e89073a` T13's test change, `4bc0b10` T14 (`check_node_definitions` + Gate 3 fix + `test_seed_label_split.py` fail-closed guard), `608c1dc` C1 (`check_negation_proximity` fix), `a5a7499` I4/I1/I5 (un-wipeable-litter defense, stale-comment correction, orphan-node rejection)). **MERGED 2026-08-01:** ff-merged to local `main` (`40c8c35..a5a7499`), which is now **125 commits ahead of origin**, still push-gated, never pushed. Merge gate verified by the coordinator, not taken from reports: unit 2552 passed / 6 skipped / 3 xfailed run twice (the known ~1-in-6 intermittent did not appear); integration 33 passed / 24 skipped / 1 failed, the pre-baselined `test_cluster_5_reproducers.py::test_retrieval_candidates_record_carries_session_id` by exact name; live graph 32 nodes / 30 rels with `display_name`, `entity_type`, `seed_version` and `embedding` all 32/32, and `MistIdentity` 1 / `MistTrait` 9 / `MistCapability` 5 / `MistPreference` 6 / `User` 1. I4's two layers were each proven independently -- the `SeedNode` validator rejects `seed_version`/`entity_type`/`updated_at`, and a poisoned node built via `model_construct` and fact-referenced so it reaches the write path still lands the applier's own stamps with `created_at` absent from the property map. `mist-memory/` (own local repo, no remote): `02a6bdc` (T13, node definitions in `seed/{mist,user}.md` + the T10 live-write capture) on top of `e7e4a99`/`34d4514`. Do NOT push. Four review findings are deferred with rulings, none blockers: **I2** a `seed_version` bump strands the previous version's content (the wipe scopes on the new version and never matches the old) -- R1.6 or whoever first bumps owns it; **I3** `reseed()` is non-atomic and `cmd_seed` takes no pre-wipe snapshot -- every live run in R1.4 took a manual backup first, make that a documented requirement; **I6** a third seed document breaks `bootstrap_vault_from_seed`, failing soft so the vault half silently stops updating; **I7** no gate covers embeddings, the one failure that recurred twice on live data -- genuinely wanted, deferred only because a fifth gate landing without the adversarial pass the other four received is precisely how C1 happened.
**Status:** make-mist-usable Phase 2 / Sub-project A (MIS-120). Chain landed: F1/F2/F3 -> C1/C2 (bitemporal engine) -> deep review (71 findings fixed) -> C3 (extraction accuracy) -> MIS-124 (ontology v1.4.0 MECE) -> R1.0/R1.1 (determinism substrate) -> self-model dedup (applied live) -> R1.2 (proof-first log_regenerator) -> R1.3 (vault->graph fact retirement) -> R1.3.1 (vault write-policy correction) -> R1.4 (seed-utterance migration, ff-merged 2026-08-01) -> **I7 seed embedding gate + `ensure_initial_epoch` wiring + version-stamp collapse + R1.4.5 golden log, all ff-merged to local `main` @ `987651f` on 2026-08-03. Current work: remediating the whole-codebase reachability review that ran alongside them -- see header and "Known live bugs".** Memory state: curated-profile recall works and reasons (Phase 1, validated 2026-06-09); conversational fact capture accurate but model-bounded (rel precision 0.833 documented near-miss -> MIS-125); durability now genuinely closed at the live level for the seeded profile -- the node-definition gap that would have broken it (T10's finding) is fixed and proven against an actual wipe-and-recreate cycle, not just unit tests. NEXT (after merge): R1.5 (LEARNING staleness) -> R1.6 (`live == rebuilt` closure + new ADR, which also inherits the two known-open items in the header: Gate 3's 5 content mismatches, and legacy bookkeeping properties not surviving a wipe). The rel-precision residual remains deferred to an extraction-quality decision (constrained/grammar-guided decoding or a larger model, MIS-125), unrelated to R1.4. ADR-011 codifies the three-bucket vault write pattern; ADR-017 at v1.1.1.

---

## Current Status

### Backend
- **Status:** CONTAINERIZED (Docker + CUDA 12.4). All development against the container; Windows native venv is corrupted.
- **Server:** FastAPI WebSocket on port 8001.
- **LLM:** Gemma 4 E4B Q5_K_M dense (carteakey-full recipe) via llama-server (llama.cpp OpenAI-compatible API). Selected 2026-04-16 via gauntlet (ADR-008 revised). Serving at `http://mist-llm:8080`.
- **LLM ctx_size:** 32K configured on llama-server; effective attention window ~8K (Gemma's trained context). Cluster 6's `context_budget.context_window=8192` default respects this.
- **StreamingLLMProvider abstraction:** `LlamaServerProvider` (primary), `OllamaProvider` (fallback), optionally wrapped by `InstrumentedStreamingLLMProvider` (Cluster 5) for JSONL observability.
- **ConversationHandler:** Cluster 3 persona injection + Cluster 6 `ContextBudgetPlanner` + `_build_request` Pydantic-dump helper (Cluster 5). `conversation_max_tokens=1024` (up from 400; Cluster 6 fix for Bug E).
- **ExtractionPipeline:** 6 stages + Stage 1.5 `SubjectScopeClassifier` (Cluster 1) — classifies each utterance as `user-scope | system-scope | third-party | unknown` between pre-processing and ontology extraction. Metadata threaded into `EXTRACTION_USER_TEMPLATE` as a `subject_scope` hint.
- **Voice Pipeline:** VAD -> Whisper -> Gemma 4 E4B -> Chatterbox Turbo TTS with streaming parallelism. ~4-5s TTFA.
- **Audio Transport:** Binary WebSocket frames (MIST protocol: 16-byte header + PCM16), RMS normalization (-20 dBFS), interrupt fade-out.
- **Log Streaming:** WebSocketLogHandler with per-logger gating, token bucket rate limiter, request ID propagation.
- **Persistent Logging:** `./logs/mist-backend.log` at DEBUG level (survives container removal).
- **Debug JSONL Observability:** `DebugJSONLLogger` with **7** record phases (`turn`, `extraction`, `llm_call`, `retrieval_candidates`, `llm_request_raw`, `reconciliation`, `vault`). **5 carry their own env gate** (`llm_call`, `retrieval_candidates`, `llm_request_dump`, `vault`, `reconciliation`); `turn` and `extraction` are gated by the master `MIST_DEBUG_JSONL` only. The module docstring still lists only the original five -- it predates the `reconciliation` and `vault` phases. See Cluster 5 artifacts below. **Correlation is currently degraded:** `event_id` is never passed to `record_vault_op` or `record_llm_request_dump`, so those records carry `"event_id": null` (register P8).
- **Knowledge Graph:** Extraction + curation pipeline + hybrid retrieval (graph + vector + vault sidecar, RRF merge). ADR-009 provenance separation structurally enforced (Cluster 2). MIST identity retrieval injects persona (Cluster 3). Ontology **v1.4.0** (verified at runtime via `version_stamps.ONTOLOGY_VERSION`, which DERIVES the stamp from the active ontology object rather than restating it) defines **51 edge types of which 39 are extractable**, and 30 node types of which 21 are extractable; each edge type declares reconciliation semantics (cardinality / temporal_class / contradicts / progression_supersedes) consumed by the schema-driven `ReconciliationEngine` (C1/C2); v1.3.0 (2026-06-12, C3) added `RECOMMENDS` + `HAS_HABIT` and retired the universal `started` / `ended` / `duration` relationship props (superseded by the bitemporal interval, which is the canonical fact-time channel); v1.2.1 (2026-06-12) removed the USES<->DISLIKES contradicts pair and USES progression (behavior is orthogonal to sentiment/competence -- the pairs erased co-true beliefs). Fact edges are append-only bitemporal versions keyed on `version_key` with 3-arm currency filters on every user-facing read. Validator constraints AND extractor `ALLOWED_*` sets derive from the ontology (Inv-A6; undirected predicates validate as unordered pairs). Drift guards standing: scorer frozensets, validator constraints, extractor allowlists, and the extraction-prompt sha256 <-> `extraction_version` pin.
- **Knowledge Seed:** 32-node baseline (1 MistIdentity + 9 MistTraits + 5 MistCapabilities + 6 MistPreferences + 1 User + 10 anchor entities; 20 identity relationships + 10 anchor relationships = 30 facts; all 32 embedded), now seeded via the versioned seed source (`mist-memory/seed/{mist,user}.md`, `mist_admin.py seed` -> `load_seed_documents` + `reseed`). **Safe to re-seed** as of R1.4 T11-T14: `SeedNode` (T11) + the applier writing ontology type labels and descriptive properties (T12) + the real source carrying node definitions (T13) means a wipe-and-recreate cycle now correctly restores every label and property -- proven twice live, including one run that performed a genuine full wipe. `scripts/seed_data.yaml` is deleted, and **that deletion IS on `main`** (verified 2026-08-03: `git ls-tree main scripts/` has no match, and the file is absent from the working tree). The "on this branch, not on `main`" qualifier was written pre-merge and was never updated when `feat/r1.4-seed-source-and-data-gate` was ff-merged; the deletion commit is `84a5bd9`. T13 restored the file only transiently/untracked to script the node extraction, then re-deleted it -- the extraction itself is preserved in `mist-memory` commit `02a6bdc` and in git history. Two known-open, non-blocking items: Gate 3 fails on 5/30 real facts (genuine content drift, not a gate bug) and a full wipe drops legacy `provenance`/`confidence`/etc. bookkeeping properties that predate the `SeedFact`/`SeedNode` model -- see header.
- **Vault Layer (Cluster 8, in progress):** NOTE -- the Phase 5/6/8 text in this bullet is HISTORICAL as of R1.3 + R1.3.1. Per-turn session-note appending and the `DERIVED_FROM`->`VaultNote` provenance edge described below are BOTH DELETED. Current behavior: one MIST-authored session note per session, written once at session end (or by startup catch-up) as an LLM synthesis sourced from the event-store log, via the single `VaultWriter.write_session_note` full-render path; entity provenance is `EXTRACTED_FROM`->`ConversationContext` carrying `source_utterance_id`. `RebuildStamps` still ride that edge family but remain WRITE-ONLY (no consumer reads them; deferred to R1.4/R1.6). The Phase 9 slug derivation and Phase 10 seed bootstrap paragraphs below remain accurate. Retained verbatim for archaeology: `backend/vault/` package with `VaultWriter` (serialized `asyncio.Queue` consumer for session-note appends, identity/user upserts), `VaultSidecarIndex` (sqlite-vec `vec0` + FTS5 + RRF hybrid query over two-tier chunks), `VaultFilewatcher` (watchdog daemon thread with 500ms debounce + asyncio bridge + 60s mtime audit job + MIST-write coordination for user-edit detection), Pydantic frontmatter models for the four `mist-*` note types, and `AuthoredBy` 5-state authorship enum. Wired through `VaultConfig` / `SidecarIndexConfig` / `FilewatcherConfig` on `KnowledgeConfig`. **Phase 5 integrated:** single server-owned VaultWriter built and started in `server.py` lifespan, plumbed through `VoiceProcessor -> ModelManager -> KnowledgeIntegration -> ConversationHandler`, with per-turn vault append after event-store write (failure-isolated per ADR-010 Invariant 6). **Phase 6 integrated:** `vault_note_path` is pre-allocated synchronously at `handle_message` Step 0 (via `_get_or_allocate_vault_path`) and threaded through `_extract_knowledge_async` -> `ExtractionPipeline.extract_from_utterance` -> `CurationPipeline.curate_and_store` -> `CurationGraphWriter.write`. Every upserted entity now emits a `DERIVED_FROM` edge to a `:__Provenance__:VaultNote {path}` node (MERGE-idempotent on path). New `VaultNote` ontology node type registered as bridging; `DERIVED_FROM` edge extended to permit `VaultNote` targets and `MistIdentity` sources. The graph is now formally rebuildable from the vault. **Phase 8 integrated:** rebuild-determinism stamps. New `RebuildStamps` frozen dataclass (`ontology_version`, `extraction_version`, `model_hash`) constructed by `build_curation_pipeline` from `KnowledgeConfig` and injected into `CurationGraphWriter`. Every `DERIVED_FROM`->`VaultNote` edge now carries the three stamps + `derived_at` timestamp on both ON CREATE and ON MATCH branches so re-extractions land the current stamps. New config fields `KnowledgeConfig.extraction_version` (default `"2026-04-17-r1"`, env `EXTRACTION_VERSION`) and `KnowledgeConfig.model_hash` (default `"gemma-4-e4b-q5-k-m-carteakey-full-v1"`, env `MIST_MODEL_HASH`). **Phase 9 integrated:** retrieval routing + slug improvement. QueryClassifier extended with a `historical` intent (regex patterns matching "what did we discuss"/"remember when"/"last time"/etc.) routed to the vault sidecar; `hybrid` now produces three-way RRF merges across graph + vector + vault sidecar via `_merge_rrf_three_way`. New `QueryIntentConfig` fields per ADR-010 weight table (`rrf_vault_weight=0.4` hybrid; historical-specific `0.2/0.1/0.7` graph/vector/vault). `KnowledgeRetriever` accepts an optional `vault_sidecar: SidecarIndexProtocol` plumbed top-down through `VoiceProcessor -> ModelManager -> KnowledgeIntegration -> build_conversation_handler -> build_knowledge_retriever`; `_vault_sidecar_retrieve` wraps `query_hybrid` and converts vec0+FTS5 results to `RetrievedFact` rows. Session slug derivation now extracts significant words from the FIRST USER UTTERANCE (stopwords + short tokens filtered, top 5 retained) with a 4-char SHA-256(session_id) suffix for guaranteed per-session uniqueness — produces filenames like `2026-04-22-vault-architecture-mist-a3f1.md` instead of opaque `2026-04-22-<sanitized-session-id>.md`. **Phase 10, AS IT ACTUALLY STANDS TODAY (corrected 2026-08-03 -- the prior text was re-certified as accurate while describing two functions that no longer run):** seed vault bootstrap. `mist_admin seed` calls `bootstrap_vault_from_seed(vault_writer, documents: list[SeedDocument], rendered_at)` (`backend/knowledge/admin.py:153`), which R1.4 T10 repointed off the retired `scripts/seed_data.yaml` dict and onto the versioned seed source. Each document's body is written VERBATIM -- `seed/mist.md`'s body IS `identity/mist.md`'s body -- rather than assembled from structured per-field dicts. The identity document goes through `VaultWriter.upsert_identity_body`; the user document through `VaultWriter.upsert_user` keyed on `source_path.stem`. **`VaultWriter.upsert_identity` (writer.py:272) is DEAD -- zero production callers**, retained only behind the `VaultWriterProtocol`. **`emit_seed_vault_provenance` no longer exists in production -- zero references outside two test docstrings that record its retirement.** R1.4 retired the last `DERIVED_FROM`->`VaultNote` path, so no `:__Provenance__:VaultNote` node or per-entity provenance edge is created by seeding any more; entity provenance is `EXTRACTED_FROM`->`ConversationContext` carrying `source_utterance_id`. `--no-vault-bootstrap` (`mist_admin.py:1494`) still opts out; bootstrap still auto-skips when `config.vault.enabled` is False. Filewatcher + sidecar share the same lifecycle. Phase 11 (CLI subcommands `vault-status` / `vault-reindex` / `vault-rebuild` / `vault-migrate`) is next.
- **Graph Regeneration (Sub-project A / R1):** `backend/knowledge/regeneration/` -- `log_regenerator.py` (cache-driven log->graph rebuild into an isolated staging graph; `ColdCacheError` on any extraction-cache miss, no in-loop LLM; self-model copy-forward + cross-layer edge re-derivation) + `rebuild_gate.py` (rebuild-twice byte-identical determinism gate + divergence report). Driven by `scripts/mist_admin.py graph-rebuild-from-log --dry-run`; fenced off from live by `backend/knowledge/eval_isolation.py:assert_rebuild_target_not_live` + `docker-compose.staging-neo4j.yml`. PROOF-FIRST as of R1.2 (2026-06-29): determinism proven at the unit level; NOT yet run against live data (cold-cache refusal -- warm-up is the documented prerequisite). **R1.3 (vault->graph fact retirement) COMPLETE 2026-07-30:** every vault-file-edit -> Neo4j-fact write path is now deleted (`GraphRegenerator` class, `GraphStore.upsert_user`, orphan-marking, `extract_from_file`, the vault-rebuild `--scope`/`--retry-orphaned` CLI modes); the only surviving `GraphRegenerator` reference is the quarantined `backend/knowledge/regeneration/graph_regenerator.py` (legacy utterance-based, byte-unchanged, not this class). **R1.3.1 (vault write-policy correction) COMPLETE 2026-07-31:** per-turn vault appending removed, session notes are end-of-session synthesis from the event-store log plus a startup catch-up pass, and the two session-id namespaces are collapsed into one (`EventStore.start_session` takes the chat-layer id; `_es_session_ids` deleted). **R1.4 (seed-utterance migration) is NEXT and LOAD-BEARING**, not optional follow-up: the Phase-1 curated-profile facts the now-deleted `upsert_user` used to write will NOT survive a graph rebuild until migrated onto seed-utterances, and that migration must run and validate before any live `live == rebuilt` cutover (R1.6). **R1.4 now carries a hard precondition discovered during R1.3.1's final review:** the 11 legacy event-store sessions predate the namespace collapse and carry old `uuid4` ids, so the moment R1.4 re-seeds `ConversationContext` nodes keyed on event-store ids, all 11 become matching catch-up candidates, find no note at their derived path (the slug hash differs -- `-9199` from the event-store id vs the real note's `-37a8` from the retired external id), and write a SECOND note for a session that already has one. The existing note is `authored_by: user-edit`, so the R1.3.1 user-edit guard will REFUSE to overwrite it -- meaning the duplicate lands alongside rather than clobbering -- but the duplicate is still wrong. Drain or age-bound those 11 rows BEFORE re-seeding. Real cutover + `live == rebuilt` closure = R1.6, which also owns the new ADR formalizing graph-wins-for-facts (ADR-010 Inv-5/Inv-6 stay formally unamended through R1.3 by design).
- **Tests:** **3069 unit tests passing, 7 skipped, 3 xfailed, 0 failed** (verified by a full container run on 2026-08-25, extraction-cache Phase 1 fix-wave landing -- see header; this line last read 2679 as of 2026-08-03 and had gone stale relative to the header's own 2996/3068 figures by the time this correction was made -- exactly the drift this line's own rule exists to prevent). Integration, re-verified 2026-08-25 by the coordinator (not this pass's own run) against current HEAD with a disposable staging Neo4j up (`docker compose -f docker-compose.yml -f docker-compose.staging-neo4j.yml --profile staging up -d mist-neo4j-staging`; auto-torn-down afterward via `--profile staging rm -sfv mist-neo4j-staging` -- its state is tmpfs, wiped on removal): `tests/integration/knowledge/` -> **12 passed, 14 skipped, 0 failed.** The rebuild coverage specifically -- `test_log_regenerator.py` + `test_golden_log_rebuild.py` -- is **8 passed, 0 failed**, including `test_cache_driven_rebuild_builds_entity_graph`, `test_cold_cache_refuses`, `test_rebuild_twice_byte_identical`, and `test_rebuild_twice_is_byte_identical_over_a_log_with_real_turns`. Every skip requires the separate *eval* Neo4j instance, which was NOT brought up this pass -- **the eval-gated tests did not run**; that gap is not closed by this verification and should not be read as closed. Live graph checked 32/30 both before and after. Superseded here the file's own earlier note ("last recorded 2026-08-03: 58 collected... 2 failed"), which predated this branch's changes to the replay path entirely. **There is exactly ONE authoritative unit-test count in this file, here.** A previous revision carried three mutually inconsistent counts in three places; if you update this number, update it here and nowhere else, and re-read it from a real run rather than from a task report. Run inside container: `MSYS_NO_PATHCONV=1 docker compose exec -T mist-backend python -m pytest tests/unit/ -q`. Suite warnings are not a stable count on this branch (see header) and are not tracked here. **Suite was fully green as of `40c8c35`** -- the long-standing `test_filewatcher.py` debounce flake was closed by replacing 13 fixed-sleep waits with a bounded `_wait_for` poll helper (see the 2026-08-19 PRIOR ENTRY). Verified 5/5 across both fixed (`-p no:randomly`) and default random ordering. When adding filewatcher tests, use `_wait_for` rather than `asyncio.sleep`, and pass `settle=` whenever the assertion is exactly-N or negative.

### Frontend (Tauri 2.x + React 19 + react-three-fiber)
- **Repo:** separate git repository nested at `./mist-frontend/` inside this repo (own .git, no remote configured; intentional per `feedback_no_push_docs`).
- **Status:** Production-ready as of 2026-05-08 spatial-app reframe. FE/BE integration Wave 1 shipped 2026-05-10 (handshake, heartbeat, state_cycle, turn-streaming, vad_status, log streaming, health_status, error discrimination) on branch `integration/v1`. Subsequent waves cover backend tool-call events, cards, graph_subgraph, and frontend visual polish.
- **Protocol contracts:** ADR-016 (LLM-mediated frontend tool calls; BE-decided routing) + ADR-017 (WebSocket message contract; discriminated events, lifecycle, error model). Both live in `knowledge-vault/Decisions/`.
- **Flutter Desktop (`mist_desktop/`):** Decommissioned 2026-05-11. Git history at commit `e18c092` preserves the Flutter source. Workstream `mist-ai-frontend-audit-remediation` archived under `knowledge-vault/_archive/legacy-flutter-frontend/`.

### Code Quality
- Full pre-commit suite: black, ruff (D102 strict), bandit, codespell, AI-slop pattern checker, trim whitespace, fix end-of-files, large file + merge conflict + private key detection.
- AI-slop pattern checker enforces no emoji/unicode-decorative/arrow symbols in new code.
- CI configured via GitHub Actions.

---

## MVP Knowledge Integration — Cluster Status

**Workstream:** `mist-ai-knowledge-integration-mvp-validation` — structurally COMPLETE 2026-04-22. All 8 clusters shipped; workstream closed with `/vault-end-session` on the Cluster 8 closure note. Full detail in the knowledge-vault workstream note at `knowledge-vault/Projects/mist-ai/workstreams/mist-ai-knowledge-integration-mvp-validation.md`.

**All eight architectural clusters complete (2026-04-22).** Cluster roll-up:

| Cluster | Scope | Closure date | Gauntlet artifact |
|---|---|---|---|
| 1 | Ontology expansion + subject-scope classifier | 2026-04-21 | post-cluster-1-gauntlet-report-2026-04-21.md |
| 2 | Graph provenance separation (ADR-009) | 2026-04-20 | post-cluster-2-gauntlet-report-2026-04-20.md |
| 3 | Identity layer + persona injection + AI-slop filter + dual temperature | 2026-04-21 | post-cluster-3-gauntlet-report-2026-04-21.md |
| 4 | Deterministic rails (Bugs A, C, G, K) | 2026-04-20 | post-cluster-4-gauntlet-report-2026-04-20.md |
| 5 | Observability (llm_call + retrieval_candidates + llm_request_raw JSONL phases) | 2026-04-21 | v6-cluster-5-diagnostic-report-2026-04-21.md |
| 6 | Context budget (ContextBudgetPlanner) + max_tokens=1024 fix | 2026-04-21 | post-cluster-6-gauntlet-report-2026-04-21.md |
| 7 | Existing-data migration | Absorbed into Cluster 8 Phase 10 (seed vault bootstrap) | — |
| 8 | Vault-native memory (ADR-010, 12-phase) | 2026-04-22 | post-cluster-8-gauntlet-report-2026-04-22.md |

**Phase 4 acceptance gates** (all cleared at MVP close):

- Relationship correctness >= 80% — CLEARED (92% on v1-mist-scope-inputs.jsonl; V6 `mist-identity USES X` edges landing post-Cluster-1)
- Post-session retrieval semantic content >= 80% — CLEARED (9/10 user-facing facts post-Cluster-2)
- Emoji violations = 0 — CLEARED (held across V4+V5+V6)
- Empty responses < 10% — CLEARED (0/30 V6 post-Cluster-6)
- LLMRequest validation errors = 0 — CLEARED
- Unit tests >= 900 green — CLEARED (1488 at post-MVP close)

Plan artifact for overnight post-MVP run: `~/.claude/plans/peaceful-greeting-bee.md`.

**Bug status (P1/P2 from 2026-04-17 gauntlet):**
- A (83% NULL provenance) — CLEARED (Cluster 4)
- B (identity drift, emoji leak, AI slop) — CLEARED (Cluster 3)
- C (LLMRequest tool_calls schema) — CLEARED (Cluster 4)
- E (empty LLM responses) — CLEARED (Cluster 6; root cause: max_tokens=400 truncating tool-call JSON)
- G (reserved-namespace guard) — CLEARED (Cluster 4)
- I (LEARNING->USES slippage) — CLEARED (Cluster 1: scope classifier + prompt rebalance)
- J (MIST-tooling attributed to Raj USES) — CLEARED (Cluster 1: validator accepts MistIdentity source on USES/DEPENDS_ON/WORKS_WITH; normalizer forces MistIdentity type on reserved names)
- K (prompt-injection written as fact) — CLEARED (Cluster 4 pre-filter + prompt tightening held through Cluster 1; declarative-framing residual noted as P1)
- N (retrieval returns only provenance plumbing) — STRUCTURALLY RESOLVED (Cluster 2)

---

## Active Work

### Current Focus
make-mist-usable Phase 2 / Sub-project A (MIS-120). **Latest landed on `main`:** audit finding
Q2-1 close-out at `330ff1c`, 2026-08-19 (see header PRIOR ENTRY) -- `main` HEAD is `330ff1c`,
confirmed via `git rev-parse main` (and `origin/main` matches: both `330ff1c`) from the
extraction-cache branch. `git merge-base HEAD main` returns the same value here but is the WRONG
command to cite for this claim -- it names the branch point, which would still read `330ff1c` if
`main` had since advanced past it; `git rev-parse main` is what actually establishes current
`main` HEAD. **In flight, not yet merged:** extraction-cache Phase 1 (spec D1, D2, D3, D10) on
branch `feat/extraction-cache-production-writer`, COMPLETE as of this entry (see header) -- this
is the cache warm-up enabler the previous wording of this line was still waiting on: the
production extraction path now writes the cache row every rebuild needs. NEXT, in order: merge
this branch -> extraction-cache Phase 2/3 (D4, then D5/D6/D7 -- coupled per spec section 8) ->
R1.6 (`live == rebuilt` GREEN closure gate + the new ADR formalizing graph-wins-for-facts,
superseding ADR-010 Inv-5/Inv-6). Phase 4 (D8) is independent and can land any time after Phase 1;
it is not part of that merge-blocking sequence. R1.4 and R1.4.6 (seed-utterance
migration, hydrator work) are ALSO landed on `main` before `330ff1c` -- this line previously read
as if R1.3 (2026-07-30) were still the latest landed work, which understated 6+ commits/weeks of
progress; the header's PRIOR ENTRY chain is the accurate history, not this paragraph, whenever the
two disagree. R1 specs/plans are local/gitignored under `docs/superpowers/`. The dated "Recently
Completed" entries below are the pre-Phase-2 MVP archive (closed 2026-04-22).

### Recently Completed (2026-04-22, overnight autonomous)
- **Ontology additive expansion (Commits A1-A4):** 4 new node types (`Date`, `Milestone`, `Metric`, `Document`) + 4 new edge types (`OCCURRED_ON`, `HAS_METRIC`, `REFERENCES_DOCUMENT`, `PRECEDED_BY`) under ontology v1.0.0 (additive under major). Validator constraints (`RELATIONSHIP_CONSTRAINTS`), extractor `ALLOWED_*` frozensets, storage traversal allowlist (`_USER_FACING_REL_TYPES`), extraction system-prompt enumeration, and 3 new few-shot examples all updated atomically. Standing ontology-consistency guard test ensures future additions can't drift validator/ontology apart. Commits `baeef03` -> `54a10d5`.
- **Scorer drift repair (Commit B1):** `scripts/eval_harness/scorers.py` resynced with current extractable ontology — closes 5-week drift from Cluster 1 and this morning's Phase A additions. Added `tests/unit/test_eval_harness_scorers.py` as a standing parity guard (set-equality against ontology, bidirectional diff message, membership landmark tests). Commit `916407f`.
- **Docker flash-attn fix (Commit C1):** `docker/backend/Dockerfile` install of `flash-attn==2.8.3` was silently skipping for months due to missing build deps (`psutil`, `ninja`, `wheel`). Added explicit `pip install ninja packaging wheel psutil` before the flash-attn line + replaced silent `|| echo "skipped"` with loud `[FLASH-ATTN BUILD FAILED]` error (still exits 0 for build resilience). Post-fix rebuild verified: `flash_attn-2.8.3` compiles in ~20s. Stack restart pending user action (denied in auto mode). Commit `e45d8b5`.
- **V7 tool-heavy probe set (Commit D1):** `data/ingest/v7-tool-heavy-inputs.jsonl` -- 25 queries with labeled expected_behavior (20 positive + 5 negative controls) to unblock `mist-ai-tool-calling-production-rigor`. Design doc at `scripts/eval_harness/v7_probe_set_design.md`. Each line is force-added over gitignore because it's engineered research data, not runtime output. Commit `8a300b9`.
- **V6 gauntlet rerun (E1, folded into this commit):** 30/30 OK, 0 empty, 0 emoji, 0 LLMRequest errors. No regressions vs post-Cluster-8 baseline. Document (2), Date (1), Metric (1) node types produced spontaneously under the expanded system prompt on first run; Milestone not produced (conversation content doesn't motivate it). No new typed edges yet (producer-side, not validator-side — morning followup). Report at `data/ingest/post-ontology-expansion-gauntlet-report-2026-04-22.md` (gitignored).

### Recently Completed (2026-04-21, end of day)
- **Cluster 1 (Ontology + subject-scope classifier):** 8 commits (`4dc7204` -> `3b10a24`) on main, pushed to origin. Extended validator `RELATIONSHIP_CONSTRAINTS` to accept Organization + MistIdentity as source for USES/DEPENDS_ON/WORKS_WITH. Added 4 new MIST-scope predicates: `IMPLEMENTED_WITH`, `MIST_HAS_CAPABILITY`, `MIST_HAS_TRAIT`, `MIST_HAS_PREFERENCE`. Added `MistIdentity` as extractable entity type (13 total). New `SubjectScopeClassifier` module running as Stage 1.5 AFTER significance + dedup gates, writing `subject_scope` metadata to PreProcessedInput, threaded into extraction user template. Rewrote `EXTRACTION_SYSTEM_PROMPT` removing user-centric bias; 3 user / 3 system / 1 third-party / 1 empty example balance. Normalizer `RESERVED_NAMES` now remaps both id AND entity_type to MistIdentity. Cluster 3 integration: `get_mist_identity_context` UNIONs HAS_* (seed) and MIST_HAS_* (extracted) into one merged set. Cluster 2 integration: `_USER_FACING_REL_TYPES` extended with new edges so multi-hop traversal expands through them. Bug J closure evidence in V6: `mist-identity -[USES]-> lancedb/neo4j/llamacpp/sentence-transformers` landed (all dropped pre-Cluster-1). V1 probe = 11/12 (92%). V6 = 0/30 empty, 0 emoji, 0 Bug C. +44 net new tests (1022 -> 1066).
- **Cluster 6 (Context budget + max_tokens fix):** V6 empty-response rate 53% -> 0%. `LLMConfig.conversation_max_tokens=1024` (up from 400) at all three ConversationHandler invoke sites fixes the "GHOST turn" failure mode (tool-call JSON truncation). `ContextBudgetPlanner` with TokenCounter + HistoryStrategy protocols + SlidingWindowStrategy default provides defense-in-depth. Commits c4c4d71, d354f30, 997517f, c800e35. +32 net new tests.
- **Cluster 5 (Observability):** Three JSONL record phases added to `DebugJSONLLogger` (`llm_call`, `retrieval_candidates`, `llm_request_raw`) each with its own env gate. `InstrumentedStreamingLLMProvider` wraps any concrete provider transparently; `llm_call_context` ContextVar threads caller metadata (`session_id`/`event_id`/`call_site`/`pass_num`). All factories wired. Commits 27af364, f5d0ec4, ab85115, e7ca7e2 + polish 3c8f0b2. +44 net new tests.
- **Cluster 3 (Identity + AI-slop filter + dual temperature):** 6 deliverables — config split (`conversation_temperature=0.7`), slop detector library, pref-no-ai-slop seed, QueryClassifier identity intent at priority 0, `retrieve_mist_context()` + `MistContext` renderer with HARD RULES framing, response post-filter with regen + strip_fixable fallback. Bug B closed: 0 emoji across 46 V4+V5+V6 turns; consulting-voice markdown drift -56%. Commits f306788 -> 6124e43. +90 net new tests.

### Previously Completed (2026-04-20)
- **Cluster 4 (Deterministic rails):** Bug A fix (ON CREATE SET e.provenance='extraction'); Bug C fix (`list[dict[str, Any]]` widening in LLMRequest.messages); Bug G (RESERVED_NAMES table in EntityNormalizer); Bug K two-layer fix (pre-filter + prompt tightening).
- **Cluster 2 (ADR-009 graph provenance separation):** 5 writer sites migrated to `:__Provenance__` base label; retrieval multi-hop filter anchored at `:__Entity__`; `mist_admin graph-reset --include-derived`; `graph-stats` three-section output. Canonical V6 turn-30 probe returns 9/10 user-facing facts vs morning's 0.

### Previously Completed (2026-04-16 -> 2026-04-08)
- Gemma 4 E4B selected as production model via gauntlet (ADR-008 revised)
- Model backend migration: Ollama -> llama-server via `StreamingLLMProvider` abstraction
- Binary WebSocket audio transport (MIST protocol, ~7x bandwidth reduction)
- Personality system (YAML per voice profile)
- FRIDAY default voice profile

### Blockers
None. MVP closed; tool-calling workstream unblocked by V7 probe set.

---

## Debug Observability Quick-Start

`DebugJSONLLogger` writes structured JSONL records to the path set by `MIST_DEBUG_JSONL`. Three phase-specific gates layered on top:

```bash
# From Git Bash on Windows: MSYS_NO_PATHCONV=1 prefix is REQUIRED
# (see reference_docker_exec_path_mangling memory) or /app/... gets path-translated.
MSYS_NO_PATHCONV=1 docker compose exec -T \
  -e MIST_DEBUG_JSONL=/app/data/ingest/session.jsonl \
  -e MIST_DEBUG_LLM_JSONL=1 \
  -e MIST_DEBUG_RETRIEVAL_JSONL=1 \
  -e MIST_DEBUG_LLM_REQUESTS=1 \
  mist-backend python -m scripts.mist_admin replay \
  /app/data/ingest/v6-inputs.jsonl \
  --session-id diagnostic \
  --output /app/data/ingest/report.jsonl
```

**Emitted phases when gates are open:**
- `phase: "turn"` — per-turn wrapper with event_id/session_id/utterance + retrieval summary + llm_passes + total_turn_ms. (Pre-Cluster-5 infrastructure.)
- `phase: "extraction"` — per-turn extraction stats (entities_count, avg_confidence, graph_writes). (Pre-Cluster-5 infrastructure.)
- `phase: "llm_call"` — full request/response at every provider.invoke(). Content/tool_calls/usage/latency_ms. call_site-tagged: `chat.initial`, `chat.final`, `chat.regen`, `extraction.ontology`, `extraction.internal_derivation`.
- `phase: "retrieval_candidates"` — full graph + vector candidate pools from `KnowledgeRetriever.retrieve()` BEFORE RRF merge + rank truncation. Gate: `MIST_DEBUG_RETRIEVAL_JSONL=1`.
- `phase: "llm_request_raw"` — pre-validation LLMRequest kwargs dump on Pydantic ValidationError. Gate: `MIST_DEBUG_LLM_REQUESTS=1`.

All records carry `ts_iso` + `session_id` + `event_id` for cross-record joins.

---

## Dependency Injection Contract

All classes depending on external systems (Neo4j, LLM, embeddings, event store, vector store, debug logger) accept dependencies as required constructor parameters. Factories in `backend/factories.py` own all wiring with real implementations.

**Factory entry points:**
- `build_conversation_handler(config, llm_provider=None)` — composition root for chat. Reads `DebugJSONLLogger.from_env()` once, logs active phase gates, threads the logger through `build_llm_provider` (wraps with `InstrumentedStreamingLLMProvider` when `llm_call_enabled`) and `build_knowledge_retriever` (forwards `debug_logger`). Returns a `ConversationHandler` with all dependencies wired.
- `build_extraction_pipeline(config, graph_store=None, llm_provider=None, include_curation=True, include_internal_derivation=True)` — extraction + curation + internal derivation pipeline.
- `build_knowledge_retriever(config, graph_store=None, vector_store=None, embedding_provider=None, debug_logger=None)` — hybrid retriever (graph + vector + RRF).
- `build_llm_provider(config, debug_logger=None)` — provider with optional instrumentation.

---

## Architecture Overview

### Docker Stack
```
docker-compose.yml              # 3 services: mist-backend, mist-neo4j, mist-llm
docker-compose.override.yml     # Dev mode volume mounts (backend/tests/scripts/voice_profiles bind-mounted)
docker/backend/Dockerfile       # CUDA 12.4 + Python 3.11 + Chatterbox
```

**Volume mounts** (backend code hot-reloadable):
- `./data:/app/data` (graph snapshots, JSONL diagnostics, event store SQLite)
- `./logs:/app/logs` (persistent logs)
- `./backend:/app/backend`
- `./tests:/app/tests`
- `./scripts:/app/scripts`
- `./voice_profiles:/app/voice_profiles`

### Backend Structure
```
backend/
├── server.py              # WebSocket server (port 8001)
├── voice_processor.py     # Voice pipeline orchestration
├── audio_protocol.py      # MIST binary frame builder
├── log_handler.py         # WebSocketLogHandler
├── request_context.py     # ContextVar propagation
├── sentence_detector.py   # Streaming TTS sentence boundary detection
├── debug_jsonl_logger.py  # Cluster 5: 5-phase JSONL sink with env gates
├── factories.py           # Composition root
├── errors.py              # MistError hierarchy
├── interfaces.py          # Protocols (EmbeddingProvider, VectorStoreProvider, GraphConnection, EventStoreProvider)
├── chat/
│   ├── conversation_handler.py   # Persona + budget + slop + post-filter (Clusters 3, 5, 6)
│   ├── mist_context.py           # Cluster 3 MistContext dataclasses + renderer
│   ├── slop_detector.py          # Cluster 3 pattern catalogue
│   └── context_budget.py         # Cluster 6 planner + TokenCounter + HistoryStrategy
├── llm/
│   ├── provider.py                   # Abstract StreamingLLMProvider ABC
│   ├── llama_server_provider.py      # Primary concrete provider
│   ├── ollama_provider.py            # Fallback concrete provider
│   ├── instrumented_provider.py      # Cluster 5 wrapper + llm_call_context ContextVar
│   └── models.py                     # LLMRequest/LLMResponse/ToolCall Pydantic models
└── knowledge/
    ├── config.py                     # KnowledgeConfig + nested configs
    ├── models.py                     # RetrievalResult, RetrievedFact, QueryIntent, etc.
    ├── embeddings.py                 # EmbeddingGenerator (Sentence Transformers)
    ├── extraction/                   # 6-stage extraction pipeline + ontology constraints + validator
    ├── curation/                     # Dedup + reconciliation.py (plan_edge planner + ReconciliationEngine) + intervals.py + graph writer + confidence + scheduler (conflict_resolver.py deleted in C2)
    ├── ingestion/                    # Markdown ingestion for vector store
    ├── retrieval/
    │   ├── knowledge_retriever.py    # Hybrid retrieval + identity-intent routing (Cluster 3)
    │   └── query_classifier.py       # Intent classification (live/relational/factual/hybrid/identity/historical)
    ├── regeneration/                 # Tombstone: vault->graph regenerator retired (R1 will rebuild from utterances; factory raises)
    ├── eval_isolation.py             # F1: fail-closed eval Neo4j (host,port) allowlist
    └── storage/
        ├── neo4j_connection.py
        ├── graph_executor.py         # Async/sync boundary
        └── graph_store.py            # GraphStore + get_mist_identity_context (Cluster 3)
```

### Frontend Structure
Frontend lives in a separate git repository nested at `./mist-frontend/` (own .git; Tauri 2.x + React 19 + react-three-fiber). See that repo for its internal layout. Backend integration is contract-only via ADR-016 + ADR-017 (v1.1.1).

---

## Configuration

### Environment Variables (.env / .env.example)
```bash
# Neo4j (NEO4J_URI is HARDCODED in docker-compose.yml environment to the in-network
# bolt://mist-neo4j:7687 -- the .env value is for host-side tools and must not leak in)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Rebuild-determinism stamps (drift guard pins EXTRACTION_VERSION to the prompt sha256)
ONTOLOGY_VERSION=1.3.0
EXTRACTION_VERSION=2026-06-12-r4
MIST_MODEL_HASH=gemma-4-e4b-q5-k-m-carteakey-full-v1

# LLM backend
LLM_BACKEND=llamacpp                       # llamacpp (default) | ollama (fallback)
LLM_SERVER_URL=http://mist-llm:8080
MODELS_DIR=./models                        # Host path; mounted read-only into mist-llm
LLM_MODEL_FILE=unsloth/gemma-4-E4B-it-Q5_K_M.gguf
LLM_CTX_SIZE=32768                         # llama-server ctx_size; effective attention ~8K
LLM_TEMPERATURE=0.0                        # Extraction default
LLM_CONVERSATION_TEMPERATURE=0.7           # Conversation default (Cluster 3 split)
LLM_CONVERSATION_MAX_TOKENS=1024           # Cluster 6 Bug E fix (was hardcoded 400)
MODEL=gemma-4-e4b

# Cluster 6 context budget
MIST_CTX_BUDGET_ENABLED=true
MIST_CTX_BUDGET_WINDOW=8192
MIST_CTX_BUDGET_OUTPUT_RESERVE=512
MIST_CTX_BUDGET_SAFETY=256
MIST_CTX_BUDGET_RETRIEVAL_RATIO=0.4
MIST_CTX_BUDGET_HISTORY_STRATEGY=sliding_window

# Cluster 5 observability (all off by default)
# MIST_DEBUG_JSONL=/app/data/ingest/debug.jsonl
# MIST_DEBUG_LLM_JSONL=1
# MIST_DEBUG_RETRIEVAL_JSONL=1
# MIST_DEBUG_LLM_REQUESTS=1

# Voice / TTS
TTS_ENABLED=true
TTS_ENGINE=chatterbox
VOICE_PROFILE=friday

# Feature flags
ENABLE_KNOWLEDGE_INTEGRATION=true

# Event store (Layer 1)
EVENT_STORE_DB_PATH=/app/data/event_store.db
EVENT_STORE_AUDIO_DIR=/app/data/audio

# Vector store (Layer 2)
VECTOR_STORE_DATA_DIR=/app/data/vector_store
```

### Critical Settings
- `LLM_BACKEND=llamacpp` (Gemma 4 E4B primary; `ollama` available for fallback)
- `LLM_MODEL_FILE=unsloth/gemma-4-E4B-it-Q5_K_M.gguf` (carteakey-full recipe per ADR-008)
- `MIST_CTX_BUDGET_ENABLED=true` (Cluster 6 default; disable only to confirm budget-related regressions)
- Docker data root: `D:\Users\rajga\DockerData` (not default C:)
- Git Bash on Windows: prefix `docker compose exec` with `MSYS_NO_PATHCONV=1` when passing `/app/...` paths in env vars

---

## Tech Stack

### Backend
- Python 3.11+
- FastAPI + Uvicorn (WebSocket server)
- Docker (nvidia/cuda:12.4.0-devel-ubuntu22.04)
- PyTorch 2.6.0+cu124 (flash attention enabled)
- llama-server / llama.cpp (LLM inference — Gemma 4 E4B Q5_K_M via GGUF, OpenAI-compatible API)
- `StreamingLLMProvider` abstraction: `LlamaServerProvider` primary, `OllamaProvider` fallback, `InstrumentedStreamingLLMProvider` decorator
- openai + httpx (LLM client)
- Whisper (STT — base model)
- Chatterbox Turbo (TTS — zero-shot voice cloning, MIT)
- Neo4j 5.x (knowledge graph; `:__Entity__` user-facing, `:__Provenance__` audit-trail per ADR-009; bitemporal versioned fact edges per C1/C2)
- LanceDB (vector store Layer 2 -- legacy; sidecar sqlite-vec covers vault retrieval, see KNOWN_ISSUES)
- sqlite-vec + FTS5 (vault sidecar index, RRF hybrid query)
- Sentence Transformers all-MiniLM-L6-v2 (384-dim embeddings)
- SQLite (event store Layer 1)
- Pydantic v2 (LLMRequest/Response + config)

### Frontend (separate git repository nested at `./mist-frontend/`)
- Tauri 2.x (cross-platform desktop shell)
- React 19 + TypeScript strict
- three.js + @react-three/fiber + drei (3D composition)
- Vite (build), native WebSocket via Tauri shell

---

## Development Workflow

### Starting the Stack
```bash
docker compose up -d               # Full stack
docker compose logs -f mist-backend
```

### Running Tests (inside container; native venv is corrupted)
```bash
docker compose exec -T mist-backend python -m pytest tests/unit/                       # Full unit suite
docker compose exec -T mist-backend python -m pytest tests/integration/                # Integration (Neo4j + llama-server must be up)
docker compose exec -T mist-backend python -m pytest tests/unit/chat/ -v               # Targeted
```

### Admin CLI (`mist_admin`)
```bash
# Inside container:
docker compose exec -T mist-backend python -m scripts.mist_admin stack-status
docker compose exec -T mist-backend python -m scripts.mist_admin graph-stats
docker compose exec -T mist-backend python -m scripts.mist_admin graph-reset --include-derived --confirm
docker compose exec -T mist-backend python -m scripts.mist_admin seed
docker compose exec -T mist-backend python -m scripts.mist_admin chat "utterance" --session-id sid
docker compose exec -T mist-backend python -m scripts.mist_admin replay /app/data/ingest/v6-inputs.jsonl --session-id sid --output /app/data/ingest/report.jsonl
```

### Rebuilding
```bash
docker compose build mist-backend          # After Dockerfile/requirements changes
docker compose build --no-cache mist-backend
```

### Code Quality
```bash
python scripts/check_ai_slop.py --critical-only
pre-commit run --all-files
black backend/
ruff check backend/ --fix
# Frontend formatting / lint lives in mist-frontend repo (npm scripts).
```

---

## Testing

### Backend Tests
- **Count:** 1488 unit tests + 1 platform-skipped + 3 xfailed (at post-MVP 2026-04-22)
- **Runner:** pytest inside Docker container
- **Command:** `docker compose exec -T mist-backend python -m pytest tests/unit/`
- **Note:** Tests must run inside container

### Integration Reproducers
Landed per-cluster for regression protection:
- `tests/integration/test_cluster_3_reproducers.py` (7 tests) — persona injection, post-filter regen, identity-intent routing, temperature split
- `tests/integration/test_cluster_5_reproducers.py` (6 tests) — all three observability phases emitting end-to-end
- `tests/integration/test_cluster_6_reproducers.py` (4 tests) — budget-driven history pruning, max_tokens config wiring

### Standing Drift Guards
- `tests/unit/knowledge/extraction/test_validator.py::TestValidatorOntologyConsistency` — every extractable edge in the ontology has a validator constraint; constraint source/target sets mirror `EdgeTypeDefinition` exactly.
- `tests/unit/test_eval_harness_scorers.py::TestScorerOntologyParity` — `scripts/eval_harness/scorers.py` frozensets match `backend/knowledge/ontologies/v1_0_0.py` `EXTRACTABLE_*` lists bidirectionally. Prevents silent mis-scoring of new extractable types.

---

## Evaluation

### V6 Gauntlet (ontology extraction, 30-turn cohesive conversation)
- **Latest result (2026-04-22 post-ontology-expansion):** 30/30 OK, 0 empty, 0 emoji, 0 LLMRequest errors; Document/Date/Metric new types produced; no regression on hard gates. Report: `data/ingest/post-ontology-expansion-gauntlet-report-2026-04-22.md`.
- **Canonical run protocol:** `graph-reset --include-derived --confirm` -> `seed` -> `replay data/ingest/v6-inputs.jsonl ...` -> `graph-stats` -> write report.

### V7 Tool-Heavy Probe Set (tool-call decision accuracy, 25 single-turn probes)
- **Purpose:** Unblocks `mist-ai-tool-calling-production-rigor` workstream with 20 positive probes (tool expected) + 5 negative controls (tool use = false positive).
- **Input:** `data/ingest/v7-tool-heavy-inputs.jsonl` (force-added over gitignore as engineered research data).
- **Design doc:** `scripts/eval_harness/v7_probe_set_design.md`.
- **Acceptance criteria:** tool-selection precision >= 0.90, recall >= 0.90, 0/5 false positives on negatives.
- **Run:** `docker compose exec -T mist-backend python -m scripts.mist_admin replay /app/data/ingest/v7-tool-heavy-inputs.jsonl --session-id v7-probe --output /app/data/ingest/v7-report.jsonl`. Dedicated scorer against debug-JSONL tool-call stream is a morning followup.

### Eval-Harness Module
- `scripts/eval_harness/` — Phase 3 orchestrator + scorers for 6 test categories (schema_conformance, tool_selection, personality, rag_integration, coherence, speed).
- `scorers.py` frozensets are a mirror of the ontology (intentional, to let the harness run without a backend import at module-load time); parity is now guarded by `tests/unit/test_eval_harness_scorers.py`.

---

## Docker

### Image
- Base: `nvidia/cuda:12.4.0-devel-ubuntu22.04` + Python 3.11 venv at `/opt/venv`.
- Non-root execution under `appuser` UID 1000 (Phase 2 P0 follow-up): avoids root-owned bind-mount artifacts on `./data`, `./logs`, `./mist-memory`. `/home/appuser/.cache/{huggingface,torch}` pre-created and chowned so named volumes inherit correct permissions.
- Flash-attn: `flash-attn==2.8.3` now compiles (post-2026-04-22 fix, commit `e45d8b5`). Build deps `ninja / packaging / wheel / psutil` installed before the flash-attn pip line. **Stack restart required to activate** — current running container still uses PyTorch SDPA. `docker compose down && docker compose up -d` (user-gated) then `docker compose exec -T mist-backend python -c "import flash_attn; print(flash_attn.__version__)"` to verify.
- Dep-resolver pre-existing conflict (non-blocking warning): `chatterbox-tts 0.1.7` pins `numpy<2.0.0 / transformers==5.2.0` but image has `numpy 2.4.3 / transformers 4.57.6`. Unchanged by recent work; file a followup if TTS breaks.

### Cache volumes
- Named volumes: `mist-hf-cache` -> `/home/appuser/.cache/huggingface`, `mist-torch-cache` -> `/home/appuser/.cache/torch`.
- Older `/root/.cache` named volumes exist from the pre-non-root era (orphaned after Phase 2 P0 mount-path migration). Safe to `docker volume prune` once confirmed no container uses them.

### Healthchecks
- `mist-backend`: `/health` endpoint, 30s interval.
- `mist-neo4j`: cypher-shell probe.
- `mist-llm` (llama-server): `/health` probe.

### Docker data root
- `D:\Users\rajga\DockerData` (not default `C:\`). Windows Docker Desktop config.

---

## Gauntlet Workflow (Cluster Validation)

Each cluster validates acceptance via re-running the V4 (5-utterance smoke), V5 (11-utterance breadth), and V6 (30-turn cohesive session) gauntlets against the merged code.

**Canonical protocol:**
1. `mist_admin graph-reset --include-derived --confirm` — wipe graph (snapshots saved to `data/graph_snapshots/`).
2. `mist_admin seed` — restore 32-entity baseline.
3. `mist_admin replay /app/data/ingest/v6-inputs.jsonl --session-id <name> --output /app/data/ingest/<report>.jsonl` with observability env vars set per Debug section above.
4. Analyze the JSONL diagnostic file + per-turn replay output; write a report under `mist.ai/data/ingest/` (gitignored per policy).
5. Compare against the baseline from the prior cluster's gauntlet report.

**Gauntlet input files:** `data/ingest/v4-inputs.jsonl`, `v5-inputs.jsonl`, `v6-inputs.jsonl` (committed).

**Gauntlet reports:** `data/ingest/post-cluster-<N>-gauntlet-report-YYYY-MM-DD.md` (gitignored by `data/ingest/` convention).

---

## Next Steps

### Sub-project A -- C3 (extraction accuracy) COMPLETE (2026-06-12)
- **assertion_kind signal landed end-to-end.** Shared `derive_assertion_kind` + explicit-kind gate in the engine, bucket scoring in the F2 scorer, and the extraction prompt emits it. Same-turn cease/assert arbitration fixed with live-Cypher proof: cease bucket 7/7, retract 5/5, assert perfect. This gives the planner an intra-turn arbitration signal that the prior same-turn collision classes lacked.
- **Ontology v1.3.0.** Added `RECOMMENDS` + `HAS_HABIT`; retired the universal `started` / `ended` / `duration` relationship props (the bitemporal interval is the canonical fact-time channel that superseded them).
- **Prompt precision rules (r2 -> r3 -> r4).** assertion_kind emission, date-entity discrimination, RECOMMENDS/HAS_HABIT coverage, HAS_HABIT cadence requirement, and no structural-edge over-extraction from prepositional scope.
- **F2 relationship precision 0.672 baseline -> ~0.83** (r3 0.831 / r4 0.812 -- a flash-attn near-tie band). The 0.90 headline gate is a DOCUMENTED NEAR-MISS, not an extraction-quality failure: the residual is ~70% entity-id/type canonicalization (the model extracts the correct relationship but labels an endpoint differently than the gold's arbitrary canonical choice) plus flash-attn near-tie drift. Prompt iteration cannot close it -- prompt-byte changes reshuffle the near-tie probes by ~+/-0.02 (see KNOWN_ISSUES production-reproducibility). Supporting numbers: typing ~0.89, RELATED_TO 0, entity precision ~0.83-0.86, negative-controls 0. The named architectural lever to reach 0.90 is the entity-canonicalization sub-project (follow-up B).
- **V7 tool-decision routing PASS.** recall 0.950 (up from the 0.650 baseline), precision 1.000, FP 0/5, deterministic (25/25 tool decisions reproducible). Driven by the `acceptable_tools` dual-acceptance mechanism + typed-fact routing guidance.
- **Determinism foundation.** F2 is now measured via a byte-reproducible EXTRACTION-ONLY harness (`mist_admin.py replay --extraction-only`). Full-chat non-reproducibility root-caused to chat-path stochasticity + flash-attention (NOT MIST code); shipped an injectable-clock fix (production behavior unchanged by default).

### Immediate (current queue as of 2026-08-03, in this order)

**The three sections below this one are HISTORICAL, frozen at the 2026-06-12 C3 close. This is
the live queue.**

1. **Reachability mechanisms** (task 13) -- because ~20 findings are one defect class and three of
   them were produced while actively hunting it. **B: module-identity test** (asserts no two
   `sys.modules` entries resolve to one file; catches the namespace-package double-import that
   would have made the session-id fix a silent no-op). **A: composition-root completeness check**
   over `server.py`'s lifespan -- every `build_*()` param with a falsy default must be supplied or
   explicitly exempted with a justification; this targets `server.py:443` directly, which is where
   the defects actually occurred (asserting on the factories would miss it -- the factories are
   correct and the CALL SITE is wrong). **C: AST reachability scan** with a justified exemption
   file. **D: runtime zero-result alarm** (depends on the D3 scheduler-discard fix).
2. **`server.py:443`** -- the one line behind three dead features. Needs an assertion shape
   covering all three: no caller / null optional dependency / wrong shared-state instance.
3. **R1.4.6 hydration** -- SEQUENCED BEFORE R1.5. A seeded graph is structurally a DIFFERENT
   OBJECT from a used one: all 30 live seed edges carry exactly two properties (`seed_version`,
   `updated_at`) because `seed/applier.py` bypasses reconciliation, while a usage edge carries
   twelve. The seed is not a testing ground because it never went through the code path being
   tested. Hydrate via the **LIVE path** (`handle_message` -> extraction -> curation ->
   reconciliation), NOT via `log_regenerator` -- hydrating through the rebuild path would make
   R1.6's `live == rebuilt` gate compare a replay to a replay. Self-model stays seeded (authored
   identity; R1.0 partitioned it precisely so it is preserved, not re-derived).
   Spec: `docs/superpowers/specs/2026-08-03-r1.4.6-hydration-design.md`.
4. **R1.5** -- LEARNING staleness. `last_asserted_at` first, then the rebuild pass. Option A
   (write-time close at rebuild) was chosen after reversing twice; the decisive reframe was that
   R1.5 changes no live behaviour under either option, so the question is which substrate R1.6
   inherits.
5. **R1.6** -- `live == rebuilt` closure + the new ADR formalizing graph-wins-for-facts
   (superseding ADR-010 Inv-5/Inv-6). Inherits the `ONTOLOGY_VERSION` consistency question, Gate
   3's five content-drift mismatches, and the legacy bookkeeping properties that do not survive a
   wipe.

### Historical -- Sub-project A sequencing as written at the C3 close (2026-06-12)
1. **Follow-up B -- entity-canonicalization sub-project.** **[COMPLETE 2026-06-14 as MIS-124.** Typing cleared 0.875 -> 0.909; rel precision 0.812 -> 0.833, a documented near-miss whose residual was re-diagnosed as model extraction quality rather than canonicalization. The lever is spent; do not chase it with prompt or taxonomy edits. Follow-up is MIS-125.**]**
2. **R1 -- regenerator rebuild.** Utterance->graph regenerator; vault->graph fact path retired (R1.3). **`backend/knowledge/regeneration/` is NOT a tombstone** -- correcting this file's prior claim. It holds three modules, two of them live R1.2 machinery: `log_regenerator.py` (cache-driven log->graph rebuild into isolated staging, `ColdCacheError` on miss, no in-loop LLM) and `rebuild_gate.py` (rebuild-twice byte-identical determinism gate + divergence report). Only `graph_regenerator.py` is the quarantined legacy class, deliberately retained with guard raises its own docstring instructs maintainers not to remove. Graph is truth for FACTS, vault is truth for PROSE (Phase-2 truth model).
3. **Deferred observability (from C3).** Temporal emission / date-fill / resolver split; confidence-threshold analysis. Document-only at C3 close; sequence after B/R1.

### Architectural findings parked as R1 inputs (from 2026-06-12 deep review; document-only, no code yet)
1. **LEARNING progression dead-end** -- nothing supersedes LEARNING itself, and it has no temporal decay, so abandoned-learning facts stay current forever. **CORRECTED 2026-08-01:** the original deep-review wording said LEARNING "is progression-superseded by USES/SKILLED_IN". BOTH halves are false and were false when written -- there is no `SKILLED_IN` predicate in the ontology in any v1.x version (`git log -S "SKILLED_IN" --all` returns zero commits touching code), and `USES.progression_supersedes` was `("STRUGGLES_WITH",)`, never `("LEARNING",)`, before being stripped entirely in v1.2.1 (`v1_0_0.py:1239-1241`, commit `4b5048f`, landed in that same review). LEARNING is progression-superseded by `EXPERT_IN` alone (`:1303`). The conclusion stands; the premise does not. Left uncorrected, this line re-infected every `/mist-status` that read it. **Status: the spoken half shipped in C3 Task 8 (`assertion_kind=cease`); the silent half is R1.5, accepted 2026-08-01 (`docs/superpowers/specs/2026-08-01-r1.5-staleness-design.md`), gated behind R1.4.5 (golden log).**
2. **WORKS_AT SINGLE same-turn dual employers** -- two WORKS_AT in ONE turn: planner closes neither (same valid_from); both stay current. C3's assertion_kind arbitration covers cease/assert collisions; the SINGLE-cardinality dual-assert case still needs intra-turn cardinality arbitration.
3. **Interrupt wire/memory divergence** -- on barge-in the WS wire shows the truncated response but conversation memory keeps the full pre-interrupt text (febe-observability-5).
4. **Inv-A9 cross-process write hole** -- curation writes are asyncio.Lock-serialized within one process only; `mist_admin` CLI (separate process) can interleave (concurrency-async-5).

### Standing follow-ups
1. **Push gate:** local `main` is **19 commits ahead of origin** and has never been pushed; push when the make-mist-usable workstream completes. (The "120+" this line used to claim was never re-read after the R1.4 merge history was rewritten.)
2. **Stack restart pending:** docker-compose.yml changed (env_file block + hardcoded in-network NEO4J_URI + LLM_TEMPERATURE passthrough) -- takes effect on next `docker compose up -d`.
3. `mist-ai-context-compression-multi-session`, frontend integration waves, personality growth: parked pending Sub-project A completion.

### Long-term
1. Command Center architecture (orchestrating agentic teams)
2. Vision integration (Gemma 4 vision)
3. GTX 1070 dual-GPU addition (parked post-voice-integration)
4. Mobile app (TBD; Tauri Mobile or separate native shell — out of scope for current roadmap)

---

## Known live bugs (from the 2026-08-02 reachability review; NOT yet fixed unless marked)

These are wrong in production right now. Each was verified against code or live data by the
review coordinator, not accepted on a hunter's report. Register IDs are given so the full
evidence is findable.

| ID | Bug | Why it matters |
|---|---|---|
| **R4** | **`Neo4jConnection.health_check` is never called.** `server.py`'s health loop derives the frontend's `agent` boolean from `voice_processor.models.knowledge.enabled` -- a construction-time flag, not a probe. | **Neo4j can be down while the UI reports healthy.** The probe that would catch it exists and is dead. Nine methods share the name `health_check`; all four production call sites resolve to LLM providers or the sidecar. |
| **R5** | **`LanceDBVectorStore.delete_by_source` has no caller.** `ingestion/pipeline.py:8` and `:110` both instruct "caller should first call `delete_by_source(source_id)`". None does. | **Re-ingesting a changed document appends a second full set of chunks instead of replacing them** -- silent duplicate accumulation, and duplicate hits back through vector search. |
| **L1** | **Self-model health score permanently 0.** `health.py:76-81` matches `:__Entity__ WHERE knowledge_domain='internal'`; live count is 0 because the 21 self-model nodes are `:__SelfModel__`, disjoint since the R1.0 partition migration. | `overall` is capped at 85 regardless of graph state. A regression the migration never updated. `test_health.py`'s fake dispatches on the RETURN-clause alias, which is how it survived. |
| **L2 / P1** | **The rebuild never scopes turns by epoch.** `get_all_turns_for_reextraction` implements an `ontology_version` filter and the column is written on every turn, but `log_regenerator.py:218` passes only `after_event_id`. R1.4's `origin` discriminator is likewise never filtered on. | `rebuild()` selects EVERY turn ever logged, then demands all of them be cache-present under the CURRENT epoch triple -- so any turn predating a stamp bump is a guaranteed `ColdCacheError` that aborts the rebuild before writing a node. Passes today only because the live log is empty. **The epoch's scoping role in `graph = f(seed, log, epoch)` is not exercised at all.** Fix is one keyword argument. A rebuild also replays `origin='test'` probe traffic into the canonical graph. |
| **D1 / D2 / D3** | Three curation features disabled by **one line**, `server.py:443`. `SelfReflectionJob` gets `event_store=None` and returns zeros on line 1; `SkillDerivationJob` reads a different `ToolUsageTracker` instance than the one receiving `.record()`; `CurationScheduler._loop:135` discards every `JobResult`. | No internal knowledge is derived from conversation history; no Skill/KNOWS/MistCapability is ever derived; `GraphHealthScorer`'s seven sub-scores reach `logger.info` and nothing else. D2 needs a **durable channel**, not just parameter passing -- the records are in-memory and die on restart. |
| **P6** | **`session_id` was pinned to the literal `"default"` for process life** -- `set_session_id` had zero callers. **FIXED 2026-08-03** (`8d6c161`, `eb7a8b4`). | Kept here because its consequences are still in the data: existing session notes carry `session_id: default`, and `writer.py:69-71` documents patching around it. |
| **L6** | `llm_parameters` is hardcoded to `{"temperature": 0.7}` rather than read from the provider -- and it is read back. | A column claiming to record what the LLM was given, recording something else. |
| **L5** | Centrality/community GDS projections name `RELATES_TO`, which does not exist in the ontology (`RELATED_TO` is correct). | Inert today (both jobs disabled), but if enabled the failure is caught by a handler logging "GDS plugin may not be installed", misattributing a typo to a missing dependency. GDS is genuinely absent too, so both modes coincide. |

**Dead but not a live bug** (documented capability absent in production): `search_documents` has
zero call sites, so document RAG does not exist at conversation time -- Raj ruled 2026-08-03 that
this is **to be BUILT, not retired**, since MIST pulling and saving a document is a real intended
capability distinct from the Obsidian vault. Also dead: `KnowledgeRouter` (entire class, 31 tests,
zero callers -- its DISCARD tier would drop filler like "ok"/"thanks" before the LLM extractor,
which sits upstream of the extraction-noise work), `ToolOutputClassifier`, `SourceMetadata` (8
signatures, always None, so external-source provenance is never written), and
`ExtractionConfig.additional_instructions` (~38 lines of extraction prompt encoding a RETIRED
ontology, including the "User is the SUBJECT performing actions" line that was the documented root
cause of Bug J -- and it is the first thing a maintainer opens to tune extraction).

## Known Issues
- GPU contention between llama-server and Chatterbox adds ~1.1x TTS overhead on single GPU
- Binary audio transport implemented but not E2E validated yet (pending manual test)
- 45 open P3 items in KNOWN_ISSUES.md from 2026-03-22 audit (3 closed by the 2026-06-12 deep review; opportunistic resolution, tracked in `mist-ai-technical-debt-p3` parked workstream)
- Git Bash on Windows path-mangles unix-absolute paths in env vars passed to `docker compose exec`; prefix with `MSYS_NO_PATHCONV=1`

---

## Important Files

### Documentation
- **CLAUDE.md** — AI integration guidelines (never push to remote)
- **README.md** — Project overview and setup
- **REPOSITORY_STRUCTURE.md** — File organization
- **CONTRIBUTING.md** — Code quality standards
- **KNOWN_ISSUES.md** — P3 backlog from backend audit (45 open)
- **TESTING.md** — Test conventions
- **tests/CLAUDE.md** — Backend test AI guidance
- Frontend test guidance lives in the mist-frontend repo

### Configuration
- **.env** — Environment variables (never commit)
- **.env.example** — All config with defaults
- **.gitattributes** — Line ending normalization (WSL2/Windows)
- **pyproject.toml** — Python tool configuration
- **.pre-commit-config.yaml** — Pre-commit hooks

### Plan Artifacts
- `~/.claude/plans/cluster-execution-roadmap.md` (canonical) + mirror at `mist.ai/.local/plans/cluster-execution-roadmap.md`
- `~/.claude/plans/2026-04-21-cluster-3-identity-layer.md` (completed)

### Vault Artifacts (persistent memory)
- `knowledge-vault/Projects/mist-ai/workstreams/mist-ai-knowledge-integration-mvp-validation.md` — authoritative workstream state (closed 2026-04-22)
- `knowledge-vault/Projects/mist-ai/sessions/` — session notes; most recent covers Cluster 8 closure + post-MVP overnight
- `knowledge-vault/Decisions/ADR-008-revised-model-backend-selection.md`, `ADR-009-graph-provenance-separation.md`, `ADR-010-memory-storage-architecture.md` (all accepted post-Cluster-8)

---

## Quick Reference

| Area | Status | Notes |
|---|---|---|
| Backend | CONTAINERIZED, non-root | Docker + CUDA 12.4, Gemma 4 E4B via llama-server |
| Knowledge integration | MVP COMPLETE + hardened | make-mist-usable Phase 1 landed 2026-06-09; deep review 2026-06-12 |
| Bitemporal engine (C1/C2) | LANDED + hardened | version_key append-only edges, schema-driven ReconciliationEngine, 3-arm currency filters |
| Ontology **v1.4.0** | 51 edge types (39 extractable), 30 node types (21 extractable) | cardinality/temporal_class/contradicts/progression semantics per edge type; v1.4.0 (MIS-124) added the `Abstraction` supertype + `parent_type` overlap mechanism and retired Topic/Milestone |
| Unit tests | **2679 + 7 skipped + 3 xfailed** | Authoritative count is the Tests bullet under Current Status -- do not maintain a second one here; this row mirrors it |
| Eval isolation (F1) | FAIL-CLOSED | (host,port) allowlist; disposable tmpfs eval Neo4j via `--profile eval` |
| Extraction harness (F2/F3) | LANDED | extraction-only byte-reproducible harness (`replay --extraction-only`); full-chat non-determinism is flash-attn, not MIST code |
| Extraction accuracy (C3 + MIS-124) | COMPLETE | `extraction_version=2026-06-14-r5` (r4 was C3; MIS-124 bumped it); typing 0.909 PASS, F2 rel precision 0.833 = documented near-miss vs 0.90 -- residual is model extraction QUALITY, not canonicalization; that lever is spent -> MIS-125 (constrained decoding or a larger model). V7 PASS R=0.950 P=1.000 FP 0/5 |
| Reachability | **UNSOLVED, in progress** | ~20 features built/tested/wired to nothing, 6 shapes. `TestProductionCallerExists` covers shape 1 only. Mechanisms A-E designed; see "Known live bugs" and the findings register |
| TTS | Chatterbox Turbo | 0.74x RTF, 3.9GB VRAM |
| Frontend | Wave 1 integrated | Nested `./mist-frontend/` repo; ADR-017 v1.1.1 |
| Code Quality | FULL SUITE | black, ruff, bandit, codespell, AI-slop, pre-commit |
| Docker | COMPLETE | 3-service stack + eval profile (mist-backend, mist-neo4j, mist-llm, mist-neo4j-eval) |
| CI/CD | CONFIGURED | GitHub Actions |
