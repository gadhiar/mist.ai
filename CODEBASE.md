# MIST.AI Codebase Context

**Last Updated:** 2026-08-02 (**Epoch wiring COMPLETE + ff-merged to local `main` @ `f4aeadd`; version-stamp authority collapse ACCEPTED and folded into R1.4.5.** `main` now 7 ahead of origin, NOT pushed.

**`ensure_initial_epoch` is now called from production.** `ConversationHandler.__init__`, immediately after `initialize()` -- the one path production always opens the event store on (`mist_admin.py` also constructs one, but only when a subcommand runs). `now_iso` from the injected clock. Placed inside the existing `try` because every realistic failure mode already fails `initialize()` one line up, so a separate handler would guard nothing while adding a bare `except Exception` the convention forbids.

**The tests are the point, not the one-liner.** `tests/unit/chat/test_conversation_handler_epoch.py` (6 tests). `TestProductionCallerExists` asserts that constructing a handler THE WAY PRODUCTION DOES leaves an epoch in the ledger -- the assertion every one of this branch's dead-wired features lacked. **Mutation-proved BOTH directions: commenting out the call fails 5 of the 6 new tests, and all 7 PRE-EXISTING epoch tests in `test_store.py` still pass without it.** That second half is the defect class demonstrated concretely -- those tests were correct and thorough about the method's behaviour and completely blind to whether anything invoked it. Also covers clock discipline (`activated_at` equals an injected fixed instant), idempotency across two handlers on one file db, and a disabled-store side-effect boundary. Unit **2588 passed / 6 skipped / 3 xfailed**.

**LIVE VERIFIED by restarting `mist-backend`:** log reads `Event store: wrote provisional initial epoch 1 (ontology=1.4.0, extraction=2026-06-12-r1)`; `epoch_ledger` went 0 -> 1 row, `provisional=1`. Verifying THIS fix by unit tests alone would have been self-defeating.

**The restart surfaced a second drift, and challenging it produced a better answer than the one first proposed.** `.env:48` pins `EXTRACTION_VERSION=2026-06-12-r1`, overriding the code's own default of `2026-06-14-r5` (`config.py:606` and `:662`), while `vault/writer.py:60` hardcodes r5 and `graph_writer.py:259` falls back to `"1.2.1"` under a comment reading "no hardcoded version literal" -- **four authorities.** The first instinct was "decide which value is right." Raj challenged the premise and was correct: **version stamps here are PURELY DESCRIPTIVE and nothing branches on them** -- the active ontology is chosen by Python import, not by the env var; `extraction/prompts.py` has zero references to `extraction_version`; every consumer only writes the value; and `canonical_serialize.py:39` deliberately EXCLUDES the stamp triple so the determinism proof reads "same log + same epoch => same facts". While data is seeded and regenerable, a bump is a scripted regeneration, not a migration. **What survives is that the VALUE does not matter but CONSISTENCY does:** `cache_key = sha256(event_id|ontology_version|extraction_version|model_hash)` is the sole mechanism where a label becomes a behaviour, and disagreement there is a hard `ColdCacheError` -- in the exact mechanism R1.4.5 depends on. Accepted fix, folded into R1.4.5: collapse to one authority, have the golden log's cache generator derive its stamps from the same place the rebuild reads them so they cannot drift by construction, drop the `.env` pins so env cannot silently override a code default with a STALER value, and remove the `graph_writer` fallback. Timing is fortunate -- **no extraction-cache database exists yet**, so there are zero mislabelled entries and the fix currently costs nothing; after R1.4.5 authors its cache it would invalidate every entry. Note `epoch_id 1` captured the drifted value and `ensure_initial_epoch` is idempotent, so collapsing the authorities must also supersede that row (it is `provisional=1`, explicitly R1.6's to redefine). **This downgrades the `ONTOLOGY_VERSION` drift R1.6 inherited from a correctness problem to a consistency-of-authority problem.**

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

**Not merged.** Coordinator owns the whole-branch review and the fast-forward. PRIOR ENTRY -- 2026-07-31 (R1.4, T10: **NOT SHIPPABLE AS DESIGNED, BLOCKED** -- the live-data-loss incident that led to the work above. Branch `feat/r1.4-seed-source-and-data-gate`, 9 tasks landed (T1-T9), T10 (retirements + live verification) found a live-data-loss defect during its own verification and stopped per design rather than merging over it. T1-T9 built a versioned seed source (`mist-memory/seed/{mist,user}.md`, `SeedFact`/`SeedDocument` models, `apply_seed_documents`/`reseed`, three `seed-verify` gates) to replace `scripts/seed_data.yaml`. **T10's live run of `reseed()`'s wipe-then-recreate cycle proved the new seed source cannot express NODE DEFINITIONS -- only FACTS (edges).** `SeedFact` carries subject/predicate/object/valid_from/valid_to and nothing else; the applier's `_MERGE_NODE` sets only `id`/`created_at`/`updated_at`/`seed_version`. The old `apply_seed` carried `entity_type`/`display_name`/description/pronouns/self_concept via structured per-entity YAML dicts and applied ontology-specific labels (`SET n:{label}`); the new one has no equivalent. This was invisible through T1-T9 because MERGE preserves untouched properties on a match, and every seed run before T10 matched richly-labelled nodes left over from the old `apply_seed` path. Only a genuine delete-and-recreate exposes it -- which is exactly what `reseed()`'s wipe does once a node carries the `seed_version` stamp from a prior run, and T10's live verification (run seed, run it again for idempotency) is what triggered that for the first time. **Live consequence, confirmed and then reverted:** all 32 nodes stripped to a bare partition label (`__Entity__`/`__SelfModel__`) plus four properties, zero ontology labels (`MistIdentity`/`MistTrait`/`MistCapability`/`MistPreference`/`User`/`Organization`/`Technology`/etc. all gone), zero `display_name`/`entity_type`/`pronouns`/`self_concept`. `GraphStore.ensure_mist_identity()` (called at backend startup) started raising `ConstraintValidationFailed` instead of no-op MERGE-ing, meaning the next backend restart would have hard-failed. Persona injection (`get_mist_identity_context()`) was degraded live. **Restored** from a pre-task full-graph JSON backup (`data/graph_snapshots/pre-r1.4-task10-full-backup-2026-07-31.json`, all labels + properties, taken before T10's first live write) -- verified node-for-node, `ensure_mist_identity()`'s exact production query re-tested clean post-restore. Graph is back to its R1.3.1 state; **do not re-run `mist_admin.py seed` or `tests/integration/knowledge/test_seed_label_split.py` until the model gap is fixed -- both wipe-and-recreate through the same defective applier and will immediately re-strip the graph.** Fix requires a T1/T4-level model change (a node-definition concept on `SeedFact`/`SeedDocument` carrying type + descriptive properties) -- explicitly out of scope for a same-task patch; not attempted. `scripts/seed_data.yaml` (`git rm`'d, UNCOMMITTED) is now understood to be the only remaining full node-definition source outside git history and the JSON backup -- do not let its staged deletion land without first extracting what the model-change work needs, or resolve it via `git restore --staged --worktree scripts/seed_data.yaml`. T10 additionally landed two independently-shippable, non-applier-touching fixes not yet committed: (1) `VaultWriter.upsert_identity_body` + `bootstrap_vault_from_seed` repointed onto `documents: list[SeedDocument]`, verbatim-body rendering for both `identity/mist.md` and `users/<id>.md` (old `upsert_identity`/structured-dict path kept, unused, undeleted -- ~10 pre-existing tests depend on it); (2) the T3 `origin` session-provenance discriminator's write-AND-read wiring closed end to end (`MIST_SESSION_ORIGIN` env -> `EventStoreConfig.session_origin` -> `ConversationHandler` -> `EventStore.start_session`; `ConversationSession.origin` field added -- the column existed since T3 but had no read path at all). A live-only Gate 3 (`seed-verify`'s containment check) defect was also found and ruled fixable independently (kebab-id vs display-name substring mismatch, 29/30 facts false-fail) but the fix was not implemented -- work stopped when the node-definition defect was found first. Full detail, all live numbers, and the complete file list: `.superpowers/sdd/2026-07-31-r1.4-seed-source-and-data-gate/task-10-report.md` (gitignored, local only). PRIOR ENTRY -- 2026-07-31 (R1.3.1 -- vault write-policy correction -- COMPLETE, ff-merged to local `main` @ `9e48a92` from `feat/r1.3.1-vault-write-policy` (17 commits, 8 tasks). Removed per-turn vault appending entirely: R1.3 deleted the `DERIVED_FROM->VaultNote` contract that was the ONLY justification for it -- the appended note was the anchor that edge pointed at -- so Bucket 2 is now synthesis-only. A session note is written ONCE at session end by the new `SessionSynthesizer` (`backend/chat/session_synthesizer.py`) from the session's turns READ FROM THE EVENT-STORE LOG, not live memory; the same code path therefore also serves `SessionNoteCatchup` (`backend/vault/session_catchup.py`, `run_forever()` wired into `server.py` lifespan) for sessions whose process died before session end. Four cost gates: skip sessions with no graph state (one Cypher), defer to live traffic, bounded per-pass retry persisted as `status: skipped` stubs, production event store only. Deleted `append_turn_to_session` / `update_entities_extracted` / `append_session_synthesis` / `mark_session_completed` / `_append_turn_sync` / `peek_turn_count` and the `SENTINEL` / `VaultConfig.append_sentinel` / `MIST_VAULT_APPEND_SENTINEL` leftovers; `MistSessionFrontmatter` dropped `turn_count` / `participants` / `append_sentinel_offset`, gained `title` + the `skipped` status. `write_session_note` derives `date:` from the path stem (`YYYY-MM-DD-<slug>`) and RAISES on a non-canonical path -- the old `datetime.now()` fallback drifted across UTC midnight, and the single session note MIST had ever written was itself a specimen of that bug (path allocated 06-09 at 23:59, note stamped 06-10 after midnight; corrected). **Also collapsed the two session-id namespaces:** `EventStore.start_session` now TAKES the chat-layer session id instead of minting its own `uuid4`, and `ConversationHandler._es_session_ids` is deleted outright -- which fixed a PRE-EXISTING bug where the live extraction path wrote the external id and the replay path wrote the internal one into `ConversationContext.conversation_id`, independently re-traced and confirmed. `MIST.md` (injected into MIST's context every turn) corrected and ADR-011 amended (amendment, not rewrite; `knowledge-vault` @ `3a94278`, committed local, NOT pushed). Process note worth keeping: TWO features on this branch passed their own tests while doing NOTHING -- catch-up synthesized zero notes because three id spaces were conflated, and `run_forever` ran at most once per boot because `is_conversation_active` was bound to `handler.sessions`, a map that only ever GROWS (its sole remover `clear_session` has zero production callers). Both were plan defects, not implementation defects, and both were found by adversarial live tracing, not by tests. Six separate guards were caught green with live regressions wired in. Final verification: unit **2404 collected / 2396 passed / 5 skipped / 3 xfailed**; integration 58 collected / 33 passed / 2 failed (BOTH still the same pre-existing baselined pair from R1.3, exact file:line match) / 23 skipped. **SMOKE-TESTED LIVE 2026-07-31** (first time this feature has ever executed -- the container had been up 26h, booted before the wiring existed): backend restarted clean, catch-up scheduled, ONE pass ran, 11 candidates found, LLM probed once, graph queried, all 11 skipped, ZERO synthesis calls, vault byte-identical, no exceptions, no spin. FOLLOW-ON 2026-07-31 (`40c8c35`, on `main`): the `test_filewatcher.py` debounce flake is FIXED -- all 13 fixed `asyncio.sleep(debounce_ms/1000 + margin)` waits replaced with a bounded `_wait_for(predicate, settle=, timeout=)` poll. The margin was never the issue (measured fire latency 100.7-103.0ms against a 250ms budget, ~147ms headroom), so failures required a >147ms event-loop stall that no fixed sleep can absorb. `_wait_for` returns rather than asserting, so callers' existing assertions remain the sole failure diagnostic; `settle` waits past the debounce window after the predicate first holds, used wherever the assertion is exactly-N or negative (polling for `>= 1` would otherwise return before a late duplicate arrived and pass vacuously -- which is exactly what the collapse tests exist to catch). Four pure-negative tests deliberately KEEP their fixed sleeps: with no positive condition to poll, the sleep only gives the watcher a chance to misbehave, and a stall makes a spurious call less likely, not more. Suite now 5/5 green across both fixed and random ordering (was 5 failures in 7 runs); `test_filewatcher.py` runtime 5.56s -> 4.31s. KNOWN OPEN: (1) **R1.4 GAINED A PRECONDITION:** the 11 legacy event-store sessions carry PRE-COLLAPSE uuids, so once R1.4 re-seeds `ConversationContext` on event-store ids they all become matching candidates, find no note at their derived path, and write a SECOND note for a session that already has one -- including the real `-37a8` note, which is `authored_by: user-edit`. Drain or age-bound those rows BEFORE re-seeding. (3) Minor, follow-up: a REFUSED session-note write is reported as success (`_handle_write_session_note` returns `str(path)` unconditionally), so the vault-op debug stream records `ok=True` for a write that did not happen. PRIOR ENTRY -- 2026-07-30 (R1.3 -- vault->graph fact retirement -- COMPLETE, ff-merged to `main` @ `cdcae55` from `feat/r1.3-vault-graph-fact-retirement` (24 commits, 10 tasks). Retired every path by which editing a vault markdown file wrote a FACT into Neo4j: deleted `GraphStoreProtocol.upsert_user` / `GraphStore.upsert_user`, `mark_orphaned_by_provenance_path` / `get_orphaned_provenance_paths`, `ExtractionPipeline.extract_from_file`, `GraphRegeneratorConfig`, and the whole `GraphRegenerator` class (`backend/knowledge/curation/graph_regenerator.py`) -- its bus-event responsibility moved onto `VaultFilewatcher._do_reindex`. Dropped `vault_note_path` threading from the curation path; retired the `--scope` / `--retry-orphaned` vault-rebuild CLI modes in `mist_admin` (both were already call-time dead by the time T8 removed them). Entity provenance re-anchored from `DERIVED_FROM->VaultNote` onto `EXTRACTED_FROM->ConversationContext`; `RebuildStamps` now ride that edge family across every extraction but remain WRITE-ONLY -- no consumer reads them, a real decision deferred to R1.4/R1.6. `apply_seed`'s `DERIVED_FROM->VaultNote` seeding path is DELIBERATELY RETAINED (R1.4's scope, not this branch's). Two rewritten integration tests (`test_phase3_production_wiring_smoke.py`, `test_vault_edit_read_path.py`, renamed from `test_adr010_invariant5.py`) assert the retirement by graph node/edge COUNT DELTA, not by shape -- nine mutations proved shape-based guards permeable before the delta form landed; the unit-level twin (`tests/unit/vault/test_filewatcher_graph_noop.py`) went through 4 mutation-hardening rounds to close 10/10 escape routes, including patching `GraphStore.__init__` on the class object (not a module binding) after two rounds of import-laundering escapes via `backend.factories` re-exports. Final verification (T10): unit 2358 passed / 5 skipped / 3 xfailed, 2366 collected, 0 errors (was 2426 at branch start; net -68 is the sum of nine independently-reconciled per-task deltas). Integration 58 collected / 0 errors / 33 passed / 23 skipped / 2 failed -- BOTH BASELINED, pre-existing, not this branch's: `test_cluster_5_reproducers.py::test_retrieval_candidates_record_carries_session_id` (`session_id` not reaching retrieval-candidate records) and `knowledge/test_seed_label_split.py::test_seed_yields_only_entity_nodes` (expects 31 seeded `:__Entity__` nodes, live graph has 11 -- state drift, plausibly the 2026-06-29 self-model dedup 41->21; asserts against the live DB, so the real fix is likely moving it onto the F1 isolated eval Neo4j). Two grep retirement proofs both clean against their expected-survivor lists (writer.py's graph->vault `upsert_user`/`upsert_user_snapshot`, the quarantined `backend/knowledge/regeneration/` package, `apply_seed`, the ontology's `VaultNote` node type, and the sidecar's `subject="VaultNote"` prompt-assembly pseudo-facts). One orphaned observation: `backend/knowledge/curation/bucket1_reader.py` (file->graph-edge parser built for the now-deleted `GraphRegenerator`) has no production caller left, only its own unit test; it is inert (returns a dataclass, performs no graph write) and not a retirement failure, but worth pruning or repurposing when R1.4 lands. NEXT: **R1.4 (seed-utterance migration) is LOAD-BEARING** -- the Phase-1 curated-profile facts written by the now-deleted `upsert_user` will not survive a graph rebuild until migrated to seed-utterances -- then R1.5 (LEARNING staleness) -> R1.6 (`live == rebuilt` GREEN closure + the new ADR formalizing graph-wins-for-facts, superseding ADR-010 Inv-5/Inv-6; R1.3 deliberately leaves ADR-010 formally unamended). PRIOR ENTRY -- 2026-07-29 (doc refresh after a ~1-month gap; content reflects HEAD @ `6a05cbd` on `main`, tree clean, 61 ahead of origin; last code work 2026-06-29. SINCE MIS-124, Sub-project A advanced into R1 -- the utterance->graph regenerator: R1.0 (`:__SelfModel__` partition) + R1.1 (deterministic identity resolver + canonical content-equality) landed 2026-06-15; the self-model partition migration was applied to live mist-neo4j and the self-model dual-seed dedup designed 2026-06-23; the dedup SHIPPED + APPLIED TO LIVE (self-model 41 -> 21 nodes, persona de-doubled, `identity/mist.md` now a graph no-op) and R1.2 (proof-first `log_regenerator`: cache-driven log->graph rebuild into isolated staging + rebuild-twice byte-identical determinism gate GREEN + `graph-rebuild-from-log --dry-run` CLI) landed 2026-06-29. extraction_version + ontology UNCHANGED since MIS-124 (`2026-06-14-r5` / v1.4.0 -- R1 is regeneration + determinism, not extraction). Tests ~2399 -> 2426 unit green. NEXT: R1.3 (vault->graph fact retirement) -> R1.4 (seed-utterance migration + Phase-1 data gate, LOAD-BEARING) -> R1.5 (LEARNING staleness) -> R1.6 (`live == rebuilt` GREEN closure + new ADR). PRIOR ENTRY -- 2026-06-14 (MIS-124 -- ontology v1.4.0 MECE taxonomy + entity canonicalization -- COMPLETE on `feat/mis-124-ontology-v1.4.0-mece-taxonomy`. `extraction_version = "2026-06-14-r5"`, ontology v1.4.0. Retired Topic (->Concept) + Milestone (->Event, event_type=milestone); added the `Abstraction` supertype with a generic `parent_type` mechanism + "emit parent when no child clearly fits" fallback over the 7 abstract leaves. Overlap-handling architecture: a canonical (id,type) resolver generalizing the normalizer `RESERVED_NAMES` (retired-type coercion, bounded curated registry, Metric value/unit + string numeric-first id canonicalization, parent fallback) + hierarchy-aware validator/scorer/dedup (accepts-parent-iff-accepts-Concept, `types_match`, cluster-widened Tier-3 dedup) + a gated specificity floor against fallback gaming. Third-party facts now tracked (Person/Org-sourced) with `CONFIDENCE_EXTERNAL.third_party_penalty` applied. F2 re-baseline (extraction-only, re-adjudicated gold 2026-06-14): **TYPING ACCURACY CLEARED 0.875 -> 0.909 (PASS)**; rel precision 0.812 -> 0.833 (DOCUMENTED NEAR-MISS of the 0.90 gate); rel recall 0.846, entity precision 0.854, RELATED_TO 0.000, neg-controls 0, specificity 1.000. The remaining rel-precision residual is model entity-extraction QUALITY (event-naming consistency, metric structured-field emission, occasional predicate choice) + small-model prompt sensitivity / flash-attn near-tie drift -- NOT a canonicalization gap; the canonicalization lever is spent (it closed typing and reduced live-graph fragmentation). Follow-up: constrained/grammar-guided decoding or a larger model (extraction-quality frontier, separate decision). Tests ~2309 -> 2399. PRIOR ENTRY -- 2026-06-13 (C3 -- Sub-project A extraction accuracy -- COMPLETE; `extraction_version = "2026-06-12-r4"`, ontology v1.3.0. assertion_kind signal landed end-to-end (shared `derive_assertion_kind` + explicit-kind gate in the engine, bucket scoring in the F2 scorer, prompt emits it); same-turn cease/assert arbitration fixed with live-Cypher proof (cease 7/7, retract 5/5, assert perfect). Ontology v1.3.0 adds RECOMMENDS + HAS_HABIT and retires the universal `started`/`ended`/`duration` relationship props (superseded by the bitemporal interval). F2 relationship precision 0.672 -> ~0.83 (r3 0.831 / r4 0.812, flash-attn near-tie band) -- a DOCUMENTED NEAR-MISS of the 0.90 gate, residual is ~70% entity-id/type canonicalization + flash-attn drift, not extraction quality (closed by follow-up B). V7 tool-decision PASS: recall 0.950 (from 0.650), precision 1.000, FP 0/5, deterministic 25/25. F2 now measured via a byte-reproducible extraction-only harness; full-chat non-reproducibility root-caused to chat-path stochasticity + flash-attention (not MIST code). Tests 2252 -> ~2309. Prior entry (2026-06-12): deep review of the 109-commit unpushed span -> 71 confirmed findings fixed across 10 batches (ontology v1.2.1 semantics, currency-filter gaps, production filewatcher writer wiring, temporal_status->CEASE, confidence forwarding, event-loop offloading, F2 gold-corpus user-anchor correction); tests 2158 -> 2252.)))
**Branch:** `feat/r1.4-seed-source-and-data-gate` (branched from `main` @ `40c8c35`), 17 commits ahead (T1-T14 plus three whole-branch-review fixes: ...`b966938` T9, `84a5bd9` T10, `849ac3c` T11, `4fca9d1` T12, `e89073a` T13's test change, `4bc0b10` T14 (`check_node_definitions` + Gate 3 fix + `test_seed_label_split.py` fail-closed guard), `608c1dc` C1 (`check_negation_proximity` fix), `a5a7499` I4/I1/I5 (un-wipeable-litter defense, stale-comment correction, orphan-node rejection)). **MERGED 2026-08-01:** ff-merged to local `main` (`40c8c35..a5a7499`), which is now **125 commits ahead of origin**, still push-gated, never pushed. Merge gate verified by the coordinator, not taken from reports: unit 2552 passed / 6 skipped / 3 xfailed run twice (the known ~1-in-6 intermittent did not appear); integration 33 passed / 24 skipped / 1 failed, the pre-baselined `test_cluster_5_reproducers.py::test_retrieval_candidates_record_carries_session_id` by exact name; live graph 32 nodes / 30 rels with `display_name`, `entity_type`, `seed_version` and `embedding` all 32/32, and `MistIdentity` 1 / `MistTrait` 9 / `MistCapability` 5 / `MistPreference` 6 / `User` 1. I4's two layers were each proven independently -- the `SeedNode` validator rejects `seed_version`/`entity_type`/`updated_at`, and a poisoned node built via `model_construct` and fact-referenced so it reaches the write path still lands the applier's own stamps with `created_at` absent from the property map. `mist-memory/` (own local repo, no remote): `02a6bdc` (T13, node definitions in `seed/{mist,user}.md` + the T10 live-write capture) on top of `e7e4a99`/`34d4514`. Do NOT push. Four review findings are deferred with rulings, none blockers: **I2** a `seed_version` bump strands the previous version's content (the wipe scopes on the new version and never matches the old) -- R1.6 or whoever first bumps owns it; **I3** `reseed()` is non-atomic and `cmd_seed` takes no pre-wipe snapshot -- every live run in R1.4 took a manual backup first, make that a documented requirement; **I6** a third seed document breaks `bootstrap_vault_from_seed`, failing soft so the vault half silently stops updating; **I7** no gate covers embeddings, the one failure that recurred twice on live data -- genuinely wanted, deferred only because a fifth gate landing without the adversarial pass the other four received is precisely how C1 happened.
**Status:** make-mist-usable Phase 2 / Sub-project A (MIS-120). Chain landed: F1/F2/F3 -> C1/C2 (bitemporal engine) -> deep review (71 findings fixed) -> C3 (extraction accuracy) -> MIS-124 (ontology v1.4.0 MECE) -> R1.0/R1.1 (determinism substrate) -> self-model dedup (applied live) -> R1.2 (proof-first log_regenerator) -> R1.3 (vault->graph fact retirement) -> R1.3.1 (vault write-policy correction) -> **R1.4 (seed-utterance migration) code-complete, live-verified, and whole-branch-reviewed (2 rounds, C1 + I4/I1/I5 fixed) as of 2026-07-31, awaiting the coordinator's final verification and merge -- see header.** Memory state: curated-profile recall works and reasons (Phase 1, validated 2026-06-09); conversational fact capture accurate but model-bounded (rel precision 0.833 documented near-miss -> MIS-125); durability now genuinely closed at the live level for the seeded profile -- the node-definition gap that would have broken it (T10's finding) is fixed and proven against an actual wipe-and-recreate cycle, not just unit tests. NEXT (after merge): R1.5 (LEARNING staleness) -> R1.6 (`live == rebuilt` closure + new ADR, which also inherits the two known-open items in the header: Gate 3's 5 content mismatches, and legacy bookkeeping properties not surviving a wipe). The rel-precision residual remains deferred to an extraction-quality decision (constrained/grammar-guided decoding or a larger model, MIS-125), unrelated to R1.4. ADR-011 codifies the three-bucket vault write pattern; ADR-017 at v1.1.1.

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
- **Debug JSONL Observability:** `DebugJSONLLogger` with 5 record phases (`turn`, `extraction`, `llm_call`, `retrieval_candidates`, `llm_request_raw`). Each gated by its own env var. See Cluster 5 artifacts below.
- **Knowledge Graph:** Extraction + curation pipeline + hybrid retrieval (graph + vector + vault sidecar, RRF merge). ADR-009 provenance separation structurally enforced (Cluster 2). MIST identity retrieval injects persona (Cluster 3). Ontology **v1.3.0** carries **39 extractable relationship types**, each declaring reconciliation semantics (cardinality / temporal_class / contradicts / progression_supersedes) consumed by the schema-driven `ReconciliationEngine` (C1/C2); v1.3.0 (2026-06-12, C3) added `RECOMMENDS` + `HAS_HABIT` and retired the universal `started` / `ended` / `duration` relationship props (superseded by the bitemporal interval, which is the canonical fact-time channel); v1.2.1 (2026-06-12) removed the USES<->DISLIKES contradicts pair and USES progression (behavior is orthogonal to sentiment/competence -- the pairs erased co-true beliefs). Fact edges are append-only bitemporal versions keyed on `version_key` with 3-arm currency filters on every user-facing read. Validator constraints AND extractor `ALLOWED_*` sets derive from the ontology (Inv-A6; undirected predicates validate as unordered pairs). Drift guards standing: scorer frozensets, validator constraints, extractor allowlists, and the extraction-prompt sha256 <-> `extraction_version` pin.
- **Knowledge Seed:** 32-node baseline (1 MistIdentity + 9 MistTraits + 5 MistCapabilities + 6 MistPreferences + 1 User + 10 anchor entities; 20 identity relationships + 10 anchor relationships = 30 facts; all 32 embedded), now seeded via the versioned seed source (`mist-memory/seed/{mist,user}.md`, `mist_admin.py seed` -> `load_seed_documents` + `reseed`). **Safe to re-seed** as of R1.4 T11-T14: `SeedNode` (T11) + the applier writing ontology type labels and descriptive properties (T12) + the real source carrying node definitions (T13) means a wipe-and-recreate cycle now correctly restores every label and property -- proven twice live, including one run that performed a genuine full wipe. `scripts/seed_data.yaml` is deleted (commit `84a5bd9` on this branch, not on `main`; T13 restored it only transiently/untracked to script the node extraction, then re-deleted it -- the extraction itself is preserved in `mist-memory` commit `02a6bdc` and this branch's git history). Two known-open, non-blocking items: Gate 3 fails on 5/30 real facts (genuine content drift, not a gate bug) and a full wipe drops legacy `provenance`/`confidence`/etc. bookkeeping properties that predate the `SeedFact`/`SeedNode` model -- see header.
- **Vault Layer (Cluster 8, in progress):** NOTE -- the Phase 5/6/8 text in this bullet is HISTORICAL as of R1.3 + R1.3.1. Per-turn session-note appending and the `DERIVED_FROM`->`VaultNote` provenance edge described below are BOTH DELETED. Current behavior: one MIST-authored session note per session, written once at session end (or by startup catch-up) as an LLM synthesis sourced from the event-store log, via the single `VaultWriter.write_session_note` full-render path; entity provenance is `EXTRACTED_FROM`->`ConversationContext` carrying `source_utterance_id`. `RebuildStamps` still ride that edge family but remain WRITE-ONLY (no consumer reads them; deferred to R1.4/R1.6). The Phase 9 slug derivation and Phase 10 seed bootstrap paragraphs below remain accurate. Retained verbatim for archaeology: `backend/vault/` package with `VaultWriter` (serialized `asyncio.Queue` consumer for session-note appends, identity/user upserts), `VaultSidecarIndex` (sqlite-vec `vec0` + FTS5 + RRF hybrid query over two-tier chunks), `VaultFilewatcher` (watchdog daemon thread with 500ms debounce + asyncio bridge + 60s mtime audit job + MIST-write coordination for user-edit detection), Pydantic frontmatter models for the four `mist-*` note types, and `AuthoredBy` 5-state authorship enum. Wired through `VaultConfig` / `SidecarIndexConfig` / `FilewatcherConfig` on `KnowledgeConfig`. **Phase 5 integrated:** single server-owned VaultWriter built and started in `server.py` lifespan, plumbed through `VoiceProcessor -> ModelManager -> KnowledgeIntegration -> ConversationHandler`, with per-turn vault append after event-store write (failure-isolated per ADR-010 Invariant 6). **Phase 6 integrated:** `vault_note_path` is pre-allocated synchronously at `handle_message` Step 0 (via `_get_or_allocate_vault_path`) and threaded through `_extract_knowledge_async` -> `ExtractionPipeline.extract_from_utterance` -> `CurationPipeline.curate_and_store` -> `CurationGraphWriter.write`. Every upserted entity now emits a `DERIVED_FROM` edge to a `:__Provenance__:VaultNote {path}` node (MERGE-idempotent on path). New `VaultNote` ontology node type registered as bridging; `DERIVED_FROM` edge extended to permit `VaultNote` targets and `MistIdentity` sources. The graph is now formally rebuildable from the vault. **Phase 8 integrated:** rebuild-determinism stamps. New `RebuildStamps` frozen dataclass (`ontology_version`, `extraction_version`, `model_hash`) constructed by `build_curation_pipeline` from `KnowledgeConfig` and injected into `CurationGraphWriter`. Every `DERIVED_FROM`->`VaultNote` edge now carries the three stamps + `derived_at` timestamp on both ON CREATE and ON MATCH branches so re-extractions land the current stamps. New config fields `KnowledgeConfig.extraction_version` (default `"2026-04-17-r1"`, env `EXTRACTION_VERSION`) and `KnowledgeConfig.model_hash` (default `"gemma-4-e4b-q5-k-m-carteakey-full-v1"`, env `MIST_MODEL_HASH`). **Phase 9 integrated:** retrieval routing + slug improvement. QueryClassifier extended with a `historical` intent (regex patterns matching "what did we discuss"/"remember when"/"last time"/etc.) routed to the vault sidecar; `hybrid` now produces three-way RRF merges across graph + vector + vault sidecar via `_merge_rrf_three_way`. New `QueryIntentConfig` fields per ADR-010 weight table (`rrf_vault_weight=0.4` hybrid; historical-specific `0.2/0.1/0.7` graph/vector/vault). `KnowledgeRetriever` accepts an optional `vault_sidecar: SidecarIndexProtocol` plumbed top-down through `VoiceProcessor -> ModelManager -> KnowledgeIntegration -> build_conversation_handler -> build_knowledge_retriever`; `_vault_sidecar_retrieve` wraps `query_hybrid` and converts vec0+FTS5 results to `RetrievedFact` rows. Session slug derivation now extracts significant words from the FIRST USER UTTERANCE (stopwords + short tokens filtered, top 5 retained) with a 4-char SHA-256(session_id) suffix for guaranteed per-session uniqueness — produces filenames like `2026-04-22-vault-architecture-mist-a3f1.md` instead of opaque `2026-04-22-<sanitized-session-id>.md`. **Phase 10 integrated:** seed vault bootstrap (absorbs Cluster 7 migration). `mist_admin seed` now extends to `bootstrap_vault_from_seed` (async helper in `backend/knowledge/admin.py`) which calls `VaultWriter.upsert_identity` (rendered from seeded MistTraits/Capabilities/Preferences) and `VaultWriter.upsert_user` (rendered from the seeded user dict via `_build_user_body_markdown`). After the writes, `emit_seed_vault_provenance` MERGE-creates two `:__Provenance__:VaultNote` nodes (one per bootstrap note) and emits per-entity `DERIVED_FROM` edges from each seeded entity (mist-identity + traits/caps/prefs -> identity/mist.md; user + anchor entities -> users/<id>.md). Edges carry `event_id='seed'` literal (no Phase 8 stamps -- seed entities are deterministic via re-run, not extraction-rebuild). New `--no-vault-bootstrap` flag opts out; bootstrap also auto-skips when `config.vault.enabled` is False. Filewatcher + sidecar share the same lifecycle. Phase 11 (CLI subcommands `vault-status` / `vault-reindex` / `vault-rebuild` / `vault-migrate`) is next.
- **Graph Regeneration (Sub-project A / R1):** `backend/knowledge/regeneration/` -- `log_regenerator.py` (cache-driven log->graph rebuild into an isolated staging graph; `ColdCacheError` on any extraction-cache miss, no in-loop LLM; self-model copy-forward + cross-layer edge re-derivation) + `rebuild_gate.py` (rebuild-twice byte-identical determinism gate + divergence report). Driven by `scripts/mist_admin.py graph-rebuild-from-log --dry-run`; fenced off from live by `backend/knowledge/eval_isolation.py:assert_rebuild_target_not_live` + `docker-compose.staging-neo4j.yml`. PROOF-FIRST as of R1.2 (2026-06-29): determinism proven at the unit level; NOT yet run against live data (cold-cache refusal -- warm-up is the documented prerequisite). **R1.3 (vault->graph fact retirement) COMPLETE 2026-07-30:** every vault-file-edit -> Neo4j-fact write path is now deleted (`GraphRegenerator` class, `GraphStore.upsert_user`, orphan-marking, `extract_from_file`, the vault-rebuild `--scope`/`--retry-orphaned` CLI modes); the only surviving `GraphRegenerator` reference is the quarantined `backend/knowledge/regeneration/graph_regenerator.py` (legacy utterance-based, byte-unchanged, not this class). **R1.3.1 (vault write-policy correction) COMPLETE 2026-07-31:** per-turn vault appending removed, session notes are end-of-session synthesis from the event-store log plus a startup catch-up pass, and the two session-id namespaces are collapsed into one (`EventStore.start_session` takes the chat-layer id; `_es_session_ids` deleted). **R1.4 (seed-utterance migration) is NEXT and LOAD-BEARING**, not optional follow-up: the Phase-1 curated-profile facts the now-deleted `upsert_user` used to write will NOT survive a graph rebuild until migrated onto seed-utterances, and that migration must run and validate before any live `live == rebuilt` cutover (R1.6). **R1.4 now carries a hard precondition discovered during R1.3.1's final review:** the 11 legacy event-store sessions predate the namespace collapse and carry old `uuid4` ids, so the moment R1.4 re-seeds `ConversationContext` nodes keyed on event-store ids, all 11 become matching catch-up candidates, find no note at their derived path (the slug hash differs -- `-9199` from the event-store id vs the real note's `-37a8` from the retired external id), and write a SECOND note for a session that already has one. The existing note is `authored_by: user-edit`, so the R1.3.1 user-edit guard will REFUSE to overwrite it -- meaning the duplicate lands alongside rather than clobbering -- but the duplicate is still wrong. Drain or age-bound those 11 rows BEFORE re-seeding. Real cutover + `live == rebuilt` closure = R1.6, which also owns the new ADR formalizing graph-wins-for-facts (ADR-010 Inv-5/Inv-6 stay formally unamended through R1.3 by design).
- **Tests:** **2396 unit tests passing, 5 skipped, 3 xfailed** (2404 collected, 0 errors -- 2026-07-31 R1.3.1 landing; was 2358 at the R1.3 landing, net +38 across R1.3.1's eight tasks). Integration: 58 collected, 0 errors, 33 passed, 23 skipped, 2 failed (both pre-existing and baselined, unchanged since R1.3 -- see header). Run inside container: `docker compose exec mist-backend python -m pytest tests/unit/`. **Suite is fully green as of `40c8c35`** -- the long-standing `test_filewatcher.py` debounce flake was closed by replacing 13 fixed-sleep waits with a bounded `_wait_for` poll helper (see the header entry). Verified 5/5 across both fixed (`-p no:randomly`) and default random ordering. When adding filewatcher tests, use `_wait_for` rather than `asyncio.sleep`, and pass `settle=` whenever the assertion is exactly-N or negative.

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
make-mist-usable Phase 2 / Sub-project A (MIS-120). Latest landed: R1.3 (vault->graph fact retirement) on `main` @ `cdcae55`, 2026-07-30. NEXT: R1.4 (seed-utterance migration + Phase-1 data gate, LOAD-BEARING) -> R1.5 (LEARNING staleness) -> R1.6 (`live == rebuilt` GREEN closure + new ADR); enabler = a cache warm-up path so the rebuild dry-run runs against real data (R1.2 currently refuses on a cold cache). R1 specs/plans are local/gitignored under `docs/superpowers/`. The dated "Recently Completed" entries below are the pre-Phase-2 MVP archive (closed 2026-04-22).

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

### Immediate (Sub-project A sequencing, post-C3)
1. **Follow-up B -- entity-canonicalization sub-project.** The named architectural lever to take F2 relationship precision from ~0.83 to 0.90 (the C3 near-miss residual is ~70% endpoint canonicalization) and to reduce live-graph fragmentation. Tracked in Linear.
2. **R1 -- regenerator rebuild.** Utterance->graph regenerator (vault->graph path retired; tombstone in `backend/knowledge/regeneration/`), seed-utterance migration, new ADR. Graph is truth for FACTS, vault is truth for PROSE (Phase-2 truth model).
3. **Deferred observability (from C3).** Temporal emission / date-fill / resolver split; confidence-threshold analysis. Document-only at C3 close; sequence after B/R1.

### Architectural findings parked as R1 inputs (from 2026-06-12 deep review; document-only, no code yet)
1. **LEARNING progression dead-end** -- nothing supersedes LEARNING itself, and it has no temporal decay, so abandoned-learning facts stay current forever. **CORRECTED 2026-08-01:** the original deep-review wording said LEARNING "is progression-superseded by USES/SKILLED_IN". BOTH halves are false and were false when written -- there is no `SKILLED_IN` predicate in the ontology in any v1.x version (`git log -S "SKILLED_IN" --all` returns zero commits touching code), and `USES.progression_supersedes` was `("STRUGGLES_WITH",)`, never `("LEARNING",)`, before being stripped entirely in v1.2.1 (`v1_0_0.py:1239-1241`, commit `4b5048f`, landed in that same review). LEARNING is progression-superseded by `EXPERT_IN` alone (`:1303`). The conclusion stands; the premise does not. Left uncorrected, this line re-infected every `/mist-status` that read it. **Status: the spoken half shipped in C3 Task 8 (`assertion_kind=cease`); the silent half is R1.5, accepted 2026-08-01 (`docs/superpowers/specs/2026-08-01-r1.5-staleness-design.md`), gated behind R1.4.5 (golden log).**
2. **WORKS_AT SINGLE same-turn dual employers** -- two WORKS_AT in ONE turn: planner closes neither (same valid_from); both stay current. C3's assertion_kind arbitration covers cease/assert collisions; the SINGLE-cardinality dual-assert case still needs intra-turn cardinality arbitration.
3. **Interrupt wire/memory divergence** -- on barge-in the WS wire shows the truncated response but conversation memory keeps the full pre-interrupt text (febe-observability-5).
4. **Inv-A9 cross-process write hole** -- curation writes are asyncio.Lock-serialized within one process only; `mist_admin` CLI (separate process) can interleave (concurrency-async-5).

### Standing follow-ups
1. **Push gate:** branch is 120+ commits ahead of origin; push when the make-mist-usable workstream completes.
2. **Stack restart pending:** docker-compose.yml changed (env_file block + hardcoded in-network NEO4J_URI + LLM_TEMPERATURE passthrough) -- takes effect on next `docker compose up -d`.
3. `mist-ai-context-compression-multi-session`, frontend integration waves, personality growth: parked pending Sub-project A completion.

### Long-term
1. Command Center architecture (orchestrating agentic teams)
2. Vision integration (Gemma 4 vision)
3. GTX 1070 dual-GPU addition (parked post-voice-integration)
4. Mobile app (TBD; Tauri Mobile or separate native shell — out of scope for current roadmap)

---

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
| Ontology v1.3.0 | 39 predicates w/ reconciliation semantics | cardinality/temporal_class/contradicts/progression; v1.3.0 (C3) added RECOMMENDS + HAS_HABIT, retired universal started/ended/duration props |
| Unit tests | **~2309 + 5 skipped + 3 xfailed** | Run inside container; + live-Cypher bitemporal boundary test vs eval Neo4j |
| Eval isolation (F1) | FAIL-CLOSED | (host,port) allowlist; disposable tmpfs eval Neo4j via `--profile eval` |
| Extraction harness (F2/F3) | LANDED | extraction-only byte-reproducible harness (`replay --extraction-only`); full-chat non-determinism is flash-attn, not MIST code |
| Extraction accuracy (C3) | COMPLETE | `extraction_version=2026-06-12-r4`; F2 rel precision 0.672 -> ~0.83 (0.90 gate = documented near-miss, residual is entity canonicalization -> follow-up B); V7 PASS R=0.950 P=1.000 FP 0/5 |
| TTS | Chatterbox Turbo | 0.74x RTF, 3.9GB VRAM |
| Frontend | Wave 1 integrated | Nested `./mist-frontend/` repo; ADR-017 v1.1.1 |
| Code Quality | FULL SUITE | black, ruff, bandit, codespell, AI-slop, pre-commit |
| Docker | COMPLETE | 3-service stack + eval profile (mist-backend, mist-neo4j, mist-llm, mist-neo4j-eval) |
| CI/CD | CONFIGURED | GitHub Actions |
