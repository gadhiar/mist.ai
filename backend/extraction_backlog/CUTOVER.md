# Epoch cutover runbook

Changing the extraction model is a NEW epoch. The whole event log is re-extracted
through the backlog under the new model, a staging graph is built from it and
checked, and only then does the new epoch become active. The old graph stays
live, frozen, until the swap.

Code: `backend/extraction_backlog/cutover.py` (lifecycle, check, seed-only
probe), `ExtractionDispatcher._fill_step` (filling),
`EventStore.promote_epoch_cutover` and `EventStore.promote_epoch_cutover_seed_only`
(the two promotion transactions), `backend/extraction_backlog/admin.py` (CLI).

When the conversation log has never held a turn that was extracted into the live
graph (for example, the log is empty), skip sections 4 and 5: see section 6A and
its precondition.

Every command below is marked:

- [VERIFIED-IN-REPO] the command exists with these arguments; the reference names
  the file it is defined in. "Verified in repo" is NOT "run against the live
  stack": nothing in this runbook has been run against the live stack.
- [UNIT-TESTED] also exercised by the unit tier (`tests/unit/extraction_backlog/`,
  `tests/unit/event_store/test_epoch_cutover.py`) against fakes.
- [UNVERIFIED] not established from the repository. Read it as a proposal.

---

## 0. States

    begin    -> filling   candidate recorded in `epoch_cutover`; NOT in epoch_ledger
    (fill)   -> ready     every logged turn has a candidate cache row
    rebuild  -> checked   staging graph built twice under the candidate passed its gates
    promote --graph-swapped
             checked -> promoted
                          candidate appended to epoch_ledger + first-hand activation
                          + applied markers through the rebuilt turn, one SQLite
                          transaction (section 6)
    promote --seed-only-graph
             ready | checked -> promoted
                          read-only probe finds no stamped and no unseeded element,
                          no apply marker exists; candidate appended to epoch_ledger
                          + first-hand activation marking nothing, one SQLite
                          transaction (section 6A; operator precondition there)
    abandon  -> abandoned deletes nothing

While a cutover is open (filling, ready or checked):

- The dispatcher applies NOTHING to the live graph. It only infers and caches under
  the candidate's stamps. New turns are still logged, and are inferred under the
  candidate as they arrive.
- The active epoch's pending turns stay pending. The extraction service now serves
  the candidate model, so the active epoch cannot progress. This is expected.
- `extraction_status` (WebSocket), `GET /extraction/status` and `cutover status`
  show progress. The WebSocket `cutover` block reports the exact state, including
  `checked` -- `backend/extraction_contract/models.py`'s `CutoverStatus.state` now
  carries a `checked` value (v2 (2), MIS-171), so a checked cutover no longer needs
  to be reported as `ready`; the CLI shows the same state.

Every transition logs one INFO line: `epoch_cutover transition cutover_id=... from=... to=...`.

---

## 1. Preconditions

1. The candidate model is serving on the extraction host (the lead confirms
   gpt-oss-20b on the GTX 1070 host first).
2. You know the BARE model hash the service will report in `/v1/info`. It is the
   service's `EXTRACTION_MODEL_HASH` environment value
   (`backend/extraction_service/settings.py`, `EXTRACTION_MODEL_HASH`;
   `docker-compose.extraction.yml` requires it). [VERIFIED-IN-REPO]
3. The service's `extraction_version` is the code's `EXTRACTION_VERSION`
   (`backend/extraction_service/app.py` imports it from
   `backend/knowledge/version_stamps.py`). `cutover begin` defaults to the same
   value. [VERIFIED-IN-REPO]
4. `MIST_EXTRACTION_INFERENCE=service` on the backend (the dispatcher does not fill
   in `off` mode). [VERIFIED-IN-REPO: `backend/extraction_backlog/settings.py`]

The admin CLI reads the event store and cache named by the backend's environment,
so run it inside the backend container. [UNVERIFIED: `docker compose exec` form
not run from this branch]

    docker compose exec mist-backend python -m backend.extraction_backlog.admin status

---

## 2. Begin

    docker compose exec mist-backend python -m backend.extraction_backlog.admin \
      cutover begin --model-hash <BARE_MODEL_HASH>

[VERIFIED-IN-REPO: `admin.py`, `_add_cutover_parser`] [UNIT-TESTED]

Optional: `--extraction-version V` and `--ontology-version O` (default: the code's
`EXTRACTION_VERSION` / `ONTOLOGY_VERSION`). The candidate's stored `model_hash` is
composed from the bare hash and the backend's embedding model through
`compose_model_hash`.

Refused (exit 2, nothing written) when a cutover is already open, when the ledger is
empty, or when the candidate stamps equal the active epoch's.

---

## 3. Point the service at the candidate, then wait for `ready`

Switch the extraction service to the candidate model (set its
`EXTRACTION_MODEL_HASH` and model file, restart it). See
`docker/extraction/README.md` for the service's variables. [UNVERIFIED: the exact
restart procedure on the GTX 1070 host is the lead's]

Until `/v1/info` matches the candidate, the dispatcher reports `epoch_mismatch` and
sends no job. [UNIT-TESTED]

Watch progress:

    docker compose exec mist-backend python -m backend.extraction_backlog.admin cutover status

It prints `covered=N total=M`. The state becomes `ready` when every logged turn has
a candidate cache row. Filling continues after that for turns logged later.
[UNIT-TESTED]

Dead-lettered turns under the candidate count as covered (they are cached as
`extraction_failed` skips) and replay as no-ops.

---

## 4. Check: build staging twice and gate it

Start the disposable staging Neo4j. Its data is on tmpfs, so it is wiped when the
container is removed. [VERIFIED-IN-REPO: header of `docker-compose.staging-neo4j.yml`]

    docker compose -f docker-compose.yml -f docker-compose.staging-neo4j.yml \
      --profile staging up -d mist-neo4j-staging

Run the check:

    docker compose exec mist-backend python -m backend.extraction_backlog.admin \
      cutover rebuild --staging-uri bolt://mist-neo4j-staging:7687 \
      --min-seed-nodes <N> --expect-turns <N> --min-replay-edges <N>

[VERIFIED-IN-REPO: `admin.py`; the real wiring is `build_rebuild_deps_from_env` in
`cutover.py`] [UNIT-TESTED with a fake LogRegenerator and fake gates only: a real run
needs Neo4j and has not been done]

`bolt://mist-neo4j-staging:7687` is in the default rebuild allowlist
(`DEFAULT_REBUILD_NEO4J_ENDPOINTS` in `backend/knowledge/eval_isolation.py`).
`assert_rebuild_target_not_live` runs before any staging write and again inside
every `LogRegenerator.rebuild`; a live target is refused.

Sizing the floors. `--expect-turns` is the number of turns the rebuild selects:
turns whose session origin is `real` and whose `ontology_version` equals the
candidate's (`LogRegenerator.rebuild` calls `get_all_turns_for_reextraction` with
`CANONICAL_ORIGINS`). `--min-seed-nodes` and `--min-replay-edges` are the same
floors `python scripts/mist_admin.py graph-rebuild-from-log` takes; size them from
the corpus.

Gates (all from `backend/knowledge/regeneration/rebuild_gate.py`): rebuild-twice
identical, turns processed (both builds), canonical form non-vacuous, replay edges
non-vacuous, self-model applied. `live == rebuilt` is NOT a gate (a new model is
expected to differ); its summary line is stored in the report as
`live_vs_rebuilt`. The check also refuses if the rebuild's turn selection changed
while it ran.

Exit codes: 0 passed -> `checked`; 1 rebuild-twice disagreed; 2 refused; 4 a
non-vacuity or self-model gate failed. On anything but 0 the cutover is `ready`
with the report recorded; fix and re-run. A re-run of a `checked` cutover demotes
it to `ready` first.

`cutover status` shows `rebuilt_through_event_id`: the last turn the staging graph
contains. Promotion marks exactly the turns up to and including it as applied.

---

## 5. Swap the live graph

Stop the backend first, so nothing writes the live graph during the swap and no
turn is applied under the old epoch after it: [UNVERIFIED: not run from this branch]

    docker compose stop mist-backend

Turns logged after `rebuilt_through_event_id` are safe: they stay apply-pending and
the dispatcher applies them to the new graph after promotion.

### 5.1 Back up the live stores and graph

    export MIST_BACKUP_ROOT=/mnt/backup/mist
    python -m scripts.backup.dump --label pre-cutover-<CUTOVER_ID>

[VERIFIED-IN-REPO: `scripts/backup/README.md` section 1]

### 5.2 Capture the staging graph

`graph-backup` reads the graph named by `NEO4J_URI`
(`scripts/mist_admin.py`, `cmd_graph_backup` -> `_connect` -> `config.neo4j`), so
point it at staging (host port 7689):

    NEO4J_URI=bolt://localhost:7689 python scripts/mist_admin.py graph-backup \
      --output /mnt/backup/mist/cutover-<CUTOVER_ID>-staging.json

[VERIFIED-IN-REPO: `graph-backup --output` in `scripts/mist_admin.py`]
[UNVERIFIED: the `NEO4J_URI` override has not been run against staging]

Do this before removing the staging container: its data is on tmpfs.

### 5.3 Replace the live graph with the staging graph -- UNRESOLVED

The existing restore tools REFUSE the live graph by design, and this runbook does
not loosen or bypass either guard:

- `python scripts/mist_admin.py graph-restore` calls `assert_neo4j_dev_isolated`
  on its target before reading the artifact (`cmd_graph_restore`); its docstring:
  "Restoring INTO live is an operator decision that belongs with the operator, not
  a flag on this command."
- `python -m scripts.backup.restore` requires a `MIST_RESTORE_TARGET` marker that
  "the live state root will never carry" (`scripts/backup/README.md` section 4.2)
  and also runs `assert_neo4j_dev_isolated` on the graph URI.

Recommended method (lead, 2026-09-26): the volume swap below. Restore into a fresh
allowlisted instance, repoint `mist-neo4j` at that instance's volume, and keep the
old live volume untouched as the rollback. It is a host operation that Raj confirms
when a cutover actually runs; nothing in this repository performs it. Both
candidates are still unverified:

- [UNVERIFIED] Neo4j offline copy: stop `mist-neo4j`, then load a
  `neo4j-admin database dump` of the checked graph into the live data volume with
  `neo4j-admin database load --overwrite-destination=true`. The staging instance
  is on tmpfs, so the dump would have to come from a non-tmpfs instance the
  staging artifact (5.2) was restored into.
- [UNVERIFIED, RECOMMENDED] Volume swap: restore the staging artifact (5.2) into a fresh,
  dev-allowlisted Neo4j instance with the existing `graph-restore`, stop it, and
  re-point the live `mist-neo4j` service at that instance's data volume.

Whatever method is chosen, verify the result before promoting:

    python scripts/mist_admin.py graph-stats

[VERIFIED-IN-REPO: `graph-stats` in `scripts/mist_admin.py`; read-only per
`scripts/backup/README.md` section 4.8]

---

## 6. Promote

Only after the swap. `--graph-swapped` is your statement that the live graph IS
the checked staging graph; the code never writes live Neo4j itself.

With the backend stopped, run the CLI in a one-off container: [UNVERIFIED: the
`docker compose run` form has not been run from this branch]

    docker compose run --rm mist-backend python -m backend.extraction_backlog.admin \
      cutover promote --graph-swapped

[VERIFIED-IN-REPO: `admin.py`] [UNIT-TESTED]

Refused (exit 2, nothing written) unless the cutover is `checked` and the flag is
given, and if the active epoch changed since `begin`. Otherwise ONE SQLite
transaction:

1. appends the candidate to `epoch_ledger` (it becomes the active epoch);
2. writes the new epoch's backlog activation first-hand, so T2a's automatic
   first-activation rule does not run for it;
3. marks applied exactly the turns up to and including `rebuilt_through_event_id`
   in log order;
4. sets the cutover to `promoted`.

A crash anywhere inside leaves none of it written; run it again. [UNIT-TESTED:
`test_a_crash_between_ledger_append_and_activation_writes_neither`]

The command prints the `MIST_MODEL_HASH` value to use. Set it in the backend's
environment BEFORE starting it: live graph writes are stamped from
`KnowledgeConfig` (`MIST_MODEL_HASH`, and the code's `EXTRACTION_VERSION`), not
from the epoch ledger (`build_curation_pipeline` in `backend/factories.py`).
[VERIFIED-IN-REPO]

    docker compose up -d mist-backend

The dispatcher then applies the turns after `rebuilt_through_event_id` to the new
live graph, in order, from the candidate cache rows (no new inference for turns
already filled). [UNIT-TESTED]

Keep the extraction service on the candidate model: it is now the active epoch's.

---

## 6A. Promote over a seed-only live graph (no rebuild, no swap)

**Operator precondition.** Use this path ONLY when the conversation log has never
held a turn that was extracted into the live graph: in practice, when
`conversation_turn_events` is empty, or when you have independent evidence that no
extraction ever ran against this graph. Then the live graph holds only what the
seed applier wrote, and the candidate epoch can start from it as it is: sections
4 (rebuild) and 5 (swap) are skipped. An empty log is also where `cutover rebuild`
cannot run (its floors must be >= 1). The code does not enforce this
precondition; the checks below cannot see every extraction write.

Check the log is empty (read-only):

    docker compose exec mist-backend python -m backend.extraction_backlog.admin cutover status

It must print `covered=0 total=0`. `total` counts every logged turn, unfiltered
(`_print_cutover` in `admin.py` -> `BacklogStore.fill_scan` ->
`EventStore.list_turn_keys_in_replay_order`). [VERIFIED-IN-REPO] [UNIT-TESTED:
the `covered=... total=...` line] [UNVERIFIED: not run against the live stack].
The top-level `status` command does not print the logged-turn count.

What the probe proves: at least one node, no node or relationship WITHOUT
`seed_version`, and no node or relationship WITH `ontology_version`,
`extraction_version` or `model_hash`. The seed applier sets `seed_version` on the
nodes (`backend/knowledge/seed/applier.py:210`) and edges (`:78`) it writes.
[VERIFIED-IN-REPO] Any element an extraction writer CREATES lacks `seed_version`,
so the probe sees it.

What the probe and the apply-marker check CANNOT see [VERIFIED-IN-REPO]:

- an extraction write that mutates an existing seeded element without stamping
  it: `backend/knowledge/extraction/internal_derivation.py:407-410` (Stage 9
  UPDATE) and `:418-421` (DEPRECATE) on a `:__SelfModel__` node;
  `backend/knowledge/curation/reconciliation.py:754-758` (the ON MATCH append on
  an existing edge); `backend/knowledge/curation/skill_derivation.py:174` and
  `:238` (edges MERGEd with no stamp, invisible when they match an existing
  seeded edge) and `:187-190`, `:215-218` (SET on an existing node, no stamp);
- a turn applied without a marker: `backend/chat/conversation_handler.py:2035-2049`
  runs extraction in process whenever no dispatcher is attached and writes no
  `extraction_applied` row; a turn recorded as legacy at the backlog's first
  activation gets no marker whether or not an earlier path applied it
  (`BacklogStore.ensure_activation`).

Hence the precondition above: these checks back it up, they do not replace it.

Stop the backend first, as for section 6, so no live-path writer can touch the
graph between the probe and the promotion, and no turn is logged during it:
[UNVERIFIED: not run from this branch]

    docker compose stop mist-backend
    docker compose run --rm mist-backend python -m backend.extraction_backlog.admin \
      cutover promote --seed-only-graph

[VERIFIED-IN-REPO: `admin.py`, `_add_cutover_parser`] [UNIT-TESTED:
`tests/unit/extraction_backlog/test_cutover_seed_only.py`, with a fake probe]
[UNVERIFIED: the real probe (`probe_live_graph_from_env`) has not been run against
Neo4j]

`--seed-only-graph` and `--graph-swapped` are mutually exclusive; giving both is
refused by argparse (exit 2) before anything is opened. [UNIT-TESTED]

Refused (exit 2, `[cutover] REFUSED: <reason>`, nothing written) when: [UNIT-TESTED]

- no cutover is open, or it is not `ready` or `checked`;
- the fill is incomplete (a logged turn has no candidate cache row);
- `extraction_applied` holds ANY row, under ANY epoch, `applied` or `curated`
  (a recorded apply; an empty table does not prove no turn was applied, see
  above);
- the active epoch is no longer the cutover's source epoch;
- the live-graph probe fails: it must find at least one node, every node and
  relationship carrying `seed_version`, and none carrying `ontology_version`,
  `extraction_version` or `model_hash`. A probe that cannot connect or errors is a
  refusal too.

The probe is ONE read-only Cypher statement (`SEED_ONLY_PROBE_CYPHER` in
`cutover.py`) run on the LIVE graph named by the backend's environment
(`config.neo4j`) through `Neo4jConnection.execute_query`. It runs only after the
state, fill, marker and epoch checks pass, and before the transaction. A unit test
refuses any write clause in it. [UNIT-TESTED: textually, and over a fake
connection]

The fill is checked before the transaction, not inside it (the candidate cache is
a separate database). A turn logged after that check is not in the candidate
cache, so after promotion the dispatcher infers it fresh under the new epoch, in
its place in the log. With the backend stopped, no turn should be logged then.

Otherwise ONE SQLite transaction (`EventStore.promote_epoch_cutover_seed_only`)
re-checks the state, the source epoch and the markers, then:

1. appends the candidate to `epoch_ledger` (prev = the source epoch);
2. writes the new epoch's backlog activation first-hand with
   `turns_at_activation` = the logged turns and `marked_applied` =
   `legacy_unextracted` = 0, so T2a's automatic first-activation rule does not
   run for it;
3. writes NO applied marker;
4. sets the cutover to `promoted`, its `check_report` recording
   `promotion_mode: seed_only_graph`, the probe counts, and any earlier check
   report under `prior_check_report`.

A crash anywhere inside leaves none of it written; run it again. [UNIT-TESTED:
`TestPromoteSeedOnly::test_a_crash_between_ledger_append_and_activation_writes_nothing`]

Next, as in section 6: set the printed `MIST_MODEL_HASH` in the backend's
environment, then recreate the backend so its writers are stamped with the new
epoch: [UNVERIFIED: not run from this branch]

    docker compose up -d --force-recreate mist-backend

The dispatcher then applies EVERY logged turn to the live graph, in log order,
from the candidate cache rows (no new inference for turns already filled).
[UNIT-TESTED: `test_every_turn_is_applied_in_log_order_from_the_candidate_cache`]

---

## 7. Abandon

    docker compose exec mist-backend python -m backend.extraction_backlog.admin cutover abandon

[VERIFIED-IN-REPO] [UNIT-TESTED]

Deletes nothing. The candidate's cache rows are keyed by its stamps, so no other
epoch reads them; a later cutover to the same stamps reuses them. Point the
extraction service back at the active epoch's model afterwards, or the active
epoch stays in `epoch_mismatch`.

---

## 8. Known limits

- Gate 3 (duplicate) uses an in-memory cache with a wall-clock TTL
  (`ExtractionPipeline._check_dedup`, `dedup_cache_ttl_seconds`). Filling re-runs
  the whole log in quick succession, so its duplicate decisions can differ from
  the ones the live path made.
- The rebuild replays only `origin='real'` turns logged under the candidate's
  `ontology_version`; the backlog fills every logged turn. Promotion marks every
  logged turn up to `rebuilt_through_event_id` applied, including out-of-scope
  turns before it, which the swapped graph does not contain.
- The self-model gate compares the live and rebuilt `:__SelfModel__` node counts.
  Stage 9 runs on the live path and never on rebuild, so live self-model nodes
  written by Stage 9 make that gate fail.
- The seed-only probe and the promotion transaction are not atomic with each
  other: Neo4j and SQLite share no transaction. A live-graph write between the
  probe and the COMMIT is not seen. Stopping the backend first (section 6A)
  closes that window for the backend's own writers; the transaction re-checks
  the SQLite side (state, source epoch, apply markers).
- The probe requires EVERY node to carry `seed_version`, so a node any other
  writer created (including Stage 9 self-model nodes) refuses the seed-only path
  even if it carries no extraction stamp.
- The probe cannot see unstamped mutations of seeded elements, and the marker
  check cannot see in-process or legacy applies (section 6A lists the file:line
  sites). The seed-only path rests on the operator precondition in 6A.
