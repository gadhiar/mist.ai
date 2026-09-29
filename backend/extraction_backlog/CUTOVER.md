# Epoch cutover runbook

Changing the extraction model is a NEW epoch. The whole event log is re-extracted
through the backlog under the new model, a staging graph is built from it and
checked, and only then does the new epoch become active. The old graph stays
live, frozen, until the swap.

Code: `backend/extraction_backlog/cutover.py` (lifecycle, check, seed-only
probe), `ExtractionDispatcher._fill_step` (filling),
`EventStore.promote_epoch_cutover` and `EventStore.promote_epoch_cutover_seed_only`
(the two promotion transactions), `backend/extraction_backlog/admin.py` (CLI).

When no extraction has ever run against the live graph, skip sections 4 and 5:
see section 6A and its precondition. That path needs an EMPTY conversation log,
and because an empty log alone does not establish the precondition when the log
has been reset or replaced, 6A reseeds the live graph with the backend stopped
and probes it before promoting.

Code anchors in this runbook are symbols. Where a line matters, the reference
gives the grep that finds it, run from the repository root; line numbers drift.

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
                          the conversation log is empty, no apply marker exists,
                          read-only probe finds no stamped and no unseeded element;
                          candidate appended to epoch_ledger + first-hand
                          activation marking nothing, one SQLite transaction
                          (section 6A; operator precondition there)
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

`status` prints, besides the backlog counts and any open cutover (whose
`candidate:` line carries its `ontology_version`, `extraction_version` and
`model_hash`): [VERIFIED-IN-REPO: `_status` in `admin.py`] [UNIT-TESTED:
`TestAdminStatusCli` in `tests/unit/extraction_backlog/test_status.py`]

    epoch N: ontology_version=... extraction_version=... model_hash=...
    code: ONTOLOGY_VERSION=... EXTRACTION_VERSION=... configured model_hash=<MIST_MODEL_HASH> (MIST_MODEL_HASH; composed <hash>)
    writer stamps: match

or `writer stamps: MISMATCH (<field>, ...)` followed by the dispatcher's own
refusal text with both stamp triples. The writer stamps are what a backend
started from this environment writes (`writer_stamps_from_config(
KnowledgeConfig.from_env())`); the verdict is the dispatcher's guard
(`ExtractionDispatcher._writer_stamp_mismatch`), compared with the ACTIVE epoch
(not an open cutover's candidate). On a MISMATCH the dispatcher applies and
dispatches nothing. Exit 0, or 1 when the epoch ledger is empty.

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

`status` (section 1) must then print `writer stamps: match` against the new
epoch; on `MISMATCH` the dispatcher stays stalled and applies nothing.
[UNIT-TESTED: the `status` line only]

The dispatcher then applies the turns after `rebuilt_through_event_id` to the new
live graph, in order, from the candidate cache rows (no new inference for turns
already filled). [UNIT-TESTED]

Keep the extraction service on the candidate model: it is now the active epoch's.

---

## 6A. Promote over a seed-only live graph (no rebuild, no swap)

**Operator precondition.** Use this path ONLY when no extraction has ever run
against THIS live graph. Then the live graph holds only what the seed applier
wrote, and the candidate epoch can start from it as it is: sections 4 (rebuild)
and 5 (swap) are skipped. The code cannot prove the precondition. It checks an
empty conversation log, an empty `extraction_applied` and a read-only graph
probe; the procedure in 6A.3 adds the reseed that makes those checks evidence.
An empty log is also where `cutover rebuild` cannot run (its floors must be
>= 1), so this is the only path for it.

### 6A.1 Why a reseed, an empty log and the probe are the evidence

Neither the log nor the probe is evidence alone. The event log
(`event_store.db`) has been reset before, and a reset log is empty whatever ran
against the graph before the reset. The probe cannot see an unstamped SET on a
seeded element, or an unstamped `:__SelfModel__`-only element a reseed adopted
(6A.2). Taken together with a reseed of the live graph, done
AFTER the last log reset and with the backend stopped, they are:

1. **The reseed erases every earlier write to a seed element, or leaves
   something the probe counts, or refuses to run, except on an unstamped
   `:__SelfModel__`-only element it adopts.**
   `python scripts/mist_admin.py seed --no-vault-bootstrap` (`cmd_seed` ->
   `reseed`; 6A.3 step 3 says why the flag) deletes every edge carrying the current `seed_version`, then every node carrying it that
   has no relationship left, then MERGEs both back from the seed source
   (`wipe_seed_version`, `_WIPE_EDGES`, `_WIPE_NODES`, `apply_seed_documents`:
   `grep -nE '^_WIPE_(EDGES|NODES)|^def (reseed|wipe_seed_version|apply_seed_documents)' backend/knowledge/seed/applier.py`).
   [VERIFIED-IN-REPO] So every element that carried the current
   `seed_version` before the reseed is deleted and re-created, except a seed
   node that some relationship without the current `seed_version` still
   touches. [VERIFIED-IN-REPO]

   The re-created seed elements are not all new, though. The MERGEs match on
   the node id and on (subject, type, object), not on `seed_version`, so a
   non-seed element that survived the wipe and matches a seed node or fact
   would be adopted: it would get `seed_version` and keep its other
   properties. Since MIS-177 D1 the reseed refuses, before its wipe, a graph
   holding most such elements (6A.2, "adopted elements"); it can still adopt
   an unstamped `:__SelfModel__`-only one. [VERIFIED-IN-REPO]

   No writer outside the seed applier and the staging seeder sets
   `seed_version` (`grep -rln seed_version backend/knowledge/curation
   backend/knowledge/extraction backend/chat` lists only
   `curation/reconciliation.py`, which reads it and never sets it; the
   reconciliation engine writes a clamped copy of a seed edge with
   `seed_origin_version` plus extraction stamps, not `seed_version`, and a
   reseed refuses a graph holding that copy). [VERIFIED-IN-REPO] So an
   element extraction created survives the wipe without `seed_version`, and
   so does the relationship that kept a seed node alive, and the probe counts
   both unless the re-apply adopted them. The reseed refuses to run over any
   of them that touches an `:__Entity__` node or carries `extraction_version`,
   `model_hash` or `provenance = 'extraction'`. An adopted element outside
   those passes the probe if it is also unstamped, carrying whatever was
   written to it before the reseed.

   So a reseeded graph that passes the probe carries no write from before the
   reseed, except on an unstamped `:__SelfModel__`-only element the reseed
   adopted. That needs a seed source defining the same `:__SelfModel__` node
   id, or a fact of the same type between the same two `:__SelfModel__`
   nodes, as an unstamped extraction write.
   [UNVERIFIED: whether the current seed source does; it lives in
   `mist-memory/seed/` (`load_seed_documents`), which is not in the
   repository]
2. **The empty log shows nothing was extracted after the reseed.** Both
   extraction paths act only on a turn in the log: the dispatcher reads the
   log, and the in-process path runs only with the `event_id` `append_turn`
   returned (`ConversationHandler`: `grep -n 'elif event_id:'
   backend/chat/conversation_handler.py`). The curation scheduler's
   `SelfReflectionJob`, which applies Stage 9 operations, reads its turns from
   the log too (`get_turns_since`). [VERIFIED-IN-REPO] An empty log at
   promotion, with no log reset after the reseed, means no turn existed after
   it for any of them. Promotion checks the log before the probe and again
   inside its transaction.
3. **Stopping the backend excludes the writers that need no turn.** The
   curation-scheduler jobs in 6A.2 run on timers inside the live backend and
   write seed nodes without any logged turn. So the backend stays STOPPED from
   the reseed through promotion.

The argument also rests on two things no command checks: [UNVERIFIED]

- every `seed_version` in the live graph is the one the reseed wipes (the wipe
  is scoped to one version, and the probe checks that `seed_version` is
  present, not which). A read-only `MATCH (n) WHERE n.seed_version IS NOT NULL
  RETURN DISTINCT n.seed_version` (and the same over relationships) would show
  it; it has not been run;
- nothing outside the backend (an operator script) wrote the live graph after
  the reseed. Such writers have not been enumerated.

### 6A.2 What the probe and the apply-marker check cannot see

[VERIFIED-IN-REPO] The same list, with the same greps, is the comment above
`SEED_ONLY_PROBE_CYPHER` in `cutover.py`.

What the probe proves: at least one node, no node or relationship WITHOUT
`seed_version`, and no node or relationship WITH `ontology_version`,
`extraction_version` or `model_hash`. The seed applier sets `seed_version` on
every node and edge it writes (`apply_seed_documents`, `_MERGE_EDGE`:
`grep -nE '"seed_version": seed_version,|SET r\.seed_version' backend/knowledge/seed/applier.py`).
Any node or edge an extraction writer CREATES, stamped or not (for example the
unstamped KNOWS and HAS_CAPABILITY edges `SkillDerivationJob` MERGEs:
`grep -nE 'MERGE \((u|m)\)-' backend/knowledge/curation/skill_derivation.py`),
lacks `seed_version` when it is written, and the probe sees it. A reseed
refuses to run over most such elements and can adopt the rest (below).

It cannot see an unstamped SET on an existing SEEDED element:

- nodes:
  - `InternalKnowledgeDeriver._apply_operation` (Stage 9; also reached from
    `SelfReflectionJob` through `derive`): the UPDATE and DEPRECATE branches,
    and the CREATE-op MERGE's ON MATCH (`updated_at`, `confidence`, and a type
    label) when it matches a seeded `:__SelfModel__` node:
    `grep -nE 'op_type == "(UPDATE|DEPRECATE)"|ON MATCH SET' backend/knowledge/extraction/internal_derivation.py`;
  - `SkillDerivationJob._update_skill` and the update branch of
    `_ensure_capability`:
    `grep -n 'SET e.proficiency' backend/knowledge/curation/skill_derivation.py`;
  - `CurationGraphWriter._upsert_entity`'s ON MATCH when the statement carries
    no EXTRACTED_FROM clause (`with_conversation_provenance` false; with it, the
    EXTRACTED_FROM edge and its ConversationContext node lack `seed_version` and
    are counted):
    `grep -n 'ON MATCH SET e.confidence' backend/knowledge/curation/graph_writer.py`;
- edges (seed edges join `:__Entity__` and `:__SelfModel__` nodes: the label
  union in `_MERGE_EDGE`, `grep -n 'MATCH (s:' backend/knowledge/seed/applier.py`):
  - `ReconciliationEngine._apply`'s CLOSE_TRANSACTION branch (`recorded_until`,
    `is_latest_belief = false`, `updated_at`: a seed edge closed by
    supersession) and REINFORCE branch (`confidence`, `evidence`,
    `updated_at`), both by elementId, and `_apply_structural`'s ON MATCH
    (`confidence`, `evidence`, `updated_at`) when its MERGE matches a seed edge:
    `grep -nE 'is ActionKind\.(CLOSE_TRANSACTION|REINFORCE):|ON MATCH SET' backend/knowledge/curation/reconciliation.py`.
    Whether the same turn also creates an element the probe does see (a new
    version, a clamped copy) depends on the actions the planner chose; nothing
    here relies on it;
- adopted elements. The seed applier's `_MERGE_NODE` MERGEs on the node id and
  `_MERGE_EDGE` on (subject, type, object); neither keys on `seed_version`
  (`grep -nE '"MERGE \((n|s)' backend/knowledge/seed/applier.py`). So a reseed
  would give `seed_version`, and the seed's provenance, and for an edge its
  `valid_from`, `valid_to`, `source_type` and `confidence`, to any non-seed
  element that survived the wipe and matches a seed node or fact, keeping
  every property the seed does not set. [VERIFIED-IN-REPO]
  - MIS-177 D1: `seed` (`reseed`, and `apply_seed_documents`) refuses, before
    any write or wipe and with no override, a graph holding (a) anything the
    graph-reset guard counts (`admin.RESET_GUARD_CYPHER`: an `:__Entity__`
    node lacking `provenance = 'seed'` or `seed_version` or carrying an
    extraction stamp, or a relationship touching an `:__Entity__` node
    without `seed_version` or with a stamp), (b) any node or relationship, in
    any partition, carrying `extraction_version` or `model_hash`, or (c) any
    with `provenance = 'extraction'`:
    `grep -nE '_assert_seed_target_holds_only_seed\(|^SEED_GUARD_STAMP_PROPERTIES' backend/knowledge/seed/applier.py`.
    [UNIT-TESTED: `tests/unit/knowledge/seed/test_seed_guard.py`]
    [UNVERIFIED: `tests/integration/knowledge/test_seed_guard_eval.py` has
    not been run against Neo4j from this branch]
  - A clamped copy of a seed edge (`ReconciliationEngine._apply_append`) has
    the same type and endpoints as the seed edge it copies, so the reseed
    would adopt it: reset `valid_to` to the seed fact's, so the retired
    belief reads as current again, and let the next reseed's wipe delete it.
    Its target is always an `:__Entity__` node and it has no `seed_version`
    (`grep -nE 'MATCH \(t:__Entity__|r.seed_origin_version = ' backend/knowledge/curation/reconciliation.py`),
    so (a) refuses the reseed. So do the KNOWS edge and the `user` node
    `SkillDerivationJob` MERGEs without a stamp
    (`grep -nE 'MERGE \(u:|MERGE \(u\)-' backend/knowledge/curation/skill_derivation.py`).
  - An UNSTAMPED `:__SelfModel__`-only element is outside (a), (b) and (c),
    so a reseed can still adopt it, and it then passes the probe with
    anything written to it before the reseed. Examples are the HAS_CAPABILITY
    edge `SkillDerivationJob` MERGEs and the edge Stage 9 MERGEs from
    MistIdentity, neither with any property:
    `grep -n 'MERGE (m)-\[:HAS_CAPABILITY\]' backend/knowledge/curation/skill_derivation.py`,
    `grep -n 'MERGE (m)-\[:{rel_type}\]->(e)' backend/knowledge/extraction/internal_derivation.py`.
    The nodes those two writers create carry `ontology_version` only, which
    the probe refuses but the seed guard deliberately does not test (the
    startup MistIdentity node carries it too; the comment above
    `SEED_GUARD_STAMP_PROPERTIES` gives the greps). This needs a seed source
    that defines the same node id, or an edge of the same type between the
    same two nodes. [UNVERIFIED: whether the current seed source
    (`mist-memory/seed/`, not in the repository) defines one]
- curation-scheduler jobs, which need no logged turn:
  - `ConfidenceDecayJob` (`confidence`, and `status = 'archived'`, on active
    `:__Entity__` nodes of a decay-enabled `knowledge_domain`), `OrphanDetector`
    (`status = 'archived'`) and `EmbeddingMaintenance` (`embedding`,
    `embedding_updated_at`; registered with `enabled=False`):
    `grep -n 'SET e\.' backend/knowledge/curation/confidence_decay.py backend/knowledge/curation/orphan_detector.py backend/knowledge/curation/embedding_maintenance.py`;
  - they, `SkillDerivationJob` and `SelfReflectionJob` are built by
    `build_curation_scheduler` (`backend/factories.py`:
    `grep -nE 'name="(confidence_decay|orphan_detection|embedding_maintenance|skill_derivation|self_reflection)"' backend/factories.py`)
    and started in the backend's lifespan (`backend/server.py`:
    `grep -nE 'build_curation_scheduler\(|curation_scheduler\.start\(' backend/server.py`).
    `CurationScheduler.start` skips them only under `MIST_HYDRATION_ISOLATION`
    or with `MIST_CURATION_SCHEDULER_ENABLED` off (`curation_scheduler_enabled`,
    default on).

The apply-marker check (`extraction_applied` empty) cannot see a turn applied
without a marker: the in-process path above writes none, and a turn recorded as
legacy at the backlog's first activation gets none whether or not an earlier
path applied it (`BacklogStore.ensure_activation`). The log-empty check covers
both: with no logged turn there is no such turn.

Hence the precondition and 6A.1: these checks back it up, they do not replace
it.

### 6A.3 Procedure

None of the `docker compose` forms below has been run from this branch
[UNVERIFIED]; the commands themselves are [VERIFIED-IN-REPO] where marked.

1. **Begin and reach `ready`.** Begin as in section 2 while the backend runs
   (`MIST_EXTRACTION_INFERENCE=service`). With the log empty there is no turn to
   fill, and the dispatcher's first fill step moves the cutover to `ready`
   (`ExtractionDispatcher._fill_step`: no uncovered turn -> `ready`).
   [UNIT-TESTED: `test_an_empty_log_fills_to_ready_and_promotes_with_zero_turns`,
   with a fake service] Then:

       docker compose exec mist-backend python -m backend.extraction_backlog.admin cutover status

   It must print `state=ready` and `covered=0 total=0` (`total` counts every
   logged turn: `_print_cutover` -> `BacklogStore.fill_scan`). [UNIT-TESTED: the
   `covered=... total=...` line]

2. **Stop the backend**, and keep it stopped through step 5 (6A.1, point 3):

       docker compose stop mist-backend

3. **Reseed the live graph** (6A.1, point 1). This must come after the last log
   reset:

       docker compose run --rm mist-backend python scripts/mist_admin.py seed --no-vault-bootstrap

   [VERIFIED-IN-REPO: `cmd_seed`, subcommand `seed`, in `scripts/mist_admin.py`;
   it calls `reseed(..., allow_live=True)`] If it refuses with
   `SeedTargetNotSeedOnlyError` (6A.2, "adopted elements"), it has written
   nothing and the precondition does not hold: do not continue on this path.
   After a successful reseed it backfills seed embeddings
   unless given `--no-embeddings`; that writes only `embedding`, which the probe
   does not read. `--no-vault-bootstrap` keeps the reseed to the graph: the
   vault bootstrap runs after `cmd_seed` has already called
   `connection.disconnect()`, and touches only the vault directory. It writes
   the notes `identity/mist.md` and `users/<id>.md` (`users/user.md` for the
   seed's `user.md`) through `VaultWriter` (`admin.bootstrap_vault_from_seed`);
   before that, `VaultWriter.start` creates the vault subdirectories
   (`sessions`, `identity`, `users`, `decisions`, `meta`) and, when the vault
   has no `.git`, runs `git init` and an empty initial commit there, since
   `git_auto_init` defaults to True (`MIST_VAULT_GIT_AUTO_INIT`):
   `grep -nE '_ensure_directories_sync|_maybe_git_init|"commit"' backend/vault/writer.py`,
   `grep -n git_auto_init backend/knowledge/config.py`. `VaultWriter` neither
   reads nor writes the graph or the event store (`grep -nE
   'event_store|EventStore|neo4j|Neo4j' backend/vault/writer.py` prints
   nothing), so the bootstrap cannot change the graph, the event store, the
   probe or the log-empty result, and is not part of the cutover.
   [VERIFIED-IN-REPO: `grep -nE 'connection.disconnect\(\)|no_vault_bootstrap' scripts/mist_admin.py`]

4. **Probe, read-only:**

       docker compose run --rm mist-backend python -m backend.extraction_backlog.admin cutover probe

   [VERIFIED-IN-REPO: `_cutover_probe` in `admin.py`, `check_seed_only` in
   `cutover.py`] [UNIT-TESTED: `TestProbeCommand` in
   `tests/unit/extraction_backlog/test_cutover_seed_only.py`, with a fake probe
   and with the real probe over a fake connection that refuses writes]
   [UNVERIFIED: the real probe has not been run against Neo4j]. It runs the
   log-empty check and the graph probe (the probe even when the log check
   fails) and prints one line per check and one `  violation: ...` line per
   violation, for example:

       [probe] conversation log: PASS (0 logged turns)
       [probe] live graph: PASS (node_count=32 nodes_without_seed_version=0 nodes_with_extraction_stamp=0 relationship_count=30 relationships_without_seed_version=0 relationships_with_extraction_stamp=0)
       [probe] both checks pass (exit 0). The probe cannot see every write; the operator precondition in CUTOVER.md 6A still applies.

   Exit codes: 0 both checks pass; 2 either refuses (`conversation log:
   REFUSED (N logged turn(s))` and/or `live graph: REFUSED (...)`); 1 the probe
   could not run (Neo4j unreachable, or any `GraphProbeError`: `live graph: NOT
   RUN: ...`), whatever the log check found. It writes nothing to the event
   store, the cache or the graph, and works with or without an open cutover.
   Do not promote on anything but exit 0.

5. **Promote:**

       docker compose run --rm mist-backend python -m backend.extraction_backlog.admin \
         cutover promote --seed-only-graph

   [VERIFIED-IN-REPO: `admin.py`, `_add_cutover_parser`] [UNIT-TESTED:
   `tests/unit/extraction_backlog/test_cutover_seed_only.py`, with a fake probe]

6. **Restart on the new epoch.** As in section 6: set the printed
   `MIST_MODEL_HASH` in the backend's environment, then recreate the backend so
   its writers are stamped with the new epoch:

       docker compose up -d --force-recreate mist-backend

   `status` (section 1) must print `writer stamps: match`. [UNIT-TESTED: the
   `status` line]

`--seed-only-graph` and `--graph-swapped` are mutually exclusive; giving both is
refused by argparse (exit 2) before anything is opened. [UNIT-TESTED]

`cutover promote --seed-only-graph` exits 0 on promotion and 2 on any refusal,
printing `[cutover] REFUSED: <reason>` and writing nothing, when: [UNIT-TESTED]

- no cutover is open, or it is not `ready` or `checked`;
- the fill is incomplete (a logged turn has no candidate cache row);
- `extraction_applied` holds ANY row, under ANY epoch, `applied` or `curated`
  (a recorded apply; an empty table does not prove no turn was applied, see
  6A.2);
- the active epoch is no longer the cutover's source epoch;
- the conversation log is not empty: `conversation_turn_events` holds any row,
  from any session origin (`real`, `test`, `seed`). The reason names the count:
  `the conversation log holds N logged turn(s) (conversation_turn_events, any
  session origin): seed-only promotion requires an empty log, ...; use the
  graph-swapped path`. This check runs BEFORE the probe, so a non-empty log
  never reaches Neo4j;
- the live-graph probe fails: it must find at least one node, every node and
  relationship carrying `seed_version`, and none carrying `ontology_version`,
  `extraction_version` or `model_hash`. A probe that cannot connect or errors is a
  refusal too (exit 2 here, where `cutover probe` exits 1).

The probe is ONE read-only Cypher statement (`SEED_ONLY_PROBE_CYPHER` in
`cutover.py`) run on the LIVE graph named by the backend's environment
(`config.neo4j`) through `Neo4jConnection.execute_query`. `promote` runs it only
after the state, fill, marker, epoch and log checks pass, and before the
transaction. A unit test refuses any write clause in it. [UNIT-TESTED: textually,
and over a fake connection]

The fill is checked before the transaction, not inside it (the candidate cache is
a separate database); with an empty log there is no turn for it to cover.

Otherwise ONE SQLite transaction under `BEGIN IMMEDIATE`
(`EventStore.promote_epoch_cutover_seed_only`) re-checks the state, the source
epoch, the markers and the empty log, so a turn logged after the first log check
(during the probe, say) is refused there with the same reason and nothing is
written [UNIT-TESTED: `test_e_a_turn_logged_during_the_probe_...`,
`test_e_a_turn_logged_after_the_probe_...`, and
`test_a_turn_another_connection_commits_before_the_transaction_is_refused`],
then:

1. appends the candidate to `epoch_ledger` (prev = the source epoch);
2. writes the new epoch's backlog activation first-hand with
   `turns_at_activation` = `marked_applied` = `legacy_unextracted` = 0, so T2a's
   automatic first-activation rule does not run for it;
3. writes NO applied marker;
4. sets the cutover to `promoted`, its `check_report` recording
   `promotion_mode: seed_only_graph`, the probe counts, and any earlier check
   report under `prior_check_report`.

A crash anywhere inside leaves none of it written; run it again. [UNIT-TESTED:
`TestPromoteSeedOnly::test_a_crash_between_ledger_append_and_activation_writes_nothing`]

On success it prints, for example:

    [cutover] promoted cutover 1 to epoch 2 over a seed-only live graph (32 node(s), 30 relationship(s), all seed-stamped) and an empty conversation log: no turn was marked applied. The dispatcher extracts and applies turns logged from now on, in log order, under the new epoch.

followed by the `MIST_MODEL_HASH` line (section 6). After the restart the
dispatcher extracts every turn logged from then on under the new epoch and
applies it in log order. [UNIT-TESTED:
`test_turns_logged_after_promotion_are_extracted_and_applied_in_log_order`]

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
  probe and the COMMIT is not seen. Keeping the backend stopped from the reseed
  through promotion (section 6A.3) closes that window for the backend's own
  writers, including the curation scheduler; the transaction re-checks the
  SQLite side (state, source epoch, apply markers, empty log).
- The probe requires EVERY node to carry `seed_version`, so a node any other
  writer created (including Stage 9 self-model nodes) refuses the seed-only path
  even if it carries no extraction stamp.
- Seed-only promotion refuses any logged turn, of any session origin, including
  `test` and `seed` sessions that no extraction path would pick up. A log that
  holds turns goes through sections 4 to 6.
- `seed` refuses a graph holding non-seed `:__Entity__` data, or any element
  carrying `extraction_version`, `model_hash` or `provenance = 'extraction'`
  (MIS-177 D1, 6A.2), but can still adopt an unstamped
  `:__SelfModel__`-only element that matches a seed node or fact, which then
  passes the probe.
- The probe cannot see unstamped SETs on seeded elements, and the marker check
  cannot see in-process or legacy applies (section 6A.2 names the symbols and
  the greps). An empty log shows only that the log is empty now, not that it
  was never reset. The seed-only path rests on the operator precondition and
  the reseed-then-probe procedure in 6A; writers outside the backend are not
  enumerated.
