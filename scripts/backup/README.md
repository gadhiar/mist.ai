# MIST.AI backup and restore runbook

Read this at 3am. It assumes you did not write any of it and that you are not calm.

Three commands exist:

    python -m scripts.backup.dump       take a backup      (read-only, safe to run any time)
    python -m scripts.backup.restore    put one back       (DESTRUCTIVE, four gates, see below)
    python -m scripts.backup.prune      delete old ones    (deletes nothing without --confirm)

Exit codes are the same shape everywhere: `0` it worked, `2` it refused and changed nothing,
`1` it failed part way and the message says what state that leaves.

WHAT HAS AND HAS NOT BEEN EXERCISED. Four separate statements, because they have four different
evidence bases:

- The three commands above, and every refusal they can make, are covered by
  `tests/unit/backup/` (222 tests: `python -m pytest tests/unit/backup -q`), including a full
  synthetic round trip that restores a captured graph and compares embeddings for exact equality.
- The `backend/` code this runbook also names -- `load_artifact`,
  `restore_graph_from_artifact` and the `graph-stats` helpers -- is NOT covered by that suite. It
  is covered by `tests/unit/knowledge/`, which is a different tier of the same run.
- Two of the fixes that suite covers are proved only against SIMULATED faults, and in neither case
  is the simulation the defect: the Neo4j schema-rejection rule is modelled from the Cypher manual
  and has never been observed against a real server, and the read-only vault handler is exercised
  on Linux through a read-only DIRECTORY when the defect it exists for is a Windows `WinError 5`
  on read-only git objects. Section 4.9 says what that leaves unproved.
- Nothing here has been run against the live stack or the dev-hydration stack on the host from
  this branch. The rehearsal below is therefore a rehearsal, not a replay of something already
  done -- and for the two items above it is the only real gate there is.

---

## 1. Take a backup

    export MIST_BACKUP_ROOT=/mnt/backup/mist
    python -m scripts.backup.dump

`MIST_BACKUP_ROOT` is REQUIRED and there is no default. An unset variable is a refusal
(exit 2), not a fallback path. That is deliberate: the tool this replaces defaulted to
`data/graph_snapshots/`, inside the live state root, so one `rm -rf data` took the stores and
every graph backup together.

It may point anywhere that does not die with this machine, including an external drive, with no
code change:

    export MIST_BACKUP_ROOT=/run/media/raj/backup-usb/mist
    python -m scripts.backup.dump --label pre-upgrade

The destination is refused if it is the repository, sits inside the repository, or is live state
(`scripts/backup/destination.py`). A backup kept in the working tree is destroyed by the same
deletion that destroys the working tree.

The dump is read-only against live state: the SQLite stores are opened `mode=ro` and copied
through `sqlite3.Connection.backup()`, and the graph is read with two `MATCH` queries. It is safe
to run while the backend is serving.

Optional flags: `--output DIR` (overrides `MIST_BACKUP_ROOT`), `--label NAME` (default: the UTC
timestamp), `--state-root DIR`, `--vault-root DIR`, `--verbose`.

An artifact is built in `<label>.partial` and renamed only after its `manifest.json` is written.
So a directory WITHOUT the suffix is complete, and a directory WITH it is a dump that died.

---

## 2. What is captured, and what is not

Captured, in one artifact directory:

| Path in the artifact         | What it is                                                     |
| ---------------------------- | -------------------------------------------------------------- |
| `stores/event_store.db`      | conversation sessions, turn events, epoch ledger                |
| `stores/extraction_cache.db` | cached extraction results                                       |
| `stores/vault_sidecar.db`    | vault chunk index, with its embeddings                          |
| `graph.json`                 | every node and relationship, all partitions, INCLUDING embeddings |
| `vault/`                     | a copy of the whole `mist-memory/` tree, `.git` INCLUDED          |
| `manifest.json`              | layout version, `created_at`, per-file sha256, row counts, uncounted tables, graph counts, both vault file counts, git HEAD, producer stamps |

The three stores are captured BY NAME, never by glob. The live `./data` also holds
`event_store.pre-r1.4-backup-2026-07-31.db` and `event_store.pre-reset-backup-2026-06-09.db`; a
glob would sweep both in, and a hurried restore could then load a months-old event store.

Graph embeddings travel as exact `list[float]`, not as strings. A restore returns the same floats,
which is why the rehearsal in section 4 checks equality rather than similarity.

### The vault leg: two counts, and why `.git` is captured

`.git` IS CAPTURED, AND THAT IS DELIBERATE. `copy_vault` passes no `ignore=` to
`shutil.copytree` (`scripts/backup/dump.py`, `copy_vault`). The live `mist-memory/` is a git
repository with NO REMOTE and NO UPSTREAM, so its commits exist nowhere else -- 15 of them when
last measured on the host, a figure not re-measurable from a worktree because `mist-memory/` is
gitignored and untracked (`git ls-files mist-memory` -> empty). Excluding `.git` would introduce
a new data-loss mode inside a tool whose whole job is to remove them.

The cost of keeping it is recurring rather than one-off, and is worth knowing rather than
discovering: `digest_artifact_files` rglobs the whole artifact, so every git loose object is
sha256'd at capture -- and re-digested by `verify_artifact_files` at EVERY restore preflight. A
vault whose git history grows makes every preflight slower, not just every dump.

That is why the vault is counted twice. The `vault` block of `manifest.json` carries both keys,
and both the dump and the restore print both numbers:

| Manifest key        | What it counts                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------- |
| `corpus_file_count` | Files with no `.git` path segment. The notes. The only one of the two that answers "did my notes come back". |
| `file_count`        | EVERY file in the tree, `.git` plumbing INCLUDED. A measure of the artifact, never of the corpus.         |

Both legs count through one function -- `count_vault_files` in `scripts/backup/dump.py`, which
`scripts/backup/restore.py` imports rather than reimplementing -- so a capture and a restore
cannot report the same tree differently.

The reason for two numbers is one specific misreading. The MIS-140 rehearsal reported "117 vault
files" as though that measured the corpus. Measured on the host, the 117 was 13 notes and 104 git
objects: 89% plumbing. Neither number is printed bare now. The line reads
`13 vault corpus file(s) (117 including .git plumbing)`.

#### A layout-2 artifact may or may not carry `corpus_file_count`

`BACKUP_LAYOUT_VERSION` stays at **2**. Adding a manifest key is backward compatible: a reader
that does not know `corpus_file_count` ignores it, and `file_count` keeps the meaning it has
always had. REDEFINING `file_count` to mean the corpus would not have been compatible, which is
why it was not done.

The consequence a reader has to handle: `BackupManifest` has no version field of its own
(`scripts/backup/manifest.py`), so nothing in a layout-2 manifest distinguishes one written
before this key from one written after. A layout-2 artifact MAY OR MAY NOT carry
`corpus_file_count`, depending on when it was taken. Read it with a default rather than by
subscript:

    manifest["vault"].get("corpus_file_count")   # None on a layout-2 artifact taken before the key

### How a captured store is verified (layout version 2)

`PRAGMA integrity_check` is the gate. Each copy is reopened and checked; a copy that fails is a
damaged file and the dump stops with the artifact left as `.partial`. The per-table row counts in
`manifest.json` are REPORTING, not a gate, because a `SELECT COUNT(*)` that fails says what the
verifying process lacks rather than what the file holds.

`vault_sidecar.db` is the concrete case. It holds the sqlite-vec virtual table `vault_chunks_vec`,
which no connection can query without the extension loaded (`no such module: vec0`), so counting it
used to fail the whole dump on a byte-correct copy (MIS-153). Its four backing tables are ordinary
tables and count either way. The dump now loads sqlite-vec for every store; where the extension is
unavailable, the tables that could not be counted are listed in that store's `uncounted_tables` and
named in a `[backup] WARNING` line, and the artifact is still written. Install `sqlite-vec>=0.1.3`
(already in `requirements.txt`) to get their counts.

Two consequences worth knowing before reading a manifest:

- `row_counts` may be PARTIAL. A table missing from it is not an empty table; check
  `uncounted_tables`, which maps each uncounted table to SQLite's own message, before reading a gap
  as a zero. The message is recorded because a missing module is only the commonest cause; where
  none is named, the tooling says so rather than guessing.
- sqlite-vec's shadow tables (`vault_chunks_vec_chunks`, `_info`, `_rowids`, `_vector_chunks00`)
  appear in `row_counts` as themselves. They are not filtered, because filtering would mean
  hardcoding one extension's internal naming.

Neither check establishes vec0 SEMANTIC validity: `COUNT(*)` on a `vec0` table counts its rowid
shadow table, not the vectors.

#### What the version bump does to older artifacts

`read_manifest` requires the layout version to match EXACTLY, so a pre-MIS-153 `layout_version: 1`
artifact is not readable by this build. Two consequences:

- `restore` refuses it. Recovering from one means checking out a commit from before the bump.
- `prune` files it under `skipped`, not `delete`. Skipped directories are never deletion
  candidates, so a v1 artifact is retained forever and does not count toward `--retain`. Remove any
  such directory by hand once you no longer want it.

This was accepted rather than overlooked: no complete artifact exists anywhere on the deployment
(`find -name manifest.json` returns nothing; the backup root holds only a `.partial` directory),
so the bump orphans nothing today. It will matter the first time the version moves again.

### NOT captured: `data/vector_store/`

It is recorded as excluded in every manifest (`"excluded": ["vector_store"]`) rather than silently
omitted.

The measured facts, taken on the live deployment during MIS-140 (not re-measurable from a
worktree, which has no `data/`):

- `du -sh data/vector_store` -> `5.0K`.
- `find data/vector_store -type f` -> exactly two files: a `_transactions` log and a `_versions`
  manifest. There are no `.lance` data fragments, so the store holds ZERO vectors today.

Do not read that as "vector_store is derived and therefore safe to drop". The honest position is
narrower: it is written by the ingestion pipeline from EXTERNAL source documents
(`grep -n "_vector_store.store_chunks" backend/knowledge/ingestion/pipeline.py` -> :224), so if it
were ever populated, its contents would not be reconstructible from the graph or the vault alone --
they would be reconstructible only by re-ingesting those same source documents, wherever they are.
It is excluded today because it is empty. If it stops being empty, this exclusion needs revisiting.

`vault_sidecar.db` IS captured even though it is derived from `mist-memory/`. The reason is cost,
not principle: it holds per-chunk embeddings, so rebuilding it re-runs the embedding model over the
whole corpus.

---

## 3. Why losing `./data` and losing the graph are separate failures

They live on different storage and fail independently.

- The SQLite stores and `mist-memory/` are on the bind mount: they are files under the repository
  working tree on this disk.
- Neo4j is on DOCKER NAMED VOLUMES: `mist-neo4j-data` and `mist-neo4j-logs`
  (`docker-compose.yml:105-107`, declared at `docker-compose.yml:179-181`). They are NOT under
  `./data`.

Consequences, each one separately true:

- `rm -rf data` destroys the stores and leaves the graph intact.
- `docker volume rm mist.ai_mist-neo4j-data` destroys the graph and leaves `./data` intact.
- A dead disk takes both.

`mist-memory/` is covered by nothing else at all: it is gitignored (`grep -n "mist-memory"
.gitignore` -> :39) and `git ls-files mist-memory` returns zero files, so the corpus exists only on
this disk until a backup copies it.

That is why one artifact covers all three, and why an artifact missing a leg is a failure rather
than a smaller success.

---

## 4. The restore rehearsal

This is the step that turns the backup from a rumour into a fact. Do it once before anything is
scheduled, and repeat it after any change to this package.

The target is the dev-hydration stack. It is a real, already-existing non-live target: its own
Neo4j (`mist-neo4j-dev`, host Bolt port 7690) and its own state root (`./dev-state`).

### 4.1 Bring the dev stack up

From the repository root:

    docker compose -f docker-compose.yml -f docker-compose.override.yml \
      -f docker-compose.dev-hydration.yml --profile dev up -d mist-neo4j-dev

Naming the service explicitly matters: a bare `up -d` would also reconcile the live services.
`mist-neo4j-dev` publishes host port 7690 (`docker-compose.dev-hydration.yml:73`), and
`localhost:7690` is already in the dev allowlist (`backend/knowledge/eval_isolation.py:69`), which
is what lets a restore run from the host shell.

### 4.2 Place the handshake marker

The restore refuses any target that has not identified itself. Identification is a file YOU create
in the target root. `./dev-state` may not exist yet -- it is created by the dev BACKEND service,
which this rehearsal does not start -- so create it first:

    mkdir -p ./dev-state
    touch ./dev-state/MIST_RESTORE_TARGET

Nothing in MIST.AI creates that file. The live state root will never carry it, however it is
spelled, symlinked or bind-mounted, which is the entire point: a denylist can only refuse the
spellings someone thought of.

### 4.3 Learn the exact token to type

Run the restore once without `--confirm-target`. It refuses (exit 2) and prints the resolved
absolute path you must type back:

    export MIST_BACKUP_ROOT=/mnt/backup/mist
    export NEO4J_PASSWORD=...        # the host shell needs the dev instance's credentials
    python -m scripts.backup.restore \
      --artifact /mnt/backup/mist/20260917T030000Z \
      --target-root ./dev-state \
      --target-graph-uri bolt://localhost:7690

COPY THE PATH THE TOOL PRINTS. Do not type a path from this document, and do not type one from
memory. The resolved form depends on the machine and the shell you are in: this host is Windows
(`D:\Users\rajga\mist.ai`), so a run from a native Windows shell resolves to a backslashed drive
path, while the same directory reached from inside a container resolves to a POSIX path such as
`/app/dev-state`. The comparison is byte-for-byte and neither form is normalised into the other,
so the only string guaranteed to be accepted is the one the refusal just printed.

The token is compared against the resolved path and never against the string you passed. Typing
`--target-root ./dev-state --confirm-target ./dev-state` is REFUSED even though both strings match
each other: retyping a relative path proves nothing about where it lands, and where it lands is the
only question that matters. Whitespace around the token is stripped; nothing else is normalised.

### 4.4 Run it

The same command again, with the path from 4.3 pasted after `--confirm-target`. The placeholder
below is NOT a value to type -- substitute the exact string the previous run printed:

    python -m scripts.backup.restore \
      --artifact /mnt/backup/mist/20260917T030000Z \
      --target-root ./dev-state \
      --target-graph-uri bolt://localhost:7690 \
      --confirm-target <PASTE THE RESOLVED PATH PRINTED IN 4.3>

Four gates run, every time, and no flag disables any of them:

1. `--target-root` is required; there is no default.
2. The typed token must equal the resolved target path.
3. The target must carry `MIST_RESTORE_TARGET`, on top of `assert_isolated_root` for the root and
   `assert_neo4j_dev_isolated` for the graph URI.
4. A pre-restore backup of the TARGET is taken first, through the same dump leg. If it fails, the
   restore does not proceed.

The first line of successful output is the pre-restore artifact path. That directory is your way
back if you have just restored into the wrong place. Write it down before reading the rest, and
see 4.7 for what to do with it.

#### The six phases

A restore is stage-then-swap, in six named phases (`run_restore` in `scripts/backup/restore.py`).
Each has a documented failure state, and
`tests/unit/backup/test_restore.py::TestAFailureInEachPhaseLeavesExactlyTheDocumentedState` pins
them.

1. PREFLIGHT. Read-only, and it checks BOTH halves. Artifact: every file the manifest names is
   re-digested (`verify_artifact_files`), then the graph leg is parsed, version-checked, decoded
   and checked for re-anchorable endpoints (`load_graph_leg`). Target: the target graph answers
   `SHOW CONSTRAINTS` and `SHOW INDEXES`; every DDL statement in the artifact names an object this
   build can parse; the volume has room for the staged copies; and no earlier restore left a
   marker behind (`assert_target_is_restorable`). Any failure here is exit 2 with the target
   bit-for-bit untouched.
2. PRE-RESTORE BACKUP. The fourth gate and the first consequential step: a full dump of the
   target through the same dump leg. If it fails, exit 2 and nothing has been overwritten.
3. STAGE. Each store is copied to `<name>.db.incoming` and the artifact's vault tree to
   `vault.incoming/`, each a sibling of the file or directory it will replace. Nothing live is
   touched, so a failure here costs nothing and needs no recovery.
4. GRAPH. `restore_graph_from_artifact` detach-deletes the target graph, replaces its schema, and
   loads nodes and relationships in batches. THE ONE NON-ATOMIC LEG.
5. COMMIT. Per store: `os.replace` onto the live name, then that store's `-wal`/`-shm` are
   unlinked, then the next store. Then the vault, as two renames: `vault` -> `vault.previous`,
   then `vault.incoming` -> `vault`.
6. CLEANUP. `vault.previous` is removed, then the progress marker is deleted.

#### The ordering guarantee, stated per fault class

THIS ORDERING USED TO BE DESCRIBED MORE BROADLY THAN IT HELD. The old text here said the whole
artifact was checked first and the graph was written last, and presented that as a general
guarantee that a failure left the target intact. It held for ARTIFACT faults and it was false for
TARGET faults: the target was not checked at all, and a graph leg running LAST failed only after
the stores and the vault had already been replaced. That is how the MIS-140 rehearsal ended with
a target holding the artifact's stores, the artifact's vault, and an empty graph. What holds now
is narrower, and is true per fault class rather than in general:

- An ARTIFACT fault -- a missing or altered file, a `format_version` this build cannot read, a
  relationship endpoint no node provides -- is caught in phase 1. Exit 2, target untouched.
- A TARGET fault -- graph down, artifact DDL this build cannot name, too little free space, a
  stale restore marker -- is also caught in phase 1, before the pre-restore backup is spent.
  Exit 2, target untouched.
- A GRAPH fault is caught in phase 4, which now runs BEFORE the swap rather than after it. The
  target's graph is destroyed; its stores and its vault are still its own.
- A COMMIT fault happens in phase 5. What the target holds then depends on how far the commit
  got, and the marker records exactly that. See 4.5 and 4.7.

Phase 1 does two artifact checks rather than one because digest validity and decodability are
different properties: a file can match its recorded sha256 exactly and still be an artifact this
build cannot read, because the version it declares is one this build does not know.

### 4.5 What is atomic here, and what is not

Read this before you need it. At 3am the useful question is which of the two lists below your
failure is in.

#### What IS atomic: `os.replace`, per store and per rename

- Each store is committed by one `os.replace` onto its live name. That rename either happened or
  it did not; there is no half-written store, and the target never sees a truncated `.db`.
- THREE STORES ARE THREE ATOMIC OPERATIONS, NOT ONE. A failure at store 2 of 3 leaves store 1
  holding the artifact's copy and stores 2 and 3 holding the target's own, so the target is a
  mixture of two points in time. `stores_committed` in the marker names exactly which ones moved.
- The vault swap is TWO RENAMES, not one, because Windows cannot rename a directory onto an
  existing one. There is therefore a window in which `vault/` DOES NOT EXIST: the target's tree is
  at `vault.previous/` and the artifact's is at `vault.incoming/`. Both trees are intact in that
  window; neither is named `vault`. The window is irreducible without transactional NTFS, which is
  deprecated.

This is atomic PER STORE AND PER RENAME. It is NOT atomic per tree, and nothing in this package
says that it is. Staging is always a sibling of its destination, which is what makes each
`os.replace` a same-volume rename and therefore atomic at all; no flag makes the staging location
configurable, because such a flag would take that guarantee away silently.

#### What is NOT atomic: phase 4, the graph

`restore_graph_from_artifact` detach-deletes the target graph and drops its schema before it
loads anything (`backend/knowledge/admin.py`). There is no staging step for a graph and no
rollback.

**A PHASE-4 FAILURE DESTROYS THE TARGET'S GRAPH.** The stores and the vault survive it -- they
are still the target's own, with the artifact's staged copies sitting beside them uncommitted --
and the pre-restore artifact is the only route back to the graph that was there. That is a
property of the design. It is not a statement about how much data the graph happens to hold
today, and it does not become less true as it holds more.

#### The failure modes that remain open

Named, because a runbook listing only the handled ones reads as a guarantee:

- A RELATIONSHIP-COUNT MISMATCH RAISED AFTER NODES ARE ALREADY LOADED. The batched relationship
  load checks `created != len(batch)` and raises `GraphArtifactError` when the server creates a
  different number (`restore_graph_from_artifact` in `backend/knowledge/admin.py`). Every node,
  the schema, and every earlier relationship batch are already in the target when that fires.
  The message says the graph is partially loaded; it is.
- A DRIVER DISCONNECT, A SERVER RESTART OR A FULL DISK DURING ANY LEG. Nothing holds a
  transaction across phases, so each of these leaves the target wherever the interrupted phase had
  reached. The marker is the record of which phase that was.
- A STORE-LEG FAILURE AT STORE 2 OF 3, as above: a target that is a mixture of two points in time
  rather than either of them. The commonest cause of a failed rename is a backend still running
  against the target and holding a store open.
- PHASE 6'S REMOVAL OF `vault.previous` IS NOT A FAILURE MODE OF THE RESTORE. It is the one step
  whose failure is caught and downgraded to a warning. By the time it runs the restore has
  SUCCEEDED -- stores, vault and graph are all the artifact's -- and `vault.previous/` is a
  redundant second copy of the target's old vault, which the pre-restore artifact also holds. So a
  failed deletion prints a `[restore] WARNING:` line naming the directory, still deletes the
  marker, and still exits 0. Remove the directory by hand.

  It is written that way deliberately. Left uncaught, `remove_tree`'s `OSError` is neither a
  `MistError` nor a `GraphArtifactError`, so `main`'s handler tuple would miss it: a restore that
  did everything right would exit with a traceback instead of 0, AND leave the marker behind --
  which makes the next restore refuse (4.6). That would turn a failed deletion of a redundant
  directory into a block on the recovery path, at the moment someone is recovering, which is the
  shape of MIS-157 itself. A recovery tool must not withhold recovery to make a point.

### 4.6 `restore.in-progress.json`: the file that says what state you are in

Written into the TARGET ROOT as soon as the pre-restore backup succeeds, rewritten after every
phase transition, and deleted only when the restore completes. Its presence means a restore
started and did not finish.

    <target-root>/restore.in-progress.json

It is JSON, indented and key-sorted, so `cat` is enough:

    cat ./dev-state/restore.in-progress.json

| Key                                   | What it tells you                                                                                                                              |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `marker_version`                      | The marker's own layout version, currently `1`. A reader that does not know the value prints the file verbatim rather than interpreting it.       |
| `started_utc`                         | When the consequential sequence began.                                                                                                           |
| `artifact_dir`                        | The artifact that run was restoring FROM.                                                                                                        |
| `artifact_label`                      | That artifact's manifest label.                                                                                                                  |
| `target_root`                         | The resolved target.                                                                                                                             |
| `pre_restore_artifact`                | THE WAY BACK: the target's own state, captured before phase 3. The first field to read.                                                          |
| `phases.staged`                       | Phase 3 completed. Staged copies exist and nothing live has been touched.                                                                        |
| `phases.graph.committed`              | Phase 4 completed. `false` beside `staged: true` means the graph leg is where it died.                                                            |
| `phases.graph.nodes`, `.relationships`| What phase 4 loaded. `0` until it completes.                                                                                                     |
| `phases.stores_committed`             | The store filenames already `os.replace`d onto their live names, in commit order. A list rather than a flag, because the commit is atomic per store. |
| `phases.vault_committed`              | Both vault renames completed.                                                                                                                    |
| `phases.vault_previous`               | Where the target's previous vault tree was renamed aside, WHILE IT STILL EXISTS. `null` before phase 5; set once the vault is renamed aside. NOT how you learn about a failed cleanup: the marker is deleted at the end of phase 6 whether or not the removal succeeded, so a failure is reported by the `[restore] WARNING:` line instead (4.5). A non-null value here means the run did not reach the end of phase 6 at all. |

The marker is rewritten through its own `.tmp` and an `os.replace`, so a crash during a rewrite
leaves the PREVIOUS marker intact rather than a truncated one.

A TARGET CARRYING A MARKER IS REFUSED, AND THERE IS NO OVERRIDE FLAG. Preflight's fourth check
prints the marker's full contents -- `pre_restore_artifact` included, so the way back is on screen
in the refusal itself -- and exits 2. The only thing that clears it is deleting the file by hand:

    rm ./dev-state/restore.in-progress.json

There is deliberately no `--force`, `--resume`, `--ignore-marker` or `--no-marker`, and none may
be added; `tests/unit/backup/test_restore.py` asserts the parser rejects all four spellings. A
flag is a bypass, and a bypass can be put in a schedule by someone who was not there when the
first restore failed. A manual delete cannot be: the act of deleting the file IS the
acknowledgement that you read it and decided what to do about the target.

### 4.7 If a restore failed: reading the marker and putting the target back

Exit 1 means a phase after the pre-restore backup failed. You have two things: the message the
tool printed, and the marker. Use the marker. The message scrolls away; the file does not.

#### Step 1: find out which phase failed

    cat <target-root>/restore.in-progress.json

Read `phases` from the top. The last entry recorded as complete is the last phase that finished,
so the phase after it is the one that died. `stores_committed` is a list rather than a flag, so a
SHORT list there is a phase-5 failure part way through the stores.

| Marker state                                                       | What the target holds                                                                             | What to do                                                          |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| No marker, and the run exited 2                                    | Its own stores, vault and graph. Nothing was overwritten.                                          | Nothing. Fix what the refusal named and re-run.                      |
| `staged: false`                                                    | Its own stores, vault and graph. Phase 3 failed and removed its own staged copies.                 | Delete the marker, use an intact artifact, re-run. No recovery needed. |
| `staged: true`, `graph.committed: false`                           | ITS OWN stores and vault. A graph that is empty or partially loaded.                               | Case A.                                                              |
| `graph.committed: true`, `stores_committed` shorter than the artifact's store list | The artifact's graph. A MIXTURE of the artifact's stores and its own. Its own vault. | Case B.                                                              |
| `stores_committed` complete, `vault_committed: false`              | The artifact's graph and stores. Its own vault.                                                    | Case B.                                                              |
| `vault_committed: true`, `vault_previous` non-null                 | The artifact's graph, stores and vault. Phase 5 landed and phase 6 did not run.                     | Case C.                                                              |

CASE A -- THE GRAPH LEG FAILED. The stores and the vault on disk are still the target's own, so
the loss is confined to the graph. Two routes, and they are not equivalent:

- If the artifact is sound and the failure was transient (server restart, driver disconnect),
  delete the marker and re-run the SAME restore. Phase 4 clears before it loads, so a partially
  loaded graph is not an obstacle to re-running it.
- If the ARTIFACT is the problem, put the target's own graph back from the pre-restore artifact,
  as in step 2.

CASE B -- THE COMMIT LEG FAILED PART WAY. The target is a mixture of two points in time, which is
the one state not to leave it in. Stop whatever is running against it first -- a backend holding a
store open is the usual reason a rename failed -- then do step 2.

CASE C -- ONLY THE CLEANUP IS OUTSTANDING. The restore itself landed. There are two ways you get
here, and only one of them leaves a marker:

- The run exited 0 and printed `[restore] WARNING: ... could not be removed`. Phase 6 could not
  delete `vault.previous/`, said so, and finished anyway. There is NO marker -- it is deleted even
  on this path, so that a directory nobody needs cannot refuse your next restore.
- The run was killed between phase 5 and phase 6. Then the marker is still there, with
  `vault_committed: true` and `vault_previous` non-null.

Either way the target is correct. Check `vault/` holds what you expect, then clear what is left:

    rm -rf <target-root>/vault.previous
    rm -f <target-root>/restore.in-progress.json

#### Step 2: restore the pre-restore artifact back

The pre-restore artifact is an ordinary backup artifact, so putting the target back is this same
command pointed at it. READ `pre_restore_artifact` OUT OF THE MARKER FIRST: the next run
overwrites the marker with its own.

    cat <target-root>/restore.in-progress.json        # copy pre_restore_artifact somewhere
    rm <target-root>/restore.in-progress.json         # preflight refuses until this is gone
    python -m scripts.backup.restore \
      --artifact <THE pre_restore_artifact PATH FROM THE MARKER> \
      --target-root ./dev-state \
      --target-graph-uri bolt://localhost:7690 \
      --confirm-target <THE RESOLVED PATH, AS IN 4.3>

Four things to know before running it:

- IT TAKES ANOTHER PRE-RESTORE BACKUP FIRST, of the damaged target. That is not waste: it is the
  only copy of the half-restored state, and it is what you would want if this recovery is also
  wrong. All four gates run again; none of them is skipped because the situation is an emergency.
- LEFTOVER `.incoming` FILES DO NOT BLOCK IT. Phase 3 overwrites a staged store and removes a
  staged vault tree before copying, so the failed run's staging is reused ground rather than an
  obstacle.
- A LEFTOVER `vault.previous/` IS NOT REMOVED BY THE RE-RUN when `vault/` is missing, because the
  removal sits behind the "target has a vault to rename aside" branch. If the failed run died
  between the two vault renames, rename the tree you want back into place by hand BEFORE
  re-running, and delete the other afterwards. Both trees are intact; only the names are wrong.

  CHECK ITS DATE BEFORE YOU REASON ABOUT IT. A `vault.previous/` you find is not necessarily from
  the run that just failed: phase 6 can decline to remove one (4.5), and a re-run with `vault/`
  present renames the live tree over that name only after clearing it. An operator who assumes the
  directory belongs to this run will reconstruct the wrong failure. The marker's `started_utc`, and
  the directory's own mtime, are what tell you which run left it.
- IT PUTS BACK ONLY WHAT THE DUMP LEG CAPTURES. `data/vector_store/` is excluded from every
  artifact (section 2), and a store the manifest records as absent is left as the target's own.

If the graph is the ONLY thing you need back and the stores and vault are intact, the graph-only
path in 4.11 is faster. It also has none of the four gates, so choose it deliberately rather than
because it is shorter.

### 4.8 Verify

Three graph numbers, one vector, and the vault count. Run all five checks; a restore that gets
the counts right and the vectors wrong is the failure mode that hides behind every similarity
threshold in the codebase.

(a) What the artifact says it holds:

    python -c "import json;m=json.load(open('/mnt/backup/mist/20260917T030000Z/manifest.json'));print(m['graph'])"

(b) What LIVE holds now, READ-ONLY. `cmd_graph_stats` calls ten counting helpers
(`scripts/mist_admin.py:514-527`) and none of them writes: `grep -n "execute_write"
backend/knowledge/admin.py` returns no hit between `:624-742` or `:1228-1300`, the two ranges those
helpers occupy.

    docker compose exec mist-backend python scripts/mist_admin.py graph-stats

Live counts drift after the capture, so expect them to be equal only if nothing has been ingested
since. What must match exactly is (a) against (c).

(c) and (d) What the restored dev graph holds, and whether one embedding came back bit-for-bit.
From the host shell, in the same environment that ran the restore:

    python - <<'PY'
    import json
    from dataclasses import replace
    from backend.knowledge.config import get_config
    from backend.knowledge.eval_isolation import assert_neo4j_dev_isolated
    from backend.knowledge.graph_artifact import load_artifact
    from backend.knowledge.storage.neo4j_connection import Neo4jConnection

    ARTIFACT = "/mnt/backup/mist/20260917T030000Z/graph.json"
    URI = "bolt://localhost:7690"

    with open(ARTIFACT, encoding="utf-8") as handle:
        artifact = load_artifact(json.load(handle))
    assert_neo4j_dev_isolated(URI)
    connection = Neo4jConnection(replace(get_config().neo4j, uri=URI))
    connection.connect()
    try:
        nodes = connection.execute_query("MATCH (n) RETURN count(n) AS n")[0]["n"]
        rels = connection.execute_query("MATCH ()-[r]->() RETURN count(r) AS r")[0]["r"]
        sample = next(n for n in artifact["nodes"] if "embedding" in n["properties"])
        rows = connection.execute_query(
            "MATCH (n {id: $id}) RETURN n.embedding AS embedding", {"id": sample["id"]}
        )
    finally:
        connection.disconnect()

    print("nodes        artifact", len(artifact["nodes"]), "restored", nodes)
    print("relationships artifact", len(artifact["relationships"]), "restored", rels)
    print("sampled node", sample["id"])
    print("embedding EXACTLY equal:",
          list(rows[0]["embedding"]) == list(sample["properties"]["embedding"]))
    PY

THE EMBEDDING COMPARISON IS EXACT FLOAT-LIST EQUALITY. It is `==` on two `list[float]`, element by
element. It is NOT cosine similarity and must never be relaxed into one: a vector that is merely
close has been silently rewritten, and every similarity check in this codebase would still pass on
it.

(e) THE VAULT, COUNTED THE WAY THE MANIFEST COUNTS IT. The restore's own output line already
prints both numbers -- `N vault corpus file(s) (M including .git plumbing)` -- and the artifact's
manifest carries both keys. Compare corpus against corpus:

    python -c "import json;m=json.load(open('/mnt/backup/mist/20260917T030000Z/manifest.json'));print(m['vault'])"

`corpus_file_count` is the number to check against what you expect of your notes. Do NOT read
`file_count` as a corpus size: on the live tree it is roughly nine parts git plumbing to one part
note. On an artifact taken before that key existed it is absent altogether -- see section 2.

The rehearsal has passed when: nodes and relationships match between (a) and (c), the embedding
line prints `True`, the restored `corpus_file_count` equals the artifact's, and `ls ./dev-state`
shows the three stores and a `vault/` tree from the artifact, with no `restore.in-progress.json`,
no `*.incoming` and no `vault.previous/` left behind.

### 4.9 What the unit tier cannot prove, and why this rehearsal is the gate

Two of the fixes this section describes are proved only against SIMULATED faults. In neither case
is the simulation the defect, and the gap is worth carrying into the rehearsal rather than
discovering after it.

THE NEO4J SCHEMA-REJECTION RULE IS MODELLED, NOT OBSERVED. `replace_graph_schema` drops the
target's schema and re-applies the artifact's rather than using `CREATE ... IF NOT EXISTS`, and
the stated reason is the Cypher manual's description of `IF NOT EXISTS`: it creates nothing and
throws nothing when an object of that name, or an equivalent constraint under another name,
already exists -- so it is silent in the same-name-DIFFERENT-definition case, which is the one
case that must not pass unnoticed (`backend/knowledge/admin.py`, `replace_graph_schema`). THAT
SERVER BEHAVIOUR HAS NEVER BEEN OBSERVED AGAINST A REAL NEO4J FROM THIS BRANCH. The unit tier
drives a fake connection, so it proves the code takes the drop-and-recreate path; it does not
prove what a real server would have done with the alternative. Restoring into a dev graph that
ALREADY carries schema is what would test it, and 4.1 leaves `mist-neo4j-dev` in that state if it
has been hydrated before.

THE READ-ONLY VAULT HANDLER IS PROVED ON THE WRONG OPERATING SYSTEM. The defect it exists for
(MIS-157) is a Windows `WinError 5` raised by `shutil.rmtree` on the read-only loose objects git
writes under `.git/objects`. On Linux a read-only FILE is removable, so THE CONTAINER TIER CANNOT
REPRODUCE THE DEFECT AT ALL. What the Linux test does instead is make the CONTAINING DIRECTORY
read-only (`chmod 0o555`), which raises a genuine `PermissionError` through the same handler. That
proves the handler is wired into `remove_tree` and that the retry succeeds once the attribute is
cleared. It proves nothing about the Windows read-only-file case.

So the host rehearsal is the only evidence either of these will get. Run 4.4 through 4.8 TWICE
against a vault that has a real `.git`: the first run finds no `vault/` in a fresh `./dev-state`,
so phase 6 has nothing to remove, and it is the SECOND run that renames a vault full of read-only
git objects to `vault.previous/` and then removes it.

### 4.10 Clean up

The dev graph now holds a copy of live. That is fine -- it is the dev instance -- but say so out
loud to anyone using it for hydration, and either re-hydrate it or drop the volume:

    docker compose -f docker-compose.yml -f docker-compose.override.yml \
      -f docker-compose.dev-hydration.yml --profile dev stop mist-neo4j-dev

Leave `./dev-state/MIST_RESTORE_TARGET` in place; the marker is not consumed, and a target that
stays marked is a rehearsal you can repeat.

### 4.11 The OTHER restore command, and why this is not it

`python scripts/mist_admin.py graph-restore ARTIFACT [--confirm]` also exists, and it also
detach-deletes the graph it is pointed at. Know about it so you do not reach for it by accident at
3am.

What it is: the GRAPH-ONLY developer path. It restores `graph.json` and nothing else -- no stores,
no vault -- and its only guard is `assert_neo4j_dev_isolated` on the target URI
(`scripts/mist_admin.py:474-479`). It has NO handshake marker, NO typed confirmation token and NO
pre-restore backup: `--confirm` is a bare flag, so a repeated shell command restores again with no
further question asked.

Use it when you are iterating on a dev graph and want one leg back quickly. Do NOT use it for
disaster recovery. The four-gate `python -m scripts.backup.restore` is the command for that, and
the gates are the difference: they are what makes a restore into the wrong target survivable.

---

## 5. Retention

    export MIST_BACKUP_ROOT=/mnt/backup/mist
    python -m scripts.backup.prune --retain 7            # prints the plan, deletes nothing
    python -m scripts.backup.prune --retain 7 --confirm  # applies it

Three rules, and the third is the one that matters:

1. Age comes from each artifact's own `manifest.json` `created_at`, never from directory mtime and
   never from the directory name. An mtime records when a directory was last DISTURBED -- by a
   virus scanner, a copy to a new disk, an rsync -- which says nothing about what it holds.
2. It never prunes below 1. `--retain 0` is raised to 1 and the output says so.
3. It NEVER deletes a directory without a MIST.AI backup manifest. If `MIST_BACKUP_ROOT` is
   pointed at your Documents folder, prune deletes NOTHING and lists every directory it skipped.
   Unfamiliar names in the SKIP list mean the root is wrong; fix the variable, do not delete
   anything by hand to "help" it.

`.partial` directories are protected by rule 3 for free: a dump renames into place only after
writing the manifest, so a partial has no manifest to be dated by.

WHAT TO DO ABOUT ACCUMULATING PARTIALS, since prune will never reap them. Each one is a dump that
DIED, so it is evidence rather than litter. Read the dump's exit code and log first: several
partials in a row mean the dump leg is failing and the backup you believe you have is not being
taken. Once a COMPLETE artifact newer than the partial exists, delete it by hand:

    rm -rf /mnt/backup/mist/<label>.partial

Deleting it also frees that `--label` for reuse, which the dump refuses while the directory exists.

---

## 6. Scheduling the dump

NOTHING IS ARMED. This goal ships no cron file, no systemd unit, no scheduled task and no timer,
and none should be added until the rehearsal in section 4 has passed once on the host.

The dump leg is schedulable BY CONSTRUCTION: it is read-only against live state, takes no input,
prompts for nothing, and returns deterministic exit codes (0 written, 2 destination refused and
nothing written, 1 failed with a `.partial` directory left behind). The invocation a schedule would
use is exactly the one in section 1:

    MIST_BACKUP_ROOT=/mnt/backup/mist python -m scripts.backup.dump

followed by

    MIST_BACKUP_ROOT=/mnt/backup/mist python -m scripts.backup.prune --retain 7 --confirm

A schedule wired up before a restore has been rehearsed produces a directory that fills up with
files nobody has proved are usable. That is the condition this package exists to end, so the order
is not negotiable: rehearse first, arm second.

The restore leg is NOT schedulable and must never be automated. It requires a typed confirmation
token equal to the resolved target path and a handshake marker placed by hand, and both of those
requirements exist precisely to keep a machine from running it.
