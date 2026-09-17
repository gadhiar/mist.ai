# MIST.AI backup and restore runbook

Read this at 3am. It assumes you did not write any of it and that you are not calm.

Three commands exist:

    python -m scripts.backup.dump       take a backup      (read-only, safe to run any time)
    python -m scripts.backup.restore    put one back       (DESTRUCTIVE, four gates, see below)
    python -m scripts.backup.prune      delete old ones    (deletes nothing without --confirm)

Exit codes are the same shape everywhere: `0` it worked, `2` it refused and changed nothing,
`1` it failed part way and the message says what state that leaves.

WHAT HAS AND HAS NOT BEEN EXERCISED. Every code path named here is covered by
`tests/unit/backup/` (165 tests: `python -m pytest tests/unit/backup -q`), including a full
synthetic round trip that restores a captured graph and compares embeddings for exact equality.
None of it has been run against the live stack or the dev-hydration stack on the host from this
branch. The rehearsal below is therefore a rehearsal, not a replay of something already done.

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
| `vault/`                     | a copy of the whole `mist-memory/` tree                          |
| `manifest.json`              | layout version, `created_at`, per-file sha256, row counts, graph counts, git HEAD, producer stamps |

The three stores are captured BY NAME, never by glob. The live `./data` also holds
`event_store.pre-r1.4-backup-2026-07-31.db` and `event_store.pre-reset-backup-2026-06-09.db`; a
glob would sweep both in, and a hurried restore could then load a months-old event store.

Graph embeddings travel as exact `list[float]`, not as strings. A restore returns the same floats,
which is why the rehearsal in section 4 checks equality rather than similarity.

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

The refusal names the RESOLVED path, for example `/home/raj/mist.ai/dev-state`. Copy it.

The token is compared against the resolved path and never against the string you passed. Typing
`--target-root ./dev-state --confirm-target ./dev-state` is REFUSED even though both strings match
each other: retyping a relative path proves nothing about where it lands, and where it lands is the
only question that matters. Whitespace around the token is stripped; nothing else is normalised.

### 4.4 Run it

    python -m scripts.backup.restore \
      --artifact /mnt/backup/mist/20260917T030000Z \
      --target-root ./dev-state \
      --target-graph-uri bolt://localhost:7690 \
      --confirm-target /home/raj/mist.ai/dev-state

Four gates run, every time, and no flag disables any of them:

1. `--target-root` is required; there is no default.
2. The typed token must equal the resolved target path.
3. The target must carry `MIST_RESTORE_TARGET`, on top of `assert_isolated_root` for the root and
   `assert_neo4j_dev_isolated` for the graph URI.
4. A pre-restore backup of the TARGET is taken first, through the same dump leg. If it fails, the
   restore does not proceed.

The first line of successful output is the pre-restore artifact path. That directory is your way
back if you have just restored into the wrong place. Write it down before reading the rest.

Then, in order: the artifact's file digests are re-checked against its manifest, the stores are
replaced, the vault tree is replaced wholesale (not merged), and the graph is loaded last -- the
graph leg detach-deletes the target first, so it goes last, and a failure earlier leaves the target
with its own graph rather than none.

### 4.5 Verify

Three numbers and one vector. Run all four checks; a restore that gets the counts right and the
vectors wrong is the failure mode that hides behind every similarity threshold in the codebase.

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

The rehearsal has passed when: nodes and relationships match between (a) and (c), the embedding
line prints `True`, and `ls ./dev-state` shows the three stores and a `vault/` tree from the
artifact.

### 4.6 Clean up

The dev graph now holds a copy of live. That is fine -- it is the dev instance -- but say so out
loud to anyone using it for hydration, and either re-hydrate it or drop the volume:

    docker compose -f docker-compose.yml -f docker-compose.override.yml \
      -f docker-compose.dev-hydration.yml --profile dev stop mist-neo4j-dev

Leave `./dev-state/MIST_RESTORE_TARGET` in place; the marker is not consumed, and a target that
stays marked is a rehearsal you can repeat.

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
