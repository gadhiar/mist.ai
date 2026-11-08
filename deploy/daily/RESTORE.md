# Restore drill and promotion

A backup counts only once it has been restored and compared with what it captured. The drill
does that without touching the running instance: it restores one artifact into a fresh Docker
volume and a standby directory, then compares counts. Promotion makes a verified drill the
instance, which is also how the instance is recovered from a backup.

Every step is an `instance.py` command. Never run `restore-drill.yml` by hand: the commands
check, before any call that uses the drill volume, that it is the labelled volume `drill-start`
created, that it is not the instance's graph volume, and that no container outside the drill
project mounts it. Running the file directly skips all three.

In the commands below, `<root>` is the instance root, `<drill root>` the drill's directory and
`<label>` the backup's directory name in `MIST_BACKUP_HOST_PATH`. Each command exits 0 done, 2
refused with nothing changed, or 1 failed part-way with the state named. No command of this CLI
runs a command that could delete a volume: each one passes a guard first that refuses `-v`,
`--volumes`, `prune`, `down` and every `volume` subcommand but `inspect` and `create`, and exits
1 naming the command.

Run one drill per instance at a time. Every drill of an instance runs in the one compose project
`mist-<instance>-drill`, and its Neo4j container keeps the volume it was created on: a second
drill started beside it would restore into, and verify, the first drill's graph. So:

- `drill-start` refuses while an earlier drill's containers remain, or while it cannot list
  them. Run `drill-end` for the earlier drill first.
- `drill-restore` refuses while another drill's Neo4j container remains: when the drill
  project's `mist-neo4j-dev` mounts any volume but this drill's at `/data`, or its mounts cannot
  be read. Run `drill-end`, which removes every container of the drill project and keeps every
  volume, then restore again.
- `drill-restore` refuses a drill that has already verified: once `ops/` holds a `match` for
  this drill newer than its `drill.json`. Promote it, or end it and start a new drill.
- `drill-verify` reads the `/data` mount of the drill project's `mist-neo4j-dev` after verify
  runs. Unless it is this drill's volume, it writes no `ops/` record and exits 1, naming the
  volume it found.

---

## Before you start

- Pick a drill root outside the instance root, outside every registered live root and outside
  every git work tree, on the same drive as the instance root. `drill-start` refuses the first
  three; `promote` refuses another drive, because it moves the standby in by rename. The
  directory must be empty or not exist yet.
- Pick the artifact: a directory under `MIST_BACKUP_HOST_PATH` that holds a `manifest.json`.
  The scheduler names each backup by its UTC start time, so the latest name is the newest.
- The drill adds a Neo4j (up to `MIST_DAILY_NEO4J_MEM_LIMIT`, 3g by default) and a restore
  container (up to `MIST_DAILY_BACKUP_MEM_LIMIT`, 4g) to what already runs. On a host short of
  memory, stop `mist-llm` first by turning gaming mode on, and turn it off when the drill ends:

```
python <root>/bin/instance.py gaming-mode on --root <root>
```

---

## The drill

1. Start it. This creates the volume `mist-<instance>-drill-<UTC stamp>`, labelled
   `org.mist.role=drill` and `org.mist.instance=<instance>`, and writes
   `<drill root>/drill.json`:

```
python <root>/bin/instance.py drill-start --root <root> --drill-root <drill root> --artifact <label>
```

2. Place the restore handshake. Create an empty file
   `<drill root>/standby/state/MIST_RESTORE_TARGET` by hand. No command creates it, and none
   ever will: the handshake is a person saying "this directory may be overwritten".

3. Run the restore once without a token. Restore refuses (exit 2) and prints the exact
   `--confirm-target` value to pass, which is its view of the standby path inside the container:

```
python <root>/bin/instance.py drill-restore --root <root> --drill-root <drill root>
```

4. Run it again with the printed token. Restore takes its pre-restore backup into
   `<drill root>/pre-restore/`, then restores the graph into the drill volume and the stores and
   vault into `<drill root>/standby/`:

```
python <root>/bin/instance.py drill-restore --root <root> --drill-root <drill root> --confirm-target <token>
```

5. Verify. This compares node and relationship counts, every node's embedding, each store's
   per-table row counts and the vault's file count with the artifact, prints the counts, and
   writes them to `<root>/ops/drill-<UTC stamp>.json`. Exit 0 is a match, 1 a mismatch:

```
python <root>/bin/instance.py drill-verify --root <root> --drill-root <drill root>
```

6. End it. This removes the drill's containers and keeps the volume and the drill root:

```
python <root>/bin/instance.py drill-end --root <root> --drill-root <drill root>
```

If you turned gaming mode on, turn it off:

```
python <root>/bin/instance.py gaming-mode off --root <root>
```

Scheduled backups are armed only after a drill verifies `match`: rehearse first, arm second.
Arm them once, after the first drill passes:

```
python <root>/bin/instance.py arm-backups --root <root>
```

A drill you will not promote leaves its volume and drill root behind. Both are yours to delete
once you no longer need them; no command of this CLI deletes a volume.

A mismatch means this artifact does not restore to what it captured. Keep the drill root and the
`ops/drill-*.json` record, do not promote, and do not arm backups until a drill matches.

---

## Promotion

Promotion replaces the instance's graph, state and vault with a verified drill's. Run steps 1 to
5 of the drill, so that the newest `ops/drill-*.json` record for this drill root is a `match`
newer than its `drill.json`. Skip `drill-end`; `promote` removes the drill's containers itself.

```
python <root>/bin/instance.py promote --root <root> --drill-root <drill root>
```

`promote` refuses with nothing changed when there is no such `match`, when the drill root is on
another drive than the instance root, when either standby directory is missing, or when
`daily.env` does not hold exactly one `MIST_DAILY_NEO4J_VOLUME=` line. Otherwise, in this order:

1. It removes the drill project's containers (`rm -s -f`, never `-v`).
2. It lists the containers that still mount the drill volume, and refuses (exit 2) if there are
   any: the instance has not been stopped and nothing has moved.
3. It stops the instance (`stop`).
4. It removes the instance's containers (`--profile backup rm -f`, never `-v`), so `up` creates
   a Neo4j container on the promoted volume instead of restarting the old one on the old volume.
   The volumes stay.
5. It moves `<root>/state` and `<root>/mist-memory` into `<root>/retired/<UTC stamp>/`.
6. It moves `<drill root>/standby/state` and `<drill root>/standby/mist-memory` into their
   places.
7. It deletes the promoted `state/MIST_RESTORE_TARGET`, which would otherwise leave the live
   state a valid restore target.
8. It writes the instance marker into the new `state/` and `mist-memory/`.
9. It rewrites the `MIST_DAILY_NEO4J_VOLUME=` line of `daily.env`, and only that line, to the
   drill volume.
10. It runs `up`.
11. It finds the instance's `mist-neo4j` container by its compose labels and reads its `/data`
    mount. Unless that is the drill volume, it exits 1 naming the volume it found.

A failure in step 3 or 4 exits 1 with nothing moved, and `up` restarts the instance. A failure
after step 4 exits 1 and names every step already done. Then read `/health`:

```
python <root>/bin/instance.py status --root <root>
```

`status` prints `/health`'s status and each check's reason; `deploy/daily/README.md`, Reading
/health, says what each reason means.

Nothing is deleted. The previous state and vault stay in `<root>/retired/<UTC stamp>/`, and the
previous graph volume stays under its old name. To go back, stop the instance, move the two
directories back from `retired/`, and set `MIST_DAILY_NEO4J_VOLUME` back to the old volume name
in `daily.env` before running `up`.
