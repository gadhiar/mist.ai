# MIST.AI daily instance

`deploy/daily/compose.yml` is the compose project `mist-daily`: the MIST.AI instance used every
day, run from a pinned release image, with its state in an instance root outside every git work
tree. Development never restarts it, and no file in a working tree reaches it.

The services keep the workstation stack's names (`mist-backend`, `mist-neo4j`, `mist-llm`) and
its host ports 8001, 7687 and 7474, so the live-endpoint denylists in
`backend/knowledge/eval_isolation.py` protect this instance with no change. A fourth service,
`mist-backup`, runs the backup scheduler under profile `backup`, so nothing starts it until
backups are armed. Two more, `mist-extraction-llm` and `mist-extraction`, run night extraction
on this GPU under profile `extraction`, which only `extraction apply` starts (see Night
extraction).

The compose contract is pinned by `tests/unit/deploy/test_daily_compose.py`, which parses the
YAML; the image layout it relies on is pinned by `tests/unit/deploy/test_backend_dockerfile.py`.

---

## Instance root layout

The instance root is a host directory outside every git work tree. `MIST_DAILY_ROOT` names it.

```
<root>/
  .mist-live-instance    marker: {"v": 1, "instance": "daily", "created_at": ...}
  daily.env              secrets and host paths, from daily.env.example; never committed
  state/                 mounted at /app/data (event store, extraction cache, sidecar,
                         vector store, audio); holds a copy of the marker
  mist-memory/           mounted at /app/mist-memory (the vault, with its .git); holds a copy
                         of the marker
  logs/                  mounted at /app/logs; logs/neo4j/ is mounted at mist-neo4j's /logs;
                         the host CLI writes the extraction logs here
  ops/                   mounted at /app/ops: read-only in mist-backend, read-write in
                         mist-backup; release records, backup status, gaming-mode marker,
                         extraction mode and paused marker
  models/                the llama-server weights, mounted read-only at /models
  release/<tag>/         deploy/ and scripts/daily/ extracted from the release tag
  retired/<UTC stamp>/   state/ and mist-memory/ replaced by a promotion, kept until deleted
  bin/instance.py        the active release's CLI, run by the sign-in task
  secrets/               mounted read-write at /app/secrets in mist-backend only: the Home
                         Assistant and Spotify tokens; in no backup (see Capabilities)
```

The graph does not live in the root: it is the external Docker volume named by
`MIST_DAILY_NEO4J_VOLUME`, so `docker compose down -v` cannot delete it, and promoting a restored
volume changes one line of `daily.env`. The model and embedding caches are the project-scoped
volumes `mist-hf-cache` and `mist-torch-cache`.

Every bind mount starts at `MIST_DAILY_ROOT`, `MODELS_DIR` or `MIST_BACKUP_HOST_PATH`, so none is
relative to a repository. Every published port binds 127.0.0.1 only: the WebSocket has no
authentication, and a client that sends no `Origin` header passes the allowlist. `mist-llm`
publishes no port at all; the backend reaches it inside the project, and a dev stack reaches it
over the external network `MIST_LLM_SHARE_NETWORK`, which `mist-backend` and `mist-neo4j` do
not join.

---

## Variables

`daily.env` is passed to compose with `--env-file`, which resolves every `${VAR}` in
`compose.yml`, and compose also gives it to `mist-backend` as its `env_file`. The
`environment` literals in `compose.yml` win over any line in it: the store paths, the capability
store and token paths (see Capabilities), `MIST_OPS_DIR=/app/ops`, `MIST_SESSION_ORIGIN=real`,
`TTS_ENABLED=false` (the code defaults TTS on) and `EMBEDDING_DEVICE=cpu` (the embedding model
must hold no GPU memory). A required variable that is unset makes compose refuse the whole file
before anything starts.

`daily.env.example` lists every variable with a placeholder or its default.

| Variable | Read by | Required | Default |
|---|---|---|---|
| `MIST_RELEASE_TAG` | compose: the `mist-backend:<tag>` image | yes | |
| `MIST_DAILY_ROOT` | compose: every bind mount under the root, and the env file | yes | |
| `MIST_INSTANCE_NAME` | the instance CLI | written by `init` | |
| `NEO4J_USERNAME` | compose: Neo4j auth and the backup; the backend through `env_file` | no | `neo4j` |
| `NEO4J_PASSWORD` | compose: Neo4j auth and the backup; the backend through `env_file` | yes | |
| `MIST_DAILY_NEO4J_VOLUME` | compose: name of the external graph volume | yes | |
| `MIST_DAILY_NEO4J_HEAP` | compose: Neo4j initial and maximum heap | no | `1g` |
| `MIST_DAILY_NEO4J_PAGECACHE` | compose: Neo4j page cache | no | `512m` |
| `MODELS_DIR` | compose: host directory mounted at `/models` | yes | |
| `LLM_MODEL_FILE` | compose: the model llama-server loads | no | as `docker-compose.yml` |
| `LLM_CTX_SIZE` | compose: llama-server context size | no | `32768` |
| `LLM_CACHE_RAM_MIB` | compose: llama-server prompt-cache cap | no | `2048` |
| `MODEL` | the backend through `env_file` | no | code default |
| `MIST_LLM_SHARE_NETWORK` | compose: external network `mist-llm` joins | no | `mist-llm-share` |
| `MIST_ALLOWED_ORIGINS` | compose: browser origins the backend admits | no | the three Tauri origins |
| `MIST_EXTRACTION_INFERENCE` | none: compose sets it on `mist-backend`, so a `daily.env` line is ignored | no | `service` |
| `MIST_EXTRACTION_SERVICE_URL` | none: compose sets it on `mist-backend`, so a `daily.env` line is ignored | no | `http://mist-extraction:8090` |
| `MIST_HEALTH_BACKLOG_MAX_AGE_S` | compose: the backend's overdue age for the extraction backlog | no | `86400` |
| `MIST_TIMEZONE` | the backend through `env_file`; alarms, timers and the clock need it | no | UTC; alarms, timers and clock unavailable |
| `MIST_CAPABILITIES_ENABLED` | the backend through `env_file` | no | `false` |
| `MIST_CAPABILITIES` | the backend through `env_file` | no | all five |
| `MIST_HA_MCP_URL` | the backend through `env_file`: lights | no | none, and lights is not built |
| `MIST_LIGHTS_DEFAULT_TARGET` | the backend through `env_file`: lights | no | none |
| `MIST_SPOTIFY_CLIENT_ID` | the backend through `env_file`: music | no | none, and music is not built |
| `MIST_SPOTIFY_DEVICE` | the backend through `env_file`: music | no | none |
| `MIST_RING_FALLBACK_AFTER_S` | the backend through `env_file`: alarms | no | `30` |
| `MIST_EXTRACTION_WINDOW` | the instance CLI: the night window, `HH:MM-HH:MM` in host local time | no | `23:59-10:00` |
| `MIST_DAILY_EXTRACTION_MODEL_FILE` | compose: the model the extraction llama-server loads | no | `ggml-org/gpt-oss-20b-MXFP4.gguf` |
| `MIST_DAILY_EXTRACTION_CTX_SIZE` | compose: extraction llama-server context size | no | `65536` |
| `MIST_DAILY_EXTRACTION_NCMOE` | compose: MoE layers whose experts stay on the CPU | no | `21` |
| `MIST_DAILY_EXTRACTION_CACHE_RAM_MIB` | compose: extraction llama-server prompt-cache cap | no | `2048` |
| `EXTRACTION_MODEL_HASH` | compose: the model identity `mist-extraction` stamps | for extraction | none, and extraction stays paused |
| `EXTRACTION_LLM_DEVICE` | compose: the GPU name in `mist-extraction`'s serving identity | for extraction | none, and extraction stays paused |
| `MIST_BACKUP_HOST_PATH` | compose: host directory mounted at `/backup` | yes | |
| `MIST_BACKUP_INTERVAL_HOURS` | compose: the backup scheduler | no | `24` |
| `MIST_BACKUP_RETAIN` | compose: the backup scheduler | no | `7` |
| `MIST_BACKUP_CHECK_MINUTES` | compose: the backup scheduler | no | `10` |
| `MIST_BACKUP_RETRY_MINUTES` | compose: the backup scheduler | no | `60` |
| `MIST_BACKUP_MIN_NODE_RATIO` | compose: the backup scheduler | no | `0.5` |
| `MIST_DAILY_BACKEND_MEM_LIMIT` | compose: `mem_limit` of `mist-backend` | no | `8g` |
| `MIST_DAILY_NEO4J_MEM_LIMIT` | compose: `mem_limit` of `mist-neo4j` | no | `3g` |
| `MIST_DAILY_LLM_MEM_LIMIT` | compose: `mem_limit` of `mist-llm` | no | `9g` |
| `MIST_DAILY_BACKUP_MEM_LIMIT` | compose: `mem_limit` of `mist-backup` | no | `4g` |
| `MIST_DAILY_EXTRACTION_LLM_MEM_LIMIT` | compose: `mem_limit` of `mist-extraction-llm` | no | `14g` |
| `MIST_DAILY_EXTRACTION_MEM_LIMIT` | compose: `mem_limit` of `mist-extraction` | no | `1g` |
| `MIST_DAILY_WS_PORT` | compose: host port of the WebSocket; rehearsal only | no | `8001` |
| `MIST_DAILY_BOLT_PORT` | compose: host port of Neo4j Bolt; rehearsal only | no | `7687` |
| `MIST_DAILY_NEO4J_HTTP_PORT` | compose: host port of the Neo4j browser; rehearsal only | no | `7474` |

The three port variables exist only so the same file can run as a rehearsal project on other
ports. The instance CLI refuses any non-canonical port for an instance named `daily`, and refuses
for every instance a port on the eval, dev or staging allowlists (7688, 7689, 7690 and 8002).

---

## Release lifecycle

A release is an annotated tag `daily-YYYY.MM.DD[.N]` on main. The instance CLI,
`scripts/daily/instance.py`, drives each step:

1. `release-build <tag>` pipes `git archive <tag>` to `docker build`, making
   `mist-backend:<tag>` with the tag and commit baked in as `MIST_RELEASE_TAG` and
   `MIST_RELEASE_COMMIT` and as the OCI labels `org.opencontainers.image.version` and
   `org.opencontainers.image.revision`. From the same archive it builds `mist-extraction:<tag>`
   with `docker/extraction/Dockerfile`, labelled with the commit as its revision. It records both
   image IDs in `ops/release-<tag>.json` (`image_id` and `extraction_image_id`) and extracts the
   archive's `deploy/` and `scripts/daily/` into `release/<tag>/`.
2. `switch <tag>` takes one backup when backups are configured and the instance runs, rewrites
   the `MIST_RELEASE_TAG` line of `daily.env`, and brings the project up from
   `release/<tag>/deploy/daily/compose.yml`. A change on main therefore reaches the instance
   only through `switch`.
3. `switch <previous tag>` is the rollback. It restores the code, not a data migration the newer
   release made; the pre-switch backup is the way back from that. To roll back to a release
   without night extraction, whose `ops/release-<tag>.json` records no `extraction_image_id`,
   go in this order: `extraction off`, then `unregister-signin-task.ps1`, then
   `switch <previous tag>`, then that release's `register-signin-task.ps1` from
   `release/<previous tag>/deploy/daily/windows/`, without `-NightExtraction`. Unregistering
   first ends the "MIST extraction window" task, whose ticks would otherwise fail every 5
   minutes against a release with no night extraction, and `extraction off` leaves the mode
   `off` if you later switch forward again.

No working tree is involved at runtime: there are no source bind mounts, and
`pull_policy: never` means compose neither builds nor pulls (a pull of `mist-backend:<tag>` could
fetch an unrelated public image of that name). `up` and `switch` refuse when the tag's image ID
differs from the recorded one, and `status` reports the running image's tag, ID and revision
label, and any container whose compose config file is not the release's.

---

## Commands

Every command is `python <root>/bin/instance.py <command> --root <root>`, or the same file in
the repository's `scripts/daily/`. The project is `mist-<instance>`, with the instance name
taken from the root's marker, and every compose call passes
`-f <root>/release/<MIST_RELEASE_TAG>/deploy/daily/compose.yml --env-file <root>/daily.env`.

Exit codes: 0 done; 2 refused, nothing changed, except that `extraction apply` refuses a
missing identity variable after it has stopped extraction and written the `unconfigured`
paused marker; 1 failed part-way, and the message names the state left. `backup-now` returns
the backup attempt's own exit code, `accept-backup-baseline` the scheduler's, and
`drill-restore` and `drill-verify` those of restore and verify.

`switch`, `up`, `arm-backups`, `backup-now` and `drill-start` first check the configuration,
and refuse with exit 2 before anything starts:

- an instance named `daily` whose `daily.env` sets `MIST_DAILY_WS_PORT`,
  `MIST_DAILY_BOLT_PORT` or `MIST_DAILY_NEO4J_HTTP_PORT` to anything but 8001, 7687 and 7474;
- any instance with one of those ports in 7688, 7689, 7690 or 8002;
- a `MIST_BACKUP_HOST_PATH` or drill root inside the instance root, inside a registered live
  root or a directory holding `.mist-live-instance`, or inside a git work tree;
- a `mist-backend:<tag>` image whose ID differs from `ops/release-<tag>.json`. The fix is
  `release-build <tag>`, which rebuilds the image from the tag and records it again. This is
  the one check that asks Docker, with `docker image inspect`; it changes nothing.

| Command | What it does |
|---|---|
| `switch TAG [--skip-backup]` | Refuses a tag with no `release/<tag>/`, a tag with no `mist-backend:<tag>` image, and a `rehearsal-` tag on the instance named `daily`. When `MIST_BACKUP_HOST_PATH` is set and `mist-backend` is running, it runs `backup-now` on the current release first, and stops with that exit code if the backup fails. It then stops and removes the extraction containers, found by their project and service labels with plain `docker`, so the target release need not define them, keeping each one's last log lines in `logs/extraction-last-<service>.log`. Then it rewrites only the `MIST_RELEASE_TAG=` line of `daily.env`, copies every `*.py` file of `release/<tag>/scripts/daily/` into `bin/`, and runs `up`. |
| `up [--wait-minutes N]` | Waits up to N minutes (default 10) for `docker info`, creates the share network if it is missing, and starts `mist-neo4j`, `mist-llm` and `mist-backend` in one call, leaving `mist-llm` out while `ops/gpu-yielded` exists. Only when `ops/backups-armed` exists does it start `mist-backup`, in a second call, so a failure there (exit 1) leaves the first three running. Then it runs `extraction apply`; one that does not converge is reported and exits 1 with the chat services left running, and a release that records no `extraction_image_id` gets a note instead, with exit 0. Last, it names every container of the project whose compose config-file label is not the release's compose file, and exits 1. |
| `down` | `docker compose ... --profile backup --profile extraction stop`. It stops every container and removes nothing: no command of this CLI runs `down -v`, `rm -v` or `volume rm`. Every command passes `volume_safety.py`, which `instance.py` loads from beside itself, before it runs; the guard refuses `-v`, `--volumes`, `prune`, `down` and every `volume` subcommand but `inspect` and `create`, and the CLI exits 1 naming the command. |
| `status` | Prints the running backend's image tag, its image ID with `match` or `MISMATCH` against the release record, and its `org.opencontainers.image.revision` label; each container from another compose file (then exit 1), and it exits 1 too when the project's containers cannot be listed, since then that check could not run; the `gaming-mode status` line; the `extraction status` line, or `extraction: none in release <tag>` for a release without night extraction; `/health`'s status and each check's status and reason; and `/usage/turns-per-day?days=7`'s counts and how many of those days had a real turn. |
| `gaming-mode on [--by manual\|watcher]` | Yields the GPU to a game: see Gaming mode below. |
| `gaming-mode off [--wait-minutes N]` | Ends the yield, waiting up to N minutes (default 5) for `mist-llm`, then runs `extraction apply`. |
| `gaming-mode status` | Prints `yielded since <time> by <who>`, or `normal`, from the marker alone. |
| `extraction on\|off\|auto` | Writes the mode to `ops/extraction-mode`, then runs `extraction apply`. When apply refuses, the mode stays written and the command exits 1. See Night extraction. |
| `extraction apply` | Starts or stops `mist-extraction-llm` and `mist-extraction` to match the mode, the window, gaming mode and the two identity variables: see Night extraction. Exit 0 when it converged, or when it started nothing because `mist-backend` is not running; 2 refused (a malformed window or mode, no `mist-extraction:<tag>` image or one whose ID differs from the release record, or a missing identity variable, after the stop); 1 when a stop or start did not converge. |
| `extraction status` | Prints one line: the mode, the window and whether now is `inside` or `outside` it; `running yes`, `no`, `partial`, or `unknown` when the containers cannot be listed; `paused` with the marker's reason, `expired <reason>` past its `until`, `unreadable` or `no`; the image as `match`, `MISMATCH` or `missing`; and `configured yes\|no`. It prints no variable's value. A malformed window or mode prints `extraction: settings refused (...)` instead. |
| `arm-backups` | Writes `ops/backups-armed` and starts `mist-backup`; from then on every `up` starts it too. The deploy plan arms backups only after a restore drill passes. |
| `disarm-backups` | Removes `ops/backups-armed`, then stops `mist-backup`. |
| `backup-now` | One attempt, armed or not: `docker compose ... --profile backup run --rm mist-backup python -m scripts.backup.scheduler --once`. |
| `accept-backup-baseline --label LABEL` | Makes a reviewed suspect backup the baseline: `docker compose ... --profile backup run --rm mist-backup python -m scripts.backup.scheduler --accept-suspect LABEL`, which moves `suspect/LABEL` back into the backup root and records it as the last success. See `backup_suspect` under Reading /health. |
| `drill-start --drill-root D --artifact NAME` | Starts a restore drill of the backup `NAME` in `MIST_BACKUP_HOST_PATH`. Refuses a drill root that is not empty or fails the path checks above, a volume name that already exists, and any container of the drill project that remains from an earlier drill (one drill per instance at a time). Creates the volume `mist-<instance>-drill-<UTC stamp>` with labels `org.mist.role=drill` and `org.mist.instance=<instance>`, and writes `D/drill.json` (`volume`, `artifact`, `started_at`). |
| `drill-restore --drill-root D [--confirm-target TOKEN]` | Runs `scripts.backup.restore` in the drill project into the drill volume and `D/standby/`, with `--confirm-target` only when given, so the first run prints the token restore wants. Refuses a drill that already verified a `match`, and a drill project whose `mist-neo4j-dev` mounts another volume at `/data` or whose mounts cannot be read. |
| `drill-verify --drill-root D` | Runs `scripts.backup.verify` against the restored drill and writes its counts to `ops/drill-<UTC stamp>.json`. Exit 0 match, 1 mismatch. Then reads the `/data` mount of the drill project's `mist-neo4j-dev`; unless it is the drill volume, it writes no record and exits 1. |
| `drill-end --drill-root D` | Removes the drill's containers (`rm -s -f`, no `-v`). The volume and the drill root stay until you delete them. |
| `promote --drill-root D` | Makes a verified drill the instance: the volume becomes `MIST_DAILY_NEO4J_VOLUME`, `D/standby/state` and `D/standby/mist-memory` replace `state/` and `mist-memory/`, which move to `retired/<UTC stamp>/`, and the instance comes back up. Before anything moves it stops the instance and removes its containers (`--profile backup rm -f`, no `-v`), so `up` creates `mist-neo4j` on the promoted volume; after `up` it reads that container's `/data` mount and exits 1 unless it is the drill volume. |

The drill commands run `deploy/daily/restore-drill.yml` from the release as project
`mist-<instance>-drill`, with `MIST_DRILL_NEO4J_VOLUME` and `MIST_DRILL_ROOT` set in that
command's environment only. Before any call that uses the volume, `drill-restore`,
`drill-verify`, `drill-end` and `promote` refuse a volume without both drill labels, the volume
`MIST_DAILY_NEO4J_VOLUME` names, and a volume any container outside the drill project mounts.
`promote` also refuses without a `match` record newer than `drill.json`, and a drill root on
another drive than the instance root. The runbook, with the order and every gate, is
`deploy/daily/RESTORE.md`; never run `restore-drill.yml` by hand.

`compose.yml` declares `${MIST_BACKUP_HOST_PATH:?...}`, so compose refuses the file unless
`daily.env` sets it. In practice `MIST_BACKUP_HOST_PATH` is therefore always set, and `switch`
always backs up a running instance first. Pass `--skip-backup` to switch without that backup,
for example when the backup destination is offline; the rollback then has no pre-switch
artifact to restore a newer release's data migration from.

---

## Gaming mode

Gaming mode gives the RTX 4070 SUPER's memory to a game while the rest of MIST keeps running.

What yields: `gaming-mode on` writes `ops/gpu-yielded`
(`{"v": 1, "since": "<UTC time>", "by": "manual"|"watcher"}`) through a temporary file and a
rename, waits two backend polls (twice `MIST_GPU_YIELD_POLL_S` from `daily.env`, 2 s by
default, so 4 s), then stops `mist-llm`. The backend sees the marker within one poll and
releases Whisper, its one GPU model in this instance: the embedding model runs on the CPU by
literal, and TTS is off. Waiting before the stop means `/health` never sees a stopped LLM while
the backend still reads normal. On a release with night extraction it then stops
`mist-extraction` and `mist-extraction-llm` too, by the stop path in Night extraction, with
the paused marker's reason `gaming`.

- If the stop fails twice, `on` removes the marker it wrote and exits 1, so nothing reports
  `yielded` while the LLM still holds the GPU. A marker that was already there is left.
- A second `on` keeps the first `since`, skips the wait and still stops `mist-llm`.
  `on --by manual` over a watcher's yield takes it over (`by` becomes `manual`), so the watcher
  cannot end a hold you asked for; `on --by watcher` never changes a manual one.

What keeps running: the API (`/health`, `/usage`, WebSocket connections), the event log, the
extraction backlog, backups, the use count and every curation job that needs no LLM. Turns
still queue in the backlog, but nothing drains it: extraction runs on this GPU at night, in
the daily project, so it stops with the LLM, and `gaming-mode off` applies it again.
`self_reflection`, the one curation job that uses the LLM, is deferred and covers the deferred
hours when it next runs.

A turn gets the yielded reply. While yielded, every `audio` and `text` message gets the error
envelope "MIST has yielded the GPU to a game and will answer when gaming mode ends.", and no
turn is recorded. `/health` reads `yielded`, with the `llm` check `yielded` (`gpu_yielded`),
and `restart_recommended` stays false.

`off` waits for the LLM. `gaming-mode off` starts `mist-llm` and polls `docker compose ... ps`
until compose reports it `healthy`, for up to `--wait-minutes` (default 5), and only then
removes the marker. While the LLM loads, turns still get the yielded reply. On a release with
night extraction it then runs `extraction apply`, which starts extraction again if the mode
and window want it; an apply that does not converge makes `off` exit 1 after the marker is
gone.

- If it never becomes healthy, `off` exits 1 and leaves the marker, so `/health` keeps saying
  `yielded`, which is still true.
- If Whisper does not reload after the marker goes, for example because a game still holds the
  memory, the `llm` check reads `down` with `resume_failed`, and the backend retries with a
  backoff up to 60 s.
- `up` leaves `mist-llm` stopped while the marker exists, so a restart during a game does not
  load the model.

The residual: the backend process keeps its CUDA context while yielded, which PyTorch frees
only when the process exits. The scratch rehearsal (H3) records how large it is.

### The gaming watcher

`instance.py gaming-watch --root <root>` turns gaming mode on and off by itself, on Windows.
Every `MIST_GAME_POLL_S` seconds (10 by default) it lists the running programs and asks whether
one of them is a game's own process. A launcher, client, crash reporter, anti-cheat, overlay,
redistributable or background app never counts, so MIST stays up while those sit in the
background.

What counts, by two rules in this order:

1. **A Java game.** `java.exe` or `javaw.exe` counts when the folder it runs for lies under a
   `MIST_GAME_RUNTIME_DIRS` entry, a game instance or server folder. That folder is its
   `--gameDir` argument (a Minecraft client), or else its working directory (a server started
   from its own folder). It does not count when its arguments run an installer
   (`--installServer`, or a `-jar` file named like `*installer*.jar`), or when the folder lies
   under a `MIST_GAME_EXCLUDE` directory. Where the Java itself is installed plays no part, so
   a launcher's own Java, or any other Java program, does not count.
2. **A game from a library.** Any program counts when it was started from under a
   `MIST_GAME_DIRS` entry, a game library, unless an exclusion names it. A new game installed
   into a library counts with no configuration change.

What never counts, from a library either: the exclusion table that ships in `gamewatch.py`
names generic classes once for every user, by folder (the store clients' own launcher and
client folders, VR runtimes, shared redistributables, anti-cheat) and by file name (launchers,
a client lobby, crash reporters, web helpers, overlays, installers, updaters,
redistributables, uninstallers, anti-cheat). A non-game particular to one PC, such as a
wallpaper app or a mod tool kept in a library, goes in `MIST_GAME_EXCLUDE`. A game whose own
file name matches a shipped pattern, such as a title with "launcher" or "setup" in its
executable's name, is missed; turn gaming mode on by hand for it.

The three folder variables, each separated by `;` and set without quotes (spaces inside a
directory are kept, backslashes are literal, and only the host CLI reads them):

- `MIST_GAME_DIRS`: the game libraries.
- `MIST_GAME_RUNTIME_DIRS`: the folders that hold game instances and servers run on Java.
- `MIST_GAME_EXCLUDE`: host exclusions. An entry with `\`, `/` or `:` is a directory, and any
  other is a file name pattern, matched ignoring case.

For example:

```
MIST_GAME_DIRS=E:\SteamLibrary\steamapps\common;E:\Games
MIST_GAME_RUNTIME_DIRS=E:\Minecraft\Instances;E:\Minecraft\Servers
MIST_GAME_EXCLUDE=E:\SteamLibrary\steamapps\common\Some Wallpaper App;modtool*.exe
```

A path is under a directory when it is that directory or lies inside it, ignoring case:
`E:\Games2\x.exe` is not under `E:\Games`. When entries nest, the lower index wins.
`gaming-watch` refuses (exit 2) while `MIST_GAME_DIRS` and `MIST_GAME_RUNTIME_DIRS` are both
unset or empty; either alone is enough.

- When a game starts and there is no marker, it runs `gaming-mode on --by watcher`.
- When no game has run for `MIST_GAME_RESUME_DELAY_S` seconds (120 by default), it runs
  `gaming-mode off`, but only for its own marker (`by: watcher`). It never ends a manual gaming
  mode: `gaming-mode on` by hand holds until `gaming-mode off` by hand.
- If someone runs `gaming-mode off` while a game runs, it does not turn gaming mode on again
  until it has seen a scan with no game.
- Its `off` keeps scanning while `mist-llm` loads; a game started meanwhile stops `mist-llm`
  again and the marker stays.
- A failed `on` or `off` is logged and decided again at the next scan.

A game running elevated (as administrator), or as another user, may go unseen: the watcher
runs without elevation and skips any program it cannot open, and it never reads an elevated
Java's working directory, so such a Java counts only through `--gameDir`. Turn gaming mode on
by hand for such a game.

The watcher needs a 64-bit Python, and refuses (exit 2) on any other: it reads a Java
process's working directory through the x64 process layout. It reads no 32-bit Java's
folder, so a 32-bit Java counts only through `--gameDir`.

It logs each change to `logs/gaming-watch.log` under the root, rotated at 1 MiB with three old
files kept. Its first line gives how many entries each folder variable holds and the position
of each configured directory that does not exist. A log line names a game only by the variable
and position of the entry it counted under, as `MIST_GAME_DIRS[0]` for the first library or
`MIST_GAME_RUNTIME_DIRS[1]` for the second instance folder, never by the program's name, its
path or its arguments; a Minecraft client's arguments carry its account's access token.

To see what counts and why, run `instance.py gaming-scan --root <root>`. It lists the running
programs once, prints to the terminal, and writes no log; it needs no release switched in, so
it runs before the first `switch` too. It prints, in this order:

- each configured entry as `MIST_GAME_DIRS[0] <entry>`, with ` (missing)` after a directory
  that does not exist;
- `game: yes` or `game: no`;
- `counts MIST_GAME_DIRS[<i>]: <program>` for each program that counts from a library, and
  `counts MIST_GAME_RUNTIME_DIRS[<j>]: <program> for <folder>` for each Java game, with the
  folder it counted for;
- `excluded <reason>: <program>` for each program under a library that an exclusion removed,
  the reason naming the shipped folder or file name pattern, or the `MIST_GAME_EXCLUDE` entry;
- `not counted <reason>: <program>` for each other Java process, the reason being
  `no runtime folder`, `installer`, `folder unreadable` or a `MIST_GAME_EXCLUDE` entry.

The scan prints no argument other than a folder taken from `--gameDir`, and no other line of
`daily.env`. When MIST steps down for something that is not a game, the scan names it: add it
to `MIST_GAME_EXCLUDE`. It exits 0, 2 on the refusals `gaming-watch` makes, and 1 when
`gamewatch.py` cannot be loaded or listing fails.

`gamewatch.py` sits beside `instance.py` in `bin/`, where `switch` installs both. Without it,
`gaming-watch` exits 1 naming the file, and every other command still runs. On Windows every
Docker call the CLI makes starts without a console window, so the watcher can run under
`pythonw.exe`.

To start it at every sign-in, add `-GamingWatch` to `register-signin-task.ps1` (see "Start
after power-on"):

```powershell
powershell -ExecutionPolicy Bypass -File deploy\daily\windows\register-signin-task.ps1 -Python <python.exe> -Root <root> -GamingWatch
```

It registers a second task, "MIST gaming watch", which runs
`<pythonw.exe> <root>/bin/instance.py gaming-watch --root <root>` with the same sign-in
trigger and settings as the first, including no time limit. `pythonw.exe` comes from the
directory of `-Python`, and the script refuses before registering anything when it is not
there. `unregister-signin-task.ps1` removes all three tasks when present: this one, "MIST
daily instance" and "MIST extraction window".

---

## Night extraction

Extraction runs on this GPU at night, in the daily project, beside `mist-llm`:
`mist-extraction-llm` is the llama-server that loads the extraction model, and
`mist-extraction` is the service `mist-backend` calls at `http://mist-extraction:8090` (compose
sets `MIST_EXTRACTION_INFERENCE=service` and that URL on `mist-backend`). By day turns queue in
the extraction backlog; in the window the backlog drains. Both services sit under profile
`extraction` with `restart: "no"`, so neither the chat services' `up -d` nor a Docker restart
starts them. Only `extraction apply` does, run by the task below, by `up`, by `gaming-mode off`
and by `extraction on|off|auto`.

Below, `<dc>` is `docker compose -p mist-<instance> -f
<root>/release/<MIST_RELEASE_TAG>/deploy/daily/compose.yml --env-file <root>/daily.env --profile
extraction`; without `--profile extraction`, compose does not resolve the two services.

Every `extraction` command refuses with exit 2 on a release whose `ops/release-<tag>.json`
records no `extraction_image_id`, one built before night extraction.

### The window

`MIST_EXTRACTION_WINDOW` in `daily.env` sets it as `HH:MM-HH:MM`, 24-hour, on the host's
clock: the host's local time, the time zone Windows runs in, not `MIST_TIMEZONE`. Unset or
empty, it is `23:59-10:00`. The start is inside the window and the end is not, and a window
whose end comes before its start crosses midnight. Only the host CLI reads the variable. A
value that is not `HH:MM-HH:MM`, or one with equal ends, makes `extraction apply` refuse
(exit 2) and `extraction status` print `settings refused`; for always or never, use
`extraction on` or `extraction off`.

### The modes

`ops/extraction-mode` holds one word, written by `extraction on`, `extraction off` or
`extraction auto`:

- `on`: extraction runs at any hour; the window is ignored.
- `off`: extraction never runs.
- `auto`: extraction runs inside the window.

A root with no file reads as `auto`. `init` writes `off` when the file is absent, on a new
root and on a second `init` of an older root alike, so a new instance starts with extraction
off until its epoch cutover (`backend/extraction_backlog/CUTOVER.md`); a mode set since is
kept.

Whatever the mode, extraction does not run while gaming mode is on (`ops/gpu-yielded`), and
does not start while `EXTRACTION_MODEL_HASH` or `EXTRACTION_LLM_DEVICE` is unset in
`daily.env`; a value still in its `<placeholder>` form counts as unset. They are the model
identity `mist-extraction` stamps on each result and the GPU name in its serving identity. When
the mode and window want extraction but one is unset, `extraction apply` stops it, writes the
paused marker with reason `unconfigured`, and exits 2 naming the variable, never a value: set
it, then run `extraction apply`.

### What `extraction apply` does

Each call reads the mode, the window, `ops/gpu-yielded` and the identity variables afresh,
and takes the two services there, whatever an earlier call or a crash left, so calls in any
order converge without a lock.

To stop, it writes the paused marker `ops/extraction-paused`
(`{"v": 1, "since": <UTC>, "reason": <r>, "until": <UTC or null>}`) with the reason `gaming`,
`mode`, `window` or `unconfigured`, in that order of precedence. For `window`, `until` is the
next window start plus 15 minutes; for the others it is null. When the services run and no
current marker was there before, it waits two backend polls, as `gaming-mode on` does. Then
it stops `mist-extraction` with a 10 s timeout, then `mist-extraction-llm`, saves each one's
last 200 log lines to `logs/extraction-last-<service>.log`, replacing the previous copy, and
removes both containers by name. The service stops first, so a call in flight ends as a
transport error, which costs the turn no attempt.

To start, it first refuses (exit 2) unless the local `mist-extraction:<tag>` image has the ID
the release recorded; the fix is `release-build <tag>`. When both containers already run
healthy from the current release's compose file and configuration, it removes the marker and
does nothing more. While `mist-backend` is not running it starts nothing: it prints a note,
writes no marker and exits 0, and `up` applies again once the backend runs. Otherwise it writes
the marker with reason `starting`, whose `until` is the start's 600 s deadline, and removes any
container made by another release or an older configuration, for example after a
`MIST_DAILY_EXTRACTION_NCMOE` change in `daily.env`. It then runs `<dc> up -d` for
`mist-extraction-llm`, then for `mist-extraction`, and waits up to 600 s for both to report
`healthy`. After each step it reads the desired state again and takes the stop path if
extraction is no longer wanted. It removes the marker only once both are healthy and still
wanted. A start that fails removes both containers and the marker, so `/health` reads
extraction as down, and exits 1.

Each start, and each stop of a running service, appends `<UTC> start|stop <reason>` to
`logs/extraction-window.log`; a failed start appends `stop failed`.

While the marker is current (well formed, and its `until` null or still ahead), `/health`
reports `extraction_service` as `paused` (`extraction_paused`), which counts as up, so the
instance reads `healthy` all day while extraction waits for night. While gaming mode is on, the
check reads `yielded` (`gpu_yielded`) instead, checked first. A marker past its `until`, or one
that is malformed, reads as absent, and the check reports what the service is doing. The
`extraction_backlog` check is not paused: see Reading /health.

### The extraction window task

The task "MIST extraction window" keeps the window with no command typed. Register it with
`register-signin-task.ps1` and `-NightExtraction` (see Start after power-on):

```powershell
powershell -ExecutionPolicy Bypass -File deploy\daily\windows\register-signin-task.ps1 -Python <python.exe> -Root <root> -NightExtraction
```

It runs `<pythonw.exe> <root>/bin/instance.py extraction apply --root <root>` every 5 minutes
all day, from 00:00, and at sign-in. A tick missed while the PC slept runs when it wakes, a
tick never starts while the last run still goes, and a run is ended after 15 minutes.
`pythonw.exe` comes from the directory of `-Python`, and the script refuses before registering
anything when it is not there. Running the script again without `-NightExtraction` leaves the
task in place.

Without the task, nothing starts extraction at the window's start or stops it at the window's
end unless a command that applies runs then, and a `window` marker expires 15 minutes after
the next start, so `/health` then reads `extraction_service` as `unreachable`.

### Gaming

`gaming-mode on` stops extraction too, after `mist-llm`, with reason `gaming`: a game
outranks the mode. `gaming-mode off` waits for `mist-llm`, removes `ops/gpu-yielded`, then runs
`extraction apply`, which starts extraction again only if the mode and window want it. The
gaming watcher's `off` passes its game check to that apply, so a game started meanwhile keeps
extraction stopped, with reason `gaming`.

### Checking it

`extraction status` prints one line: the mode, the window and whether now is inside it,
whether both services run, the paused reason, the image match and `configured yes|no` (see
Commands). `status` prints the same line, and for a release without night extraction prints
`extraction: none in release <tag>` where `extraction status` refuses.

### Turning extraction off

`extraction off` writes mode `off` and stops both services with reason `mode`; `/health` reads
`extraction_service` as `paused`. The task keeps running, and each tick leaves extraction
stopped. Turns keep queueing, so once the oldest pending turn is a day old,
`extraction_backlog` reads `backlog_overdue`.
`extraction auto` returns to the window, and `extraction on` runs extraction at any hour. To
remove the task as well, run `unregister-signin-task.ps1`, which removes all three tasks, then
`register-signin-task.ps1` again without `-NightExtraction`.

---

## Capabilities

The backend can run five capabilities: clock, alarms, timers, lights (Home Assistant over MCP)
and music (Spotify). They are off until `daily.env` turns them on, and
`GET http://127.0.0.1:8001/capabilities` reports whether the host is enabled, the power level,
and each capability's state and reason.

Where their data lives. `compose.yml` sets four literals in `mist-backend`, so no `daily.env`
line moves them:

- `MIST_CAPABILITIES_DB_PATH=/app/data/capabilities.db`: the store `capabilities.db` sits in
  `state/` beside the others, and every backup captures it with its row counts.
- `MIST_HA_TOKEN_FILE=/app/secrets/ha_token` and
  `MIST_SPOTIFY_TOKEN_PATH=/app/secrets/spotify_token.json`: the tokens sit in
  `<root>/secrets/`, which `mist-backend` mounts read-write at `/app/secrets` (the Spotify
  token is rewritten each time it is refreshed).
- `MIST_HA_TOKEN=` is empty, so a token put in `daily.env` is ignored; the file is its one
  source.

The secrets folder. `init` creates `<root>/secrets/`; for a root made before this release, run
`init --root <root>` again, which fills in what is missing and rewrites nothing. No other
service mounts it, so `mist-backup` cannot read it and `<root>/secrets/` is not backed up. Put
the tokens there:

- `ha_token`: a Home Assistant long-lived access token, created in Home Assistant under the
  user MIST signs in as (its profile, Security tab), written alone into the file.
- `spotify_token.json`: written by the Spotify authorization, run once in the running backend
  after `MIST_SPOTIFY_CLIENT_ID` is set:
  `docker compose -p mist-daily exec mist-backend python -m backend.capabilities.music.authorize`.
  It prints an address to open in a browser and asks for the address the browser lands on.
  `--check` afterwards refreshes the token and prints how many devices Spotify reports.

A missing or unreadable `ha_token` makes lights `unavailable` with the reason; the other
capabilities run.

The `daily.env` rows, all commented out in `daily.env.example`:

- `MIST_CAPABILITIES_ENABLED`: `true` turns the host on; `false` or unset builds none.
- `MIST_CAPABILITIES`: which of the five to run, comma-separated; all five when unset.
- `MIST_TIMEZONE`: the IANA zone alarms, timers and the clock use. While it is unset or unknown
  those three are `unavailable`, so an alarm never rings at an hour from a guessed zone.
- `MIST_HA_MCP_URL` and `MIST_LIGHTS_DEFAULT_TARGET`: lights is built only with the URL; the
  target is the light or area a command that names none acts on.
- `MIST_SPOTIFY_CLIENT_ID` and `MIST_SPOTIFY_DEVICE`: music is built only with the client ID;
  the device is where playback goes when no Spotify device is active.
- `MIST_RING_FALLBACK_AFTER_S`: how long past an alarm's time an endpoint that cannot reach the
  backend waits before it rings on its own; 30 when unset, 5 to 300.

Activation. Set the rows, then run `up` to apply a changed `daily.env` on the same release, or
`switch <tag>` for a new release (which runs `up` itself). Either recreates `mist-backend` with
the new environment.

At the off level, while gaming mode is on, the backend keeps running: alarms and timers already
set ring and time out on schedule, because the scheduler needs no model. Spoken messages are
refused, because Whisper is released. A typed message goes to the capability host's model-free
step alone: one it handles (a ringing alarm or timer, a yes or no to a waiting proposal, or a
fast-path command such as "set a timer for 10 minutes") is answered with no model call, and any
other typed message gets the refusal: the yielded message, or the resume-failed message when
the speech model could not be reloaded after gaming mode. With capabilities off
(`MIST_CAPABILITIES_ENABLED=false`, so no host), every typed message gets that refusal.

Rollback, behaviour: set `MIST_CAPABILITIES_ENABLED=false` in `daily.env` and run `up`. No host
is built, and the WebSocket and `/capabilities` answer as disabled. Rollback, code:
`switch <previous tag>`.

Rollback, data: stop the instance with `down`, then delete `state/capabilities.db` and its
`-wal` and `-shm` files. That drops every alarm, timer, event and action-log row; the pre-switch
backup holds the last copy. While capabilities stay off, nothing recreates the file, so the next
backup lacks a store the last success held and is `backup_suspect` (the success rule,
`scripts/backup/scheduler.py`, `stores the last success held have no file`). Review its
`manifest.json`, then run `accept-backup-baseline --label <label>` to make it the baseline.

Rollback, tokens: delete `<root>/secrets/ha_token` and revoke the token in Home Assistant;
delete `<root>/secrets/spotify_token.json` and remove the app's access in Spotify's account
settings. Neither is in any backup, so after a restore onto a new instance root both are
issued again (`deploy/daily/RESTORE.md`).

---

## Reading /health

`GET http://127.0.0.1:8001/health` always answers HTTP 200. Its `status` is `healthy`,
`degraded`, `yielded` or `unhealthy`, and each check carries a `status` and a `reason`;
`instance.py status` prints them all. The reasons below are the checks this instance adds or
changes, with what each means and what to do.

| Check | Reason | Status | Meaning | Remedy |
|---|---|---|---|---|
| `extraction_service`, `extraction_backlog` | `unavailable_at_boot` | down | The extraction dispatcher raised at boot. | Read the boot error in `logs/mist-backend.log`, fix it, and restart `mist-backend`. |
| `extraction_service`, `extraction_backlog` | `disabled_by_config` | null | No dispatcher, or `MIST_EXTRACTION_INFERENCE=off`. | Compose sets the mode and URL on `mist-backend`, so a `daily.env` line neither causes nor fixes this. Run `extraction status`: it refuses on a release without night extraction. Otherwise read the boot lines of `logs/mist-backend.log`. |
| `extraction_service` | `extraction_paused` | paused | `extraction apply` stopped extraction on purpose, or is starting it: `ops/extraction-paused` is current (see Night extraction). Reads as up. | None if intended; `extraction status` names the reason. |
| `extraction_service` | `gpu_yielded` | yielded | Gaming mode is on, so extraction is stopped with `mist-llm`. Checked before the paused marker. | `gaming-mode off`, which runs `extraction apply`. |
| `extraction_service` | `unreachable` | down | `mist-extraction` does not answer `GET /v1/health`, and no current paused marker explains it. Turns queue and nothing dead-letters. | Run `extraction status` to see the mode, the window and what runs, then `extraction apply`, which starts the services when they are wanted or says why not. If the "MIST extraction window" task is missing, register it (see Night extraction). |
| `extraction_service`, `extraction_backlog` | `query_failed` | down | The service's health call failed or answered invalid JSON; or the backlog could not be read. | Read `logs/mist-backend.log` for the cause; for the backlog, check `state/` on disk. |
| `extraction_service` | `service_degraded` | down | The service answered, but its status is not `ok` or it cannot reach its own LLM. | Read `<dc> logs mist-extraction-llm` while it runs, or `<root>/logs/extraction-last-mist-extraction-llm.log` after a stop (see Night extraction for `<dc>`). |
| `extraction_service` | `epoch_mismatch` | down | The service's `/v1/info` extraction version or model hash differs from the active extraction epoch, so nothing is dispatched. | Bring back the night setting the active epoch was bound with: restore a changed `MIST_DAILY_EXTRACTION_*`, `EXTRACTION_MODEL_HASH` or `EXTRACTION_LLM_DEVICE` line of `daily.env` to the release's value, or `switch` to the release whose extraction version the epoch carries, then run `extraction apply`. To keep a new setting instead, cut over: a serving change is a new epoch (`backend/extraction_backlog/CUTOVER.md` section 9; `backend/extraction_backlog/dispatcher.py`, FAILURES). |
| `extraction_backlog` | `stalled` | down | The dispatcher state is `stalled`: the backend's writer stamps differ from the active epoch, or the service's contract version is incompatible or its `/v1/info` unparsable. | Read the logged reason in `logs/mist-backend.log`. For writer stamps, set `MIST_MODEL_HASH` or switch to the release whose versions match the epoch, then restart `mist-backend`. |
| `extraction_backlog` | `backlog_overdue` | down | The oldest pending turn is older than `MIST_HEALTH_BACKLOG_MAX_AGE_S`: 86,400 s (24 h) by default in this compose file, since turns queue by day and drain at night. | Run `extraction status`: a mode of `off`, `configured no` or an image other than `match` keeps extraction from running; `logs/extraction-window.log` shows when it last started and stopped. It clears as the backlog drains. |
| `extraction_backlog` | `dead_lettered` | down | At least one turn is dead-lettered and waits for a retry. | `docker compose -p mist-daily exec -T mist-backend python -m backend.extraction_backlog.admin retry-dead-letters`, prefixed with `MSYS_NO_PATHCONV=1` under Git Bash. |
| `backup` | `disabled_by_config` | null | `MIST_BACKUP_STATUS_PATH` is unset; not the case in this compose file. | None. |
| `backup` | `backup_never_run` | down | Backups are configured but no status file exists yet. | `arm-backups` once a drill has passed, or `backup-now` for one attempt. |
| `backup` | `query_failed` | down | `ops/backup-status.json` cannot be read or lacks a key. | The next attempt overwrites it; run `backup-now`. |
| `backup` | `backup_suspect` | down | The last dump failed the success rule (an empty graph or vault, a lost store, or fewer nodes than `MIST_BACKUP_MIN_NODE_RATIO` of the last success), so it was moved to `suspect/` and prune did not run; no success is recent. | Inspect `<backup root>/suspect/<label>/manifest.json`, where the backup root is `MIST_BACKUP_HOST_PATH`, against the last success's counts in `ops/backup-status.json`. If the shrink is expected, run `accept-backup-baseline --label <label>`; otherwise find what emptied the graph, vault or store before the next attempt. |
| `backup` | `backup_failed` | down | The last attempt failed and no success is recent. | Read `docker compose -p mist-daily logs mist-backup`, fix the destination, run `backup-now`. |
| `backup` | `backup_overdue` | down | The last success is older than `MIST_BACKUP_MAX_AGE_HOURS` (default 26). | Check that `mist-backup` runs (`status`); `arm-backups` if it was disarmed. |
| `llm` | `gpu_yielded` | yielded | Gaming mode is on. | `gaming-mode off`. |
| `llm` | `resume_failed` | down | Gaming mode ended but Whisper did not reload; the backend retries. | Close what holds the GPU's memory; the backend retries on its own. If it persists, restart `mist-backend`. |

Any check reads `ok` with status `up` when nothing above applies.

---

## Start after power-on

Docker Desktop starts at sign-in by its own "Start when you sign in" setting, and brings back
the containers, which have `restart: unless-stopped`. A Windows sign-in task, "MIST daily
instance", then runs `<root>/bin/instance.py up --root <root>`, which waits up to 10 minutes
for Docker and checks that the release is the one recorded. The build ships two ways to start
before anyone types a command:

- (a) Automatic sign-in with an immediate lock. `register-signin-task.ps1 -LockAtSignIn` makes
  the task's first action lock the workstation (`rundll32.exe user32.dll,LockWorkStation`),
  ahead of `up`. What it needs from Raj: turning Windows automatic sign-in on himself, and
  Docker Desktop's "Start when you sign in". The build stores no credential and changes no
  sign-in setting. With automatic sign-in, Windows keeps the password itself: as an LSA secret
  when it is set through Sysinternals Autologon, or in plain text in the Winlogon
  `DefaultPassword` registry value when it is set by hand. Use Autologon. The lock comes when
  Task Scheduler starts the task, seconds after the desktop shows, and never if the task fails
  to start. As a backstop, set a screen saver with "On resume, display logon screen", which
  locks the session after its wait even when the task did not.
- (c) Sign-in only. `register-signin-task.ps1` without `-LockAtSignIn`. What it needs from
  Raj: signing in after power-on, and Docker Desktop's "Start when you sign in". The instance
  starts once he signs in.

A third way, a dedicated Docker Engine in its own WSL2 distribution that needs no sign-in, is
not built here. Deploy decision A4 chooses between (a) and (c).

Register the task from the repository or from `release/<tag>/deploy/daily/windows/`:

```powershell
powershell -ExecutionPolicy Bypass -File register-signin-task.ps1 -Python <python.exe> -Root <root> [-LockAtSignIn] [-NightExtraction]
```

`-NightExtraction` also registers "MIST extraction window" (see Night extraction), and
`-GamingWatch` "MIST gaming watch" (see Gaming mode).

The script needs `<root>/bin/instance.py`, which `switch` puts there, and prints the task's
`schtasks /query` line. The task runs as the signed-in user, only while that user is signed in,
with no time limit and no battery conditions. Task Scheduler keeps each run's exit code:
`schtasks /query /tn "MIST daily instance" /v` shows it as Last Result.
`unregister-signin-task.ps1` removes all three tasks when present ("MIST daily instance",
"MIST gaming watch" and "MIST extraction window"), for a rollback; it does not stop the
instance, and `extraction off` stops extraction. Then, after a `switch` to an older release,
register again with that release's `register-signin-task.ps1`, without `-NightExtraction`
(the order is in Release lifecycle).

Either way, the time from power-on until the instance records its first status is a gap in the
uptime ledger: `/usage/uptime` counts it as `unrecorded`, never as up.

---

## No scripted traffic to the daily instance

Nothing scripted talks to the daily instance: no eval, gauntlet, hydration, smoke run or replay,
and not `mist_admin chat` either. The backend sets `MIST_SESSION_ORIGIN=real` for the whole
process, so the origin of a turn is not per connection: every turn the instance records counts
as real use, whoever sent it. Scripted work goes to a dev stack, which reaches this instance's
`mist-llm` over the share network and nothing else.

---

## The root docker-compose.yml is the workstation stack

The repository root's `docker-compose.yml` is the workstation stack, not this instance. Do not
start it while the daily instance runs:

- its `mist-backend` and `mist-neo4j` cannot bind ports 8001, 7687 and 7474, which the daily
  instance holds;
- its `mist-llm`, which publishes 8080 (`docker-compose.yml:137-138`), would load a second model
  onto the GPU, where it would compete with the instance's llama-servers for the 12 GB card's
  memory. The night rehearsal ran the instance's own two llama-servers together, `mist-llm`
  beside `mist-extraction-llm` at `MIST_DAILY_EXTRACTION_NCMOE` 21: a peak `memory.used` of
  10,105 MiB, a contended chat time to first token of at most 0.459 s, and a chat-idle
  extraction decode median of 24.74 tokens per second over three trials;
- once the original data has moved into the instance root, it starts an empty MIST, because its
  mounts are relative to the repository (`./data`, `./logs`, `./mist-memory`).

---

## Memory limits

Every service has a `mem_limit` from a variable, and Neo4j has an explicit heap and page cache
instead of sizing itself from the Docker VM, which is shared with other containers. The defaults
are 8g backend, 3g Neo4j, 9g LLM, 4g backup, 14g `mist-extraction-llm` and 1g `mist-extraction`.

- Neo4j's and the LLM's come from the scratch rehearsal's rerun of 6 Nov 2026. Each holds the
  service's measured peak plus the remaining growth of its own cache.
  - Neo4j peaked at 1570 MiB on a near-empty graph, and its page cache fills to 512m as the store
    grows: 2082 MiB, which leaves 32.2% of 3g free.
  - llama-server peaked at 4602 MiB after a handful of prompts, and its prompt cache may still
    grow to the 2048 MiB that `--cache-ram` allows: 6650 MiB, which leaves 27.8% of 9g free. At
    the end of the rerun, `memory.stat` counted 5374 MiB of file pages, mostly the memory-mapped
    model, which the kernel can reclaim, and 312 MiB of anonymous memory.
- The backend's and backup's are estimates. The backend's has been measured on text turns only.
- `mist-extraction-llm`'s 14g has a measured peak of 9.716 GiB from the night rehearsal, at
  `MIST_DAILY_EXTRACTION_NCMOE` 21, which leaves 30.6% of 14g free, more than 1 GiB. The limit
  holds at most about 11 GiB of expert tensors (all 24 layers' experts on the CPU), the 2048 MiB
  prompt-cache cap and 1 GiB for the server.
- `mist-extraction`'s 1g is an estimate with no breakdown and no measured peak, because the
  first night run never started it.

The six limits add up to more than the Docker VM's memory, on purpose: each one bounds a single
service, and what must fit the VM is the sum of what the services use.

Check the limits against `docker stats` on the running instance. Headroom is (limit - peak) /
limit: raise any limit with less than 25% headroom over its observed peak. Every service also
logs through the `json-file` driver with `max-size` 20m and `max-file` 5.

---

## Room devices

A room device (a voice satellite on the home LAN) reaches the backend through
`deploy/daily/compose.devices.yml`, an overlay that is off unless `daily.env` sets
`MIST_DAILY_DEVICE_BIND`. `compose.yml` itself stays loopback-only and knows nothing of devices.

With the bind set, every compose call `instance.py` makes adds
`-f <root>/release/<tag>/deploy/daily/compose.devices.yml --profile devices`, which:

- gives `mist-backend` `MIST_DEVICE_LISTENER=on` and mounts `MIST_DEVICE_DIR` read-only at
  `/app/devices`; the backend publishes no new port;
- adds `mist-device-forward`, which publishes `<MIST_DAILY_DEVICE_BIND>:<MIST_DAILY_DEVICE_PORT>`
  and relays bytes to the backend's port 8443 on the project network. TLS ends in the backend,
  so the forwarder holds no key.

The variables:

- `MIST_DAILY_DEVICE_BIND`: the PC's LAN address. `instance.py` refuses anything but one private
  IPv4 address (10/8, 172.16/12, 192.168/16), and refuses the bind without `MIST_DEVICE_DIR`.
- `MIST_DAILY_DEVICE_PORT`: the host port on that address, 8443 when unset. It is refused when
  it is 8001, an eval, dev or staging port, or no port at all.
- `MIST_DEVICE_DIR`: the absolute host path of the devices directory, outside every git work
  tree and never under `<root>/secrets/`.
- `MIST_DEVICE_FORWARD_IMAGE`: workstation only, never in `daily.env`. The daily instance runs
  the forwarder from `mist-backend:<tag>`; the workstation stack takes the overlay by a second
  `-f` and sets this to its own backend image.

### Provisioning

`scripts/daily/devices.py` runs on host Python with the standard library and `openssl`. It keeps
two directories apart: the devices directory (`registry.json`, `tls/` with the server
certificate and key and the CA certificate), which `mist-backend` mounts, and the CA directory
(`<MIST_DAILY_ROOT>/devices-ca`, holding `ca.key`), which nothing mounts. Each command refuses a
directory inside a git work tree.

```
python scripts/daily/devices.py init --ip <pc-lan-ip>
python scripts/daily/devices.py add example-room --area example-room --out <private-dir>
python scripts/daily/devices.py list
python scripts/daily/devices.py revoke example-room
python scripts/daily/devices.py rotate example-room --out <private-dir>
python scripts/daily/devices.py cert --ip <new-pc-lan-ip>
```

- `devices.py init` makes the CA and a server certificate naming the PC's address.
- `devices.py add` registers a device and writes its token, the server address and port, and
  `mist_ca.pem` into the private directory's `secrets.yaml`, for the device's firmware build.
  The token is never printed; the registry keeps only its sha256.
- `devices.py revoke` refuses the device from the next registry reload on; the backend re-reads
  the file when it changes.
- `devices.py rotate` gives a device a new token; reflash it from the private directory.
- `devices.py cert` reissues the server certificate when the PC's address changes; devices keep
  their `mist_ca.pem`.

### up, status and rollback

`up` starts `mist-device-forward` in its own call, after the other services, and retries it every
30 seconds until `--wait-minutes` runs out, counted from the start of `up`, because the LAN
address can arrive after sign-in. A forwarder that does not start fails `up` with exit 1 and
leaves every other service running. With the bind unset, `up` removes a leftover forwarder with
`docker rm -f` (no `-v`), so it stops publishing the device port.

The config-file check holds `mist-backend` and the forwarder to `compose.yml` with
`compose.devices.yml` while the bind is set, and `mist-backend` to `compose.yml` alone while it
is unset; the other services may carry either.

`status` prints `device-forward: running`, `exited`, `absent` or `unknown` after the config-file
lines while the bind is set, and names a leftover forwarder while it is unset.

To roll back, comment out `MIST_DAILY_DEVICE_BIND` and run `up`: the forwarder is removed, and
`mist-backend` is recreated without the listener or the mount. Unset the bind before switching to
a release that has no `compose.devices.yml`.
