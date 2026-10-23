# MIST.AI daily instance

`deploy/daily/compose.yml` is the compose project `mist-daily`: the MIST.AI instance used every
day, run from a pinned release image, with its state in an instance root outside every git work
tree. Development never restarts it, and no file in a working tree reaches it.

The services keep the workstation stack's names (`mist-backend`, `mist-neo4j`, `mist-llm`) and
its host ports 8001, 7687 and 7474, so the live-endpoint denylists in
`backend/knowledge/eval_isolation.py` protect this instance with no change. A fourth service,
`mist-backup`, runs the backup scheduler under profile `backup`, so nothing starts it until
backups are armed.

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
  logs/                  mounted at /app/logs; logs/neo4j/ is mounted at mist-neo4j's /logs
  ops/                   mounted at /app/ops: read-only in mist-backend, read-write in
                         mist-backup; release records, backup status, gaming-mode marker
  models/                the llama-server weights, mounted read-only at /models
  release/<tag>/         deploy/ and scripts/daily/ extracted from the release tag
  retired/<UTC stamp>/   state/ and mist-memory/ replaced by a promotion, kept until deleted
  bin/instance.py        the active release's CLI, run by the sign-in task
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
`environment` literals in `compose.yml` win over any line in it: the store paths,
`MIST_OPS_DIR=/app/ops`, `MIST_SESSION_ORIGIN=real`, `TTS_ENABLED=false` (the code defaults TTS
on) and `EMBEDDING_DEVICE=cpu` (the embedding model must hold no GPU memory). A required variable
that is unset makes compose refuse the whole file before anything starts.

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
| `MIST_EXTRACTION_INFERENCE` | the backend through `env_file` | no | `service` |
| `MIST_EXTRACTION_SERVICE_URL` | the backend through `env_file` | no | code default |
| `MIST_USAGE_TIMEZONE` | the backend through `env_file` | no | UTC |
| `MIST_BACKUP_HOST_PATH` | compose: host directory mounted at `/backup` | yes | |
| `MIST_BACKUP_INTERVAL_HOURS` | compose: the backup scheduler | no | `24` |
| `MIST_BACKUP_RETAIN` | compose: the backup scheduler | no | `7` |
| `MIST_BACKUP_CHECK_MINUTES` | compose: the backup scheduler | no | `10` |
| `MIST_BACKUP_RETRY_MINUTES` | compose: the backup scheduler | no | `60` |
| `MIST_BACKUP_MIN_NODE_RATIO` | compose: the backup scheduler | no | `0.5` |
| `MIST_DAILY_BACKEND_MEM_LIMIT` | compose: `mem_limit` of `mist-backend` | no | `8g` |
| `MIST_DAILY_NEO4J_MEM_LIMIT` | compose: `mem_limit` of `mist-neo4j` | no | `2g` |
| `MIST_DAILY_LLM_MEM_LIMIT` | compose: `mem_limit` of `mist-llm` | no | `4g` |
| `MIST_DAILY_BACKUP_MEM_LIMIT` | compose: `mem_limit` of `mist-backup` | no | `4g` |
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
   `org.opencontainers.image.revision`. It records the image ID in `ops/release-<tag>.json` and
   extracts the archive's `deploy/` and `scripts/daily/` into `release/<tag>/`.
2. `switch <tag>` takes one backup when backups are configured and the instance runs, rewrites
   the `MIST_RELEASE_TAG` line of `daily.env`, and brings the project up from
   `release/<tag>/deploy/daily/compose.yml`. A change on main therefore reaches the instance
   only through `switch`.
3. `switch <previous tag>` is the rollback. It restores the code, not a data migration the newer
   release made; the pre-switch backup is the way back from that.

No working tree is involved at runtime: there are no source bind mounts, and
`pull_policy: never` means compose neither builds nor pulls (a pull of `mist-backend:<tag>` could
fetch an unrelated public image of that name). `up` and `switch` refuse when the tag's image ID
differs from the recorded one, and `status` reports the running image's tag, ID and revision
label, and any container whose compose config file is not the release's.

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
  onto the GPU, and one llama-server is what the 12 GB card holds;
- once the original data has moved into the instance root, it starts an empty MIST, because its
  mounts are relative to the repository (`./data`, `./logs`, `./mist-memory`).

---

## Memory limits

Every service has a `mem_limit` from a variable, and Neo4j has an explicit heap and page cache
instead of sizing itself from the Docker VM, which is shared with other containers. The defaults
(8g backend, 2g Neo4j, 4g LLM, 4g backup) are unmeasured estimates: check them against
`docker stats` on the running instance, and raise any limit with less than 25% headroom over its
observed peak. Every service also logs through the `json-file` driver with `max-size` 20m and
`max-file` 5.
