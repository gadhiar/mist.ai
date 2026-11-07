# Worktree dev stack on a synthetic seed

`docker-compose.dev.yml` at the repository root runs a MIST backend, its own Neo4j and a one-shot
seeder from any git worktree. It does not touch the daily instance or the workstation stack:

- It binds only the code directories (`./backend`, `./src`, `./dependencies`, `./scripts`,
  `./tests`, `./voice_profiles`), read-only. It never binds `./data`, `./mist-memory`, `./logs`
  or `.env`, and it reads no env file.
- Every store (event store, vault, sidecar, vector store, logs, model cache) is on the project's
  own `dev-state` volume, and the graph is on `neo4j-dev-data`. The project name scopes all
  three, so two worktrees never share state.
- Every port binds 127.0.0.1.
- The backend's only external network is the share network, where it reaches the daily
  llama-server by name. The daily graph and the daily backend are not on that network.

This directory is the seed the stack starts from. `vault/` is copied into an empty dev vault and
`seed/` is applied to the graph. Both describe a synthetic MIST self-model and an invented "Dev
User". The same layout serves `scripts/daily/instance.py init --seed tests/fixtures/dev-instance`
for a rehearsal instance root; `tests/unit/knowledge/seed/test_dev_seed_fixture.py` pins it.

## 1. Pick a backend image and create the share network

The stack builds nothing and pulls nothing (`pull_policy: never`). Set `MIST_BACKEND_IMAGE` to an
image already on the host, for example the workstation build (`docker compose build mist-backend`
from the repository root) or a release image from `instance.py release-build`:

```bash
docker image ls | grep mist-backend
export MIST_BACKEND_IMAGE=mist-backend:<tag>
```

The code comes from the worktree's read-only binds, so an older image runs this worktree's code,
as long as its dependencies still match `requirements.txt`.

The backend joins the external network `mist-llm-share`, the one the daily compose attaches its
llama-server to. It must exist before `up`. Running the daily instance means it already does;
otherwise create it once:

```bash
docker network create mist-llm-share
```

`MIST_LLM_SHARE_NETWORK` names a different network, if the daily instance uses one.

## 2. Start the stack

From the worktree root, with a project name unique to this worktree:

```bash
docker compose -f docker-compose.dev.yml -p <name> up -d
```

Without `-p`, compose takes the project name from the worktree's directory name, which is also
unique per worktree.

One `up` on a fresh worktree:

1. starts `mist-neo4j-dev` and waits for it to be healthy;
2. runs `mist-dev-seed`, which copies `vault/` into the empty dev vault and runs
   `python scripts/mist_admin.py seed --seed-dir /app/tests/fixtures/dev-instance/seed`. The
   first run downloads the embedding model onto `dev-state`;
3. starts `mist-backend-dev` once the seed has succeeded.

The backend listens on `ws://localhost:8002/ws`. `MIST_ALLOWED_ORIGINS` is not set, so the code's
default origins apply, and those admit `tauri dev` on port 1420. Turns are recorded with session
origin `test`, so they stay out of the real-turn counts; set `MIST_DEV_SESSION_ORIGIN` to change
that.

`mist-dev-seed` runs again on every `up`. It leaves a vault that already holds notes as it is.
Once the graph holds data from conversations, the seed guard refuses the reseed before writing
anything. On a dev-state already in use, the seeder reports the refusal and exits 0, so the
backend still starts.

To start over, discard this project's volumes, and only this project's:

```bash
docker compose -f docker-compose.dev.yml -p <name> down -v
```

## 3. A second concurrent stack

Each host port is a variable. Give a second worktree its own set:

| Variable | Default | Container port |
| --- | --- | --- |
| `MIST_DEV_WS_PORT` | 8002 | backend 8001 |
| `MIST_DEV_BOLT_PORT` | 7690 | Neo4j bolt 7687 |
| `MIST_DEV_NEO4J_HTTP_PORT` | 7477 | Neo4j HTTP 7474 |

```bash
MIST_DEV_WS_PORT=8012 MIST_DEV_BOLT_PORT=7700 MIST_DEV_NEO4J_HTTP_PORT=7487 \
  docker compose -f docker-compose.dev.yml -p <other-name> up -d
```

Inside the stack the backend always reaches `bolt://mist-neo4j-dev:7687`, which the dev endpoint
allowlist admits. A host-side tool that opens the second stack's graph on a moved bolt port
needs that port in `MIST_DEV_NEO4J_HOSTS`, because the default allowlist names
`localhost:7690` only.

The defaults are the same ports `docker-compose.dev-hydration.yml` publishes, so the hydration
overlay and a default dev stack cannot run at once.

## 4. The shared llama-server, or one of the stack's own

By default the backend uses the daily instance's llama-server
(`MIST_DEV_LLM_URL`, default `http://mist-llm:8080`). That server has one slot:
`LLAMA_ARG_N_PARALLEL=1` (`docker-compose.yml:159`, and the same in `deploy/daily/compose.yml`).
So dev turns queue behind the daily instance's turns, and a long dev turn delays them.

Profile `own-llm` starts `mist-llm-dev`, a llama-server of this stack's own, with no published
port. Point `MODELS_DIR` at the host's model directory and the backend at it:

```bash
MODELS_DIR=/path/to/models MIST_DEV_LLM_URL=http://mist-llm-dev:8080 \
  docker compose -f docker-compose.dev.yml -p <name> --profile own-llm up -d
```

Without `MODELS_DIR`, `mist-llm-dev` fails at container create on its placeholder path. The GPU
has 12 GB, so a second llama-server beside the daily one may not fit; stop one first.

## 5. Neo4j password

`MIST_DEV_NEO4J_PASSWORD` sets the dev graph's password for Neo4j and for both backend services.
Its default is a dev-only value; every port binds 127.0.0.1, so nothing beyond the host can
reach it.
