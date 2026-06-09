# WebSocket Gauntlets Runbook (V6 / V7 / V8)

How to run the live-server WebSocket-driven evaluation gauntlets cleanly,
including the operational env-var traps that have repeatedly cost
re-runs during continuous-usage hardening.

## Drivers (host-side, gitignored under `.local/`)

| Driver | Probes | Inputs | Scorer |
|---|---|---|---|
| `.local/v6_websocket_driver.py` | 30 conversational turns | `data/ingest/v6-inputs.jsonl` | qualitative (response review + Neo4j inspection) |
| `.local/v7_websocket_driver.py` | 25 tool-selection probes | `data/ingest/v7-tool-heavy-inputs.jsonl` | `scripts/eval_harness/score_v7_probe_run.py` |
| `.local/v8_websocket_driver.py` | 20 extraction-edge probes | `data/ingest/v8-edge-production-inputs.jsonl` | `scripts/eval_harness/score_v8_probe_run.py` |

All three drivers connect to `ws://localhost:8001/ws` and use the
hardcoded server-side `session_id="default"`. Driver source lives at
`.local/` (gitignored, host-side); commit production-relevant changes
to drivers via this runbook + design docs (`v7_probe_set_design.md`,
`v8_probe_set_design.md`), not via the driver files themselves.

**Two distinct session-id conventions in this runbook -- do NOT
conflate them:**

| Path | session_id used | Where set | Score command |
|---|---|---|---|
| WebSocket drivers | `default` (server hardcoded in `voice_processor.py`) | server-side; cannot be overridden by the driver | `--session-id default` |
| `mist_admin replay` (chat-path) | caller-supplied via `--session-id <tag>` | the replay process; arbitrary string | `--session-id <same tag>` |

Reading the WebSocket gauntlet section below: every score command
takes `--session-id default`. Reading the chat-path replay section:
every score command takes the same `--session-id <tag>` you passed
to `mist_admin replay`. They are not interchangeable.

## Prerequisites: WebSocket driver settings

Each driver MUST disable WebSocket keepalive pings — otherwise turns
that exceed ~20s under GPU contention will trigger
`ConnectionClosedError: keepalive ping timeout` and abort the run.

```python
async with websockets.connect(
    WS_URL,
    max_size=8 * 1024 * 1024,
    ping_interval=None,
    ping_timeout=None,
) as ws:
    ...
```

Verified set as of 2026-05-08 in all three `.local/v{6,7,8}_websocket_driver.py`.
If you create a new driver, copy this connect signature.

## Step 1: Recreate the backend with debug instrumentation

The V7 / V8 scorers join input probes against the backend's per-turn
`MIST_DEBUG_JSONL` records. The volume-mounted `docker-compose.override.yml`
exposes the env vars but defaults them to empty, so a fresh `docker
compose up` produces no debug records. Recreate explicitly with the
env vars set.

**The exact command (use Git Bash on Windows):**

```bash
MSYS_NO_PATHCONV=1 \
  TTS_ENABLED=false \
  MIST_DEBUG_JSONL=/app/data/runtime/v678-<run-tag>.jsonl \
  MIST_DEBUG_LLM_JSONL=1 \
  docker compose up -d --force-recreate mist-backend
```

**Three env-var traps to NOT skip:**

1. `MSYS_NO_PATHCONV=1` is required on Git Bash when passing
   unix-absolute paths or POSIX-style env values to `docker compose`.
   Without it, MSYS rewrites `/app/data/runtime/...` to
   `C:/Program Files/Git/app/data/runtime/...` BEFORE compose sees the
   value, and the backend writes to the wrong path. The prefix is
   required on `up` (not just `exec`).

2. `TTS_ENABLED=false` is required for any validation run that does
   not exercise voice. With TTS enabled and rendering taking 60-180s
   per turn, single turns can cross the 180s `handle_message_streaming`
   bridge timeout, deadlocking the backend mid-gauntlet.

3. `--force-recreate` wipes the in-memory `ConversationHandler.sessions`
   dict (which holds the "default" session's turn history). Without
   this, prior gauntlet runs' history bleeds into the current run's
   tool-decision context. For a clean V6 -> V7 -> V8 sequence, recreate
   between V6 and V7 if you want isolated tool-decision conditions
   (the contamination effect is real and reproducible — see
   workstream `mist-ai-voice-chat-path-unification` 2026-05-08 notes).

Then wait for healthy:

```bash
until curl -s -m 3 http://localhost:8001/health 2>/dev/null \
    | grep -q "models_loaded.:true"; do sleep 3; done
```

**Verify the env vars actually applied.** If `MSYS_NO_PATHCONV=1` was
omitted on the `up -d` line, the backend will have a path-rewritten
`MIST_DEBUG_JSONL` value pointing at `C:/Program Files/Git/app/...`,
and your gauntlet will write to the wrong file. Confirm before
running:

```bash
docker compose exec mist-backend printenv MIST_DEBUG_JSONL MIST_DEBUG_LLM_JSONL
# Expect: /app/data/runtime/v678-<run-tag>.jsonl  AND  1
# If you see C:/Program Files/Git/app/... -- recreate with the prefix.
```

## Step 2: Run drivers in sequence

```bash
# V6 — conversational gauntlet (~14 min: 30 turns x ~10s + 18s grace + 30s settle)
python -u .local/v6_websocket_driver.py

# V7 — tool-selection gauntlet (~5 min: 25 turns x ~7s + 6s grace + 15s settle)
python -u .local/v7_websocket_driver.py

# V8 — extraction-edge gauntlet (~5 min: 20 turns x ~5s + 8s grace + 30s settle)
python -u .local/v8_websocket_driver.py
```

Drivers run sequentially against the same `session_id="default"`. The
chronological order matters: V6 populates conversational history; V7
runs against that history (testing tool-decision robustness to prior
context); V8 runs after V6+V7 with extraction-focused probes.

For session-isolated runs (e.g., apples-to-apples comparison against
a chat-path baseline), recreate the backend between drivers using the
Step-1 command.

## Step 3: Score V7 / V8

```bash
# V7
MSYS_NO_PATHCONV=1 docker compose exec -T mist-backend \
  python scripts/eval_harness/score_v7_probe_run.py \
  --input data/ingest/v7-tool-heavy-inputs.jsonl \
  --debug-jsonl /app/data/runtime/v678-<run-tag>.jsonl \
  --session-id default

# V8
MSYS_NO_PATHCONV=1 docker compose exec -T mist-backend \
  python scripts/eval_harness/score_v8_probe_run.py \
  --input data/ingest/v8-edge-production-inputs.jsonl \
  --debug-jsonl /app/data/runtime/v678-<run-tag>.jsonl \
  --session-id default
```

V6 is qualitative — review the driver's output JSONL for empty
responses, emoji leakage, AI-slop patterns, and average response
length. Quick metric:

```bash
python -c "
import json
with open('data/ingest/v6-<run-tag>-websocket-output.jsonl') as f:
    rows = [json.loads(l) for l in f]
total_chars = sum(len(r.get('response','')) for r in rows)
empty = sum(1 for r in rows if not r.get('response'))
emoji_count = sum(1 for r in rows for c in r.get('response','')
                  if 0x1F300 <= ord(c) <= 0x1F9FF or 0x2600 <= ord(c) <= 0x27BF
                  or 0x1FA70 <= ord(c) <= 0x1FAFF)  # Unicode 13+ supplementary plane
errors = sum(1 for r in rows if r.get('error'))
print(f'turns: {len(rows)}, empty: {empty}, errors: {errors}, '
      f'emoji chars: {emoji_count}, avg chars: {total_chars/len(rows):.0f}')
"
```

## Isolating eval runs from live memory (throwaway-trio)

Any driver that exercises the memory layer writes into three live stores:
the event store DB, the vault corpus, and the sidecar index DB. Run against
the live container defaults and the run's synthetic utterances land in the
user's canonical memory permanently (this is how the synthetic `37A8`
corpus polluted production memory). Redirect all three to a per-run
throwaway directory before invoking the driver so live memory is untouched.

**The integration point is `scripts/mist_admin.py` (`replay` / `chat`).**
Those subcommands call `get_config()` -> `build_conversation_handler(...)`,
which drives the full retrieval + extraction + graph + (server-path) vault
pipeline. `scripts/eval_harness/run.py` is a llama-server-only model A/B
harness -- it never imports `backend/` and never touches the memory layer,
so it needs no isolation and is NOT where these env vars go.

The three stores and their override env vars (read by `*.from_env()` in
`backend/knowledge/config.py`; defaults in `docker-compose.yml` use the
`${VAR:-default}` form so they resolve to the live paths when unset):

| Store | Env var | Live default |
|---|---|---|
| Sidecar index DB | `MIST_SIDECAR_DB_PATH` | `/app/data/vault_sidecar.db` |
| Event store DB | `EVENT_STORE_DB_PATH` | `/app/data/event_store.db` |
| Vault root | `MIST_VAULT_ROOT` | `/app/mist-memory` |

Set all three to one throwaway dir for the run. Pass them with `-e` on
`docker compose exec` so they apply to the spawned process only -- the
live backend container's env is unchanged and stays up:

```bash
MSYS_NO_PATHCONV=1 docker compose exec -T \
  -e MIST_SIDECAR_DB_PATH=/app/data/eval-run/vault_sidecar.db \
  -e EVENT_STORE_DB_PATH=/app/data/eval-run/event_store.db \
  -e MIST_VAULT_ROOT=/app/data/eval-run/vault \
  mist-backend python scripts/mist_admin.py replay \
  data/ingest/<corpus>.jsonl \
  --output data/ingest/<corpus>-eval-output.jsonl \
  --session-id eval-<tag>
```

Notes:

- Use a single throwaway dir under `/app/data/` (bind-mounted, so artifacts
  survive on the host for inspection) and pick a fresh `eval-run/<tag>/`
  per run for clean isolation. `data/` is gitignored; the throwaway stores
  are disposable -- delete the dir when done.
- `get_config()` memoizes a module-global `_config` and `load_dotenv()`
  runs at import, so the overrides must be in the process env BEFORE
  `backend.knowledge.config` is first imported. The `-e` flags on `docker
  compose exec` satisfy this -- they are set on the new process before
  Python starts, so the first `get_config()` reads the throwaway paths.
- The live container is not recreated and the live backend's env is not
  modified -- only the `exec`-spawned process sees the overrides.

## Apples-to-apples chat-path replay (no live server required)

For session-isolated comparisons, run V7 via `mist_admin replay`
inside the backend container. This spawns a fresh `ConversationHandler`
with no shared session state, no WebSocket layer, and no TTS pipeline.

```bash
MSYS_NO_PATHCONV=1 docker compose exec -T \
  -e MIST_DEBUG_JSONL=/app/data/runtime/v7-chat-replay-<tag>.jsonl \
  -e MIST_DEBUG_LLM_JSONL=1 \
  mist-backend python scripts/mist_admin.py replay \
  data/ingest/v7-tool-heavy-inputs.jsonl \
  --output data/ingest/v7-chat-replay-<tag>-output.jsonl \
  --session-id v7-chat-replay-<tag>
```

Score with `--session-id v7-chat-replay-<tag>` to match the replay's
session_id (not "default" — that's the WebSocket-path session).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Backend log shows `Debug JSONL logging enabled at C:/Program Files/Git/app/...` | MSYS path-rewriting hit `MIST_DEBUG_JSONL` value | Add `MSYS_NO_PATHCONV=1` to the `docker compose up` line (not just `exec`) |
| Driver dies mid-run with `ConnectionClosedError: keepalive ping timeout` | Default `ping_interval=20s` killed the connection during a slow turn | Verify `ping_interval=None, ping_timeout=None` in the driver's `websockets.connect` |
| Single turn crosses 180s and the backend hangs | TTS rendering crossed the `handle_message_streaming` bridge timeout | Recreate backend with `TTS_ENABLED=false` for validation runs |
| Scorer reports `0/N matched against debug JSONL` | Backend env didn't apply `MIST_DEBUG_JSONL` (running container started before env was set) OR voice_processor didn't wrap the LLM provider with `InstrumentedStreamingLLMProvider` | Recreate backend with `MIST_DEBUG_JSONL=...` set; verify backend logs show "LLM provider wrapped with observability instrumentation" |
| V7 fails with FP that doesn't appear on chat-path replay | V6/V7/V8 share `session_id="default"`; prior turns' history is in scope | Recreate backend between V6 and V7 to wipe sessions; OR accept and document as session-contamination edge case |

## References

- `scripts/eval_harness/v7_probe_set_design.md` — V7 probe set + acceptance criteria
- `scripts/eval_harness/v8_probe_set_design.md` — V8 probe set + acceptance criteria
- `knowledge-vault/Projects/mist-ai/workstreams/mist-ai-voice-chat-path-unification.md` — origin of the V6/V7/V8 WebSocket-driven gauntlet pattern
- 2026-05-07 transfer notes (vault session note) — codify ops learnings from re-baseline
- 2026-05-08 transfer notes (vault session note) — V7 fix + vault-only refactor + this runbook
