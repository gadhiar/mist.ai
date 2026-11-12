MIST.AI extraction service -- deploy guide (T1b, goal mist-two-loop / MIS-171)
===============================================================================

This directory builds the image for the stateless extraction service
(`backend/extraction_service/`). The service and its dedicated llama-server
deploy identically in two places, but via two SEPARATE compose files -- one
per deployment, not one file with two profiles, because interpolation and
"what starts on this machine" differ between them:

- **extraction-local**: RTX 4070 SUPER (12 GB), voice off, switched on
  manually. Overlay file `docker-compose.extraction.yml` at the repo root,
  used alongside `docker-compose.yml`. Its defaults deploy medium effort,
  which runs instead of the chat stack, not beside it; low effort beside the
  main stack takes four shell overrides (see "Medium by default (local)").
- **extraction-host**: GTX 1070 (8 GB, Pascal), a separate machine reached
  over Tailscale, running nothing else from this repo. Standalone file
  `docker/extraction/compose.host.yml`, used alone -- `docker-compose.yml`
  is never deployed there.

The two files used to be one, with the host services gated behind an
`extraction-host` profile. That broke two ways: compose interpolates every
service in a file regardless of the active profile, so the host sidecar's
required `TS_AUTHKEY` failed `docker compose config` for
`--profile extraction-local` even though no Tailscale service was active;
and the documented host command
(`-f docker-compose.yml -f docker-compose.extraction.yml --profile
extraction-host`) started `docker-compose.yml`'s whole main stack (mist-llm,
mist-neo4j, mist-backend) on the remote GTX 1070 machine as well as the
extraction services. Splitting by deployment target fixes both:
`docker-compose.extraction.yml` now holds only the local services and needs
no Tailscale key at all, and `compose.host.yml` is a standalone file that
starts ONLY its own three services.

Both deployments run the SAME pinned llama.cpp build as the main stack's
`mist-llm` service: `ghcr.io/ggml-org/llama.cpp:server-cuda-b11151@sha256:
014f721265464f38ccb247c1338d07d852c4bae7509a4b4734d07a2bbadc765c`. This is
not a second build to track -- one pin, three services (`mist-llm`,
`mist-extraction-llm-local`, `mist-extraction-llm-host`).

Running it
----------

`docker-compose.extraction.yml` is an OVERLAY: `docker compose config`/`up`
needs `-f docker-compose.yml -f docker-compose.extraction.yml` (it joins the
main stack's network) plus `--profile extraction-local` to switch it on.
`compose.host.yml` is STANDALONE: it needs no other compose file and no
profile flag -- all three of its services start on a plain `up`.

    # Local at the defaults (medium), with the chat stack down. Naming the
    # service starts it and its llama-server, not the main stack's services
    # (needs no TS_AUTHKEY):
    docker compose -f docker-compose.yml -f docker-compose.extraction.yml \
      --profile extraction-local up -d mist-extraction-local

    # Local at low effort, beside the main stack (the earlier defaults, set
    # as shell overrides):
    EXTRACTION_REASONING_EFFORT=low EXTRACTION_LLM_CTX_SIZE=16384 \
    EXTRACTION_LOCAL_NCMOE=24 EXTRACTION_LLM_TIMEOUT_SECONDS=120 \
    docker compose -f docker-compose.yml -f docker-compose.extraction.yml \
      --profile extraction-local up -d

    # Host, standalone on the GTX 1070 machine:
    docker compose -f docker/extraction/compose.host.yml up -d

**Redeploying the host after a code change.** The service image bakes the
backend code in at build time (`docker/extraction/Dockerfile`:
`COPY backend/ /app/backend/`), and `compose.host.yml` mounts no source
directory over it. So a change anywhere under `backend/` --
`backend/extraction_service/` and `backend/extraction_contract/` included --
reaches the host only through a rebuilt image. A plain `up -d` does not
rebuild: it starts the image already built, and the host keeps running the
old code with no error to say so. After pulling the new commit on the host,
rebuild, then recreate:

    docker compose -f docker/extraction/compose.host.yml build
    docker compose -f docker/extraction/compose.host.yml up -d --force-recreate

Then check `/v1/info` (HOST_1070_RUNBOOK.md step 6): a service built from
contract 1.1.0 or later reports `contract_version` `1.1.0` and the serving
config fields (`constrained_mode`, `reasoning_effort`, `temperature`,
`ctx_size`); a response without them is still the old image.

`docker compose config` and the smoke test against a running container are
the lead's job, not this worker's (no docker, no GPU, no network in this
worktree's container).

Model
-----

Both deployments ship **gpt-oss-20b** (`ggml-org/gpt-oss-20b-MXFP4.gguf`,
11.28 GiB (12.1 GB decimal), MoE with 3.6B active parameters, Harmony chat
template), the same file
`scripts/model_bench/arms.json`'s `c9` arm benchmarks. Reasoning is always
on for this model; `EXTRACTION_REASONING_EFFORT` controls how much
(`minimal`/`low`/`medium`/`high`/`xhigh`/`max`
-- `llama_server_help_b11151.txt:639-642`). The local default is `medium`
(see "Medium by default (local)"); the host file keeps `low`, the T0 brief's
default.

**Switching to the Qwen fallback (Qwen3.5-9B) is an env change only** --
no code, no compose structure change:

    EXTRACTION_MODEL_FILE=unsloth/Qwen3.5-9B-Q8_0.gguf   # scripts/model_bench arm c5's gguf
    EXTRACTION_ADAPTER=qwen
    EXTRACTION_MODEL_HASH=qwen-3.5-9b-q8-<your-suffix>   # any string; it defines the epoch

Qwen has no `-ncmoe`/CPU-MoE story (arm `c5` runs it full-card, no
`params_required: ["ncmoe"]`), so also drop `-ncmoe`/`-lm none`/`-b`/`-ub`
overrides from the llama-server command if you switch -- this repo has not
measured Qwen3.5-9B under CPU-MoE offload because the model has no MoE
experts to offload.

Environment variables
----------------------

### Extraction service (`mist-extraction-local` / `mist-extraction-host`)

Every variable below maps 1:1 to a `backend/extraction_service/settings.py`
field via `ServiceSettings.from_env()`, apart from `EXTRACTION_LLM_DEVICE`,
which compose passes on as `EXTRACTION_SERVER_DEVICE`, and the
`EXTRACTION_SERVER_<NAME>` family, which fills `server_config`. Three fields have NO default in the
dataclass itself (`llm_base_url`, `model_hash`, `model_file`) -- compose
sets all three explicitly for both deployments regardless of whether
`from_env()`'s own `os.environ.get(..., "fallback")` would otherwise supply
one, so a deploy never silently runs on a `from_env()` fallback value.

| Variable | Local default | Host default | Notes |
|---|---|---|---|
| `EXTRACTION_LLM_BASE_URL` | `http://mist-extraction-llm-local:8080` | `http://mist-extraction-llm-host:8080` | Hardcoded per deployment -- never the backend's own LLM. |
| `EXTRACTION_MODEL_HASH` | **required, no default** (`:?`) | **required, no default** (`:?`) | Defines the epoch. Set explicitly every deploy; see EPOCH_MISMATCH in `backend/extraction_service/app.py`. |
| `EXTRACTION_MODEL_FILE` | `ggml-org/gpt-oss-20b-MXFP4.gguf` | same | Passed to `LlamaServerProvider` as the OpenAI-API `model` field. |
| `EXTRACTION_ADAPTER` | `gptoss` | same | `gptoss` / `qwen` / `gemma` -- see `backend/extraction_service/adapters.py`. |
| `EXTRACTION_REASONING_EFFORT` | `medium` | `low` | Only meaningful for `gptoss`. Sets each stage's output token cap (`STAGE_CAPS` in `backend/extraction_service/engine.py`): `low` sends scope 96, extract 2048, derive 400; `medium` and `high` send extract 32768 and keep scope 96 and derive 400. Scope and derive are always sent at `low` effort, so their caps stay low's. `medium` and `high` need a context of at least 47,104 tokens (the 32,768 extract cap plus the 14,336-token prompt reserve); the local default context, 65,536, gives it, and at the host's 16,384 default the service reports no serving identity for them. See "Effort caps and the context" and "Medium by default (local)" below. |
| `EXTRACTION_LOCATION_LABEL` | `local-4070` | `host-1070` | Reported in `/v1/info`; hardcoded per deployment, not overridable via `.env`. |
| `LLAMA_CPP_BUILD` | `b11151` | same | Reported in `ResultStamps`/`/v1/info`; shared with the compose image pin. |
| `EXTRACTION_DEBUG_PORT` | `8090` | n/a (no ports published) | Local-only, `127.0.0.1` loopback; see Tailscale exposure below. |
| `EXTRACTION_LLM_TIMEOUT_SECONDS` | `1800` | `120` | Per-LLM-call timeout: it bounds one call, not a whole `/v1/extract` request. `ServiceSettings.llm_timeout_seconds`'s own dataclass default is `30.0`s. Local: 1800 is H13's medium value, since one medium extract call that runs to its 32,768-token cap costs about 651 s. Host: at low effort the ctx-16384 re-fit measured a real cold `/v1/extract` call at 51.3-51.8s (see "Host ncmoe sizing" below), and 120 gives headroom. The value also reaches the OpenAI clients (`LlamaServerProvider(..., timeout=...)`, 5 s to connect), so a setting above 600 s is no longer cut at the SDK's 600 s read default. |
| `EXTRACTION_MAX_ATTEMPTS` | `2` | same | Maximum extraction attempts per job (first call plus repair retries on unparsable output), matching `ServiceSettings.max_attempts`'s own default. |
| `EXTRACTION_CONSTRAINED_MODE` | empty | same | Overrides the adapter's `default_constrained_mode` when set. Empty is falsy in `engine.py`'s `constrained_mode or adapter.default_constrained_mode`, so it falls through to the adapter's default -- the same effective behavior as unset. The `gptoss` default is `schema` (it was `none`); see the note below the table. |
| `EXTRACTION_LLM_DEVICE` | **required, no default** (`:?`) | **required, no default** (`:?`) | The GPU llama-server runs on, e.g. `rtx-4070-super` or `gtx-1070`. Reaches the service as `EXTRACTION_SERVER_DEVICE` and keys the extraction epoch; see "Declared llama-server settings" below. |
| `EXTRACTION_SERVER_<NAME>` | built by compose | built by compose | Not set in `.env`. One per keyed llama-server setting, from the same expression as the flag; see "Declared llama-server settings" below. |

**`EXTRACTION_CONSTRAINED_MODE` and the `gptoss` default.** `GptOssAdapter`'s
`default_constrained_mode` is `schema`, not `none`. An A/B run on the
extraction host (llama.cpp b11151, gpt-oss-20b) found that under `none` the
model wraps its answer as `<|channel|>final <|constrain|>JSON<|message|>{...}`,
llama-server's peg parser rejects that wrapper, and the scope-classification
call returns HTTP 500 on every retry, so the job carries scope `unknown` with
`scope_classification_failed`. `schema` and `json_object` both returned HTTP
200 on the same call, and the extraction payload's entities and relationships
were identical under all three modes. `schema` is the stricter of the two
working modes. The host's `.env` may therefore leave
`EXTRACTION_CONSTRAINED_MODE` empty: the adapter default it falls through to
is now the working mode.

**`EXTRACTION_LLM_TIMEOUT_SECONDS` caveat for the host.** 120s covers the
measured end-to-end `/v1/extract` figures above, but `llm_timeout_seconds`
bounds each individual LLM call, and the extraction stage alone asks for
`max_tokens=2048` (`backend/extraction_service/engine.py`). At the host's
`ncmoe=13` re-fit (16.4 decode t/s, 483 prompt t/s), a call that used the
full 2048-token budget would take roughly 145s (about 19s prompt processing
plus about 125s decode) -- over the 120s generic default. The host's own
`.env` should keep `EXTRACTION_LLM_TIMEOUT_SECONDS=300` (see
`docker/extraction/HOST_1070_RUNBOOK.md`'s "Out-of-repo timeout override,
retirement" note) rather than relying on this compose-file fallback;
120 is sized for the common low-effort case. The local file defaults to
1800 for medium, and its low-effort override, 120, relies on the 4070's much
faster decode to stay clear of this worst case.

Every other `ServiceSettings` field (`idempotency_cache_size`,
`scope_enabled`, `temperature`, `port`) has a tested default in
`settings.py` and is left unset here deliberately.
`EXTRACTION_REASONING_BUDGET_TOKENS` sets `reasoning_budget_tokens`, the
reasoning budget the `gptoss` adapter puts on the extract request and the
plan call (never on scope or derive): 8,192 by default
(`DEFAULT_REASONING_BUDGET_TOKENS`), which the local compose file passes on
as `${EXTRACTION_REASONING_BUDGET_TOKENS:-8192}` and the host file leaves to
the service's own default. An integer sets another budget, and `none` sends
no budget at all (see "No `--reasoning-budget` cap" below).

**Idempotency-cache trap (operator note).** `ServiceSettings.idempotency_cache_size`
(default 256) backs a `job_id` -> `ExtractResponse` LRU cache
(`backend/extraction_service/app.py`'s `_ResultCache`, keyed on `job_id`
alone -- `cache.knows(req.job_id)` / `cache.get_or_run(req.job_id, ...)`).
Reusing the same `job_id` across repeated `/v1/extract` calls -- for
example while manually timing something, or re-running a probe script
without regenerating IDs -- returns the CACHED result instead of
re-invoking the LLM, silently. `request_id` is not part of the cache key;
it is used only for log correlation (the "job complete"/"job failed" log
lines), so reusing it alone has no effect on caching. Always use a fresh
`job_id` when timing or re-testing the service, and regenerate `request_id`
alongside it for clean log correlation.

### llama-server (`mist-extraction-llm-local` / `mist-extraction-llm-host`)

Same `LLAMA_ARG_*` env-var convention as `mist-llm` in the root
`docker-compose.yml`. Flag spellings and line numbers below cite
`tests/unit/model_bench/fixtures/host/llama_server_help_b11151.txt`, the
real `--help` capture for this build.

- `LLAMA_ARG_MODEL=/models/${EXTRACTION_MODEL_FILE}` -- same model file the
  extraction service reports, so the two never drift.
- `LLAMA_ARG_N_GPU_LAYERS=999` -- saturating sentinel (all dense layers on
  GPU), same convention `mist-llm` uses.
- `LLAMA_ARG_JINJA=1` -- jinja templating on (line 626; already the
  server's own default, set explicitly for clarity, same as `mist-llm`).
- `LLAMA_ARG_REASONING=on` (`-rea`, line 636) -- forces reasoning on rather
  than relying on `-rea auto`'s "detect from template" behavior, which this
  worker cannot verify live (no GPU/docker). Matches
  `scripts/model_bench/README.md`'s citable reasoning for arm `c9`'s own
  `-rea on`.
- `LLAMA_ARG_THINK=deepseek` (`--reasoning-format`, line 628-634) -- puts
  thoughts in `message.reasoning_content` (line 631), the exact attribute
  `backend/llm/llama_server_provider.py` reads via `getattr(message,
  "reasoning_content", None)`.

  **A/B'd on the real host build (b11151), ctx 16384.** Three configurations
  were compared: shipped flags (`LLAMA_ARG_REASONING=on`,
  `LLAMA_ARG_THINK=deepseek`); `LLAMA_ARG_THINK` simply unset; and
  `LLAMA_ARG_CHAT_TEMPLATE=gpt-oss`. The first two behave identically (HTTP
  200, extraction succeeds); the third makes the model emit garbage and
  extraction returns HTTP 502, so it is rejected. The flags stay as shipped.

  `/props` shows `reasoning_format none` and `chat_format Content-only`
  under every one of these configurations, including the working ones --
  `/props` is NOT a reliable indicator of whether the reasoning flags are
  doing anything here. Inspecting the actual completion (not `/props`)
  shows `reasoning_content` on the response message IS populated when the
  flags are set, so the flags do have an effect; the earlier "neither flag
  seemed to have any effect" note was an incorrect inference from an
  unreliable indicator.
- `-ncmoe` / `--n-cpu-moe N` (line 124) -- see "Local ncmoe sizing" and
  "Host ncmoe sizing" below.
- `-b`/`-ub 2048`, `-lm none` -- CPU-MoE prompt-processing batch sizes and
  mmap-disabled, modeled on `scripts/model_bench/arms.json`'s `c3`/`c9`
  arms (`arg_overrides`).
- `--cache-ram ${LLM_CACHE_RAM_MIB:-2048}` -- same cap and default as
  `mist-llm` (PR #17's fix for an 8192 MiB default starving the Docker VM).
- `--temp 1.0 --top-p 1.0` -- gpt-oss/`gptoss` family sampling
  (`scripts/model_bench/README.md:562-563,610-618`). **UNVERIFIED**:
  OpenAI's stated gpt-oss recommendation as relayed by that file's own T7
  brief, not independently re-derived from a primary source (no network in
  that worker's container either, per that file's own note).

**No `--reasoning-budget` cap.** The flag is absent from both llama-server
commands. The extraction service sends the budget per request instead
(`ThinkingConfig.budget_tokens`, from
`backend/extraction_service/settings.py`'s `reasoning_budget_tokens`):
8,192 tokens on the extract request and the plan call by default, another
integer when `EXTRACTION_REASONING_BUDGET_TOKENS` sets one, and none on
scope or derive. Set the variable to `none` and the field is omitted from
the outgoing request entirely (see that field's docstring), so
llama-server's OWN default governs. That default is `-1` (unrestricted,
`llama_server_help_b11151.txt:644-645`). Whether llama-server b11151
honours the per-request budget is not measured here.
`tests/unit/test_compose_extraction.py` (local) and
`tests/unit/test_compose_extraction_host.py` (host) each assert
`--reasoning-budget` is absent from their own llama-server command.

Declared llama-server settings (`EXTRACTION_SERVER_*`)
--------------------------------------------------------

The extraction service cannot see its llama-server's launch flags, and
several of them change what the model returns. So each compose file declares
every keyed llama-server setting to the extraction service as
`EXTRACTION_SERVER_<NAME>`, built from the same `${...}` expression or literal
the flag uses. The service reads every `EXTRACTION_SERVER_*` variable
(`backend/extraction_service/settings.py`, `_server_config`) into its serving
identity, which `/v1/info` reports and every result is stamped with (MIS-196).
Because one expression feeds both containers, an override in `.env` changes the
flag and its declaration together.

| Declaration | Llama-server setting | Local value | Host value |
|---|---|---|---|
| `EXTRACTION_SERVER_IMAGE` | the service's `image:` | the pinned b11151 digest | same |
| `EXTRACTION_SERVER_DEVICE` | none (the GPU) | `${EXTRACTION_LLM_DEVICE:?...}` | same |
| `EXTRACTION_SERVER_N_GPU_LAYERS` | `LLAMA_ARG_N_GPU_LAYERS` | `999` | same |
| `EXTRACTION_SERVER_CTX_SIZE` | `LLAMA_ARG_CTX_SIZE` | `${EXTRACTION_LLM_CTX_SIZE:-65536}` | `${EXTRACTION_LLM_CTX_SIZE:-16384}` |
| `EXTRACTION_SERVER_N_PARALLEL` | `LLAMA_ARG_N_PARALLEL` | `1` | same |
| `EXTRACTION_SERVER_JINJA` | `LLAMA_ARG_JINJA` | `1` | same |
| `EXTRACTION_SERVER_REASONING` | `LLAMA_ARG_REASONING` | `on` | same |
| `EXTRACTION_SERVER_THINK` | `LLAMA_ARG_THINK` | `deepseek` | same |
| `EXTRACTION_SERVER_N_CPU_MOE` | `-ncmoe` | `${EXTRACTION_LOCAL_NCMOE:-9}` | `${EXTRACTION_HOST_NCMOE:-24}` |
| `EXTRACTION_SERVER_BATCH` | `-b` | `2048` | same |
| `EXTRACTION_SERVER_UBATCH` | `-ub` | `2048` | same |
| `EXTRACTION_SERVER_CACHE_RAM_MIB` | `--cache-ram` | `${LLM_CACHE_RAM_MIB:-2048}` | same |
| `EXTRACTION_SERVER_TEMP` | `--temp` | `1.0` | same |
| `EXTRACTION_SERVER_TOP_P` | `--top-p` | `1.0` | same |

- `EXTRACTION_SERVER_IMAGE` repeats the llama-server image string. A flag left
  at its llama-server default is not declared: the default changes only with
  the build, and the image digest pins the build.
- `EXTRACTION_SERVER_DEVICE` comes from `EXTRACTION_LLM_DEVICE`, which is
  required with no default, like `EXTRACTION_MODEL_HASH`: a missing value fails
  `docker compose config`. The 1070 runs PTX JIT-compiled kernels and the 4070
  native ones from the same image, and nothing shows the two give the same
  output, so the GPU is keyed and the two hosts never share an epoch.
- The service checks `EXTRACTION_SERVER_CTX_SIZE` against the `n_ctx`
  llama-server reports on `/props`. When they differ, or when the image,
  device or ctx size is missing, `/v1/info` reports no serving identity and
  names the reason in `serving_identity_error`.
- Not declared, each for a stated reason: `LLAMA_ARG_HOST`, `LLAMA_ARG_PORT`
  (transport), `LLAMA_ARG_METRICS`, `--no-webui` (UI and metrics), `-lm none`
  (the same weight bytes, loaded without mmap), `CUDA_CACHE_PATH` and
  `CUDA_CACHE_MAXSIZE` (a cache of the same PTX compile; the device is keyed),
  and `LLAMA_ARG_MODEL`, which is keyed through `EXTRACTION_MODEL_FILE`.

**Changing any declared setting starts a new extraction epoch.** That
includes `EXTRACTION_LLM_DEVICE`, `EXTRACTION_LLM_CTX_SIZE`, the ncmoe value
and `LLM_CACHE_RAM_MIB`. After the change the backend refuses the service,
naming each setting that differs, until a new epoch is cut over with
`admin cutover begin --settings-change` (see
`backend/extraction_backlog/CUTOVER.md`).

`tests/unit/test_compose_extraction.py` and
`tests/unit/test_compose_extraction_host.py` fail when any `LLAMA_ARG_*`,
other env var or command flag of a llama-server service is neither declared
with the identical value nor in the tests' exclusion table, and when a
declaration names no llama-server setting. Adding a flag to a llama-server
command therefore means adding its declaration in the same edit.

Context size (`EXTRACTION_LLM_CTX_SIZE`, local default `65536`, host default `16384`)
---------------------------------------------------------------------------------------

`LLAMA_ARG_CTX_SIZE` used to default to `8192` on both deployments. A real
`/v1/extract` probe (a one-sentence utterance, empty history) needed 9162
prompt tokens, and the extraction engine
(`backend/extraction_service/engine.py`) asks for `max_tokens=2048` on top of
the prompt -- 9162 + 2048 > 8192, so extraction could not succeed at the old
default. Both defaults then became `16384`, which low effort fills exactly
with the current prompt reserve (see "Effort caps and the context"). The host
file keeps it. The local file defaults to `65536` for medium effort, H13's
context; see "Medium by default (local)".

The host ncmoe fit has been re-measured at ctx 16384 -- see "Host ncmoe
sizing" below for the current table.

Effort caps and the context
---------------------------

Each stage's output token cap follows the reasoning effort the adapter sends
that stage (`STAGE_CAPS`, `stage_caps` and `GptOssAdapter.stage_effort`):

| `EXTRACTION_REASONING_EFFORT` | scope | extract | derive | context needed |
|---|---|---|---|---|
| `low` (host default) | 96 | 2048 | 400 | 16,384 |
| `medium` (local default) | 96 | 32768 | 400 | 47,104 |
| `high` | 96 | 32768 | 400 | 47,104 |

- Scope is sent at `low` effort under every setting, so its cap is always
  96: it is a classification.
- Derive is sent at `low` effort under every setting, so its cap is always
  400. No effort run calls derive, so a higher effort there would ship
  unmeasured.
- `qwen` and `gemma` send no effort, so they keep `low`'s caps whatever the
  setting says.
- The context needed is the extract cap plus `PROMPT_RESERVE_TOKENS`,
  14,336: the largest value with which low's extract cap still fits a 16,384
  context (2,048 + 14,336). The largest r6 prompt measured was 9,327 tokens,
  and the r8 prompt conventions add about 2,600 more by estimate; the
  `extraction call:` line (below) reports each call's real `prompt_tokens`.
  When the `n_ctx` llama-server reports is smaller, `/v1/info` reports no
  serving identity and `serving_identity_error` names
  `extract cap <c> + prompt reserve 14336 > observed n_ctx <n>`, and a job
  that expects an identity gets 409. `medium` and `high` need
  `EXTRACTION_LLM_CTX_SIZE` of at least 47,104; the local default, 65,536,
  gives it.
- A medium extract call can take many minutes. Size
  `EXTRACTION_LLM_TIMEOUT_SECONDS` for it: the value bounds each call both in
  the engine and in the OpenAI clients. The local default, 1800, does.

Medium by default (local)
-------------------------

`docker-compose.extraction.yml` deploys medium effort by default:

| Variable | Local default | Earlier default (the override beside the main stack) |
|---|---|---|
| `EXTRACTION_REASONING_EFFORT` | `medium` | `low` |
| `EXTRACTION_LLM_CTX_SIZE` | `65536` | `16384` |
| `EXTRACTION_LOCAL_NCMOE` | `9` | `24` |
| `EXTRACTION_LLM_TIMEOUT_SECONDS` | `1800` | `120` |

- Medium needs a context of 47,104 tokens (its 32,768 extract cap plus the
  14,336-token prompt reserve); 65,536 gives it room.
- These are the settings of H13, goal mist-extraction-quality's effort run
  on the RTX 4070 SUPER, which ran with the chat stack down. On the 12 GB
  card this configuration is measured only with the chat stack down: no run
  has measured it beside the chat model (Gemma 4 E4B). So at these defaults
  the extraction stack runs instead of chat, not beside it. Start it by
  naming the service (`... --profile extraction-local up -d
  mist-extraction-local`), as under "Running it", so `up` does not start the
  main stack's services.
- To run extraction beside the main stack, set the four earlier defaults as
  shell overrides (`EXTRACTION_REASONING_EFFORT=low
  EXTRACTION_LLM_CTX_SIZE=16384 EXTRACTION_LOCAL_NCMOE=24
  EXTRACTION_LLM_TIMEOUT_SECONDS=120`), as under "Running it". The context
  size and the ncmoe value are declared llama-server settings, so the
  override starts a different extraction epoch from the defaults (see
  "Declared llama-server settings").
- `EXTRACTION_MODEL_HASH` and `EXTRACTION_LLM_DEVICE` stay required, with no
  default.

**Two timeouts, two bounds.** The service's `EXTRACTION_LLM_TIMEOUT_SECONDS`
bounds one LLM call. The backend dispatcher's client timeout,
`MIST_EXTRACTION_REQUEST_TIMEOUT_S` (default 1800 s,
`backend/extraction_backlog/settings.py`), bounds one whole `/v1/extract`
request, which can hold several capped calls: an extract batch cut short at
its cap is split in two and retried (`backend/extraction_service/engine.py`).
1,800 s covers two medium extract calls that run to their cap (about 651 s
each), not three. A turn that caps three times can run past it: the
dispatcher's client times out and leaves. A run whose only waiting client
has left is cancelled, and its streamed LLM call is closed, so no run is
left going. The dispatcher counts the timeout as a failed attempt, and its
retry sends a fresh `job_id` (`backend/extraction_backlog/dispatcher.py`),
which starts a run of its own on llama-server's single slot
(`LLAMA_ARG_N_PARALLEL=1`); the turn may be dead-lettered after
`MIST_EXTRACTION_MAX_ATTEMPTS`. Whether llama-server b11151 stops
generating as soon as the stream closes is not measured here.

**Cancelling a job.** `POST /v1/jobs/<job_id>/cancel` cancels the job's run
if one is in flight and answers 200 with a `CancelResponse` whose `status`
is `cancelled`, `finished` or `unknown`; every request waiting on a
cancelled run gets HTTP 499 with `ErrorCode.CANCELLED`, nothing is stored,
and the same `job_id` runs again when it is sent again. The `extraction call:` lines show how close a
turn comes.

Per-call token counts (operations note)
---------------------------------------

Every LLM call the service makes, at every stage, logs one INFO line:

    extraction call: cap <max_tokens> prompt_tokens <n> completion_tokens <n> finish <reason>

- `cap` is the cap the call was sent with, which names the stage.
- `prompt_tokens` and `completion_tokens` are llama-server's own counts, so
  a run records its real prompt sizes, context lines included; a count the
  server did not report prints as `None`. `finish` is the finish reason
  (`stop`, or `length` at the cap, which also logs a length-stop line).
- The line holds counts and the finish reason only, never prompt or
  response text.
- The entry point (`backend/extraction_service/__main__.py`,
  `configure_logging()`) sends the engine logger's INFO lines and every
  logger's WARNING lines to stderr; other INFO lines, such as the job
  outcomes `app.py` logs, stay off. Read the line with
  `docker logs mist-extraction-local` (or `mist-extraction-host`).

Length stops (operations note)
------------------------------

When a call stops at its output cap (`finish_reason == "length"`), the
service logs one WARNING line before the stage handles the stop:

    extraction length stop: cap <c> completion_tokens <n> reasoning_chars <r> content_chars <k> repeat <true|false>

- `cap` is the cap the call was sent with, which names the stage, since the
  three caps differ at every effort.
- `repeat` is true when the last 200 characters of the call's tail (the last
  2,000 characters of its reasoning followed by its content) occur at least
  three times in that tail, which any loop with a period of 600 characters or
  less does. True points to a repetition loop; false to reasoning that was
  still going when the cap cut it.
- The line holds counts and the flag only, never the response text, because
  production turns are real conversations. Read it with
  `docker logs mist-extraction-local` (or `mist-extraction-host`).

Local ncmoe sizing (`EXTRACTION_LOCAL_NCMOE`, default `9`)
-------------------------------------------------------------

The default, `9`, is H13's: medium effort at context 65,536 on the RTX 4070
SUPER, run with the chat stack down. That is the only condition this
configuration has run under on the 12 GB card. No run has measured it beside
the main stack's Gemma 4 E4B, so at this default the extraction stack runs
instead of chat (see "Medium by default (local)").

The one other measurement on this card is **~10.8 GiB VRAM at `ncmoe=6`**
(per the T1b brief; not independently re-measured). At that setting it
cannot sit beside E4B (clause F, `scripts/model_bench/analyse.py:1025`:
`arm_peak_mib + voice_vram_mib + margin_mib <= total_mib` -- 10.8 GiB alone
already leaves under 1.2 GiB for Gemma 4 E4B + KV cache + voice, which does
not fit under any voice allowance).

`24`, the earlier default, is **all experts on CPU**: gpt-oss-20b's MoE layer
count is **24** by the GGUF file's own metadata (`block_count`, confirmed on
the real host build; b11151's llama-server load log prints no CUDA device or
offload lines, so the metadata is the source). It was chosen, with no
measurement at any value between 6 and 24, so that extraction could sit
beside E4B on the 12 GB card, and it is the override for a low-effort run
beside the main stack (`EXTRACTION_LOCAL_NCMOE=24`). Whether medium at a
lower offload can sit beside E4B is unmeasured; a run beside the chat model
would decide it. Latency is acceptable either way -- extraction is
asynchronous, per the T1b brief.

Host ncmoe sizing (`EXTRACTION_HOST_NCMOE`, default `24`)
--------------------------------------------------------------------------

"All experts on CPU", the local file's earlier default -- `24` is verified
via GGUF metadata (`block_count`), not the load log; see "Local ncmoe
sizing" above. `24` stays the shipped compose-file default (conservative,
all-CPU); an operator opts into a tighter value via `EXTRACTION_HOST_NCMOE`
in the host's own `.env`.

**Re-fit at ctx 16384** (9,272-token prompt, `max_tokens=2048`), gpt-oss-20b
on the GTX 1070 (Pascal, 8 GB):

| `ncmoe` | peak VRAM | headroom | prompt t/s | decode t/s | `/v1/extract` |
|---|---|---|---|---|---|
| 12 | 7632 MiB | 560 MiB | 499 | 17.7 | 51.3 s |
| 13 | 7244 MiB | 948 MiB | 483 | 16.4 | 51.5 s |
| 14 | 6840 MiB | 1352 MiB | 481 | 14.9 | 51.8 s |

No swap at any of these values. Idle VRAM is about 7560 MiB at `ncmoe=12`.
The prompt cache works. Entity count varies slightly across `ncmoe` (4 vs 3
entities on a fixed test conversation); relationships are identical. Prompt
throughput across the sweep is approximately 500 t/s (the previously
recorded ~155 t/s figure was measured at the old ctx-8192/406-token setup
and is superseded by this table).

**Recommended operator setting: `ncmoe=13`, not the bare minimum
`ncmoe=12`.** This GPU also drives the host's own display -- `12` fits with
only 560 MiB spare, which is tighter than desirable when the same card is
also rendering a desktop. `13` leaves substantially more headroom (948 MiB)
for a small, predictable decode-speed cost (16.4 vs 17.7 t/s). Set
`EXTRACTION_HOST_NCMOE=13` in the host's own `.env` to opt in; the
compose-file default stays `24` until an operator does.

Tailscale exposure model (host deployment only, `compose.host.yml`)
-----------------------------------------------------------------------

The host deployment publishes **no host ports on any service**.
Exposure is entirely through a Tailscale sidecar
(`mist-extraction-ts`, image `tailscale/tailscale`, tag `stable` pinned by
digest `sha256:c507f3a2a6ab1cabd8d809b98edeb41edbd5c3fb6ad9632ffd098b4c7d0b4065`,
resolved 2026-09-26):

- `mist-extraction-host` (the FastAPI service) sets `network_mode:
  service:mist-extraction-ts`, sharing the sidecar's entire network
  namespace. Whatever port it binds (`EXTRACTION_PORT`, default `8090`) is
  reachable over the sidecar's Tailscale interface, and nowhere else --
  the service has no `ports`, `hostname`, or `networks` of its own.
- `TS_AUTHKEY` has **no default** (`:?`, same required form as
  `EXTRACTION_MODEL_HASH`). Generate a reusable or ephemeral key in the
  Tailscale admin console before `up`.
- Sidecar state (machine identity, keys) lives on the named volume
  `mist-extraction-ts-state`, so re-authenticating is not needed on every
  restart. **Volume naming note:** running `compose.host.yml` without an
  explicit project name derives the project name from the compose file's own
  directory (`extraction`, from `docker/extraction/`), and Compose prefixes
  every named volume with it -- the volume actually created on the host is
  `extraction_mist-extraction-ts-state`, not the bare name in this file's
  `volumes:` block. Check `docker volume ls` on the host for the prefixed
  name.
- `mist-extraction-llm-host` is **not** in the sidecar's network namespace
  -- it stays on the ordinary compose network, unreachable from outside
  it. `mist-extraction-host` reaches it by compose DNS
  (`mist-extraction-llm-host:8080`) because the sidecar it borrows its
  network from is ALSO attached to that same compose network (no
  `network_mode` override on the sidecar itself). llama-server never gets
  a Tailscale identity of its own.

CUDA JIT cache (host deployment only, `compose.host.yml`)
---------------------------------------------------------

The pinned image is a cuda12 build; Pascal (sm_61, the GTX 1070's
architecture) ships PTX-only in it, so the driver JIT-compiles the SASS on
first kernel load. `CUDA_CACHE_PATH=/root/.nv/ComputeCache` on the named
volume `mist-extraction-cuda-cache` (created on the host as
`extraction_mist-extraction-cuda-cache` -- see the volume naming note under
"Tailscale exposure model" above), plus a generous
`CUDA_CACHE_MAXSIZE=2147483648` (2 GiB), makes that compile a one-time cost
across container restarts rather than a repeat on every `up`.
`mist-extraction-llm-host`'s healthcheck `start_period` is 300s (5x
`mist-llm`'s 60s) to give the first, uncached boot room to finish the JIT
compile before the health check can fail it.

Local (`mist-extraction-llm-local`) has no such volume or extended
`start_period`: the RTX 4070 SUPER is sm_89, which this cuda12 image ships
native SASS for.
