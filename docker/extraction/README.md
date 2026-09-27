MIST.AI extraction service -- deploy guide (T1b, goal mist-two-loop / MIS-171)
===============================================================================

This directory builds the image for the stateless extraction service
(`backend/extraction_service/`). The service and its dedicated llama-server
deploy identically in two places, but via two SEPARATE compose files -- one
per deployment, not one file with two profiles, because interpolation and
"what starts on this machine" differ between them:

- **extraction-local**: RTX 4070 SUPER (12 GB), beside the main stack, voice
  off, switched on manually. Overlay file `docker-compose.extraction.yml` at
  the repo root, always used alongside `docker-compose.yml`.
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

    # Local, alongside the main stack (needs no TS_AUTHKEY):
    docker compose -f docker-compose.yml -f docker-compose.extraction.yml \
      --profile extraction-local up -d

    # Host, standalone on the GTX 1070 machine:
    docker compose -f docker/extraction/compose.host.yml up -d

`docker compose config` and the smoke test against a running container are
the lead's job, not this worker's (no docker, no GPU, no network in this
worktree's container).

Model
-----

Both deployments ship **gpt-oss-20b** (`ggml-org/gpt-oss-20b-MXFP4.gguf`, 12.1
GB, MoE with 3.6B active parameters, Harmony chat template), the same file
`scripts/model_bench/arms.json`'s `c9` arm benchmarks. Reasoning is always
on for this model; `EXTRACTION_REASONING_EFFORT` controls how much
(`minimal`/`low`/`medium`/`high`/`xhigh`/`max`
-- `llama_server_help_b11151.txt:639-642`). Default `low` per the T0 brief.

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
field via `ServiceSettings.from_env()`. Three fields have NO default in the
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
| `EXTRACTION_REASONING_EFFORT` | `low` | same | Only meaningful for `gptoss`. |
| `EXTRACTION_LOCATION_LABEL` | `local-4070` | `host-1070` | Reported in `/v1/info`; hardcoded per deployment, not overridable via `.env`. |
| `LLAMA_CPP_BUILD` | `b11151` | same | Reported in `ResultStamps`/`/v1/info`; shared with the compose image pin. |
| `EXTRACTION_DEBUG_PORT` | `8090` | n/a (no ports published) | Local-only, `127.0.0.1` loopback; see Tailscale exposure below. |

Every other `ServiceSettings` field (`reasoning_budget_tokens`,
`constrained_mode`, `llm_timeout_seconds`, `max_attempts`,
`idempotency_cache_size`, `scope_enabled`, `temperature`, `port`) has a
tested default in `settings.py` and is left unset here deliberately --
`EXTRACTION_REASONING_BUDGET_TOKENS` in particular MUST stay unset (see
"No reasoning-budget cap" below).

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
commands. The extraction service sends a per-request thinking budget
(`ThinkingConfig.budget_tokens`, `None` by default per
`backend/extraction_service/settings.py`'s `reasoning_budget_tokens`
field); when `None`, the field is omitted from the outgoing request
entirely (see that field's docstring), so llama-server's OWN default
governs. That default is already `-1` (unrestricted,
`llama_server_help_b11151.txt:644-645`), so a request-level `-1` and the
server's own unset-flag default resolve to the same unbudgeted behavior.
`tests/unit/test_compose_extraction.py` (local) and
`tests/unit/test_compose_extraction_host.py` (host) each assert
`--reasoning-budget` is absent from their own llama-server command.

Local ncmoe sizing (`EXTRACTION_LOCAL_NCMOE`, default `999`)
--------------------------------------------------------------

gpt-oss-20b measured **~10.8 GiB VRAM at `ncmoe=6`** (per the T1b brief;
not independently re-measured by this worker -- no GPU, no docker, no
network). At that setting it cannot sit beside the main stack's Gemma 4
E4B on a 12 GB card (clause F,
`scripts/model_bench/analyse.py:1025`: `arm_peak_mib + voice_vram_mib +
margin_mib <= total_mib` -- 10.8 GiB alone already leaves under 1.2 GiB for
Gemma 4 E4B + KV cache + voice, which does not fit under any voice
allowance).

**No repo-held measurement exists at any OTHER `ncmoe` value for this
model on this card.** Searched `scripts/model_bench/README.md`,
`arms.json`, and `analyse.py` for "10.8"/"gpt-oss"/VRAM figures; nothing
beyond the single `ncmoe=6` point above. Per the brief's own fallback
instruction for this case, the default here is **all experts on CPU**.

gpt-oss-20b's exact MoE layer count is **UNVERIFIED in this repo** -- no
source states it (grepped for "24 layers", `num_hidden_layers`, `n_layer`,
no match anywhere in the tree). Rather than hardcode an unverified layer
count as the "all experts" value, the default is `999` -- llama.cpp clamps
`-n-cpu-moe` to the model's real MoE layer count at load time, the same way
`mist-llm`'s `LLAMA_ARG_N_GPU_LAYERS=999` in the root `docker-compose.yml`
saturates at the model's real layer count without the compose file needing
to name it.

**The lead measures the real fitting value on the 4070, beside the running
E4B, and records it as the new default here** (replace `999` with the
measured minimum `ncmoe` that satisfies clause F, or leave `999` if even
"all experts on CPU" is the answer). Latency is acceptable for this
setting either way -- extraction is asynchronous, per the T1b brief.

Host ncmoe sizing (`EXTRACTION_HOST_NCMOE`, default `999`, PROVISIONAL)
--------------------------------------------------------------------------

Same "all experts on CPU" default and same reasoning as local, but marked
**provisional**: the GTX 1070 host's RAM and CPU are not known to this
worker at all (no access to that machine), so even the qualitative
starting point (host RAM being enough for a 20B model's CPU-held experts)
is unverified, not just the exact fitting number. The lead measures on the
actual host once it is reachable and updates this default.

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
  restart.
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
volume `mist-extraction-cuda-cache`, plus a generous
`CUDA_CACHE_MAXSIZE=2147483648` (2 GiB), makes that compile a one-time cost
across container restarts rather than a repeat on every `up`.
`mist-extraction-llm-host`'s healthcheck `start_period` is 300s (5x
`mist-llm`'s 60s) to give the first, uncached boot room to finish the JIT
compile before the health check can fail it.

Local (`mist-extraction-llm-local`) has no such volume or extended
`start_period`: the RTX 4070 SUPER is sm_89, which this cuda12 image ships
native SASS for.
