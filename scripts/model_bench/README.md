# mist-model-bench: host driver (T3a)

`bench_host.py` is the host-side driver for the mist-model-bench goal. It stops MIST's live
production containers, serves a pinned llama-server build under the container name
`mist-bench-llm`, runs benchmark suites against it, and restores the production stack exactly. It
runs on the Windows host in a no-gaming session (Python 3.11.9); its `voice` subcommand shells out
to a copy of `probes/voice_vram.py` running inside the `mist-backend` container (Python 3.11.0rc1).

Invocation, from the repo root:

```
python -m scripts.model_bench.bench_host <subcommand>
```

Stdlib only, everywhere in this package, except `probes/voice_vram.py`, which runs inside the
mist-backend container and may import torch and backend/`src` modules lazily. No subprocess call
in this package builds a shell string: every `docker`/`git`/`nvidia-smi`/`python -m ...` invocation
is an argv list, `shell=False` (the `subprocess` default).

## The S1-S3 host procedure, in order

1. **`snapshot --run R`** -- `docker inspect` `mist-neo4j`, `mist-llm`, `mist-backend` and write
   `<results-root>/<run>/session/snapshot.json`. Refuses to overwrite an existing snapshot.
2. **`stop --run R [--with-neo4j]`** -- refuses without a snapshot for `R`. Stops `mist-backend`
   and `mist-llm` via `docker stop` (never `docker compose`); add `--with-neo4j` to also stop
   `mist-neo4j` (required before serving arms that declare `"stop_neo4j": true`, e.g. c3/c4 --
   their MoE loads need the VRAM Neo4j's own container would otherwise hold onto).
3. **`serve <arm> --run R [--param k=v ...]`** -- requires `--results-root`. Refuses if `mist-llm`
   is running, if a REQUIRED param (e.g. c3/c4's `ncmoe`) is missing, if the arm needs `stop_neo4j`
   and `mist-neo4j` is still running, or if a `mist-bench-llm` container already exists -- running
   OR exited (run `unserve` first). A pre-flight check that could not determine whether
   `mist-bench-llm` exists (see "Container state: three positive results, one honest unknown"
   below) refuses too, with a message to retry, rather than either assuming it is clear or assuming
   it is taken.

   Starts `mist-bench-llm` via
   `docker run -d --name mist-bench-llm ... -p 127.0.0.1:8080:8080` (no `--rm`: the driver needs to
   `docker logs` a container that fails to start before removing it itself) and waits for `/health`,
   polling the container's docker state alongside `/health` on every poll (default timeout 900s --
   CPU-MoE loads, c3/c4, are slow). If the container has exited or been removed (e.g. the pinned
   build rejected a flag -- the exact S2 failure), `serve` does not wait out the timeout: it prints
   `[FAIL]` with the exit code (or "no longer exists"), saves `docker logs` to
   `<results>/<run>/<arm>/serve_failed_<UTC>.log`, removes the container (skipped if it no longer
   exists), and exits non-zero within a few seconds. If the container's state instead cannot be
   determined (docker inspect failing, timing out, or returning something unparseable -- e.g. a
   slow docker CLI/daemon under low host memory, the exact S2 false positive this replaced) for
   longer than 180s with no confirmed reading in between, `serve` prints `[FAIL]` with a "could not
   be determined" message, tries to save `docker logs` (a `[WARN]` if that itself fails), and
   deliberately does NOT stop or remove the container, since it may be loading normally -- check it
   with `docker ps -a --filter name=mist-bench-llm` and run `unserve` yourself if you want it gone.
   A `/health` timeout with the container confirmed still running keeps its prior behaviour
   (propagates uncaught, non-zero exit) but saves the same `serve_failed_<UTC>.log` first.
4. **`run <arm> --run R [--suites ...] [--layout-pass screen|finalist] [--tuning-label X] [--rep K]
   [--param k=v]`** -- before running anything, confirms the served `mist-bench-llm` container IS
   `<arm>` (its `docker inspect` `Args` match what this driver would have built, and `/props`'
   model path ends with the arm's gguf filename); refuses otherwise. Requires
   `scripts/model_bench/decision_rules.json` to exist, be tracked in git, and be clean (another
   worker adds that file; this driver only checks the path and records its sha256). Starts the 5 Hz
   nvidia-smi sampler into `vram.csv`, runs the requested suites (default: the arm's own suite
   list) in the fixed order ttft -> correctness -> harness -> layout -> extraction, writes
   `meta.json` after each suite, and stops the sampler in a `finally` block. Refuses to overwrite
   any existing suite output -- pick a new `--run`, `--layout-pass`, or `--rep` instead.

   **`extraction` (T6, universal).** MIST's only heavy workload is semantic NLP extraction
   quality, so every arm is also measurable on MIST's own gold-labelled extraction gauntlet --
   entity typing accuracy, relation precision/recall -- regardless of whether that arm declares
   `extraction` in `arms.json`'s `suites` list (`UNIVERSAL_SUITES = ("extraction",)` in
   `validate_run_suites`; `run <arm> --run R --suites extraction` always works). A `run` with no
   `--suites` still runs exactly the arm's own declared suites -- `extraction` is an allowance, not
   a default addition. `cmd_run`'s PREFLIGHT resolves and validates the snapshot's `mist-backend`
   image (`resolve_backend_image_ref`) before any suite runs and before any file is written when
   `extraction` is among the requested suites -- a missing or incomplete `session/snapshot.json`
   prints a clean `[FAIL]` and leaves `meta.json` / `vram.csv` untouched, the same as every other
   preflight refusal. The suite then runs in a FRESH `docker run --rm` container of the resolved
   mist-backend image, sharing `mist-bench-llm`'s network namespace (`--network
   container:mist-bench-llm`, so `127.0.0.1:8080` inside that container reaches the served arm),
   `-e PYTHONHASHSEED=0 -e MIST_FIXED_CLOCK=2026-06-13T00:00:00+00:00` (determinism pins --
   `PYTHONHASHSEED` is read only at interpreter startup, so it MUST be a `docker run -e`, not an
   in-process env write; `MIST_FIXED_CLOCK` matches
   `scripts/eval_harness/extraction_probe_set_design.md:136`'s pin), the repo mounted read-only at
   `/work`, and the arm's results directory mounted writable at `/out`
   (`build_extraction_container_argv`). It never touches the production `mist-backend` container:
   no `exec`, no `start`, and the network/image targets are always the bench container and a
   resolved image id, never the literal `mist-backend` name.

   Inside that container, `python -m scripts.model_bench.probes.extraction` drives MIST's
   PRODUCTION extraction path -- the exact code `scripts/mist_admin.py replay --extraction-only`
   uses (`run_extraction_only_replay`, imported unchanged), through
   `backend.factories.build_conversation_handler` wired with the real `LlamaServerProvider`
   pointed at `--base-url` and the unit tier's injectable fakes for `graph_store` /
   `vector_store` (`tests/mocks/neo4j.FakeNeo4jConnection`,
   `tests/unit/knowledge/conftest.FakeVectorStore`) -- no Neo4j. The embedding provider is this
   module's own `HashSeededUnitEmbeddingProvider`, not
   `tests/unit/knowledge/conftest.FakeEmbeddingProvider`: the latter's tiled, all-positive vectors
   spuriously collide under cosine similarity, which silently gates a negative control
   (`ext-11-smalltalk-negative`) below the significance threshold before it ever reaches the LLM,
   for every arm (see the class docstring). Before Stage 2's LLM call, `run_probe` also overrides
   `RATE_LIMIT_MAX_PER_MINUTE` to `len(gold_probes) + 1` -- the production default (30,
   `backend/knowledge/config.py:154`) exists to protect a live conversational session's request
   budget, not a bulk gold-corpus replay, and would otherwise silently drop probes past the 30th.
   The gold corpus is `data/ingest/extraction-gold-2026-06-14.jsonl` (60 probes, adjudicated
   against ontology v1.4.0; see `scripts/eval_harness/extraction_probe_set_design.md`). Scoring is
   `scripts/eval_harness/score_extraction_run.py`, imported and called unchanged -- this suite
   never reimplements or edits the scorer. Outputs, written under the arm's results directory
   (never overwritten -- pick a new `--run` instead):
   - `<arm>/extraction.jsonl` -- one row per gold probe: id, matched/errored flag, per-item
     entity/relationship FP/FN counts.
   - `<arm>/extraction_summary.json` -- the scorer's own aggregate metrics (entity P/R, relation
     P/R, typing accuracy, RELATED_TO rate, valid-time accuracy) plus a Wilson interval per metric
     and a cluster-by-probe-id bootstrap CI (seed/B/confidence read from `decision_rules.json`'s
     `statistics.bootstrap` block, read-only), the ontology version, the gold corpus's sha256,
     `complete` / `unmatched_probe_ids`, and the effective `env` values (rate limit,
     `PYTHONHASHSEED`, `MIST_FIXED_CLOCK`).

   **Fail closed on a partial match.** If fewer probes are matched than exist in the gold corpus
   (`matched_probes != total_probes` -- a broken join, a throttled probe, anything the scorer
   could not join to a debug record; the same invariant `score_extraction_run.py`'s
   `_gates_pass` checks at line ~812), `extraction_summary.json` is still written -- with
   `complete: false` and `unmatched_probe_ids` -- but `python -m
   scripts.model_bench.probes.extraction` exits non-zero. `cmd_run` then records an `errors` entry
   and does NOT add `extraction` to `suites_completed`, so a subsequent call can retry it; the
   suite is never silently recorded as done on a partial run.

   `analyse.py`'s "Extraction quality (report-only, not a pre-registered rule)" section (after
   `Exploratory`, before `Finalist candidates`) renders each arm's `extraction_summary.json` with
   its CIs and a delta against `c0` in the same run, when both are present and complete. It
   renders NO verdict and never feeds `rules` / `exploratory_rules` -- an arm's
   `extraction_summary.json` absence renders as "no extraction_summary.json for this arm in this
   run", not a `[FAIL]"; an incomplete one (`complete: false`) renders as "incomplete: M/N probes
   matched; unmatched: [...]", with no metrics row (a partial run's precision/recall are not a
   comparable measurement, and an incomplete `c0` is not used as a delta baseline either).
5. **`unserve --run R --arm A`** -- saves `docker logs mist-bench-llm` (stdout and stderr) to
   `<arm>/server.log`, `docker stop`s `mist-bench-llm` unless its state is confirmed already
   exited, then always `docker rm`s it (needed now that `serve` no longer passes `--rm`) -- this
   handles a running container, an already-exited one, and one whose state could not be confirmed
   (attempts `docker stop` anyway rather than silently skipping it; a real failure there is
   reported, not swallowed).
6. **`restore --run R`** -- refuses if a `mist-bench-llm` container exists at all -- running OR
   exited (run `unserve` first; the name is taken either way) -- or if that check could not
   determine whether it exists (retry). `docker start`s all three
   production containers, waits for `mist-llm`'s `/health` and `/props` and for `mist-backend`'s
   `State.Health.Status == healthy`, re-inspects, and diffs against the snapshot on `Id`, `Image`,
   `Config.Cmd`, `Config.Env`, and `HostConfig`. Writes
   `session/restore_diff_<UTC>.json`, prints `restore diff: empty` or the diff, and exits non-zero
   on any difference (an `Id` change means Docker recreated the container -- always a failure) or
   an unhealthy service.

### Container state: three positive results, one honest unknown

`probe_container_state(name)` (`bench_host.py`) is how `serve`, `unserve`, and `restore` all ask
"what is `mist-bench-llm` doing" -- it runs `docker inspect -f '{{json .State}}' <name>` with its
own 15s subprocess timeout and classifies the result into exactly one of four states, never raising
on a docker failure:

- **`running`** -- inspect succeeded and parsed `State.Status` is anything other than
  `exited`/`dead` (`running`, `created`, `restarting`, ...).
- **`exited`** -- inspect succeeded and `State.Status` is `exited` or `dead`; carries `ExitCode`.
- **`absent`** -- inspect failed and its stderr says "no such object" or "no such container"
  (case-insensitive; docker uses both spellings across subcommands) -- the container is confirmed
  gone.
- **`unknown`** -- everything else: a non-zero exit without either "no such" phrase (including one
  with empty stderr -- the exact S2 failure: a slow docker CLI/daemon under low host memory printed
  nothing distinguishing), a `subprocess.TimeoutExpired`, a missing `docker` binary, empty stdout,
  or output that does not parse as JSON with a usable `Status` field.

`running`, `exited`, and `absent` are all positive results -- the driver knows what happened.
`unknown` is not: it means docker itself could not answer. At 0d3aa54 that failure escaped
`serve` as a `DockerError` and was reported as a failed serve, for a container that was actually
55 seconds into a slow 16 GB model load. The driver itself did not remove that container.

`wait_for_llama_health`'s handling of each state during `serve`'s /health wait:

- `running` -- keep waiting.
- `exited` or `absent` -- raise immediately (`ContainerExitedError`); waiting further cannot help.
- `unknown` -- print a rate-limited `[WARN]` (at most once per 30s) and keep waiting, UNLESS the
  state has been unknown continuously, with no confirmed reading in between, for more than 180s
  (`unknown_limit_s`, minimum 120s), in which case `serve` raises `ContainerStateUnknownError`,
  prints `[FAIL] <container> state could not be determined for <N>s`, tries to save `docker logs`
  (a `[WARN]` if that itself fails), and -- unlike a confirmed exit -- does NOT stop or remove the
  container, because it may be healthy. If this happens: check the container's real state yourself
  with `docker ps -a --filter name=mist-bench-llm`, and run `unserve` only if you actually want it
  gone.

Two more subcommands sit outside this per-arm loop: `vram-step --run R --label L [--seconds 10]`
(sample nvidia-smi for a labeled step -- the lead's labels are `desktop`, `mist_llm`,
`backend_idle`, `voice_peak`) and `voice --run R` (run the STT+TTS VRAM probe inside
`mist-backend`; see "Voice probe" below).

Two subcommands need neither docker nor a live server, and are meant to be run in the container
CI/dev-loop, not on the host:

- **`plan [--host-checks]`** -- dry run, no docker. Loads and validates `arms.json` (every arm
  resolves, no unknown keys, REQUIRED params are declared with a matching arg template, no base-
  inheritance cycle), then prints each arm's resolved `docker run` argv and suites, ending with
  `plan ok`. Passes with no env set (missing dirs print `[WARN]`, not a failure) -- and, since
  `docker-compose.yml`'s mist-llm digest pin is being added on a separate task branch, an
  unresolvable `compose:mist-llm`/`snapshot:mist-llm` image ref is also only a `[WARN]` here, never
  a `plan` failure. `--host-checks` (lead-only, on the host) additionally runs `docker version`, one
  `nvidia-smi` query through this driver's parser, `python -c "import yaml, openai, httpx"`, and a
  flag check against the pinned build: it runs `docker run --rm <pinned compose:mist-llm image>
  --help`, parses every flag spelling (`probes/help_flags.py`) out of that text, then checks every
  arm's built argv (every REQUIRED param filled with a dummy value) against it, printing `[ok]` or
  `[fail] arm <id>: unknown flags: [...]` per arm. If the help text parses to zero flags, that is
  itself a `[fail]`, not a silent pass. c0-old is skipped there with `[skip]`: it targets the b8808
  snapshot build (`snapshot:mist-llm`), a different llama.cpp build than the pinned image this check
  fetches `--help` from, and checking its argv against the wrong build's flags would be meaningless.
  Before running `--host-checks`, save the real `--help` output for the record (it is not committed
  -- raw host output belongs under `--results-root`, like everything else in "Results layout"
  below), e.g.:
  ```
  docker run --rm <pinned compose:mist-llm image ref> --help > <results-root>/<run>/session/llama_server_help_<UTC>.txt
  ```
  The lead captured this on 2026-09-24 against the pinned digest
  `ghcr.io/ggml-org/llama.cpp:server-cuda-b11151@sha256:014f7212...dc765c` (730 lines). A copy is
  committed as a *test* fixture, `tests/unit/model_bench/fixtures/host/llama_server_help_b11151.txt`
  -- see "Fixtures are hand-built, not recorded" below for why that one fixture is an exception.
- **`selftest`** -- no docker, no network. Exercises argument building for every arm (including
  inheritance and the REQUIRED-param refusal), the restore diff (empty, non-empty, Id-changed), the
  compose image parser (including the missing-digest refusal), the results-root-inside-repo
  refusal, and every probe parser against `tests/unit/model_bench/fixtures/host/`. Prints
  `selftest ok`.

## Environment variables (flag always wins over env)

| Flag | Env fallback | Meaning |
|---|---|---|
| `--models-dir` | `MODELS_DIR` | mounted `:ro` at `/models` inside `mist-bench-llm` |
| `--layout-dir` | `MODEL_BENCH_LAYOUT_DIR` | command-center layout spike dir (`run_host.py`, `analyse.py`) |
| `--results-root` | `MODEL_BENCH_RESULTS_ROOT` | MUST resolve outside this git work tree (see below) |

`--results-root` is checked with `Path.resolve()` + `Path.relative_to()`, not string prefix
matching, so a sibling directory that merely starts with the repo directory's name (e.g.
`mist.ai-results` next to `mist.ai`) is correctly treated as outside. This is how decision 8 is
enforced structurally: mist.ai is a public repository, and raw benchmark logs never land in the
git work tree.

## Results layout

```
<root>/<run>/session/snapshot.json          docker inspect of mist-neo4j, mist-llm, mist-backend
<root>/<run>/session/restore_diff_<UTC>.json
<root>/<run>/session/vram_steps.json        {"steps":[{label, median_mib, max_mib, total_mib, samples, t_utc}], "voice_probe": {...}|null}
<root>/<run>/<arm>/meta.json
<root>/<run>/<arm>/harness/harness/<candidate>.jsonl   (harness run with --results-dir <arm>/harness --run-name harness)
<root>/<run>/<arm>/layout/<pass>/{calls.jsonl, graded.jsonl, manifest.json}   pass = screen | finalist
<root>/<run>/<arm>/ttft.jsonl
<root>/<run>/<arm>/vram.csv
<root>/<run>/<arm>/correctness.r<K>.jsonl
<root>/<run>/<arm>/server.log
<root>/<run>/<arm>/serve_failed_<UTC>.log      docker logs of a serve that failed (container exited/absent, its state was undetermined, or /health timed out)
<root>/<run>/session/llama_server_help_<UTC>.txt   lead-saved `llama-server --help` output for --host-checks (not committed)
```

**Raw outputs are never committed.** `server.log`, `serve_failed_<UTC>.log`, the harness's
per-candidate JSONL, `calls.jsonl`, `vram.csv`, `ttft.jsonl`, and `correctness.r<K>.jsonl` all live
under `--results-root`, which the guard above forces outside the git work tree. Nothing under
`<results-root>` is part of this repository; only `arms.json` (server configs) and
`decision_rules.json` (another worker's, referenced by path and sha256 only) are tracked.

**After a failed `serve`,** read `<results-root>/<run>/<arm>/serve_failed_<UTC>.log` -- it holds
`docker logs`' stdout and stderr from `mist-bench-llm`. Whether the container still exists depends
on why `serve` failed: a confirmed exit removes it (an already-absent one has nothing to remove
either way), but a "state could not be determined" failure leaves it exactly as it was -- it may
still be loading normally. Check `docker ps -a --filter name=mist-bench-llm` yourself in that case,
and run `unserve` if you want it gone (also the fix if `serve` instead refused up front because a
stale container from an earlier attempt is still present).

## arms.json

Single source of truth for per-arm server configuration -- see the file itself, `docker-compose.yml`
(mist-llm's production args, which the common server args mirror), and the T3a brief for the full
arm table. Arms may set `"base": "<other-arm-id>"` to inherit that arm's resolved config wholesale,
then override individual keys (used both for the tuning arms a1-a4, which inherit their base's
server config exactly, and for the thinking-on variants c1-256/c1-512/c2-think512/c3-think512/
c4-think512).

An arm may also set `"arg_overrides"`, a `{flag: value}` map applied in place over `common_args`
(e.g. c3/c4's `{"-b": "2048", "-ub": "2048"}`, overriding the common `-b 1024 -ub 512` for CPU-MoE
prompt processing) instead of appending a second occurrence of the flag -- `build_server_args`
refuses an override naming a flag `common_args` does not have. `arg_overrides` is inherited through
`base` the same way `extra_args` is, so c3-think512/c4-think512/a3/a4 (all `base: c3` or `base: c4`)
carry it too. No arm's built argv may contain any flag twice (the unit tests check this;
`selftest` does not).

**Flag spellings for b11151** (`ghcr.io/ggml-org/llama.cpp:server-cuda-b11151`): the lead verified
`-rea, --reasoning [on|off|auto]`, `--reasoning-budget N`, `-ncmoe, --n-cpu-moe N`, and
`-cram, --cache-ram N` on the host on 2026-09-24 with `llama-server --help` against the pinned
digest. The lead also confirmed b11151 replaced `--mmap`/`--no-mmap` with `-lm, --load-mode
{auto|none|mmap|mlock|mmap+mlock|dio}` (`none` is the old `--no-mmap`); c3/c4 pass `-lm none` via
`extra_args` accordingly. `--temp`, `--top-p`, `--top-k`, `--min-p` and `--repeat-penalty` are
already passed to production mist-llm (`docker-compose.yml`'s mist-llm `command:` block), so their
spellings are already load-bearing there. Still **UNVERIFIED**: `--presence-penalty` (see the
response-field note below for `return_tokens` and /props' `model_path`, checked separately). These
live in `arms.json` as data, not in `bench_host.py`, specifically so the lead can correct a spelling
at step L0 without touching code. c0-old targets b8808 (`snapshot:mist-llm`, the current production
build) and gets no thinking flags at all: production's chat template default is thinking-off, and
`-rea` may not exist in b8808.

**UNVERIFIED response fields**, both flagged loudly rather than guessed at silently:
`extract_model_path_from_props()` (`bench_host.py`) assumes `/props` carries a top-level
`"model_path"` string (falling back to `default_generation_settings.model`).
`parse_correctness_response()` (`probes/correctness.py`) assumes `return_tokens: true` adds a
`"tokens": [ids]` field to `/completion`'s non-streamed response; if the deployed server instead
omits it, uses a different key, or nests it elsewhere, this raises `CorrectnessProbeError` naming
the response keys it actually saw, rather than recording an empty/wrong token list silently.

### Plan v2 (2026-09-25): c1-1024, the E4B context arms, and tokens_vs_c0

Raj moved voice to the GTX 1070 overnight, so the LLM gets the whole RTX 4070 SUPER (12 GB); the
lead's plan v2 delta (`2026-09-25-mist-model-bench-plan-v2-delta.md`) adds these arms on top of v1:

- **`c1-1024`** -- `base: c0`, thinking on with a 1024-token budget. Suites: `layout` AND `ttft`
  (unlike c1-256/c1-512, which are layout-only) -- the plan text says "layout suite", but `ttft` is
  added here so decision_rules.json's exploratory X1 can measure its P (`layout_p95_wall_ms`) and D
  (`decode_tps`) clauses on c1-1024 itself rather than borrowing another arm's speed numbers.
- **The E4B context arms**, all `base: c0` (thinking off), suites `ttft`/`correctness`/`harness`:
  `c0-ctx64k` (`--ctx-size 65536`, q8_0 KV, unchanged from c0), `c0-ctx128k` (`--ctx-size 131072`,
  q8_0 KV -- `gemma4.context_length` is 131072, per the lead), and `c0-ctx128k-q4kv` (`--ctx-size
  131072`, `-ctk q4_0 -ctv q4_0`). Their harness suite uses a new named test set,
  `"tests": "context"` (`resolve_harness_tests`, `HARNESS_CONTEXT_TESTS`) --
  `["schema_conformance", "schema_conformance_json_object", "tool_selection"]` at 10 iterations, not
  the full `default` set: these arms are report-only (decision_rules.json's exploratory X2) and the
  night's time budget does not cover a full harness pass at 64K/128K context on top of the full-card
  C4/c3 work.
- **`c0-ub1024`** -- `base: c0`, `arg_overrides: {"-b": "2048", "-ub": "1024"}`,
  `suites: ["ttft"]`. arms.json does not mark it optional (`optional` is false); the plan allowed
  at most one such server-setting arm, and running it is the lead's choice. It is expected to move
  prefill speed or VRAM; whether it does is unmeasured, and its tokens may differ (see below).
- **`tokens_vs_c0`** -- every arm added under plan v2 carries this field (schema-validated in
  `resolve_arm`, `TOKENS_VS_C0_VALUES`); analyse.py's exploratory X2 surfaces it per context arm.
  `"expected-identical-unverified"` for a context-window-only change with the same q8_0 KV and
  sampling as c0 (c0-ctx64k, c0-ctx128k -- no live run has confirmed this yet); `"may-differ"` for
  anything that changes the compute path -- q4_0 KV quantization (c0-ctx128k-q4kv) or the
  batch/ubatch sizes, which can select different GEMM kernels (c0-ub1024); `"differs-by-design"` for
  a thinking-budget change (c1-1024), expected to change completion tokens by construction.

### Plan v2: the TTFT probe reaches an arm's own context

`probes/ttft.py`'s `TTFT_TARGETS` (2048, 8192, 32000) are unchanged and keep their legacy behavior
exactly: each is CAPPED (via `cap_target`), never skipped, at `n_ctx - n_predict - 16`, so an
existing 32768-ctx arm's request sequence is byte-for-byte unchanged
(`test_run_ttft_probe_request_sequence_is_byte_identical_at_ctx_32768`). Two new targets,
`TTFT_EXTRA_TARGETS_SKIP_IF_EXCEEDED` (65000, 130000), let the 64K/128K context arms reach their own
context: a target here is used, at its raw (uncapped) length, only when it is at most `n_ctx -
n_predict - 16`; otherwise it is skipped entirely -- no row, capped or otherwise. `run_one_request`'s
default HTTP timeout is now `TTFT_HTTP_TIMEOUT_S` (900s, >= the 600s the brief requires) so a 130K-
token prefill plus 256 tokens of decode has room to finish; a larger timeout does not slow down a
fast request, it is only an upper bound on how long the probe waits before giving up.

### Plan v2: rules-sha continuity across the v1 -> v2 amendment

Plan v2's rule changes move the `decision_rules.json` sha256, and `run`'s cumulative `meta.json`
(`merge_run_meta`) refuses any call whose `decision_rules_sha256` differs from what is already on
disk for that arm dir.

**Limit, found in review:** `supersedes` only relaxes the rules-sha check. It does NOT let an arm
dir written by an earlier driver version take new calls when that dir's stored `arm_config`
differs from the current one -- and every S1/S2 arm dir in `mb1` differs (plan v2 adds
`tokens_vs_c0` to every resolved arm, and those metas also predate `arg_overrides`). Those calls
are refused on `arm_config`, as S4a's c1-512 finalist in `mb1` was. New arm dirs in an existing run
are accepted. analyse reads ONE run dir, so an arm's anchors (c0, c0-prod) must be in the same run. `decision_rules.json` now carries a top-level `"supersedes"` list naming the sha256(es) it
supersedes (`{"sha256": ..., "label": ...}`); `merge_run_meta` (still a pure, no-I/O function) takes
an explicit `superseded_rules_shas` set from its caller and allows the stored sha to differ from the
new call's sha only when the *stored* value appears in that set -- any other difference is still
refused, exactly as before. `cmd_run` reads `decision_rules.json`'s own `supersedes` list once
(`load_decision_rules_supersedes_shas`) and passes it through. Each entry in `meta.json`'s `calls`
list now also records its own `decision_rules_sha256`, so the full per-call history survives even
though the top-level `decision_rules_sha256` field only ever shows the latest call's value.
`analyse.py` reports, per arm, which rules sha its stored `meta.json` ran under: a sha equal to the
currently-analysed `decision_rules.json` produces nothing; a differing sha that is listed in
`supersedes` is an `[INFO]`, not a mismatch `[WARN]`; any other differing (unlisted) sha stays a
`[WARN]`, exactly as before this change (`compute_sha_warnings`, now returning
`(warnings, infos)`).

### Plan v2: decision_rules.json's exploratory_rules (NOT pre-registered)

`decision_rules.json` gained a top-level `"exploratory_rules"` section, entirely separate from
`"rules"` -- the pre-registered v1 rules, their `thresholds`, and `constants` are unchanged in
content (a test loads the v1 file, committed as a fixture, and asserts `rules`/`constants` compare
equal before and after). Each exploratory rule carries `"pre_registered": false` and a `"basis"`
string ("post hoc, added 2026-09-25 after S1/S2 data; plan v2"):

- **X1 `c1_1024_budget`** -- c1-1024 against R1's 85% layout bar (`r1_keep_e4b_layout_acc_min`,
  referenced by threshold key, not duplicated) plus R2's P (`layout_p95_wall_ms`) and D
  (`decode_tps`) thresholds, finalist-supersedes-screen exactly as R1/R2 do.
- **X2 `context_arms_report`** -- report-only (verdict `"n/a"`, no clauses): per context arm, ttft
  at every measured ctx target, decode_tps, arm_peak_mib, harness scores with CIs plus a
  bootstrap-CI'd delta against c0 on the same tests, the arm's `tokens_vs_c0` label, and whether its
  `correctness.r1` token ids are `identical`/`differ`/`missing` versus c0's.
- **X3 `switch_to_c2_sepvoice`/`switch_to_c3_sepvoice`/`switch_to_c4_sepvoice`** -- the same L/S1/S2/D/P
  clauses as the corresponding v1 R2 rule, with F replaced by F_sep: `arm_peak_mib(candidate) + 512
  <= total_mib`, voice excluded entirely (no lower/upper split, so no needs-review branch) --
  "voice on a separate card (GTX 1070), Raj 2026-09-25". Evaluated and reported next to the
  matching v1 R2 verdict; it never replaces or overrides it.

`analyse.py` renders these in `REPORT.md` under a separately headed `## Exploratory (NOT
pre-registered)` section, placed after the v1 `## Rules` section, and in `summary.json` under a
top-level `"exploratory_rules"` key (never mixed into `"rules"`). X1/X3 reuse `evaluate_r2`'s L/S1/S2/P
logic via module-level helpers (`_l_clause`, `_anchored_harness_clause`) that are deliberately
duplicated rather than shared by refactoring `evaluate_r2` itself, so no exploratory-rule change can
ever alter a v1 R2 verdict.

### Plan v3 / T5 (2026-09-25): c1-2048, c5 (Qwen3.5), c6 (E4B Q8_0), c3 lower quants

Per plan v3's delta (`2026-09-25-mist-model-bench-plan-v3-delta.md`), T5 adds these arms
additively -- no existing arm, `common_args`, `sampling`, `thinking_args` or
`decision_rules.json` changes; every base-commit (`61f4822`) arm still resolves
byte-for-byte identically (`test_t5_arms_additive_only.py`,
`test_t5_models_yaml_additive_only.py`).

- **`c1-2048`** (`base: c0`, thinking on, budget 2048) -- a fourth E4B thinking-budget
  point alongside the existing 256/512/1024. Runs `layout` and `ttft` (like c1-1024,
  not the layout-only c1-256/c1-512); `tokens_vs_c0: differs-by-design` -- a
  thinking-budget change is expected to change completion tokens by construction.
- **`c1-unbudgeted`** (`base: c0`, thinking on, budget -1: "unrestricted" per
  `llama_server_help_b11151.txt:644`), `layout` and `ttft`, `differs-by-design`. It first
  shipped blocked: `--reasoning-budget -1` puts a bare `-1` token in argv, which the help
  flag checker (`probes/help_flags.py`, `unknown_flags()`) read as an unknown flag. The
  checker now skips numeric tokens (`-1`, `2048`, `0.95`), so a negative value is not
  mistaken for a flag; a real unknown flag is still reported.
- **`c5`** -- Qwen3.5-9B Q8_0 (`unsloth/Qwen3.5-9B-Q8_0.gguf`), full card: no `-ncmoe`,
  32768 ctx like the other arms, thinking off, family `qwen` (the existing qwen
  sampling). `tokens_vs_c0: differs-by-design` -- a different model entirely, not a
  variant of c0. Suites `ttft`/`correctness`/`harness` (`bench-c5`, `default` tests, 10
  iterations)/`layout`. **`c5-think1024`** (`base: c5`, thinking on, budget 1024,
  `layout` suite only) carries no `tokens_vs_c0` label, matching the existing
  thinking-budget siblings (c1-256/c1-512/c2-think512/c3-think512/c4-think512), none of
  which carry one either.
- **`c6`** -- `base: c0` with only `gguf` overridden to
  `unsloth/gemma-4-E4B-it-Q8_0.gguf` (same publisher as c0's Q5_K_M); everything else
  (suites, sampling, thinking) is inherited unchanged. `tokens_vs_c0: may-differ` -- a
  different quant of the same model, the same label c0-ctx128k-q4kv uses for a
  compute-path change. `bench-c6` mirrors `bench-c0` except for the `gguf`.
- **`c3-q3`** and **`c3-iq4`** -- `base: c3` with only `gguf` overridden, to
  `unsloth/gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf` and
  `unsloth/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` respectively (both already on disk).
  Base inheritance overriding only `gguf` (confirmed here, not assumed, by reading
  `resolve_arm`: the `for key, value in raw.items(): merged[key] = value` loop replaces
  a single key in the inherited dict, so a child naming only `gguf` leaves every other
  inherited field -- including `params_required`, `param_arg_map`, `extra_args`,
  `arg_overrides`, `stop_neo4j` -- exactly as `c3` resolved them; no `bench_host.py`
  change was needed). `ncmoe` stays REQUIRED (`test_required_param_refusal`,
  `REQUIRED_PARAM_ARMS`), `-lm none` and `-b/-ub 2048` stay in place, same suites as c3.
  `bench-c3-q3`/`bench-c3-iq4` mirror `bench-c3` except for the `gguf`. `tokens_vs_c0:
  differs-by-design` -- a materially lower quant than c3's Q4_K_XL.

**Qwen3.5 thinking flags: UNVERIFIED.** `-rea on|off|auto` and `--reasoning-budget N`
exist on the pinned b11151 build (`llama_server_help_b11151.txt:636,644`), and c5/c5-
think1024 pass them exactly as c0/c1-* do. Whether Qwen3.5's bundled chat template
actually honours them cannot be checked without running the real server -- no live
b11151 llama-server has been started from this worker's container (no docker, no GPU,
no network; see "Fixtures are hand-built, not recorded" below). No chat-template
override is added here: nothing in the files available to this worker (arms.json,
this README, the help capture) says Qwen3.5's GGUF-bundled template needs one, and the
brief is explicit that a chat-template override needs a citable reason, not a guess.

**One-request check for the lead to run at first serve** (`c5`, `-rea on` vs `-rea
off`): `POST /v1/chat/completions` with a short prompt (e.g. `{"messages": [{"role":
"user", "content": "What is 2+2?"}], "max_tokens": 64}`) against the served `c5`
container, once with `-rea off` and once with `-rea on` (`--param` not needed --
`thinking.mode` is set per-arm in arms.json, so this is two separate `serve` calls, one
per arm variant, or a manual flag edit for a quick check). Confirm the response's
`choices[0].message.reasoning_content` is empty or absent under `-rea off` and
non-empty under `-rea on`. If it is empty under both, the bundled template is not
honouring `-rea`/`--reasoning-budget` for this model, and `c5-think1024`'s results
should be read as no-op runs rather than budgeted-thinking ones.

### Leaving room for scan picks

Every T5 arm above follows the same two patterns already used by c1-*/c2-think512/c3-
think512/c4-think512 (a `base` arm with one or two keys overridden) and by c0/c2/c3/c4
(a full root arm with its own `gguf`/`image`/`family`/`thinking`/`suites`/`harness`).
Adding one or two more scan-picked models later is the same shape: a new root arm
(`gguf`, `image: "compose:mist-llm"`, `family`, `thinking`, `suites`, a `harness`
block naming a new `bench-cN` candidate in `models.yaml`) plus, optionally, a
`base`-inheriting thinking-budget or lower-quant sibling. No `bench_host.py` change is
needed for either shape.

### T7 (plan v3 scan picks, 2026-09-25): c7 (Granite 4.2 8B), c8 (Spark-X2.5-4B),
### c9 (gpt-oss-20b)

Per the T7 brief, the lead's candidate scan selected three public Apache-2.0 models the
lead is downloading into `D:\Users\rajga\models\`. Same additive-only discipline as T5:
no existing arm, `common_args`, `thinking_args`, or `decision_rules.json` change; every
arm present at `044699b` (the commit this task branched from, which already contains
every T5 arm) still resolves byte-for-byte identically
(`test_t5_arms_additive_only.py`, now pointed at `044699b` instead of T5's own original
base -- see that file's module docstring). Two new `sampling` family keys are ADDED
(`granite`, `spark`), plus a third (`gptoss`); every existing family key
(`gemma`, `qwen`) stays byte-identical
(`test_common_args_sampling_and_thinking_args_are_byte_for_byte_unchanged`, updated to
check per-key equality on the keys the base commit already had, since T7 legitimately
adds new keys where T5 did not need to).

- **`c7`** -- Granite 4.2 8B (`ibm-granite/granite-4.2-8b-Q6_K.gguf`), dense 8.8B, full
  card (no `-ncmoe`), `family: granite`, thinking off (`-rea off`, the same spelling
  every other full-card arm uses), suites `ttft`/`correctness`/`harness`
  (`bench-c7`, `default` tests, 10 iterations)/`layout`. `tokens_vs_c0:
  differs-by-design` -- a different model. **`c7-think`** (`base: c7`, thinking on,
  budget 1024 -- comparable to `c5-think1024`), `layout` suite only.

  Granite 4.2's template reportedly reads `enable_thinking` and also has a `low_effort`
  option, per the brief. `low_effort` is NOT wired here: `--chat-template-kwargs
  STRING` exists on the pinned b11151 build
  (`llama_server_help_b11151.txt:590-592`, confirming llama-server has a generic
  mechanism that forwards an arbitrary JSON object to the jinja template parser), so
  the *mechanism* is citable -- but whether Granite's bundled template actually reads a
  `low_effort` key is UNVERIFIED (no live b11151 llama-server in this worker's
  container -- no docker, no GPU, no network), and the brief only asks for two arms
  here (`c7`, `c7-think`), neither of which needs it. Adding an unverified
  template-specific flag to satisfy a param the two required arms do not use would be
  guessing at behavior with no test to catch a wrong guess; left out.

  **Sampling (`bench-c7`, `granite` family): UNVERIFIED, gemma-style default.** No
  citable Granite-4.2-specific model card is available to this worker's no-network
  container. Per the brief's own fallback instruction, `granite`'s array is a byte-for-
  byte copy of `gemma`'s (temperature 1.0, top-p 0.95, top-k 64, min-p 0.0,
  repeat-penalty 1.0) under a distinct family name (not literally reusing `"family":
  "gemma"` on `c7`, so the arm's own family field states what model it targets rather
  than borrowing gemma's identity). Tool calls use `<tool_call>` tags per the brief;
  the bundled llama-server template handles them, so no chat-template override or
  parser change was needed; `models.yaml`'s `bench-c7.tool_parser: "hermes"` follows
  this file's existing naming convention for `<tool_call>`-tag models (Qwen/Hermes
  entries), which is descriptive metadata only, not independently verified against a
  live Granite response.

- **`c8`** -- Spark-X2.5-4B (`XHToken/Spark-X2.5-4B-Q8_0.gguf`), llama.cpp architecture
  `spark2_5` (supported since b10828; b11151 is newer), dense, full card. `family:
  spark` (also a gemma-style default, UNVERIFIED, same reasoning as `c7`'s `granite`
  family -- no citable Spark model card either). Thinking off (`-rea off`); thinking is
  reportedly on by default for this model, and whether `-rea off` actually maps to the
  template's `enable_thinking=false` is UNVERIFIED (same class of gap as the existing
  "Qwen3.5 thinking flags: UNVERIFIED" note above -- no live server to check the
  response's `reasoning_content` against). Suites `ttft`/`correctness`/`harness`
  (`bench-c8`, `default`, 10 iterations)/`layout`. **`c8-think`** (`base: c8`, thinking
  on, budget 1024), `layout` suite only. `tokens_vs_c0: differs-by-design`.

  `models.yaml`'s `bench-c8.tool_parser` is left `null`, not `"hermes"`: the brief
  labels this arm's tool calling higher risk than `c7`'s, since it relies on the
  harness's generic tool-call handling rather than the named `<tool_call>`-tag
  convention `c7`/Qwen/Hermes already use in this file.

- **`c9`** -- gpt-oss-20b (`ggml-org/gpt-oss-20b-MXFP4.gguf`), 12.1 GB, MoE with 3.6B
  active parameters and a Harmony chat template. Modeled on `c3`'s pattern exactly:
  `-lm none` and `-b`/`-ub` 2048 via `arg_overrides` (CPU-MoE prompt processing),
  `params_required: ["ncmoe"]` / `param_arg_map: {"ncmoe": "-ncmoe"}` (REQUIRED, same
  refusal `c3`/`c4` get from `build_server_args` when `--param ncmoe=N` is missing),
  and `stop_neo4j: true`. `family: gptoss` -- a new sampling key, `--temp 1.0 --top-p
  1.0` only (see below).

  **Reasoning effort, not a thinking budget.** gpt-oss's reasoning is always on;
  effort is set via the server flag `--reasoning-effort LEVEL`
  (`llama_server_help_b11151.txt:639-642`, one of `minimal`/`low`/`medium`/`high`/
  `xhigh`/`max`), passed through `extra_args` -- `c9`: `["-lm", "none",
  "--reasoning-effort", "low"]`; `c9-medium` (`base: c9`): the same list with
  `"medium"` in place of `"low"` (a full replacement, not a merge -- `extra_args` is a
  flat list, not itemized by flag the way `arg_overrides` is, so a child arm that needs
  a different effort value must repeat the whole list). Neither arm touches the system
  prompt or any harness/layout-runner code, so no change was needed outside
  `arms.json`.

  **Thinking mode: `{"mode": "on", "budget": -1}` -- always-on, unrestricted.**
  gpt-oss always reasons; leaving `thinking` unset (as `c0-old` does) would have left
  `layout_max_tokens()` -- which keys off `thinking`'s mode only (`"on"` -> 4096,
  anything else -> 256) -- at the 256-token budget every non-thinking arm gets. Every
  `c9`/`c9-medium` layout answer would then be truncated mid-reasoning, since the model
  reasons regardless of what `-rea` is told, making the layout suite's accuracy invalid
  for this arm rather than merely conservative. `budget: -1` (`--reasoning-budget -1`,
  "unrestricted", `llama_server_help_b11151.txt:644`) sets no cap -- gpt-oss's own
  reasoning length is effort-controlled (see below), so no other budget value has any
  more citable a meaning here, and an unrestricted budget imposes no artificial ceiling
  of its own. `-rea on` (as opposed to `-rea off`/`auto`) also avoids relying on
  `auto`'s "detect from template" behavior for a template this worker cannot run live
  to confirm.

  Suites: `c9`: `ttft`/`correctness`/`harness` (`bench-c9`, `default`, 10
  iterations)/`layout`. `c9-medium`: `layout` only, `harness: null`. `tokens_vs_c0:
  differs-by-design` on `c9`; `c9-medium` does not set its own `tokens_vs_c0` and so
  *inherits* `c9`'s resolved `"differs-by-design"` (verified, not assumed --
  `resolve_arm`'s merge starts from a copy of the resolved base arm, so an unset key on
  the child carries the base's resolved value forward; this is also true of the
  existing `c5-think1024`, which likewise inherits `"differs-by-design"` from `c5`
  despite not setting the key itself -- see
  `test_thinking_budget_siblings_inherit_tokens_vs_c0_from_their_root`, which corrects
  the record: this file previously described `c5-think1024` as carrying "no
  `tokens_vs_c0` label", which is not what `resolve_arm` actually produces).

  **Sampling (`bench-c9`, `gptoss` family): UNVERIFIED, per OpenAI's published
  guidance, not independently re-verified.** No citable gpt-oss-specific model card
  fetch was possible from this worker's no-network container; `--temp 1.0 --top-p 1.0`
  (both modes, `bench-c9` in `models.yaml`) is OpenAI's stated gpt-oss recommendation
  as relayed by the T7 brief, taken as given rather than re-derived from a primary
  source this worker could reach. `models.yaml`'s `bench-c9.tool_parser` is left
  `null`: Harmony structures tool calls through dedicated channels, not the
  `<tool_call>`-tag convention this file's `"hermes"` id names, and no dedicated
  Harmony parser id exists in this file's schema.

**Every new T7 arm passes the real-capture flag check.**
`test_every_pinned_build_arm_flag_is_found_in_the_real_capture` and
`check_all_arm_flags` (`test_host_help_flags.py`) already iterate every arm in
`arms.json` generically, so `c7`/`c7-think`/`c8`/`c8-think`/`c9`/`c9-medium` are
checked against the real `llama_server_help_b11151.txt` capture automatically;
`test_t7_scan_arms.py` additionally pins each of the six arm ids by name so a
regression here fails with a narrower, arm-named test too.

## Fixtures are hand-built, not recorded

`tests/unit/model_bench/fixtures/host/` (`sse_stream.txt`, `nvidia_smi_sample.csv`,
`voice_probe_output.json`, `correctness_response.json`, `compose_with_digest.yml`,
`compose_missing_digest.yml`) are all hand-built from documented formats. No live b11151
llama-server, real nvidia-smi output, or real voice-probe run has been recorded yet -- there is no
live stack in this worker's container (no docker, no GPU, no network). Once the lead runs `serve`
and `run` against the real stack, a follow-up should replace these with real recordings (or add
them alongside) and re-verify the parsers against actual server output, particularly the two
UNVERIFIED response-field assumptions above.

**One exception:** `llama_server_help_b11151.txt` IS a real capture, not hand-built -- the lead ran
`docker run --rm <pinned compose:mist-llm image> --help` against the pinned
`ghcr.io/ggml-org/llama.cpp:server-cuda-b11151@sha256:014f7212...dc765c` digest on the host on
2026-09-24 and saved the raw output (730 lines, scanned for personal data, none found). It is
committed here verbatim, byte-for-byte, because `probes/help_flags.py`'s parser needs a real
capture to be tested against, not a guess at the format -- an earlier hand-written excerpt
(`llama_server_help_excerpt.txt`, since removed) got the leading-indent column wrong (it assumed a
small indent; the real build's `common/arg.cpp` prints each option starting at column 0) and would
have passed its own unit tests while still finding zero flags against the real `--help` output,
which `--host-checks` already treats as a hard failure. `test_host_help_flags.py` parses this real
capture and pins its sha256, so a later accidental edit to the fixture is caught.

## Voice probe (`voice --run R`)

Runs `probes/voice_vram.py` inside `mist-backend` via `docker exec -i mist-backend python -` (the
probe source piped in on stdin -- that pipe carries only this one file, so the probe is a
self-contained script, not a package import). It constructs its own `WhisperSTT` and `ChatterboxTTS`
instances the way `backend/voice_models/model_manager.py` does (same `VoiceProfileRegistry` path,
same `VOICE_PROFILE` env var), runs one STT transcription and one TTS generation, and prints a
single JSON line with peak CUDA memory and per-half success flags. `bench_host.py` samples
nvidia-smi at 5 Hz for the probe's duration, records the sampled peak as `vram_steps.json`'s
`voice_peak` step, and stores the probe's own JSON as `vram_steps.json`'s `voice_probe`.

**Double-counting note.** Production already loads both models eagerly at `mist-backend` startup:
`backend/server.py:500-507` constructs `VoiceProcessor` and awaits `voice_processor.initialize()`,
which `backend/voice_processor.py:178` runs `ModelManager.load_all_models()` in an executor;
`ModelManager.load_all_models()` (`backend/voice_models/model_manager.py:113-152`) loads
`WhisperSTT` at line 120 and starts the Chatterbox TTS worker thread (which loads the model) at
line 190, inside `_tts_worker_chatterbox`. Because this probe constructs its own separate
`WhisperSTT`/`ChatterboxTTS` instances rather than reusing the already-running backend's, its peak
VRAM reading double-counts whatever the backend already holds resident. The `voice_peak` /
`voice_probe` numbers this feeds into `vram_steps.json` are therefore conservative (they overstate
VRAM pressure, i.e. understate headroom), not exact -- the lead should read them as an upper bound
on the true STT+TTS footprint, not the footprint itself.

## Known gaps / next steps

- `decision_rules.json` does not exist yet in this worktree (another worker's task); `run` will
  refuse with a clear `[FAIL]` until it lands.
- The harness's own `--models`/`--tests` candidate ids referenced in `arms.json` (`bench-c0`,
  `bench-c0-prod`, `bench-c2`, `bench-c3`, `bench-c4`) are owned by another worker
  (`scripts/eval_harness/models.yaml`); this driver only names them.
- `run_host.py`'s and `analyse.py`'s exact CLI flags (line ~1700 / ~1307 in
  `<command-center>/spike/layout-perception`, i.e. wherever `--layout-dir` /
  `MODEL_BENCH_LAYOUT_DIR` points on the lead's host, per the brief) were not independently
  re-read against that path from this worker's container (no access to paths outside the repo
  work tree and its git metadata); `cmd_run`'s layout suite invocation follows the brief's
  specified flags verbatim and should be spot-checked against that file before first real use.
