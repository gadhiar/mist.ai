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
   list) in the fixed order ttft -> correctness -> harness -> layout, writes `meta.json` after each
   suite, and stops the sampler in a `finally` block. Refuses to overwrite any existing suite
   output -- pick a new `--run`, `--layout-pass`, or `--rep` instead.
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
