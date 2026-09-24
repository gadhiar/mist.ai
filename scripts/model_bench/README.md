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
3. **`serve <arm> --run R [--param k=v ...]`** -- refuses if `mist-llm` is running, if a REQUIRED
   param (e.g. c3/c4's `ncmoe`) is missing, or if the arm needs `stop_neo4j` and `mist-neo4j` is
   still running. Starts `mist-bench-llm` via `docker run -d --rm ... -p 127.0.0.1:8080:8080`, and
   waits for `/health` (default timeout 900s -- `--no-mmap` MoE loads are slow).
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
   `<arm>/server.log`, then `docker stop mist-bench-llm`.
6. **`restore --run R`** -- refuses if `mist-bench-llm` is still running. `docker start`s all three
   production containers, waits for `mist-llm`'s `/health` and `/props` and for `mist-backend`'s
   `State.Health.Status == healthy`, re-inspects, and diffs against the snapshot on `Id`, `Image`,
   `Config.Cmd`, `Config.Env`, and `HostConfig`. Writes
   `session/restore_diff_<UTC>.json`, prints `restore diff: empty` or the diff, and exits non-zero
   on any difference (an `Id` change means Docker recreated the container -- always a failure) or
   an unhealthy service.

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
  `nvidia-smi` query through this driver's parser, and
  `python -c "import yaml, openai, httpx"`, printing ok/fail per check.
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
```

**Raw outputs are never committed.** `server.log`, the harness's per-candidate JSONL, `calls.jsonl`,
`vram.csv`, `ttft.jsonl`, and `correctness.r<K>.jsonl` all live under `--results-root`, which the
guard above forces outside the git work tree. Nothing under `<results-root>` is part of this
repository; only `arms.json` (server configs) and `decision_rules.json` (another worker's, referenced
by path and sha256 only) are tracked.

## arms.json

Single source of truth for per-arm server configuration -- see the file itself, `docker-compose.yml`
(mist-llm's production args, which the common server args mirror), and the T3a brief for the full
arm table. Arms may set `"base": "<other-arm-id>"` to inherit that arm's resolved config wholesale,
then override individual keys (used both for the tuning arms a1-a4, which inherit their base's
server config exactly, and for the thinking-on variants c1-256/c1-512/c2-think512/c3-think512/
c4-think512).

**UNVERIFIED flag spellings, for b11151** (`ghcr.io/ggml-org/llama.cpp:server-cuda-b11151`): `-rea`,
`--reasoning-budget`, `-ncmoe`, and the sampling flag names (`--temp`, `--top-p`, `--top-k`,
`--min-p`, `--repeat-penalty`, `--presence-penalty`). These live in `arms.json` as data, not in
`bench_host.py`, specifically so the lead can correct a spelling at step L0 without touching code.
c0-old targets b8808 (`snapshot:mist-llm`, the current production build) and gets no thinking flags
at all: production's chat template default is thinking-off, and `-rea` may not exist in b8808.

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
llama-server, real nvidia-smi output, or real voice-probe run has been recorded yet -- there is
no live stack in this worker's container (no docker, no GPU, no network). Once the lead runs `plan
--host-checks`, `serve`, and `run` against the real stack, a follow-up should replace these with
real recordings (or add them alongside) and re-verify the parsers against actual server output,
particularly the two UNVERIFIED response-field assumptions above.

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
  `D:/Users/rajga/command-center/spike/layout-perception`, per the brief) were not independently
  re-read against that path from this worker's container (no access to paths outside the repo
  work tree and its git metadata); `cmd_run`'s layout suite invocation follows the brief's
  specified flags verbatim and should be spot-checked against that file before first real use.
