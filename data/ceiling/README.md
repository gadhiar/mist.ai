# Extraction ceiling runs

Run A asks how much of the dev-lists extraction gap a stronger model in the same seat
would close. An answerer takes the model's place in the unchanged extraction service: the
service runs as built, and its model calls go to a relay that serves the answerer's
stored answers. Every request the service makes is the one gpt-oss met, with the same
prompts, batches and merge code. The code is in `scripts/eval_harness/ceiling/`.

## Files

Committed, in this directory:

| File | What it holds |
|---|---|
| `probes.jsonl` | The 40 dev-lists probes, `{"tag", "utterance"}` per line, in gold order. `make_probes` writes it from `data/gold/extraction/dev-lists-2026-11.jsonl`; a test checks it byte for byte. The answerers' service reads only this file, never gold. |
| `answers-<k>.jsonl` | Answerer k's answers, k = 1 to 4, for probe rows 10k-9 to 10k. One line per answer: `{"seq", "key", "round", "kind", "answer", "chars", "prev"}`. `key` is the SHA-256 of the request body, `chars` the length of the text the relay serves, and `prev` the SHA-256 of the line before (64 zeros on the first), so an edited, deleted, reordered or inserted line fails the load. Only the round tool's `--add` writes it. |

Local, under `.local/` (ignored by git):

| Path | What it holds |
|---|---|
| `.local/ceiling/pending-<k>.md` | The latest round's offered requests: key, tag, kind, the system prompt's and schema's file names, and every other message in full. |
| `.local/ceiling/prompts/<sha12>.txt`, `.local/ceiling/schemas/<sha12>.json` | Each system prompt and response schema a pending file names, written once, named by the first 12 hex digits of its SHA-256. |
| `.local/ceiling/answers-in-<k>.jsonl` | The answerer's input to `--add`: `{"key", "answer"}` per line. |
| `.local/ceiling/state-<k>.json` | After each round and each `--add`: the store's line count, the SHA-256 of its last line, the round and its offered requests. A store shorter than recorded, or whose recorded line changed, stops the round tool. Without this file (as after a merge) only the chain is checked. |
| `.local/eval-ceiling-a/<k>/round-<r>/<tag>/` | The bench probe's outputs and logs for one probe in one round, and the service's log per round. Nothing the round tool prints names it, and an answerer never reads it: it holds service internals. |

## Commands

All commands run from the repository root in the worker container, which has the
service's dependencies and no network.

Make the probes file:

```
python -m scripts.eval_harness.ceiling.make_probes --gold data/gold/extraction/dev-lists-2026-11.jsonl --out data/ceiling/probes.jsonl
```

Run one round for answerer k (1 to 4):

```
python -m scripts.eval_harness.ceiling.round --answerer <k>
```

It starts the relay on 127.0.0.1:18080 over `answers-<k>.jsonl`, starts the service with
`uvicorn backend.extraction_service.app:build_app_from_env --factory --host 127.0.0.1 --port 18090`
under the env below, and runs `python -m scripts.model_bench.probes.extraction_service` on
each of k's probes in turn. Each probe's first pending request is offered; its later
pending requests are superseded, because they were made only after an earlier one failed
(a pending scope call degrades to `unknown`, and the turn goes on to a degraded extract).
Stdout is one line:

```
round <r> answerer <k>: probes 10, complete <c>, offered <o> (scope <s>, extract <e>, repair <p>, derive <d>), superseded <x>, stored <n>
```

Exit 0 means nothing is pending, 3 means requests are offered, 1 is an error. Repair and
derive stay 0: the store refuses any answer the strict parse would reject, and the bench
probe sends no derivation.

Store answers, then run the next round:

```
python -m scripts.eval_harness.ceiling.round --answerer <k> --add
```

It validates each line of `answers-in-<k>.jsonl` and appends it, then prints
`added <a>, refused <m>, stored <n>` and one `refused ...` line per refusal, naming the
key's first 12 characters and the reason (for a schema failure, the JSON path and
jsonschema's message). It refuses a key already stored, a key not offered in the latest
round, an answer failing the request's own response schema, and an answer whose served
text holds `<|channel|>`. Exit 0 when every line was added, 3 when any was refused (that
request is still owed), 1 on an error. A line already stored is refused rather than
skipped, so put only new answers in the input file.

Check that answerer k is done:

```
python -m scripts.eval_harness.ceiling.round --answerer <k> --check
```

It replays k's probes into a temporary directory and writes nothing in the worktree, so it
runs in a read-only container. Exit 0 only when the store loads and nothing is pending.

Replay all four stores once over all 40 probes, twice, and compare:

```
python -m scripts.eval_harness.ceiling.replay --out .local/eval-ceiling/run-a-1
python -m scripts.eval_harness.ceiling.replay --out .local/eval-ceiling/run-a-2
python -m scripts.eval_harness.ceiling.replay --compare .local/eval-ceiling/run-a-1 .local/eval-ceiling/run-a-2
```

`--out` merges the stores (a key in two stores is refused), runs the bench probe once over
the probes file, and writes its outputs plus `requests.jsonl`, the relay log
(`seq`, `tag`, `key`, `kind`, `answered`). Each request is attributed to the probe whose
scope request it follows, because the relay is told no tag in a single bench run. It
refuses a directory that already holds a replay, and exits 0 only when nothing is
pending. `--compare` exits 0 only when the key sequences are equal and the raw rows are
equal after dropping `job_id`, `response.job_id` and `response.timings_ms`; otherwise it
prints the first differing tag and field names, never values, and exits 1. Run A's raw is
the first replay's `extraction_service_raw.jsonl`.

Optional flags, for tests and other layouts; the defaults give the commands above:

| Flag | Tool | Default |
|---|---|---|
| `--probes PATH` | round, replay | `data/ceiling/probes.jsonl` |
| `--store-dir DIR` | round, replay | `data/ceiling` |
| `--local-dir DIR` | round | `.local` |
| `--relay-port N` | round, replay | `18080` (0 picks a free port) |
| `--service-port N` | round, replay | `18090` (0 picks a free port) |

## Service env

Every run A service process gets host-1070's r6 settings, except the four identity
stamps, which name the relay so no run A result can pass for a gpt-oss one. Inherited
`EXTRACTION_*` variables and `LLAMA_CPP_BUILD` are dropped first.

| Variable | Value |
|---|---|
| `EXTRACTION_LLM_BASE_URL` | `http://127.0.0.1:18080` |
| `EXTRACTION_ADAPTER` | `gptoss` |
| `EXTRACTION_REASONING_EFFORT` | `low` |
| `EXTRACTION_LLM_TIMEOUT_SECONDS` | `300` |
| `EXTRACTION_MAX_ATTEMPTS` | `2` |
| `EXTRACTION_CONSTRAINED_MODE` | empty, which gives `schema` |
| `EXTRACTION_SCOPE_ENABLED` | `true` |
| `EXTRACTION_TEMPERATURE` | `0.0` |
| `EXTRACTION_MODEL_FILE` | `answerer-relay` |
| `EXTRACTION_MODEL_HASH` | `answerer-relay` |
| `LLAMA_CPP_BUILD` | `relay` |
| `EXTRACTION_LOCATION_LABEL` | `container-relay` |

## Caveats

- An answerer is a worker, not a bare model call. It has tools, a long context and no
  thinking cap, and it reads each system prompt once rather than per request. Run A
  measures what that worker gives in the model's seat, not what any model gives through
  an API with gpt-oss's settings.
- The relay never truncates. It always replies with `finish_reason` `stop`, so an answer
  longer than gpt-oss's 2048-token extract cap is never cut and its batch never halved.
  That flatters capacity a little; the stores record each answer's `chars`, so extract
  answers over 6,144 characters (about the cap) can be counted.
- The answerer meets no repair call: the store's checks equal the service's strict parse.
- Each run is one sample.
- Only the 40 dev-lists probes are run. The held-out set is never run, opened or scored.
