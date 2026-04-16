# MIST Model Backend Evaluation Harness

Phase 2 evaluation harness for the MIST model backend migration workstream.
Drives llama-server via its OpenAI-compatible API and scores candidate
models against MIST's real workload: ontology-constrained extraction, tool
selection, personality adherence, RAG integration, multi-turn coherence,
and speed.

Source of truth for candidate list and scoring rationale:
- `knowledge-vault/Decisions/ADR-008-revised-model-backend-selection.md`
- `knowledge-vault/Projects/mist-ai/research/2026-04-11-model-backend-cross-validation.md`
- `knowledge-vault/Projects/mist-ai/workstreams/mist-ai-model-backend-migration.md`

## Design

The harness is intentionally **standalone**. It does not import from
`backend/` and should run on any machine with `openai`, `httpx`, and
`pyyaml` installed. It does not require Docker.

```
scripts/eval_harness/
  __init__.py
  run.py                    # CLI orchestrator
  client.py                 # llama-server OpenAI wrapper + subprocess spawner
  scorers.py                # per-test scoring functions
  report.py                 # markdown report generator
  models.yaml               # 8 candidate configs (4 primary + 4 diagnostic)
  grammars/
    mist_ontology.gbnf      # GBNF grammar for 12 entity x 21 relationship ontology
  tests/
    schema_conformance.yaml # 12 extraction cases
    tool_selection.yaml     # 8 tool-calling cases
    personality.yaml        # 6 JARVIS / FRIDAY / Cortana style cases
    rag_integration.yaml    # 5 context-grounding cases
    coherence.yaml          # 5 multi-turn coherence cases
    speed.yaml              # 5 tokens-per-second cases
  results/                  # JSONL per candidate + report.md (gitignored)
```

## Installation

The harness has three Python dependencies on top of MIST's backend
environment. Install them into the venv you use for running scripts:

```
pip install openai httpx pyyaml
```

The container image already bakes `openai` and `httpx`. On the host, a
lightweight venv works:

```
python -m venv .venv-harness
.venv-harness\Scripts\activate  # Windows
pip install openai httpx pyyaml
```

## Running

### Mode 1: external server (recommended for one-off runs)

Start llama-server yourself with the right GGUF and flags, then point the
harness at the already-running endpoint:

```
# Terminal 1: start llama-server manually
llama-server -m D:/Users/rajga/models/gemma-3-12b-it-Q5_K_M.gguf \
  --host 127.0.0.1 --port 8080 --n-gpu-layers 999 \
  --ctx-size 8192 --cache-ram 4096 --kv-unified 0 \
  --chat-template gemma --no-webui

# Terminal 2: run the harness against it
python -m scripts.eval_harness.run \
  --external \
  --models gemma-3-12b-q5km \
  --tests schema_conformance,tool_selection,personality,rag_integration,coherence,speed
```

### Mode 2: spawn server per candidate (full A/B matrix)

Place all candidate GGUFs in one directory and let the harness manage
server lifecycle:

```
python -m scripts.eval_harness.run \
  --models-dir D:/Users/rajga/models \
  --llama-server-bin D:/tools/llama.cpp/llama-server.exe \
  --models gemma-4-26b-a4b-iq4xs,qwen-3.5-9b-q8,gemma-3-12b-q5km,qwen-2.5-14b-q5km
```

The harness will spawn one `llama-server` subprocess per candidate, wait
for `/health`, run every selected test, tear down the subprocess, and
move on to the next candidate.

### Useful flags

```
--list-models       print every candidate ID
--list-tests        print every registered test with its file path
--iterations N      run each case N times (for variance analysis)
--seed 42           sampling seed passed through to llama-server
--skip-report       run tests but skip markdown report generation
--log-level DEBUG   more verbose logs
```

### Environment overrides

```
LLAMA_MODELS_DIR    override models.yaml `defaults.models_dir`
LLAMA_SERVER_BIN    override models.yaml `defaults.llama_server_binary`
```

## Output

Each run creates a timestamped subdirectory under `results/`:

```
results/
  2026-04-11T21-07-33Z/
    report.md                       # human-readable comparison
    gemma-3-12b-q5km.jsonl          # per-candidate raw results
    qwen-3.5-9b-q8.jsonl
    ...
    server-logs/                    # llama-server stdout/stderr
      gemma-3-12b-q5km-server.log
      ...
```

Each JSONL line is one `CaseResult` with full response content, tool
calls, token usage, timing metrics, and the expected payload from the
test YAML. Rescore a past run without rerunning the models:

```python
from pathlib import Path
from scripts.eval_harness import report, run

defaults, candidates = run.load_models_config(run.DEFAULT_CONFIG_PATH)
test_files = run.load_test_files(list(run.DEFAULT_TEST_ORDER), run.DEFAULT_TESTS_DIR)
scores = report.rescore_from_jsonl(
    Path("scripts/eval_harness/results/2026-04-11T21-07-33Z"),
    test_files=test_files,
    candidates=candidates,
)
```

## Scoring

| Test | Scorer | Pass threshold |
|---|---|---|
| schema_conformance | JSON valid + ontology types in allow-list + source/target resolve + gold recall >= 0.8 | all structural checks + recall >= 0.8 |
| tool_selection | Tool name match + argument keyword overlap >= 0.75 | exact name + >=75% args |
| personality | Must-contain regex hits + forbidden regex absent + length envelope | must-contain >= 0.8 + no forbidden + length ok |
| rag_integration | Gold-fact recall + forbidden-fact absence | recall >= 0.7 + no forbidden |
| coherence | Non-empty + no trigram loop + no ai-slop phrase + min length + finish_reason != length | all checks |
| speed | tokens_per_second vs target floor | tps >= target |

The aggregate quality score per candidate is the mean of test mean scores.
Weights can be passed to `CandidateScores.aggregate_quality_score(weights)`
if one test should dominate.

## Adding a new model

1. Download the GGUF to `LLAMA_MODELS_DIR`.
2. Add a new entry under `candidates:` in `models.yaml`. Required fields:
   `id`, `display_name`, `tier`, `gguf`, `served_model_name`. Optional but
   strongly recommended: `chat_template`, `tool_parser`, `stop_sequences`,
   `temperature`, `top_p`.
3. Run `python -m scripts.eval_harness.run --list-models` to confirm the
   entry parsed.
4. Run `python -m scripts.eval_harness.run --models <new-id> --tests speed`
   for a fast smoke test.

## Adding a new test

1. Create `tests/<name>.yaml` matching the schema in `run.py`'s
   `_parse_test_file`.
2. If the test type needs custom scoring, add a scorer function in
   `scorers.py` and register it in `SCORER_REGISTRY`.
3. Add the test name to `DEFAULT_TEST_ORDER` in `run.py` if it should run
   by default.

## Phase 2 smoke test (at desktop)

Before running the full A/B matrix, verify Gemma 4 26B-A4B works on
Windows CUDA with llama.cpp build b8670+:

```
# 1. pin llama.cpp build
# Download the b8670+ Windows CUDA release from:
# https://github.com/ggerganov/llama.cpp/releases

# 2. download the GGUF
huggingface-cli download unsloth/gemma-4-26b-a4b-it-GGUF UD-IQ4_XS

# 3. smoke test via harness (speed test only)
python -m scripts.eval_harness.run \
  --models gemma-4-26b-a4b-iq4xs \
  --tests speed \
  --iterations 1
```

If the Gemma 4 smoke test fails due to chat-template or tool-parser
issues, fall back to the Phase 2 primary runner-up (Qwen 3.5 9B Q8_0).

## Spec corrections baked in

These are the six llama-server spec fixes from the cross-validation
research (see Section 6 of the research note):

1. `--system-prompt-file` removed upstream (llama.cpp PR #9857, Oct 2024).
   Use a system-role message instead. The harness does this in
   `run.build_messages`.
2. Base image pinned to `nvidia/cuda:12.8.1-devel-ubuntu24.04` for the
   containerized server. The harness itself does not run in a container.
3. `--kv-unified 0` added to the default shared server args in
   `models.yaml`. Correctness fix under KV quantization.
4. GBNF grammar is passed via `extra_body={"grammar": <text>}` in the
   chat completion request, not `--grammar-file` on the server. This
   lets the harness swap grammars per request without restarting the
   server. See `HarnessClient.chat`.
5. Speculative decoding is dropped from the A/B baseline. Only one
   candidate has a shipped 0.5B draft model in this matrix and the
   1.3-1.7x real speedup on memory-bandwidth-bound consumer GPUs is
   not worth the extra integration complexity for Phase 2.
6. `--cache-ram 4096` is the correct flag, not `--cram 256`. Also fixed
   in the default shared server args.

llama.cpp build pin: **>= b8670** (required for Gemma 4 MoE).

## Not in scope for V1

- Batch inference (one request at a time).
- LLM-as-judge scoring (heuristic checks only).
- Visual plots or HTML reports (markdown is enough for A/B review).
- Rerun on failures (the harness writes raw JSONL; rerun by hand).
- Adversarial prompt battery (Phase 3 if extraction quality is uneven).
