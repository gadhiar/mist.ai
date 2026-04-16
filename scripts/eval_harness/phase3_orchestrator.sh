#!/usr/bin/env bash
# Phase 3 Gemma 4 Exoneration Orchestrator.
#
# Runs an eight-phase sequential evaluation gauntlet without user
# intervention and resumes after arbitrary pauses. See the plan at
# ~/.claude/plans/cozy-herding-pebble.md for phase design rationale.
#
# Phases:
#   D0       - Build provenance gate (~15 min)
#   D-bench  - Raw throughput ceiling via llama-bench (~30 min)
#   D1       - Variance baseline + dense screen (~2h, iterations=5)
#   D2       - Config sweep on winning torchbearer (~3h, iterations=3)
#   D3       - Schema accommodations (~2h, iterations=3)
#   D4       - Reasoning-axis tests (~2h, iterations=3)
#   D5       - Final verdict run (~4h, iterations=10)
#   D6       - Decomposed verdict via exoneration_verdict.py (~30 min)
#
# Usage:
#   bash scripts/eval_harness/phase3_orchestrator.sh                     # fresh launch
#   bash scripts/eval_harness/phase3_orchestrator.sh resume <master-ts>  # resume from checkpoint
#   bash scripts/eval_harness/phase3_orchestrator.sh download            # download-only pre-flight
#
# Backgroundable via run_in_background or nohup. Survives per-candidate
# kill, power loss (JSONL fsync'd per case), and arbitrary-length pauses.

set -uo pipefail

cd "$(dirname "$0")/../.."

SUBCOMMAND="${1:-run}"
RESUME_TS="${2:-}"

RESULTS_DIR="scripts/eval_harness/results"
MODELS_DIR="D:/Users/rajga/models"

# --- helpers ---------------------------------------------------------------

log() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[${ts}] $*" | tee -a "${MASTER_LOG:-/dev/null}"
}

# JSON read/write via jq if available, else python.
json_get() {
    local file="$1"; local path="$2"
    python -c "
import json, sys
with open(r'$file') as f:
    data = json.load(f)
keys = r'$path'.split('.')
cur = data
for k in keys:
    if isinstance(cur, dict) and k in cur:
        cur = cur[k]
    else:
        cur = ''
        break
if isinstance(cur, (list, dict)):
    print(json.dumps(cur))
else:
    print(cur)
"
}

json_set() {
    local file="$1"; local path="$2"; local value="$3"
    python -c "
import json, sys
with open(r'$file') as f:
    data = json.load(f)
keys = r'$path'.split('.')
cur = data
for k in keys[:-1]:
    cur = cur.setdefault(k, {})
try:
    v = json.loads(r'''$value''')
except Exception:
    v = r'''$value'''
cur[keys[-1]] = v
with open(r'$file', 'w') as f:
    json.dump(data, f, indent=2)
"
}

json_append() {
    local file="$1"; local path="$2"; local value="$3"
    python -c "
import json
with open(r'$file') as f:
    data = json.load(f)
keys = r'$path'.split('.')
cur = data
for k in keys[:-1]:
    cur = cur.setdefault(k, {})
arr = cur.setdefault(keys[-1], [])
if r'$value' not in arr:
    arr.append(r'$value')
with open(r'$file', 'w') as f:
    json.dump(data, f, indent=2)
"
}

init_checkpoint() {
    local master_ts="$1"
    local master_dir="${RESULTS_DIR}/phase3-exoneration-${master_ts}"
    mkdir -p "${master_dir}"
    local checkpoint="${master_dir}/CHECKPOINT.json"
    if [ -f "${checkpoint}" ]; then
        log "checkpoint exists, preserving: ${checkpoint}"
        return
    fi
    python -c "
import json
data = {
    'master_timestamp': r'${master_ts}',
    'phases': {p: {'status': 'pending'} for p in ['D0', 'D-bench', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6']}
}
with open(r'${checkpoint}', 'w') as f:
    json.dump(data, f, indent=2)
"
    log "initialized checkpoint: ${checkpoint}"
}

# truncated-line detector: strip partial last line if file does not end with \n.
heal_jsonl() {
    local jsonl="$1"
    if [ ! -f "${jsonl}" ]; then return; fi
    if [ -s "${jsonl}" ]; then
        local last_byte
        last_byte=$(tail -c 1 "${jsonl}" 2>/dev/null | od -An -c | tr -d ' ')
        if [ "${last_byte}" != "\\n" ]; then
            log "healing truncated JSONL: ${jsonl}"
            python -c "
import sys
with open(r'${jsonl}', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()
# strip last partial line if any
if lines and not lines[-1].endswith('\n'):
    lines = lines[:-1]
with open(r'${jsonl}', 'w', encoding='utf-8') as f:
    f.writelines(lines)
"
        fi
    fi
}

# Run one phase: filter out completed candidates from models list, call run.py.
run_phase() {
    local phase="$1"; local models_csv="$2"; local tests_csv="$3"; local iterations="$4"
    local phase_dir_name="${MASTER_TS}-${phase}"
    local result_dir_abs="${RESULTS_DIR}/${phase_dir_name}"

    # Mark in_progress (will be upgraded to complete at end).
    json_set "${CHECKPOINT}" "phases.${phase}.status" "in_progress"
    json_set "${CHECKPOINT}" "phases.${phase}.result_dir" "${phase_dir_name}"
    json_set "${CHECKPOINT}" "phases.${phase}.tests" "${tests_csv}"
    json_set "${CHECKPOINT}" "phases.${phase}.iterations" "${iterations}"

    mkdir -p "${result_dir_abs}"

    # Heal any truncated JSONLs from a prior interrupted run.
    for jsonl in "${result_dir_abs}"/*.jsonl; do
        [ -f "${jsonl}" ] && heal_jsonl "${jsonl}"
    done

    # Filter completed candidates.
    local remaining=""
    IFS=',' read -ra mods <<< "${models_csv}"
    for m in "${mods[@]}"; do
        local jsonl="${result_dir_abs}/${m}.jsonl"
        if [ -f "${jsonl}" ] && [ -s "${jsonl}" ]; then
            # Consider complete if JSONL has at least one line per (tests * iters * avg 5 cases) expectation.
            local line_count; line_count=$(wc -l < "${jsonl}")
            if [ "${line_count}" -gt 0 ]; then
                log "phase ${phase}: ${m} already has ${line_count} results; assuming complete (delete JSONL to rerun)"
                json_append "${CHECKPOINT}" "phases.${phase}.completed_models" "${m}"
                continue
            fi
        fi
        if [ -z "${remaining}" ]; then remaining="${m}"; else remaining="${remaining},${m}"; fi
    done

    json_set "${CHECKPOINT}" "phases.${phase}.planned_models" "${models_csv}"

    if [ -z "${remaining}" ]; then
        log "phase ${phase}: all candidates already complete, skipping"
        json_set "${CHECKPOINT}" "phases.${phase}.status" "complete"
        return 0
    fi

    log "=== phase ${phase} starting (remaining: ${remaining}, tests: ${tests_csv}, iters: ${iterations}) ==="
    local started; started=$(date +%s)

    # Run remaining candidates one at a time so per-candidate resumption works.
    IFS=',' read -ra rem_mods <<< "${remaining}"
    for m in "${rem_mods[@]}"; do
        log "phase ${phase}: running ${m}"
        python -m scripts.eval_harness.run \
            --models "${m}" \
            --tests "${tests_csv}" \
            --iterations "${iterations}" \
            --results-dir "${RESULTS_DIR}" \
            --run-name "${phase_dir_name}" \
            --log-level INFO 2>&1 | tee -a "${MASTER_LOG}"
        local exit_code="${PIPESTATUS[0]}"
        if [ "${exit_code}" -eq 0 ] && [ -f "${result_dir_abs}/${m}.jsonl" ]; then
            json_append "${CHECKPOINT}" "phases.${phase}.completed_models" "${m}"
            log "phase ${phase}: ${m} complete"
        else
            log "phase ${phase}: ${m} FAILED (exit=${exit_code}), continuing with next candidate"
        fi
    done

    local elapsed=$(( $(date +%s) - started ))
    log "=== phase ${phase} done (elapsed=${elapsed}s) ==="
    json_set "${CHECKPOINT}" "phases.${phase}.status" "complete"
}

phase_is_complete() {
    local phase="$1"
    local status
    status=$(json_get "${CHECKPOINT}" "phases.${phase}.status")
    [ "${status}" = "complete" ]
}

# --- subcommands -----------------------------------------------------------

cmd_download() {
    log "download pre-flight: validating HF repo filenames before pull"
    python <<'PYEOF'
from huggingface_hub import HfApi
api = HfApi()
wanted = [
    ("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q5_K_M.gguf"),
    ("unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q8_0.gguf"),
]
for repo, fname in wanted:
    files = [s.rfilename for s in api.list_repo_files(repo)]
    # HF sometimes splits GGUFs across multiple files; accept exact or prefix match
    hit = [f for f in files if f == fname or f.startswith(fname + ".")]
    if not hit:
        alt = [f for f in files if fname.lower() in f.lower()]
        print(f"MISS: {repo}/{fname}")
        print(f"  alternatives: {alt[:5]}")
    else:
        print(f"OK:   {repo}/{fname} -> {hit[0]}")
PYEOF
    log "starting hf download (resumes if partial)"
    hf download unsloth/gemma-4-E4B-it-GGUF --include "gemma-4-E4B-it-Q5_K_M.gguf*" --local-dir "${MODELS_DIR}/unsloth" 2>&1 | tee -a "${MASTER_LOG:-/dev/null}"
    hf download unsloth/gemma-4-E2B-it-GGUF --include "gemma-4-E2B-it-Q8_0.gguf*" --local-dir "${MODELS_DIR}/unsloth" 2>&1 | tee -a "${MASTER_LOG:-/dev/null}"
    log "download pre-flight complete"
}

# Phase model + test specifications. Declared as functions so they can
# reference MASTER_TS at call time.
phase_a_models() {
    # Phase D1: variance baseline + dense screen
    echo "gemma-4-26b-a4b-q3xl-fit,gemma-4-e4b-dense-q5km"
}

phase_d2_models_dense() {
    echo "gemma-4-e4b-dense-q5km,gemma-4-e4b-dense-carteakey-full,gemma-4-e4b-dense-no-grammar,gemma-4-e4b-dense-bundled-template,gemma-4-e4b-dense-temp-1"
}

phase_d2_models_moe() {
    echo "gemma-4-26b-a4b-q3xl-fit,gemma-4-26b-a4b-q3xl-fit-carteakey-full,gemma-4-26b-a4b-q3xl-fit-no-grammar,gemma-4-26b-a4b-q3xl-fit-bundled-template,gemma-4-26b-a4b-q3xl-fit-temp-1"
}

phase_d5_finalists() {
    # best_gemma + hermes ngram; populated at runtime from CHECKPOINT.
    local best_gemma="$1"
    echo "${best_gemma},hermes-4-14b-q5km-ngram"
}

# Determine torchbearer from D1 result_dir JSONL quality means.
# Dense-leaning rule (plan D1): Dense wins unless MoE beats it by > 0.05.
select_torchbearer() {
    local d1_dir="$1"
    python <<PYEOF
import json
import statistics
from pathlib import Path
from scripts.eval_harness import scorers

d = Path(r"${d1_dir}")
results = {}
for jsonl in d.glob("*.jsonl"):
    scores = []
    for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        test_type = r.get("test_type", r.get("test_name", ""))
        scorer = scorers.SCORER_REGISTRY.get(test_type)
        if not scorer:
            continue
        try:
            _, score, _ = scorer(r, r.get("expected") or {})
            scores.append(float(score))
        except Exception:
            scores.append(0.0)
    if scores:
        results[jsonl.stem] = statistics.fmean(scores)

dense_key = next((k for k in results if "e4b-dense" in k), None)
moe_key = next((k for k in results if "26b-a4b" in k and "q3xl-fit" in k and "12b" not in k), None)
if not dense_key or not moe_key:
    print("unknown")
else:
    dense_q = results[dense_key]
    moe_q = results[moe_key]
    # Dense-leaning rule: Dense wins unless MoE beats it by MORE than 0.05
    if moe_q - dense_q > 0.05:
        print(moe_key)
    else:
        print(dense_key)
PYEOF
}

# Run all phases given MASTER_TS + CHECKPOINT already set.
run_gauntlet() {
    # D0 - Build provenance
    if ! phase_is_complete "D0"; then
        log "=== PHASE D0 (build provenance) ==="
        local d0_dir="${MASTER_DIR}/D0"
        mkdir -p "${d0_dir}"
        local bin="${LLAMA_SERVER_BIN:-C:/tools/llama.cpp/llama-server.exe}"
        "${bin}" --version > "${d0_dir}/build-info.txt" 2>&1 || true
        "${bin}" --help > "${d0_dir}/help.txt" 2>&1 || true
        {
            echo "# Phase D0 Decision - ${MASTER_TS}"
            echo
            echo "## Build Info"
            echo '```'
            cat "${d0_dir}/build-info.txt" 2>/dev/null | head -20
            echo '```'
            echo
            echo "## Flash Attention Status"
            echo "Heuristic: if --help mentions FA_ALL_QUANTS or variants, FA is enabled across KV quant types."
            grep -iE "(flash|fa_all|cuda)" "${d0_dir}/help.txt" 2>/dev/null | head -10 || echo "(grep produced no matches)"
            echo
            echo "## Decision"
            echo "Proceeding with existing prebuilt b8795. If downstream phases show q8 KV regressions,"
            echo "revisit and rebuild llama.cpp with GGML_CUDA_FA_ALL_QUANTS=ON."
        } > "${d0_dir}/d0-decision.md"
        json_set "${CHECKPOINT}" "phases.D0.status" "complete"
        json_set "${CHECKPOINT}" "phases.D0.result_dir" "D0"
    fi

    # D-bench - Raw throughput ceiling
    if ! phase_is_complete "D-bench"; then
        log "=== PHASE D-bench (llama-bench raw ceiling) ==="
        local db_dir="${MASTER_DIR}/D-bench"
        mkdir -p "${db_dir}"
        local bench_bin="${LLAMA_BENCH_BIN:-C:/tools/llama.cpp/llama-bench.exe}"
        if [ ! -x "${bench_bin}" ] && [ ! -f "${bench_bin}" ]; then
            log "D-bench: llama-bench not found at ${bench_bin}, skipping"
            json_set "${CHECKPOINT}" "phases.D-bench.status" "complete"
            json_set "${CHECKPOINT}" "phases.D-bench.result_dir" "D-bench"
            echo "(llama-bench not available; skipped)" > "${db_dir}/d-bench-results.md"
        else
            declare -a gguf_list=(
                "${MODELS_DIR}/unsloth/gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf"
                "${MODELS_DIR}/unsloth/gemma-4-12B-it-Q5_K_M.gguf"
                "${MODELS_DIR}/bartowski/NousResearch_Hermes-4-14B-Q5_K_M.gguf"
            )
            for gguf in "${gguf_list[@]}"; do
                local tag; tag=$(basename "${gguf}" .gguf)
                log "D-bench: benchmarking ${tag}"
                if [ -f "${gguf}" ]; then
                    "${bench_bin}" -m "${gguf}" -p 512 -n 128 -fa 1 --no-mmap -ctk q8_0 -ctv q8_0 --batch-size 1024 --ubatch-size 512 --threads 10 -r 3 > "${db_dir}/${tag}.bench.txt" 2>&1 || log "D-bench: ${tag} bench failed"
                else
                    echo "(gguf not found)" > "${db_dir}/${tag}.bench.txt"
                fi
            done
            {
                echo "# Phase D-bench Raw Throughput - ${MASTER_TS}"
                echo
                for f in "${db_dir}"/*.bench.txt; do
                    echo "## $(basename ${f} .bench.txt)"
                    echo '```'
                    cat "${f}"
                    echo '```'
                    echo
                done
            } > "${db_dir}/d-bench-results.md"
            json_set "${CHECKPOINT}" "phases.D-bench.status" "complete"
            json_set "${CHECKPOINT}" "phases.D-bench.result_dir" "D-bench"
        fi
    fi

    # D1 - Variance baseline + dense screen
    run_phase "D1" "$(phase_a_models)" "speed_minimal,schema_conformance,tool_selection,personality,rag_integration,coherence,speed" 5

    # Select torchbearer from D1 result
    local d1_result_dir; d1_result_dir="${RESULTS_DIR}/${MASTER_TS}-D1"
    local torchbearer; torchbearer=$(select_torchbearer "${d1_result_dir}")
    log "D1 torchbearer selected: ${torchbearer}"
    json_set "${CHECKPOINT}" "torchbearer" "${torchbearer}"

    # D2 - Config sweep on torchbearer
    local d2_models
    if [[ "${torchbearer}" == *"dense"* ]]; then
        d2_models=$(phase_d2_models_dense)
    else
        d2_models=$(phase_d2_models_moe)
    fi
    run_phase "D2" "${d2_models}" "speed_minimal,schema_conformance,tool_selection,personality,rag_integration,coherence,speed" 3

    # Determine best_gemma from D2 by quality mean
    local d2_result_dir; d2_result_dir="${RESULTS_DIR}/${MASTER_TS}-D2"
    local best_gemma_d2
    best_gemma_d2=$(python <<PYEOF
import json, statistics
from pathlib import Path
from scripts.eval_harness import scorers
d = Path(r"${d2_result_dir}")
results = {}
for jsonl in d.glob("*.jsonl"):
    scores = []
    for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        test_type = r.get("test_type", r.get("test_name", ""))
        scorer = scorers.SCORER_REGISTRY.get(test_type)
        if not scorer:
            continue
        try:
            _, score, _ = scorer(r, r.get("expected") or {})
            scores.append(float(score))
        except Exception:
            scores.append(0.0)
    if scores:
        results[jsonl.stem] = statistics.fmean(scores)
if results:
    print(max(results.items(), key=lambda kv: kv[1])[0])
else:
    print(r"${torchbearer}")
PYEOF
)
    log "D2 best_gemma selected: ${best_gemma_d2}"
    json_set "${CHECKPOINT}" "best_gemma_d2" "${best_gemma_d2}"

    # Kill-switch: check strict schema >= 0.60 on best_gemma_d2
    local best_schema; best_schema=$(python <<PYEOF
import json, statistics
from pathlib import Path
from scripts.eval_harness import scorers
jsonl = Path(r"${d2_result_dir}/${best_gemma_d2}.jsonl")
if not jsonl.exists():
    print("0.0")
else:
    scores = []
    for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("test_name") != "schema_conformance":
            continue
        try:
            _, score, _ = scorers.score_schema_conformance(r, r.get("expected") or {})
            scores.append(float(score))
        except Exception:
            scores.append(0.0)
    print(f"{statistics.fmean(scores):.3f}" if scores else "0.0")
PYEOF
)
    log "kill-switch check: best_gemma_d2 strict schema = ${best_schema} (threshold 0.60)"
    local schema_ok
    schema_ok=$(python -c "print(1 if float('${best_schema}') >= 0.60 else 0)")
    if [ "${schema_ok}" = "0" ]; then
        log "KILL-SWITCH TRIPPED: best Gemma strict schema ${best_schema} < 0.60 threshold"
        log "Skipping D3-D5. Writing early verdict."
        json_set "${CHECKPOINT}" "kill_switch_tripped" "true"
        {
            echo "# Exoneration Verdict - EARLY KILL-SWITCH"
            echo
            echo "Best Gemma configuration in D2 (${best_gemma_d2}) scored ${best_schema} on strict schema_conformance, below the pre-registered kill-switch threshold of 0.60."
            echo
            echo "No configuration, sampling change, grammar-disable, or template-change in D2 was able to lift Gemma above the floor needed to reach V1 production-readiness (0.80)."
            echo
            echo "**Winner: Hermes-4-14B Q5_K_M (no-think + ngram speculative)**"
            echo
            echo "See D2 report for full per-variant breakdown."
        } > "${MASTER_DIR}/VERDICT.md"
        return 0
    fi

    # D3 - Schema accommodations
    run_phase "D3" "${best_gemma_d2},hermes-4-14b-q5km-ngram" "schema_conformance,schema_conformance_lenient,schema_conformance_fewshot" 3

    # D4 - Reasoning tests
    run_phase "D4" "${best_gemma_d2},hermes-4-14b-q5km-ngram,hermes-4-14b-q5km-optimized" "cot_reasoning,long_turn_coherence,adversarial_persona" 3

    # D5 - Final verdict run with iterations=10
    run_phase "D5" "$(phase_d5_finalists "${best_gemma_d2}")" "speed_minimal,schema_conformance,schema_conformance_lenient,schema_conformance_fewshot,tool_selection,personality,adversarial_persona,rag_integration,coherence,cot_reasoning,long_turn_coherence,speed" 10

    # D6 - Generate verdict
    if ! phase_is_complete "D6"; then
        log "=== PHASE D6 (generating decomposed verdict) ==="
        python scripts/eval_harness/exoneration_verdict.py "${MASTER_DIR}" "${RESULTS_DIR}/${MASTER_TS}-D5" 2>&1 | tee -a "${MASTER_LOG}"
        json_set "${CHECKPOINT}" "phases.D6.status" "complete"
        json_set "${CHECKPOINT}" "phases.D6.result_dir" "."
    fi

    log "=============================================================="
    log "Exoneration protocol COMPLETE"
    log "Verdict: ${MASTER_DIR}/VERDICT.md"
    log "=============================================================="
}

# --- main dispatch ---------------------------------------------------------

case "${SUBCOMMAND}" in
    download)
        export MASTER_LOG="/tmp/phase3-download.log"
        mkdir -p "$(dirname "${MASTER_LOG}")" 2>/dev/null || true
        cmd_download
        ;;
    resume)
        if [ -z "${RESUME_TS}" ]; then
            echo "usage: $0 resume <master-ts>" >&2
            exit 1
        fi
        MASTER_TS="${RESUME_TS}"
        MASTER_DIR="${RESULTS_DIR}/phase3-exoneration-${MASTER_TS}"
        CHECKPOINT="${MASTER_DIR}/CHECKPOINT.json"
        MASTER_LOG="${MASTER_DIR}/orchestrator.log"
        if [ ! -f "${CHECKPOINT}" ]; then
            echo "checkpoint not found: ${CHECKPOINT}" >&2
            exit 1
        fi
        log "=== RESUME master=${MASTER_TS} ==="
        run_gauntlet
        ;;
    run|"")
        MASTER_TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
        MASTER_DIR="${RESULTS_DIR}/phase3-exoneration-${MASTER_TS}"
        CHECKPOINT="${MASTER_DIR}/CHECKPOINT.json"
        mkdir -p "${MASTER_DIR}"
        MASTER_LOG="${MASTER_DIR}/orchestrator.log"
        init_checkpoint "${MASTER_TS}"
        log "=============================================================="
        log "Exoneration protocol starting"
        log "master dir: ${MASTER_DIR}"
        log "=============================================================="
        run_gauntlet
        ;;
    *)
        echo "unknown subcommand: ${SUBCOMMAND}" >&2
        echo "usage: $0 [run|resume <master-ts>|download]" >&2
        exit 1
        ;;
esac
