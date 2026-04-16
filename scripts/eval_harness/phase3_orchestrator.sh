#!/usr/bin/env bash
# Phase 3 A/B evaluation orchestrator.
#
# Runs three sequential evaluation phases without user intervention:
#
#   Phase A - Ablation confirmation: 6 Gemma 4 Q5_K_XL flag variants against
#             speed + schema_conformance to verify carteakey's flag stack is
#             load-bearing.
#
#   Phase B - Expansive Gemma 4 matrix: 6 Gemma 4 optimization candidates
#             (3 original + 3 new carteakey-post-derived variants) against
#             all 7 tests (speed_minimal + 6 existing).
#
#   Phase C - Non-Gemma finalists: qwen-2.5-14b-q5km, hermes-4-14b-q5km-ngram,
#             hermes-4-14b-q5km-optimized against all 7 tests for direct
#             comparison against Phase B Gemma results.
#
# Writes a master summary markdown aggregating all three reports at the end.
# Designed to survive per-candidate failures (harness handles them per run).
# Run via: bash scripts/eval_harness/phase3_orchestrator.sh
# Or backgrounded: nohup bash scripts/eval_harness/phase3_orchestrator.sh > /dev/null 2>&1 &

set -uo pipefail

cd "$(dirname "$0")/../.."

MASTER_TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
MASTER_DIR="scripts/eval_harness/results/phase3-master-${MASTER_TS}"
MASTER_LOG="${MASTER_DIR}/orchestrator.log"
SUMMARY="${MASTER_DIR}/SUMMARY.md"

mkdir -p "${MASTER_DIR}"

log() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[${ts}] $*" | tee -a "${MASTER_LOG}"
}

run_phase() {
    local phase_name="$1"
    local models="$2"
    local tests="$3"

    log "=== ${phase_name} START ==="
    log "models: ${models}"
    log "tests: ${tests}"
    local started
    started="$(date +%s)"

    local phase_ts_before
    phase_ts_before="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

    python -m scripts.eval_harness.run \
        --models "${models}" \
        --tests "${tests}" \
        --log-level INFO 2>&1 | tee -a "${MASTER_LOG}"
    local exit_code="${PIPESTATUS[0]}"

    local elapsed=$(( $(date +%s) - started ))
    log "=== ${phase_name} DONE (exit=${exit_code}, elapsed=${elapsed}s) ==="

    # Find the most recent result directory (the one this phase created).
    local result_dir
    result_dir="$(ls -1dt scripts/eval_harness/results/2026-* 2>/dev/null | head -1)"
    log "phase result dir: ${result_dir}"
    echo "${result_dir}"
}

log "=============================================================="
log "Phase 3 orchestrator starting"
log "master dir: ${MASTER_DIR}"
log "=============================================================="

# ---------------------------------------------------------------
# Phase model/test lists -- override via env var to resume or narrow.
# Set PHASE_X_MODELS="" (empty) to skip that phase entirely.
# ---------------------------------------------------------------
PHASE_A_MODELS="${PHASE_A_MODELS-gemma-4-26b-a4b-q5xl-fit,gemma-4-q5xl-fit-no-fa,gemma-4-q5xl-minimal,gemma-4-q5xl-no-fit,gemma-4-q5xl-fit-f16kv,gemma-4-q5xl-stripped}"
PHASE_A_TESTS="${PHASE_A_TESTS-speed_minimal,schema_conformance,speed}"

PHASE_B_MODELS="${PHASE_B_MODELS-gemma-4-26b-a4b-q5xl-fit,gemma-4-26b-a4b-q3xl-fit,gemma-4-26b-a4b-iq4xs-moeoff,gemma-4-26b-a4b-q5xl-fit-128k,gemma-4-26b-a4b-q3xl-fit-target-1024,gemma-4-26b-a4b-q3xl-fit-parallel2}"
PHASE_B_TESTS="${PHASE_B_TESTS-speed_minimal,schema_conformance,tool_selection,personality,rag_integration,coherence,speed}"

PHASE_C_MODELS="${PHASE_C_MODELS-qwen-2.5-14b-q5km,hermes-4-14b-q5km-ngram,hermes-4-14b-q5km-optimized}"
PHASE_C_TESTS="${PHASE_C_TESTS-speed_minimal,schema_conformance,tool_selection,personality,rag_integration,coherence,speed}"

PHASE_A_DIR="SKIPPED"
PHASE_B_DIR="SKIPPED"
PHASE_C_DIR="SKIPPED"

if [ -n "${PHASE_A_MODELS}" ]; then
    PHASE_A_DIR="$(run_phase "PHASE A (ablation)" "${PHASE_A_MODELS}" "${PHASE_A_TESTS}")"
else
    log "PHASE A skipped (PHASE_A_MODELS empty)"
fi

if [ -n "${PHASE_B_MODELS}" ]; then
    PHASE_B_DIR="$(run_phase "PHASE B (Gemma 4 expansive matrix)" "${PHASE_B_MODELS}" "${PHASE_B_TESTS}")"
else
    log "PHASE B skipped (PHASE_B_MODELS empty)"
fi

if [ -n "${PHASE_C_MODELS}" ]; then
    PHASE_C_DIR="$(run_phase "PHASE C (non-Gemma finalists)" "${PHASE_C_MODELS}" "${PHASE_C_TESTS}")"
else
    log "PHASE C skipped (PHASE_C_MODELS empty)"
fi

# ---------------------------------------------------------------
# Master summary
# ---------------------------------------------------------------
log "writing master summary to ${SUMMARY}"

{
    echo "# Phase 3 Master A/B Summary - ${MASTER_TS}"
    echo
    echo "Autonomous orchestrator run. Combines Phase A (ablation), Phase B"
    echo "(expansive Gemma 4 matrix), Phase C (non-Gemma finalists)."
    echo
    echo "## Phase Directories"
    echo
    echo "- **Phase A (ablation):** \`${PHASE_A_DIR}\`"
    echo "- **Phase B (Gemma 4 matrix):** \`${PHASE_B_DIR}\`"
    echo "- **Phase C (non-Gemma finalists):** \`${PHASE_C_DIR}\`"
    echo
    echo "## Phase A - Ablation Report"
    echo
    if [ -f "${PHASE_A_DIR}/report.md" ]; then
        cat "${PHASE_A_DIR}/report.md"
    else
        echo "(report.md missing -- phase may have failed, check orchestrator.log)"
    fi
    echo
    echo "---"
    echo
    echo "## Phase B - Gemma 4 Expansive Matrix Report"
    echo
    if [ -f "${PHASE_B_DIR}/report.md" ]; then
        cat "${PHASE_B_DIR}/report.md"
    else
        echo "(report.md missing -- phase may have failed, check orchestrator.log)"
    fi
    echo
    echo "---"
    echo
    echo "## Phase C - Non-Gemma Finalists Report"
    echo
    if [ -f "${PHASE_C_DIR}/report.md" ]; then
        cat "${PHASE_C_DIR}/report.md"
    else
        echo "(report.md missing -- phase may have failed, check orchestrator.log)"
    fi
    echo
    echo "---"
    echo
    echo "## Orchestrator Log Tail"
    echo
    echo '```'
    tail -80 "${MASTER_LOG}"
    echo '```'
} > "${SUMMARY}"

log "=============================================================="
log "Phase 3 orchestrator COMPLETE"
log "summary: ${SUMMARY}"
log "=============================================================="
