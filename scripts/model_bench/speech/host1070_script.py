"""The host-1070 script of host step H6 and its game variant (mist-model-host plan v2, task T7).

`speech_bench plan --host-1070-script <path>` calls `write_script`, which writes three files
beside each other, outside the repository: `h6.sh` (the H6 schedule), `h6_game.sh` (one model on
the shared placement for a fixed 40-minute window, for run 8's `--with-1070` conditions) and
`extraction_load.json` (the load loop's request body). The host-1070 session copies them to
`~/mist-bench-h6/` and runs the script once with bash in WSL.

Every value the script needs is baked in here from the committed inputs and the run's pin file:
the model files with their sha256 and size, the two pinned llama.cpp images, the server flags of
`small_arms.json` and the window lengths of Run 7. The script reads models already on host-1070
and downloads nothing. It records a filtered `docker inspect` of extraction's two containers
before it stops anything and sets a trap, so any exit (a failed step, Ctrl-C, a closed session)
removes its own containers after saving their logs, starts what it stopped, waits for health and
diffs the inspect.

Stdlib only: the host runs it under a plain Python.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import core, docker_cmds

REFERENCE_MODEL = "e4b-ref"
SHARED = "gtx1070-shared"
FALLBACK = "host1070-cpu"
ALONE = "gtx1070"
GAME_PASS = "game"
LOAD_BODY_NAME = "extraction_load.json"
LOAD_BODY_CHARS = 36_000
LOAD_MAX_TOKENS = 2048
# Flags the script sets itself per server, so they are dropped from `common_args`.
_SCRIPT_FLAGS = ("-ctk", "-ctv", "-fa", "-t", "--host", "--port")
_SAMPLING_FLAGS = (
    ("temperature", "--temp"),
    ("top_p", "--top-p"),
    ("top_k", "--top-k"),
    ("min_p", "--min-p"),
)
_LOAD_SYSTEM = (
    "Extract every entity and relation from the text below and reply with one JSON object."
)


def shared_alias(model_id: str, pass_name: str) -> str:
    return core.small_alias(model_id, f"{SHARED}-{pass_name}")


def game_alias(model_id: str) -> str:
    """The alias the game variant serves under, so run 8's client can tell it apart."""
    return shared_alias(model_id, GAME_PASS)


def expected_aliases(arms: Mapping[str, object] | None = None) -> list[str]:
    """Every alias H6 may serve on port 8090, in schedule order.

    Each model's shared rows list both outcomes of the layer search (the idle and busy passes, or
    the CPU fallback), because which one appears is only known on host-1070.
    """
    doc = arms if arms is not None else core.load_small_arms()
    models = list(doc["models"])
    aliases = []
    for model_id in models:
        aliases.append(shared_alias(model_id, "idle"))
        if model_id != REFERENCE_MODEL:
            aliases.append(shared_alias(model_id, "busy"))
        aliases.append(core.small_alias(model_id, FALLBACK))
    aliases += [core.small_alias(m, ALONE) for m in models]
    return aliases


def server_args(arms: Mapping[str, object]) -> list[str]:
    """`small_arms.json`'s server flags less those the script sets per window, plus sampling."""
    common = list(arms["common_args"])
    args: list[str] = []
    i = 0
    while i < len(common):
        if common[i] in _SCRIPT_FLAGS:
            i += 2
            continue
        args.append(common[i])
        i += 1
    sampling = arms["sampling"]
    for key, flag in _SAMPLING_FLAGS:
        args += [flag, str(sampling[key])]
    thinking = {m["thinking"] for m in arms["models"].values()}
    if len(thinking) != 1:
        raise core.InputError("the host-1070 script needs one thinking setting for every model")
    args += list(arms["thinking_args"][thinking.pop()])
    return args


def load_body(filler: str) -> dict:
    """Extraction-sized work for the load loop: the filler to about 36,000 characters.

    About 9,000 tokens in and up to 2,048 out, the size compose.host.yml describes; `cache_prompt`
    is off so every request pays its full prefill, as fresh extraction text would.
    """
    text = filler
    while len(text) < LOAD_BODY_CHARS:
        text += "\n\n" + filler
    cut = text.rfind(" ", 0, LOAD_BODY_CHARS)
    text = text[: cut if cut > 0 else LOAD_BODY_CHARS]
    return {
        "messages": [
            {"role": "system", "content": _LOAD_SYSTEM},
            {"role": "user", "content": text},
        ],
        "max_tokens": LOAD_MAX_TOKENS,
        "temperature": 0,
        "cache_prompt": False,
    }


def _bash_assoc(values: Mapping[str, object]) -> str:
    return " ".join(f"[{key}]={shlex.quote(str(value))}" for key, value in values.items())


def render(
    variant: str,
    *,
    pins: Sequence[core.ModelPin],
    arms: Mapping[str, object],
    cuda_image: str,
    cpu_image: str,
) -> str:
    """The bash text of `variant` (`h6` or `game`)."""
    if variant not in ("h6", "game"):
        raise ValueError(f"unknown variant {variant!r}")
    margin = arms["placements"][SHARED]["margin_mib"]
    load_loop = (
        "while true; do curl -s -o /dev/null -X POST -H 'Content-Type: application/json' "
        f"--data @/load/{LOAD_BODY_NAME} "
        "http://mist-extraction-llm-host:8080/v1/chat/completions || sleep 1; done"
    )
    warmup = json.dumps({**core.WARMUP_BODY, "max_tokens": 1})
    values = {
        "VARIANT": variant,
        "CUDA_IMAGE": shlex.quote(cuda_image),
        "CPU_IMAGE": shlex.quote(cpu_image),
        "MARGIN_MIB": str(int(margin)),
        "MODEL_IDS": " ".join(shlex.quote(p.id) for p in pins),
        "FILES": _bash_assoc({p.id: p.file for p in pins}),
        "SHAS": _bash_assoc({p.id: p.sha256 for p in pins}),
        "SIZES": _bash_assoc({p.id: p.size for p in pins}),
        "SMALL": _bash_assoc({p.id: int(p.id != REFERENCE_MODEL) for p in pins}),
        "SERVER_ARGS": " ".join(shlex.quote(a) for a in server_args(arms)),
        "WARMUP_JSON": shlex.quote(warmup),
        "LOAD_LOOP": shlex.quote(load_loop),
        "LOAD_BODY_NAME": LOAD_BODY_NAME,
    }
    text = _TEMPLATE
    for key, value in values.items():
        text = text.replace(f"@@{key}@@", value)
    if "@@" in text:
        raise core.InputError("the host-1070 script template has an unfilled placeholder")
    return text


def _default_pins(script_path: Path) -> Path:
    # H6 writes the script to <speech>/host1070/h6.sh; the run's pin file is <speech>/host/pins.json.
    return script_path.parent.parent / "host" / "pins.json"


def write_script(script_path: Path, *, pins_path: Path | None, repo_root: Path) -> list[Path]:
    """Write `h6.sh`, `h6_game.sh` and the load body beside `script_path`; return their paths.

    Refuses, before writing anything, a missing pin file and any of the three files that already
    exists, so a script the lead has read and relayed is never silently replaced.
    """
    pins_file = pins_path if pins_path is not None else _default_pins(script_path)
    if not pins_file.is_file():
        raise core.PinError(f"no pin file at {pins_file}; pass --pins <the run's pins.json>")
    pins_doc = core.load_pins(pins_file)
    arms = core.load_small_arms(repo_root / core.SPEECH_DIR_REL / "small_arms.json")
    pins = [core.pinned_model(pins_doc, model_id) for model_id in arms["models"]]
    images = pins_doc.get("images") if isinstance(pins_doc.get("images"), dict) else {}
    cpu_image = str(images.get("cpu-llama") or docker_cmds.CPU_LLAMA_IMAGE)
    cuda_image = docker_cmds.cuda_llama_image(repo_root / "docker-compose.yml")
    filler = (repo_root / core.SPEECH_DIR_REL / core.CACHE_FILLER_PATH.name).read_text(
        encoding="utf-8"
    )

    game_path = script_path.with_name(f"{script_path.stem}_game.sh")
    body_path = script_path.with_name(LOAD_BODY_NAME)
    targets = [script_path, game_path, body_path]
    for target in targets:
        if target.exists():
            raise core.OutputExistsError(f"{target} already exists; refusing to overwrite it")

    common = {"pins": pins, "arms": arms, "cuda_image": cuda_image, "cpu_image": cpu_image}
    script_path.parent.mkdir(parents=True, exist_ok=True)
    core.write_text_exclusive(script_path, render("h6", **common))
    core.write_text_exclusive(game_path, render("game", **common))
    core.write_json_exclusive(body_path, load_body(filler.replace("\r\n", "\n").strip()))
    return targets


_TEMPLATE = r"""#!/usr/bin/env bash
# M.I.S.T speech bench, host step H6 (mist-model-host plan v2, run 7): the host-1070 rows.
# Written by `speech_bench plan --host-1070-script`; regenerate it rather than editing it.
#
# Run once with bash in WSL from ~/mist-bench-h6/, with extraction_load.json beside it and
# MODELS_DIR set to the directory holding the pinned model files:
#   MODELS_DIR=/path/to/models bash h6.sh                  (the H6 schedule)
#   MODELS_DIR=/path/to/models bash h6_game.sh <model> [ngl]   (run 8: 40 minutes, shared)
# It downloads nothing. Before it stops anything it records a filtered inspect of extraction's
# containers and sets a trap: on any exit it saves its own containers' logs, removes them, starts
# what it stopped, waits for health and diffs the inspect, which must be empty.
set -euo pipefail

VARIANT=@@VARIANT@@
H6_DIR="${H6_DIR:-$HOME/mist-bench-h6}"
MODELS_DIR="${MODELS_DIR:-}"
LOAD_BODY="$H6_DIR/@@LOAD_BODY_NAME@@"

LLM=mist-extraction-llm-host
SVC=mist-extraction-host
TS=mist-extraction-ts
BENCH=mist-bench-small
SEARCH=mist-bench-search
LOAD=mist-bench-extload
SERVE_PORT=8090
SEARCH_PORT=8099
MARGIN_MIB=@@MARGIN_MIB@@
HOST_MARGIN_MIB=1024
RETURN_SLACK_MIB=128
SHARED_IDLE_S=1500
SHARED_BUSY_S=720
ALONE_S=720
GAME_S=2400
SEARCH_S=30
HEALTH_TRIES=450
FULL_NGL=999

CUDA_IMAGE=@@CUDA_IMAGE@@
CPU_IMAGE=@@CPU_IMAGE@@
MODELS=(@@MODEL_IDS@@)
declare -A FILE=(@@FILES@@)
declare -A SHA=(@@SHAS@@)
declare -A SIZE=(@@SIZES@@)
declare -A SMALL=(@@SMALL@@)
SERVER_ARGS=(@@SERVER_ARGS@@)
WARMUP_JSON=@@WARMUP_JSON@@
LOAD_LOOP=@@LOAD_LOOP@@
INSPECT_FORMAT='{"Id":{{json .Id}},"Image":{{json .Image}},"Cmd":{{json .Config.Cmd}},"Env":{{json .Config.Env}},"Entrypoint":{{json .Config.Entrypoint}},"HostConfig":{{json .HostConfig}}}'
HEALTH_FORMAT='{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}'

declare -A SKIP_REASON=()
STOPPED=()
RUN_DIR=""
EV=""
LOG_N=0
SAMPLE_N=0
THREADS=""
CARD_TOTAL=""
STARTED_UTC=""
BEFORE_GPU=""
BEFORE_AVAIL=""
CHOSEN_NGL=""
PROBE_FITS=false
PROBE_TOTAL=""
START_CAP=""
START_LOAD_S=""
START_FA=on
START_CACHE=q8_0
START_FALLBACK=none
GPU_PID=""
FREE_PID=""

log() { printf '[h6 %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() {
  log "FAIL: $*"
  exit 1
}
now() { date +%s.%N; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", b - a }'; }
jstr() {
  local s=$1
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  printf '"%s"' "$s"
}
jnum() { if [[ -n ${1:-} ]]; then printf '%s' "$1"; else printf 'null'; fi; }
join_lines() { if [[ -s $1 ]]; then paste -sd, "$1"; fi; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR == 1 { print $1 + 0 }'; }
gpu_total() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR == 1 { print $1 + 0 }'; }
available_mib() { free -m | awk '/^Mem:/ { print $7 }'; }
state_of() { docker inspect --format '{{.State.Status}}' "$1" 2> /dev/null || echo absent; }
max_of() { awk '$1 ~ /^[0-9]+$/ { if (m == "" || $1 + 0 > m) m = $1 + 0 } END { print m }' "$1"; }
min_of() { awk '$1 ~ /^[0-9]+$/ { if (m == "" || $1 + 0 < m) m = $1 + 0 } END { print m }' "$1"; }

# Refusals come first, before anything is stopped or written.
preflight() {
  [[ -n $MODELS_DIR ]] || die "set MODELS_DIR to host-1070's models directory (the pinned files sit under it)"
  [[ -d $MODELS_DIR ]] || die "MODELS_DIR $MODELS_DIR is not a directory"
  [[ -f $LOAD_BODY ]] || die "copy @@LOAD_BODY_NAME@@ into $H6_DIR beside this script"
  [[ ! -e $RUN_DIR/before ]] || die "$RUN_DIR/before exists; the script runs once per folder"
  local c m path size sum
  for c in "$LLM" "$SVC" "$TS"; do
    [[ $(state_of "$c") == running ]] || die "$c is not running; start extraction first"
  done
  for c in "$BENCH" "$SEARCH" "$LOAD"; do
    [[ $(state_of "$c") == absent ]] || die "$c already exists; remove it before running"
  done
  for m in "${MODELS[@]}"; do
    path="$MODELS_DIR/${FILE[$m]}"
    if [[ ! -f $path ]]; then
      SKIP_REASON[$m]="model file is missing from MODELS_DIR"
      continue
    fi
    size=$(stat -c %s "$path")
    if [[ $size != "${SIZE[$m]}" ]]; then
      SKIP_REASON[$m]="model file size differs from the pin file"
      continue
    fi
    sum=$(sha256sum "$path" | awk '{ print $1 }')
    if [[ $sum != "${SHA[$m]}" ]]; then
      SKIP_REASON[$m]="model file sha256 differs from the pin file"
    fi
  done
}

record_state() {
  local dir=$1 c
  mkdir -p "$dir"
  for c in "$LLM" "$SVC"; do
    docker inspect --format "$INSPECT_FORMAT" "$c" > "$dir/$c.inspect.json"
  done
  nvidia-smi > "$dir/nvidia-smi.txt" 2>&1 || true
  free -m > "$dir/free.txt"
}

begin() {
  mkdir -p "$RUN_DIR/logs" "$RUN_DIR/samples" "$RUN_DIR/events"
  EV="$RUN_DIR/events"
  : > "$EV/searches.jsonl"
  : > "$EV/windows.jsonl"
  : > "$EV/skips.jsonl"
  STARTED_UTC=$(utc)
  THREADS=$(nproc)
  CARD_TOTAL=$(gpu_total)
  record_state "$RUN_DIR/before"
  BEFORE_GPU=$(gpu_used)
  BEFORE_AVAIL=$(available_mib)
  trap restore EXIT
  trap 'exit 130' INT TERM HUP
}

stop_extraction() {
  STOPPED+=("$1")
  log "docker stop $1"
  docker stop "$1" > /dev/null || die "could not stop $1"
}

# Logs first, then stop, then rm: nothing of a bench container is lost when it goes.
remove_own() {
  local name=$1 tag=$2 log_path
  if [[ $(state_of "$name") == absent ]]; then
    return 0
  fi
  LOG_N=$((LOG_N + 1))
  log_path="$RUN_DIR/logs/$(printf '%02d' "$LOG_N")-$tag.log"
  docker logs "$name" > "$log_path" 2>&1 || true
  docker stop "$name" > /dev/null 2>&1 || true
  docker rm "$name" > /dev/null 2>&1 || log "could not remove $name"
}

add_skip() {
  log "skip $1 on $2: $3"
  printf '{"model":%s,"placement":%s,"reason":%s}\n' "$(jstr "$1")" "$(jstr "$2")" "$(jstr "$3")" >> "$EV/skips.jsonl"
}

run_server() {
  local name=$1 port=$2 kind=$3 model=$4 alias=$5 ngl=$6 fa=$7 cache=$8 cap=$9
  local image=$CUDA_IMAGE
  local args=(docker run -d --name "$name" --memory "${cap}m" --network "container:$TS" -v "$MODELS_DIR:/models:ro")
  if [[ $kind == cuda ]]; then
    args+=(--gpus all)
  else
    image=$CPU_IMAGE
  fi
  args+=("$image" -m "/models/${FILE[$model]}" --alias "$alias" --host 0.0.0.0 --port "$port")
  args+=(-t "$THREADS" -fa "$fa" -ctk "$cache" -ctv "$cache" "${SERVER_ARGS[@]}")
  if [[ -n $ngl ]]; then
    args+=(-ngl "$ngl")
  fi
  "${args[@]}" > /dev/null || die "docker run $name ($alias) failed"
}

wait_server() {
  local name=$1 port=$2 i
  for ((i = 0; i < HEALTH_TRIES; i++)); do
    [[ $(state_of "$name") == running ]] || return 1
    if docker exec "$name" curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
      docker exec "$name" curl -sf -X POST -H 'Content-Type: application/json' \
        -d "$WARMUP_JSON" "http://127.0.0.1:$port/v1/chat/completions" > /dev/null 2>&1
      return
    fi
    sleep 2
  done
  return 1
}

# Load time runs from docker run to the reply of a one-token warm-up request. If flash attention
# or the q8_0 cache fails on the Pascal card, the window is tried again with -fa off and f16.
start_server() {
  local name=$1 port=$2 kind=$3 model=$4 alias=$5 ngl=$6 attempt fa cache t0
  for attempt in 1 2; do
    if [[ $attempt == 1 ]]; then
      fa=on
      cache=q8_0
    else
      fa=off
      cache=f16
    fi
    START_CAP=$(($(available_mib) - HOST_MARGIN_MIB))
    t0=$(now)
    run_server "$name" "$port" "$kind" "$model" "$alias" "$ngl" "$fa" "$cache" "$START_CAP"
    if wait_server "$name" "$port"; then
      START_LOAD_S=$(elapsed "$t0" "$(now)")
      START_FA=$fa
      START_CACHE=$cache
      if [[ $attempt == 1 ]]; then START_FALLBACK=none; else START_FALLBACK=fa-off-f16; fi
      return 0
    fi
    log "$alias did not start with -fa $fa and a $cache cache"
    remove_own "$name" "$alias-fa-$fa-failed"
  done
  START_LOAD_S=""
  START_FALLBACK=failed
  return 1
}

start_samplers() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$1.gpu.csv" 2> /dev/null &
  GPU_PID=$!
  (while true; do
    available_mib
    sleep 1
  done) > "$1.free.csv" 2> /dev/null &
  FREE_PID=$!
}

stop_samplers() {
  kill "$GPU_PID" "$FREE_PID" 2> /dev/null || true
  wait "$GPU_PID" "$FREE_PID" 2> /dev/null || true
}

wait_memory_returned() {
  local target=$(($1 + RETURN_SLACK_MIB)) i
  for ((i = 0; i < 60; i++)); do
    if [[ $(gpu_used) -le $target ]]; then
      return 0
    fi
    sleep 1
  done
}

start_load_loop() {
  log "load loop on"
  docker run -d --name "$LOAD" --network "container:$TS" --entrypoint sh \
    -v "$H6_DIR:/load:ro" "$CUDA_IMAGE" -c "$LOAD_LOOP" > /dev/null || die "could not start the load loop"
}

stop_load_loop() {
  log "load loop off"
  remove_own "$LOAD" load-loop
}

# One search step on port 8099, which the main PC cannot reach: serve, sample at extraction's
# peak with the load loop running, keep the count if the card keeps MARGIN_MIB free.
probe() {
  local model=$1 ngl=$2 tag peak after free started=true
  SAMPLE_N=$((SAMPLE_N + 1))
  tag="$RUN_DIR/samples/$(printf '%02d' "$SAMPLE_N")-search-$model-ngl$ngl"
  PROBE_FITS=false
  PROBE_TOTAL=""
  peak=""
  free=""
  if start_server "$SEARCH" "$SEARCH_PORT" cuda "$model" "$model@search" "$ngl"; then
    start_samplers "$tag"
    sleep "$SEARCH_S"
    stop_samplers
    after=$(gpu_used)
    peak=$(max_of "$tag.gpu.csv")
    [[ -n $peak && $peak -ge $after ]] || peak=$after
    free=$((CARD_TOTAL - peak))
    if [[ $free -ge $MARGIN_MIB ]]; then
      PROBE_FITS=true
    fi
    PROBE_TOTAL=$(docker logs "$SEARCH" 2>&1 | sed -n 's/.*offloaded [0-9]*\/\([0-9]*\) layers.*/\1/p' | tail -n 1)
    remove_own "$SEARCH" "search-$model-ngl$ngl"
  else
    started=false
  fi
  log "search $model -ngl $ngl: free $free MiB, fits $PROBE_FITS"
  printf '{"model":%s,"ngl":%s,"fa":%s,"cache":%s,"started":%s,"gpu_peak_mib":%s,"gpu_free_mib":%s,"fits":%s}\n' \
    "$(jstr "$model")" "$ngl" "$(jstr "$START_FA")" "$(jstr "$START_CACHE")" "$started" \
    "$(jnum "$peak")" "$(jnum "$free")" "$PROBE_FITS" >> "$EV/searches.jsonl"
}

# The largest layer count that keeps MARGIN_MIB free; empty when even -ngl 0 does not.
search_layers() {
  local model=$1 lo hi mid
  CHOSEN_NGL=""
  probe "$model" 0
  [[ $PROBE_FITS == true ]] || return 0
  hi=${PROBE_TOTAL:-$FULL_NGL}
  probe "$model" "$hi"
  if [[ $PROBE_FITS == true ]]; then
    CHOSEN_NGL=$hi
    return 0
  fi
  lo=0
  while [[ $((hi - lo)) -gt 1 ]]; do
    mid=$(((lo + hi) / 2))
    probe "$model" "$mid"
    if [[ $PROBE_FITS == true ]]; then lo=$mid; else hi=$mid; fi
  done
  CHOSEN_NGL=$lo
}

# One measured window on port 8090 under the alias the main PC's client waits for.
run_window() {
  local model=$1 placement=$2 pass=$3 alias=$4 kind=$5 ngl=$6 seconds=$7
  local start end gpu_before avail_before after_load="" peak="" avail_peak="" after_unload=""
  local unload_s="" status=served t0 tag
  log "window $alias for $seconds s"
  start=$(utc)
  gpu_before=$(gpu_used)
  avail_before=$(available_mib)
  if start_server "$BENCH" "$SERVE_PORT" "$kind" "$model" "$alias" "$ngl"; then
    after_load=$(gpu_used)
    SAMPLE_N=$((SAMPLE_N + 1))
    tag="$RUN_DIR/samples/$(printf '%02d' "$SAMPLE_N")-$alias"
    start_samplers "$tag"
    sleep "$seconds"
    stop_samplers
    peak=$(max_of "$tag.gpu.csv")
    [[ -n $peak && $peak -ge $after_load ]] || peak=$after_load
    avail_peak=$(min_of "$tag.free.csv")
    [[ -n $avail_peak ]] || avail_peak=$(available_mib)
    t0=$(now)
    remove_own "$BENCH" "$alias"
    wait_memory_returned "$gpu_before"
    unload_s=$(elapsed "$t0" "$(now)")
    after_unload=$(gpu_used)
  else
    status=failed
    log "window $alias failed to start"
  fi
  end=$(utc)
  {
    printf '{"model":%s,"placement":%s,"pass":%s,"alias":%s,"image":%s,"ngl":%s,' \
      "$(jstr "$model")" "$(jstr "$placement")" "$(jstr "$pass")" "$(jstr "$alias")" \
      "$(jstr "$kind")" "$(jnum "$ngl")"
    printf '"status":%s,"fallback":%s,"fa":%s,"cache":%s,"start_utc":%s,"end_utc":%s,' \
      "$(jstr "$status")" "$(jstr "$START_FALLBACK")" "$(jstr "$START_FA")" \
      "$(jstr "$START_CACHE")" "$(jstr "$start")" "$(jstr "$end")"
    printf '"window_s":%s,"load_s":%s,"unload_s":%s,"memory_cap_mib":%s,' \
      "$seconds" "$(jnum "$START_LOAD_S")" "$(jnum "$unload_s")" "$(jnum "$START_CAP")"
    printf '"gpu_before_mib":%s,"gpu_after_load_mib":%s,"gpu_peak_mib":%s,"gpu_after_unload_mib":%s,' \
      "$(jnum "$gpu_before")" "$(jnum "$after_load")" "$(jnum "$peak")" "$(jnum "$after_unload")"
    printf '"free_available_before_mib":%s,"free_available_peak_mib":%s}\n' \
      "$(jnum "$avail_before")" "$(jnum "$avail_peak")"
  } >> "$EV/windows.jsonl"
}

fits_host_memory() {
  local need=$(((${SIZE[$1]} + 1048575) / 1048576 + HOST_MARGIN_MIB))
  [[ $need -le $(available_mib) ]]
}

shared_block() {
  local model=$1
  if [[ -n ${SKIP_REASON[$model]:-} ]]; then
    add_skip "$model" gtx1070-shared "${SKIP_REASON[$model]}"
    return 0
  fi
  if ! fits_host_memory "$model"; then
    add_skip "$model" gtx1070-shared "does not fit beside extraction in host memory"
    return 0
  fi
  start_load_loop
  search_layers "$model"
  stop_load_loop
  if [[ -z $CHOSEN_NGL ]]; then
    add_skip "$model" gtx1070-shared "does not fit beside extraction"
    run_window "$model" host1070-cpu idle "$model@host1070-cpu" cpu "" "$SHARED_IDLE_S"
    return 0
  fi
  run_window "$model" gtx1070-shared idle "$model@gtx1070-shared-idle" cuda "$CHOSEN_NGL" "$SHARED_IDLE_S"
  if [[ ${SMALL[$model]} == 1 ]]; then
    start_load_loop
    run_window "$model" gtx1070-shared busy "$model@gtx1070-shared-busy" cuda "$CHOSEN_NGL" "$SHARED_BUSY_S"
    stop_load_loop
  fi
}

alone_window() {
  local model=$1
  if [[ -n ${SKIP_REASON[$model]:-} ]]; then
    add_skip "$model" gtx1070 "${SKIP_REASON[$model]}"
    return 0
  fi
  run_window "$model" gtx1070 alone "$model@gtx1070" cuda "$FULL_NGL" "$ALONE_S"
}

wait_healthy() {
  local name=$1 i status=unknown
  for ((i = 0; i < 150; i++)); do
    status=$(docker inspect --format "$HEALTH_FORMAT" "$name" 2> /dev/null || echo absent)
    if [[ $status == healthy || $status == running ]]; then
      break
    fi
    sleep 2
  done
  if [[ $status == running ]]; then status=healthy; fi
  printf '%s' "$status"
}

write_summary() {
  local code=$1 started=$2 health=$3 empty=$4 containers=$5 f="$RUN_DIR/summary.json"
  {
    printf '{\n  "schema": 1,\n  "variant": %s,\n' "$(jstr "$VARIANT")"
    printf '  "started_utc": %s,\n  "finished_utc": %s,\n' "$(jstr "$STARTED_UTC")" "$(jstr "$(utc)")"
    printf '  "exit_code": %s,\n  "threads": %s,\n  "card_total_mib": %s,\n' \
      "$code" "$(jnum "$THREADS")" "$(jnum "$CARD_TOTAL")"
    printf '  "before": {"gpu_used_mib": %s, "free_available_mib": %s},\n' \
      "$(jnum "$BEFORE_GPU")" "$(jnum "$BEFORE_AVAIL")"
    printf '  "searches": [%s],\n' "$(join_lines "$EV/searches.jsonl")"
    printf '  "windows": [%s],\n' "$(join_lines "$EV/windows.jsonl")"
    printf '  "skips": [%s],\n' "$(join_lines "$EV/skips.jsonl")"
    printf '  "restore": {"started": [%s], "health": {%s}, "inspect_diff": {"empty": %s, "containers": {%s}}}\n' \
      "$started" "$health" "$empty" "$containers"
    printf '}\n'
  } > "$f.tmp"
  mv "$f.tmp" "$f"
}

was_stopped() {
  local name
  for name in "${STOPPED[@]}"; do
    if [[ $name == "$1" ]]; then
      return 0
    fi
  done
  return 1
}

# The EXIT trap: runs once, on success, on failure and on a signal.
restore() {
  local code=$?
  trap - EXIT INT TERM HUP
  set +e
  log "restoring extraction (exit $code)"
  stop_samplers
  local c state empty=true started="" health="" containers="" status
  for c in "$BENCH" "$SEARCH" "$LOAD"; do
    remove_own "$c" "restore-$c"
  done
  for c in "$LLM" "$SVC"; do
    if was_stopped "$c"; then
      log "docker start $c"
      if docker start "$c" > /dev/null; then
        started+="${started:+,}$(jstr "$c")"
      else
        code=1
      fi
    fi
  done
  for c in "$LLM" "$SVC"; do
    status=$(wait_healthy "$c")
    [[ $status == healthy ]] || code=1
    health+="${health:+, }$(jstr "$c"): $(jstr "$status")"
  done
  record_state "$RUN_DIR/after"
  for c in "$LLM" "$SVC"; do
    if diff -u "$RUN_DIR/before/$c.inspect.json" "$RUN_DIR/after/$c.inspect.json" > "$RUN_DIR/after/$c.inspect.diff"; then
      state=true
    else
      state=false
      empty=false
    fi
    containers+="${containers:+, }$(jstr "$c"): $state"
  done
  if [[ $empty != true ]]; then
    log "the inspect diff is not empty; see $RUN_DIR/after/*.inspect.diff"
    [[ $code != 0 ]] || code=1
  fi
  write_summary "$code" "$started" "$health" "$empty" "$containers"
  log "summary at $RUN_DIR/summary.json"
  exit "$code"
}

main_h6() {
  local m
  RUN_DIR=$H6_DIR
  preflight
  begin
  stop_extraction "$SVC"
  for m in "${MODELS[@]}"; do
    shared_block "$m"
  done
  stop_extraction "$LLM"
  for m in "${MODELS[@]}"; do
    alone_window "$m"
  done
  log "every window done"
}

main_game() {
  local model=${1:-} ngl=${2:-}
  if [[ -z $model || -z ${FILE[$model]:-} ]]; then
    die "usage: bash h6_game.sh <model> [ngl]; models: ${MODELS[*]}"
  fi
  if [[ -n $ngl && ! $ngl =~ ^[0-9]+$ ]]; then
    die "ngl must be a whole number, not $ngl"
  fi
  RUN_DIR="$H6_DIR/game-$(date -u +%Y%m%dT%H%M%SZ)"
  preflight
  if [[ -n ${SKIP_REASON[$model]:-} ]]; then
    die "$model: ${SKIP_REASON[$model]}"
  fi
  fits_host_memory "$model" || die "$model does not fit beside extraction in host memory"
  begin
  stop_extraction "$SVC"
  if [[ -z $ngl ]]; then
    start_load_loop
    search_layers "$model"
    stop_load_loop
    ngl=$CHOSEN_NGL
    if [[ -z $ngl ]]; then
      add_skip "$model" gtx1070-shared "does not fit beside extraction"
      die "$model does not fit beside extraction"
    fi
  fi
  run_window "$model" gtx1070-shared game "$model@gtx1070-shared-game" cuda "$ngl" "$GAME_S"
  log "game window done"
}

case $VARIANT in
  h6) main_h6 "$@" ;;
  game) main_game "$@" ;;
  *) die "unknown variant $VARIANT" ;;
esac
"""
