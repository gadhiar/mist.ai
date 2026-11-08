"""The speech bench report: the `report` subcommand (mist-model-host plan v2, task T6).

`report` reads every suite's rows under `<results-root>/<run>/speech/` and writes two files there:
`numbers.json`, one object per run holding each figure's median, slow case (nearest-rank 90th
percentile), sample count and mark, and `REPORT.md`, the same in tables. `REPORT.md` adds the
budget of the reassessment, section 8, with each stage this bench measures put in; a levels
section answering each question the tier design asks; and what a 16 to 24 GB card would add.

Marks set a figure against its target and floor; they do not rule on it. A figure the report
computes from other figures rather than reads from rows (a sum with a carried-in stage, a rate
scaled by bandwidth, a fit on a card this PC does not have) is labelled derived and lists its
inputs, so a reader can tell the two apart.

The blind ratings are unblinded through `listening.read_key`, which refuses until Raj's
`listening/ratings.csv` exists; that refusal comes before anything is written. Runs 8 and 9 read
"not yet run" until a game session's rows exist, and `report --replace` renders both files again
when they arrive. The larger models' file sizes come from a file the lead writes in H9
(`host/larger_models.json`, or `--larger-models`); nothing here fetches anything.

Stdlib only: the host runs it under a plain Python.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import chain, core, listening, lowtier_suite, toolcall

SUITE = "report"
NUMBERS_FILE = "numbers.json"
REPORT_FILE = "REPORT.md"
LARGER_MODELS_FILE = "larger_models.json"
FIXTURE_DIR = core.SPEECH_FIXTURES_DIR / "report"
FIXTURE_ROWS = FIXTURE_DIR / "rows.jsonl"
GOLDEN_REPORT = FIXTURE_DIR / "expected_REPORT.md"
FIXTURE_RUN = "mh1"
FIXTURE_GIT_SHA = "fixture"

NOT_YET_RUN = "not yet run"
NOT_MEASURED = "not measured"
NO_TARGET = "no target"
NO_DATA = "no data"
BUDGET_SOURCE = "the reassessment, section 8"

TTS_ORDER = ("qwen3", "turbo", "original", "pocket")
STT_ORDER = ("whisper-base", "parakeet-gpu", "parakeet-cpu", "nemotron-160")
CACHE_ARMS = ("c0", "c0-ckpt", "c0-swafull")
CONDITIONS = ("idle", "decoding")
BUSY_DIR = chain.BUSY_PLACEMENT_DIR
SHARED_IDLE = "gtx1070-shared-idle"
SHARED_BUSY = "gtx1070-shared-busy"
PLACEMENT_ORDER = (
    "gpu-full",
    "gpu-4g",
    "gpu-2g",
    "cpu",
    SHARED_IDLE,
    SHARED_BUSY,
    "host1070-cpu",
    "gtx1070",
    BUSY_DIR,
)
CHAIN_GROUPS = (("command and short", ("command", "short")), ("hand-off", ("hand_off",)))

SILENCE_MS = 200
FIRST_CLAUSE_TOKENS = 10
MARGIN_MIB = 512
GATE_MIB = 2048
TODAY_BANDWIDTH_GBS = 504.0
NCMOE_STEP_MIB = 400
NCMOE_STEP_TPS = 1.4


class ReportError(core.SpeechBenchError):
    """The results cannot be reported as they stand."""


# ---------------------------------------------------------------------------
# Targets, the section 8 budget, the cards and the design questions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Target:
    """A figure's target and floor; a target with no floor is a single limit to stay under."""

    name: str
    target: float
    floor: float | None
    unit: str

    def as_dict(self) -> dict[str, object]:
        """The target as a JSON-ready mapping."""
        return {"name": self.name, "target": self.target, "floor": self.floor, "unit": self.unit}


TTS_FIRST_AUDIO = Target("speech first audio", 250, 500, "ms")
STT_FINAL = Target("final text", 80, 250, "ms")
PREFILL = Target("prefill to first token", 120, 250, "ms")
TURN_INFERENCE = Target("end-of-turn inference", 15, None, "ms")
CHAINED_TURN = Target("chained turn, commands and short answers", 2000, 3000, "ms")
BUDGET_TOTAL = Target("total, median", 865, 1470, "ms")
BUDGET_SLOW = Target("total, slow case", 1600, 2100, "ms")


@dataclass(frozen=True, slots=True)
class BudgetStage:
    """One stage of the section 8 budget, from the end of speech to first audio."""

    stage: int
    name: str
    today: str
    floor: float
    target: float
    run: str | None

    @property
    def goal(self) -> Target:
        """The stage's target and floor."""
        return Target(self.name, self.target, self.floor, "ms")


BUDGET_STAGES = (
    BudgetStage(1, "End of turn", "1,440 (wired path)", 250, 215, "run 4"),
    BudgetStage(2, "Speech to text, final", "200 to 600", 250, 80, "run 3"),
    BudgetStage(3, "Memory fetch", "unmeasured", 50, 30, None),
    BudgetStage(4, "Prefill to first token", "about 1,100 (cold, 6,100 tokens)", 250, 120, "run 2"),
    BudgetStage(5, "First speakable words", "220", 120, 110, None),
    BudgetStage(6, "Synthesis to first audio", "1,500 to 1,900", 500, 250, "run 1"),
    BudgetStage(7, "Send and start playback", "about 100", 100, 60, None),
)
BUDGET_TOTAL_TODAY = "about 3,700 to 5,400"
BUDGET_SLOW_TODAY = "unknown"


@dataclass(frozen=True, slots=True)
class Card:
    """A card the larger-card section scales to; the figures are published ones, not measured."""

    name: str
    memory_gb: int
    bandwidth_gbs: float
    bandwidth: str
    price: str


CARDS = (
    Card("RTX 4070 SUPER (today)", 12, 504, "about 504 GB/s", "n/a"),
    Card("RTX 5070 Ti", 16, 896, "896 GB/s", "about 1,200 to 1,420 USD (list price 749)"),
    Card("RTX 5080", 16, 960, "960 GB/s", "about 1,800 to 2,260 USD (list price 999)"),
    Card("RTX 4090, used", 24, 1008, "about 1,008 GB/s", "about 1,200 to 1,500 USD"),
    Card("RTX 3090, used", 24, 936, "936 GB/s", "trackers disagree"),
)
LARGER_MEMORY_GB = (16, 24)

LEVEL_QUESTIONS = (
    "Where the low level runs: 4070 slice, CPU or GTX 1070",
    "Which small model is the interaction tier",
    "Whether the small model can route its own requests up",
    "What each tier costs per request",
    "Whether both tiers stay loaded at full level",
    "What a level change costs",
    "What a low-level request costs while the top tier is writing, at full power",
    "Whether a busy extraction job on host-1070 slows a low level there",
    "What the low level keeps on the CPU",
    "What the top tier could become on a 16 to 24 GB card",
    "Whether extraction can stay on host-1070 beside a low level there",
)
LEVEL_RUNS = (
    "7, then 8 and 9",
    "7",
    "7",
    "2 and 7",
    "5",
    "7 and 8",
    "7",
    "7",
    "7",
    "5 (derived)",
    "7",
)
PLACEMENT_COLUMNS = (
    "chained turn with endpointing",
    "tool-call validity",
    "hand-off validity",
    "memory",
    "load",
    "unload",
    "game cost, run 8",
    "game cost, run 9",
)


# ---------------------------------------------------------------------------
# Marks and figures
# ---------------------------------------------------------------------------


def mark(value: float | None, target: Target) -> str:
    """The value against its target and floor: `target met`, `floor met` or `over floor`."""
    if value is None:
        return NO_DATA
    if target.floor is None:
        limit = f"{target.target:g} {target.unit}"
        return f"under {limit}" if value < target.target else f"{limit} or over"
    if value <= target.target:
        return "target met"
    if value <= target.floor:
        return "floor met"
    return "over floor"


def memory_mark(headroom: float | None, *, fits: bool = True) -> str:
    """Headroom against the 2 GiB gate and the 512 MiB margin."""
    if not fits:
        return "does not fit"
    if headroom is None:
        return NO_DATA
    if headroom >= GATE_MIB:
        return "2 GiB gate met"
    if headroom >= MARGIN_MIB:
        return "512 MiB margin met"
    return "under 512 MiB margin"


def _num(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _nums(values: Iterable[object]) -> list[float]:
    return [n for n in (_num(v) for v in values) if n is not None]


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _fig(
    label: str,
    unit: str,
    *,
    stat: str,
    n: int,
    basis: float | None,
    keys: Mapping[str, str] | None = None,
    target: Target | None = None,
    mark_text: str | None = None,
    derived: bool = False,
    inputs: Sequence[str] = (),
    **values: object,
) -> dict[str, object]:
    fig: dict[str, object] = {"keys": dict(keys)} if keys else {}
    fig.update(label=label, unit=unit, stat=stat, n=n, **values)
    if target is not None:
        fig["target"] = target.as_dict()
        fig["mark"] = mark(basis, target)
    else:
        fig["mark"] = mark_text or NO_TARGET
    fig["derived"] = derived
    if derived:
        if not inputs:
            raise ReportError(f"derived figure {label!r} names no inputs")
        fig["inputs"] = list(inputs)
    return fig


def summary_fig(label: str, values: Iterable[object], unit: str, **kw: object) -> dict:
    """Median and nearest-rank 90th percentile, with the sample count."""
    s = core.summarize(_nums(values))
    return _fig(
        label, unit, stat="summary", n=s.n, basis=s.median, median=_r(s.median), p90=_r(s.p90), **kw
    )


def value_fig(label: str, value: object, unit: str, *, n: int = 1, **kw: object) -> dict:
    """A single reading (a peak, a load time, a count) and the samples behind it."""
    v = _num(value)
    return _fig(label, unit, stat="value", n=n if v is not None else 0, basis=v, value=_r(v), **kw)


def mean_fig(label: str, values: Iterable[object], unit: str, **kw: object) -> dict:
    """A mean, for ratings on a 1-to-5 scale, where a median says little."""
    data = _nums(values)
    v = statistics.fmean(data) if data else None
    return _fig(label, unit, stat="mean", n=len(data), basis=v, value=_r(v), **kw)


def share_fig(label: str, flags: Iterable[object], **kw: object) -> dict:
    """How many of the cases hold, out of how many; `None` cases are not counted."""
    seen = [bool(f) for f in flags if f is not None]
    count = sum(seen)
    v = count / len(seen) if seen else None
    return _fig(label, "share", stat="share", n=len(seen), basis=v, count=count, value=_r(v), **kw)


def basis(fig: Mapping[str, object] | None) -> float | None:
    """The figure's headline number: its median, or its single value."""
    if not isinstance(fig, Mapping):
        return None
    return _num(fig.get("median", fig.get("value")))


def _slow(fig: Mapping[str, object]) -> float | None:
    return _num(fig.get("p90")) if fig.get("stat") == "summary" else basis(fig)


def figure_table(title: str, keys: Sequence[str], figures: list, note: str | None = None) -> dict:
    """A table with one figure per row."""
    doc: dict[str, object] = {"type": "figures", "title": title, "keys": list(keys)}
    doc["figures"] = figures
    if note:
        doc["note"] = note
    return doc


def grid_table(
    title: str, keys: Sequence[str], columns: Sequence[str], rows: list, note: str | None = None
) -> dict:
    """A table with one row per key and one measure per column, set side by side."""
    doc: dict[str, object] = {"type": "grid", "title": title, "keys": list(keys)}
    doc.update(columns=list(columns), rows=rows)
    if note:
        doc["note"] = note
    return doc


def _grid_row(keys: Mapping[str, str], cells: Mapping[str, object]) -> dict:
    return {"keys": dict(keys), "cells": dict(cells)}


# ---------------------------------------------------------------------------
# Reading the results tree
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    """Every row of a JSONL file, refusing a line that is not a JSON object."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise core.InputError(f"{path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise core.InputError(f"{path}:{number}: not a JSON object")
        rows.append(row)
    return rows


def _jsonl(path: Path) -> list[dict]:
    return read_jsonl(path) if path.is_file() else []


def _json(path: Path) -> dict | None:
    return core.load_json(path) if path.is_file() else None


@dataclass(frozen=True, slots=True)
class GameSession:
    """One `game-session` directory: run 9 when it recorded a server process, else run 8."""

    session: str
    meta: dict
    passes: list
    conditions: list
    chain: list
    e4b_return: dict | None

    @property
    def run(self) -> int:
        """8 for a light game, 9 for Minecraft with its server on this PC."""
        return 9 if self.meta.get("server_process") else 8


@dataclass(frozen=True, slots=True)
class Ratings:
    """The unblinded key and Raj's ratings."""

    key: dict
    rows: list


@dataclass(frozen=True, slots=True)
class Inputs:
    """Every row the report reads, grouped as the suites wrote them."""

    tts_samples: list
    tts_load: list
    cache: dict
    stt: list
    turn: list
    fit_steps: list
    lowtier: dict
    lowtier_memory: dict
    e4b_reload: list
    host1070: list
    chain: dict
    games: list
    ratings: Ratings | None
    larger: dict | None
    models: tuple


def _pairs(base: Path, name: str) -> dict[tuple[str, str], list]:
    found = {}
    for path in sorted(base.glob(f"*/*/{name}")):
        found[(path.parent.parent.name, path.parent.name)] = read_jsonl(path)
    return found


def _read_ratings(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(listening.RATINGS_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise core.InputError(f"{path} lacks the columns {sorted(missing)}")
        return [dict(row) for row in reader]


def load_larger_models(path: Path) -> dict:
    """The lead's file of the larger models' published file sizes, checked before any write."""
    doc = core.load_json(path)
    if doc.get("schema") != 1:
        raise core.InputError(f"{path}: schema {doc.get('schema')!r}, not 1")
    models = doc.get("models")
    if not isinstance(models, list) or not models:
        raise core.InputError(f"{path} has no models list")
    entries = list(models)
    reference = doc.get("reference")
    if reference is not None:
        entries.append(reference)
    for entry in entries:
        size = entry.get("size_bytes") if isinstance(entry, dict) else None
        if not isinstance(entry, dict) or not entry.get("id") or _num(size) is None or size <= 0:
            raise core.InputError(f"{path}: every model needs an id and a positive size_bytes")
    return doc


def load_inputs(speech: Path, *, larger_models: Path | None = None) -> Inputs:
    """Read the tree; unblinding refuses here, before anything is written, without ratings."""
    ratings = None
    if (speech / listening.SUITE).exists() or (speech / listening.KEY_SUITE).exists():
        key = listening.read_key(speech)
        ratings = Ratings(key, _read_ratings(speech / listening.SUITE / listening.RATINGS_FILE))

    larger_path = larger_models or speech / "host" / LARGER_MODELS_FILE
    if larger_models is not None and not larger_models.is_file():
        raise core.InputError(f"--larger-models {larger_models} does not exist")
    larger = load_larger_models(larger_path) if larger_path.is_file() else None

    cache = {}
    for path in sorted((speech / "cache").glob("*/requests.jsonl")):
        cache[path.parent.name] = read_jsonl(path)

    lowtier_dir = speech / lowtier_suite.SUITE
    memory = {}
    for path in sorted(lowtier_dir.glob("*/*/memory.json")):
        memory[(path.parent.parent.name, path.parent.name)] = core.load_json(path)

    games = []
    game_dir = speech / "game"
    for meta_path in sorted(game_dir.glob("*/meta.json")):
        session = meta_path.parent
        chain_rows = [r for p in sorted(session.glob("*/pass*/chain.jsonl")) for r in read_jsonl(p)]
        games.append(
            GameSession(
                session=session.name,
                meta=core.load_json(meta_path),
                passes=_jsonl(session / "passes.jsonl"),
                conditions=_jsonl(session / "conditions.jsonl"),
                chain=chain_rows,
                e4b_return=_json(session / "e4b_return.json"),
            )
        )

    fit = _json(speech / "fit" / "steps.json") or {}
    return Inputs(
        tts_samples=_jsonl(speech / "tts" / "samples.jsonl"),
        tts_load=_jsonl(speech / "tts" / "load.jsonl"),
        cache=cache,
        stt=_jsonl(speech / "stt" / "samples.jsonl"),
        turn=_jsonl(speech / "turn" / "samples.jsonl"),
        fit_steps=list(fit.get("steps", [])),
        lowtier=_pairs(lowtier_dir, "requests.jsonl"),
        lowtier_memory=memory,
        e4b_reload=_jsonl(lowtier_dir / "e4b_reload.jsonl"),
        host1070=_jsonl(speech / lowtier_suite.REMOTE_MEMORY_SUITE / "memory.jsonl"),
        chain=_pairs(speech / chain.SUITE, "turns.jsonl"),
        games=games,
        ratings=ratings,
        larger=larger,
        models=tuple(core.load_small_arms()["models"]),
    )


def _git_shas(speech: Path) -> list[str]:
    shas: set[str] = set()
    for path in speech.rglob("*"):
        if path.suffix == ".jsonl":
            shas |= {str(r["git_sha"]) for r in read_jsonl(path) if r.get("git_sha")}
        elif path.suffix == ".json" and path.name not in (NUMBERS_FILE, LARGER_MODELS_FILE):
            doc = core.load_json(path)
            if doc.get("git_sha"):
                shas.add(str(doc["git_sha"]))
    return sorted(shas)


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def _index(order: Sequence[str], value: str) -> int:
    return order.index(value) if value in order else len(order)


def _ordered(values: Iterable[str], order: Sequence[str]) -> list[str]:
    return sorted(set(values), key=lambda v: (_index(order, v), v))


def _ordered_pairs(pairs: Iterable[tuple[str, str]], models: Sequence[str]) -> list:
    return sorted(
        set(pairs),
        key=lambda p: (_index(models, p[0]), p[0], _index(PLACEMENT_ORDER, p[1]), p[1]),
    )


# ---------------------------------------------------------------------------
# Runs 1 to 5
# ---------------------------------------------------------------------------


def _run_doc(
    number: int, title: str, tables: list, notes: Sequence[str] = (), why: str = ""
) -> dict:
    doc: dict[str, object] = {"run": number, "title": title}
    doc["status"] = "measured" if tables else NOT_YET_RUN
    doc.update(tables=tables, notes=list(notes))
    if not tables:
        doc["why"] = why
    return doc


def _run1(inp: Inputs) -> dict:
    engines = _ordered((str(r.get("engine")) for r in inp.tts_samples), TTS_ORDER)
    figs = []
    for engine in engines:
        for condition in CONDITIONS:
            rows = [
                r
                for r in inp.tts_samples
                if r.get("engine") == engine and r.get("condition") == condition
            ]
            lines = [r for r in rows if r.get("kind") == "line"]
            if not rows:
                continue
            keys = {"engine": engine, "voice": str(rows[0].get("voice")), "condition": condition}
            figs += [
                summary_fig(
                    "first audio, lines",
                    (r.get("first_audio_ms") for r in lines),
                    "ms",
                    keys=keys,
                    target=TTS_FIRST_AUDIO,
                ),
                summary_fig(
                    "compute per second of speech",
                    (r.get("compute_per_speech_s") for r in lines),
                    "x",
                    keys=keys,
                ),
                value_fig(
                    "peak GPU memory reserved",
                    max(_nums(r.get("torch_peak_reserved_mib") for r in rows), default=None),
                    "MiB",
                    n=len(rows),
                    keys=keys,
                ),
                summary_fig(
                    "first audio, acknowledgements (kept out of the line figures)",
                    (r.get("first_audio_ms") for r in rows if r.get("kind") == "ack"),
                    "ms",
                    keys=keys,
                ),
            ]
    load = []
    for row in sorted(inp.tts_load, key=lambda r: _index(TTS_ORDER, str(r.get("engine")))):
        keys = {"engine": str(row.get("engine"))}
        load += [
            value_fig("load", row.get("load_s"), "s", keys=keys),
            value_fig(
                "E4B decode rate, no speech",
                row.get("decode_tps_alone"),
                "tok/s",
                n=int(row.get("decode_completions_alone") or 0),
                keys=keys,
            ),
            value_fig(
                "E4B decode rate, speech running",
                row.get("decode_tps_with_tts"),
                "tok/s",
                n=int(row.get("decode_completions_with_tts") or 0),
                keys=keys,
            ),
        ]
    tables = []
    if figs:
        tables.append(
            figure_table(
                "First audio per engine and condition",
                ("engine", "voice", "condition"),
                figs,
                "Condition `idle`: the bench model idle; `decoding`: the model writing throughout. "
                "The gap between them is what a second card would remove.",
            )
        )
    if load:
        tables.append(figure_table("Load time and the model's decode rate", ("engine",), load))
    return _run_doc(1, "speech synthesis", tables, why="no rows under `tts/`")


def _later_turns(rows: Sequence[Mapping]) -> list:
    return [r for r in rows if (r.get("turn") or 0) >= 2]


def _cached_pct(row: Mapping) -> float | None:
    cached, total = _num(row.get("cache_n")), _num(row.get("prompt_tokens"))
    return None if cached is None or not total else 100.0 * cached / total


def _run2(inp: Inputs) -> dict:
    figs = []
    for arm in _ordered(inp.cache, CACHE_ARMS):
        rows = inp.cache[arm]
        later = _later_turns(rows)
        keys = {"arm": arm}
        figs += [
            summary_fig(
                "cold time to first token",
                (r.get("ttft_ms") for r in rows if r.get("cold")),
                "ms",
                keys=keys,
            ),
            summary_fig(
                "warm time to first token, turns 2 to 10",
                (r.get("ttft_ms") for r in later),
                "ms",
                keys=keys,
                target=PREFILL,
            ),
            summary_fig(
                "prefill, turns 2 to 10", (r.get("prompt_ms") for r in later), "ms", keys=keys
            ),
            summary_fig(
                "cached share of the prompt, turns 2 to 10",
                (_cached_pct(r) for r in later),
                "%",
                keys=keys,
            ),
            summary_fig(
                "decode rate", (r.get("predicted_per_second") for r in rows), "tok/s", keys=keys
            ),
            value_fig(
                "full prompt re-processing logged",
                sum(int(r.get("full_reprocess_logged") or 0) for r in rows),
                "count",
                n=len(rows),
                keys=keys,
            ),
        ]
    tables = [figure_table("Time to first token and cache reuse per arm", ("arm",), figs)]
    return _run_doc(2, "prompt cache", tables if figs else [], why="no rows under `cache/`")


def _run3(inp: Inputs) -> dict:
    figs = []
    for engine in _ordered((str(r.get("engine")) for r in inp.stt), STT_ORDER):
        rows = [r for r in inp.stt if r.get("engine") == engine]
        keys = {"engine": engine, "device": str(rows[0].get("device"))}
        errors = sum(int(r.get("wer_errors") or 0) for r in rows)
        words = sum(int(r.get("wer_reference_words") or 0) for r in rows)
        peak = max(_nums(r.get("torch_peak_reserved_mib") for r in rows), default=None)
        figs += [
            summary_fig(
                "final text", (r.get("final_ms") for r in rows), "ms", keys=keys, target=STT_FINAL
            ),
            value_fig(
                "word error rate, all words",
                100.0 * errors / words if words else None,
                "%",
                n=len(rows),
                keys=keys,
            ),
            value_fig("peak GPU memory reserved", peak, "MiB", n=len(rows), keys=keys),
        ]
    tables = [figure_table("Final text and word error rate per engine", ("engine", "device"), figs)]
    return _run_doc(3, "speech to text", tables if figs else [], why="no rows under `stt/`")


def _turn_inference(inp: Inputs) -> dict:
    return summary_fig(
        "inference", (r.get("inference_ms") for r in inp.turn), "ms", target=TURN_INFERENCE
    )


def _run4(inp: Inputs) -> dict:
    pauses = [r for r in inp.turn if r.get("truth") == "not_finished"]
    ends = [r for r in inp.turn if r.get("truth") == "finished"]
    figs = [
        _turn_inference(inp),
        share_fig(
            "false finished at mid-thought pauses", (r.get("called_finished") for r in pauses)
        ),
        share_fig(
            "false finished, pause utterances only",
            (r.get("called_finished") for r in pauses if r.get("kind") == "pause"),
        ),
        share_fig("finished called at true ends", (r.get("called_finished") for r in ends)),
    ]
    tables = [
        figure_table(
            "Smart Turn v3.2 on the CPU",
            (),
            figs,
            "Inference is marked against 15 ms on top of 200 ms of silence.",
        )
    ]
    return _run_doc(4, "end of turn", tables if inp.turn else [], why="no rows under `turn/`")


def _fit_figs(step: Mapping) -> list:
    keys = {"step": str(step.get("label")), "part": str(step.get("part"))}
    fits = bool(step.get("fits"))
    head = step.get("headroom_mib") if fits else None
    n = int(step.get("samples") or 0)
    return [
        value_fig("peak used", step.get("max_mib"), "MiB", n=n, keys=keys),
        value_fig(
            "headroom", head, "MiB", n=n, keys=keys, mark_text=memory_mark(_num(head), fits=fits)
        ),
        value_fig(
            "by difference from the desktop step",
            step.get("by_difference_mib"),
            "MiB",
            n=n,
            keys=keys,
        ),
    ]


def _run5(inp: Inputs) -> dict:
    figs = [f for step in inp.fit_steps for f in _fit_figs(step)]
    notes = [
        f"{step.get('label')} does not fit: {step.get('message')}"
        for step in inp.fit_steps
        if not step.get("fits")
    ]
    tables = [
        figure_table(
            "Peak memory and headroom per combination",
            ("step", "part"),
            figs,
            "Headroom is the card's total less the step's peak, against a 512 MiB margin and a "
            "2 GiB gate.",
        )
    ]
    return _run_doc(5, "memory fit", tables if figs else [], notes, why="no `fit/steps.json`")


# ---------------------------------------------------------------------------
# Run 7
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Prompts:
    """Each tool prompt's kind, and which commands take two actions."""

    kinds: dict
    two_action: frozenset

    @classmethod
    def load(cls) -> Prompts:
        """Read `tool_prompts.json` once, so each table counts the same prompts by kind."""
        prompts = toolcall.load_prompts()
        return cls(
            {p.id: p.kind for p in prompts}, frozenset(p.id for p in prompts if p.two_action)
        )


def _live(rows: Sequence[Mapping]) -> list:
    return [r for r in rows if not r.get("cut")]


def _call_names(row: Mapping) -> set[str]:
    calls = row.get("raw_tool_calls") or []
    return {str((c.get("function") or {}).get("name", "")) for c in calls if isinstance(c, dict)}


def _tools(rows: Sequence[Mapping], prompts: Prompts, kinds: Sequence[str]) -> list:
    return [
        r
        for r in _live(rows)
        if r.get("kind") == "tool" and prompts.kinds.get(r.get("prompt_id")) in kinds
    ]


def _warm(rows: Sequence[Mapping]) -> list:
    return [r for r in _live(rows) if r.get("kind") == "warm"]


def _warm_ttft(rows: Sequence[Mapping], keys: Mapping | None = None) -> dict:
    return summary_fig(
        "warm time to first token, turns 2 to 10",
        (r.get("ttft_ms") for r in _later_turns(_warm(rows))),
        "ms",
        keys=keys,
        target=PREFILL,
    )


def _decode(rows: Sequence[Mapping], keys: Mapping | None = None) -> dict:
    return summary_fig(
        "decode rate", (r.get("predicted_per_second") for r in _warm(rows)), "tok/s", keys=keys
    )


def _command_valid(rows: Sequence[Mapping], prompts: Prompts, keys: Mapping | None = None) -> dict:
    return share_fig(
        "command validity, all calls",
        (r.get("tool_valid_all") for r in _tools(rows, prompts, ("command",))),
        keys=keys,
    )


def _hand_off_valid(rows: Sequence[Mapping], prompts: Prompts, keys: Mapping | None = None) -> dict:
    return share_fig(
        "hand-off validity, E01 to E04",
        (r.get("tool_valid_all") for r in _tools(rows, prompts, ("hand_off",))),
        keys=keys,
    )


def _false_hand_offs(
    rows: Sequence[Mapping], prompts: Prompts, keys: Mapping | None = None
) -> dict:
    asked = _tools(rows, prompts, ("command", "no_tool"))
    return share_fig(
        "false hand-offs on command and no-tool prompts",
        (toolcall.HAND_OFF_TOOL in _call_names(r) for r in asked),
        keys=keys,
    )


def _lowtier_figs(rows: Sequence[Mapping], prompts: Prompts, keys: Mapping) -> tuple[list, list]:
    live = _live(rows)
    speed = [
        summary_fig(
            "cold time to first token, 1,500-token prefix",
            (
                r.get("ttft_ms")
                for r in live
                if r.get("kind") == "cold" and r.get("prefix") == lowtier_suite.PREFIX_LOWTIER
            ),
            "ms",
            keys=keys,
        ),
        summary_fig(
            "cold time to first token, 6,100-token prefix",
            (
                r.get("ttft_ms")
                for r in live
                if r.get("kind") == "cold" and r.get("prefix") == lowtier_suite.PREFIX_CACHE
            ),
            "ms",
            keys=keys,
        ),
        _warm_ttft(rows, keys),
        _decode(rows, keys),
        value_fig(
            "rows cut short by the window",
            sum(1 for r in rows if r.get("cut")),
            "count",
            n=len(rows),
            keys=keys,
        ),
        value_fig(
            "replies still carrying reasoning",
            sum(1 for r in live if r.get("reasoning_flagged")),
            "count",
            n=len(live),
            keys=keys,
        ),
    ]
    two = [
        r for r in _tools(rows, prompts, ("command",)) if r.get("prompt_id") in prompts.two_action
    ]
    tools = [
        _command_valid(rows, prompts, keys),
        share_fig(
            "two-action commands, first call", (r.get("tool_valid_first") for r in two), keys=keys
        ),
        share_fig(
            "no-tool prompts answered with no call",
            (r.get("tool_valid_all") for r in _tools(rows, prompts, ("no_tool",))),
            keys=keys,
        ),
        _hand_off_valid(rows, prompts, keys),
        _false_hand_offs(rows, prompts, keys),
        summary_fig(
            "request to the last tool-call fragment",
            (r.get("tool_call_ms") for r in _tools(rows, prompts, ("command", "hand_off"))),
            "ms",
            keys=keys,
        ),
    ]
    return speed, tools


def _window(inp: Inputs, model: str, placement: str) -> dict | None:
    for row in inp.host1070:
        if (
            row.get("kind") == "window"
            and row.get("model") == model
            and row.get("placement") == placement
        ):
            return row
    return None


def _remote(inp: Inputs, model: str, placement: str) -> bool:
    mem = inp.lowtier_memory.get((model, placement)) or {}
    return mem.get("memory_method") == lowtier_suite.REMOTE_MEMORY_METHOD or bool(
        _window(inp, model, placement)
    )


def memory_cell(inp: Inputs, model: str, placement: str) -> dict | None:
    """The placement's memory by its own method: the card's share where there is one."""
    keys = {"model": model, "placement": placement}
    if _remote(inp, model, placement):
        window = _window(inp, model, placement)
        if window is None:
            return None
        if placement == "host1070-cpu":
            return value_fig(
                "host memory used at the peak (H6 summary)",
                window.get("host_used_mib"),
                "MiB",
                keys=keys,
            )
        return value_fig(
            "GTX 1070 memory taken at load (H6 summary)",
            window.get("gpu_load_mib"),
            "MiB",
            keys=keys,
        )
    mem = inp.lowtier_memory.get((model, placement))
    if mem is None:
        return None
    if _num(mem.get("gpu_by_difference_mib")) is not None:
        return value_fig(
            "GPU memory by difference", mem.get("gpu_by_difference_mib"), "MiB", keys=keys
        )
    return value_fig("container peak host memory", mem.get("host_peak_mib"), "MiB", keys=keys)


def _switch(inp: Inputs, model: str, placement: str, field: str) -> dict | None:
    keys = {"model": model, "placement": placement}
    source = _window(inp, model, placement) or inp.lowtier_memory.get((model, placement))
    if source is None:
        return None
    label = {"load_s": "load", "unload_s": "unload"}[field]
    return value_fig(label, source.get(field), "s", keys=keys)


def _memory_figs(inp: Inputs, model: str, placement: str) -> list:
    keys = {"model": model, "placement": placement}
    figs = [f for f in (memory_cell(inp, model, placement),) if f is not None]
    window = _window(inp, model, placement)
    mem = inp.lowtier_memory.get((model, placement)) or {}
    if window is not None:
        free = _num(window.get("gpu_free_at_peak_mib"))
        figs.append(
            value_fig(
                "GTX 1070 memory free at the peak",
                free,
                "MiB",
                keys=keys,
                mark_text=memory_mark(free),
            )
        )
        figs.append(
            value_fig("host memory used at the peak", window.get("host_used_mib"), "MiB", keys=keys)
        )
    elif mem and _num(mem.get("gpu_by_difference_mib")) is not None:
        figs.append(
            value_fig("container peak host memory", mem.get("host_peak_mib"), "MiB", keys=keys)
        )
    for field in ("load_s", "unload_s"):
        fig = _switch(inp, model, placement, field)
        if fig is not None:
            figs.append(fig)
    returned = (window or mem).get("memory_returned_mib") if (window or mem) else None
    figs.append(value_fig("memory returned on unload", returned, "MiB", keys=keys))
    return figs


def _chain_values(rows: Sequence[Mapping], kinds: Sequence[str]) -> list:
    return [r for r in rows if r.get("kind") in kinds and not r.get("error")]


def chained_turn_fig(rows: Sequence[Mapping], inference: Mapping, keys: Mapping | None) -> dict:
    """Turn time with endpointing added: 200 ms of silence and run 4's median inference."""
    turns = _nums(r.get("turn_ms") for r in rows)
    median = basis(inference)
    values = [] if median is None else [t + SILENCE_MS + median for t in turns]
    shown = "not yet run" if median is None else f"{median:,.0f} ms"
    return summary_fig(
        "chained turn with endpointing",
        values,
        "ms",
        keys=keys,
        target=CHAINED_TURN,
        derived=True,
        inputs=(
            "chained turn, measured: speech to text, the model to its first clause, first audio",
            f"{SILENCE_MS} ms of silence",
            f"Smart Turn inference, run 4 median: {shown}",
        ),
    )


def _chain_figs(
    rows: Sequence[Mapping], errors: int, inference: Mapping, keys: Mapping, busy: bool
) -> list:
    """One group's stage figures; `rows` holds the turns that finished, `errors` the rest."""
    figs = [
        summary_fig(
            "chained turn, measured (to first audio)",
            (r.get("turn_ms") for r in rows),
            "ms",
            keys=keys,
        ),
        chained_turn_fig(rows, inference, keys),
        summary_fig("speech to text", (r.get("stt_ms") for r in rows), "ms", keys=keys),
        summary_fig(
            "the model to its first clause",
            (r.get("first_clause_ms") for r in rows),
            "ms",
            keys=keys,
        ),
        summary_fig(
            "speech to first audio", (r.get("tts_first_audio_ms") for r in rows), "ms", keys=keys
        ),
        share_fig("tool call valid", (r.get("tool_valid") for r in rows), keys=keys),
        value_fig("turns that failed", errors, "count", n=len(rows) + errors, keys=keys),
    ]
    if busy:
        figs.append(
            summary_fig(
                "E4B decode rate during the turns",
                (r.get("busy_decode_tps") for r in rows),
                "tok/s",
                keys=keys,
            )
        )
    return figs


def _chain_group(rows: Sequence[Mapping], kinds: Sequence[str]) -> tuple[list, list]:
    group = [r for r in rows if r.get("kind") in kinds]
    return [r for r in group if not r.get("error")], [r for r in group if r.get("error")]


def _run7(inp: Inputs, prompts: Prompts) -> dict:
    inference = _turn_inference(inp)
    speed, tools, memory = [], [], []
    for model, placement in _ordered_pairs(inp.lowtier, inp.models):
        keys = {"model": model, "placement": placement}
        s, t = _lowtier_figs(inp.lowtier[(model, placement)], prompts, keys)
        speed += s
        tools += t
    pairs = set(inp.lowtier_memory) | {
        (str(r.get("model")), str(r.get("placement")))
        for r in inp.host1070
        if r.get("kind") == "window"
    }
    for model, placement in _ordered_pairs(pairs, inp.models):
        memory += _memory_figs(inp, model, placement)
    for row in inp.host1070:
        if row.get("kind") == "layers":
            keys = {"model": str(row.get("model")), "placement": str(row.get("placement"))}
            memory.append(
                value_fig(
                    "layers on the GTX 1070, searched",
                    row.get("chosen_ngl"),
                    "count",
                    n=len(row.get("probes") or []),
                    keys=keys,
                )
            )
    chained = []
    for model, placement in _ordered_pairs(inp.chain, inp.models):
        for name, kinds in CHAIN_GROUPS:
            ok, failed = _chain_group(inp.chain[(model, placement)], kinds)
            if not ok and not failed:
                continue
            keys = {"model": model, "placement": placement, "turns": name}
            chained += _chain_figs(ok, len(failed), inference, keys, placement == BUSY_DIR)
    reload_figs = []
    if inp.e4b_reload:
        reload_figs.append(
            summary_fig(
                "E4B reload after unserve", (r.get("reload_s") for r in inp.e4b_reload), "s"
            )
        )
    notes = []
    for row in inp.host1070:
        if row.get("kind") == "skip":
            notes.append(
                f"host-1070: {row.get('model')} on {row.get('placement')} not run: "
                f"{row.get('reason')}"
            )
        if row.get("kind") == "restore":
            notes.append(
                f"host-1070 restore: inspect diff empty {row.get('inspect_diff_empty')}, "
                f"exit code {row.get('exit_code')}"
            )
    tables = []
    if speed:
        tables.append(
            figure_table("Time to first token and decode rate", ("model", "placement"), speed)
        )
        tables.append(figure_table("Tool calls and hand-offs", ("model", "placement"), tools))
    if memory:
        tables.append(figure_table("Memory and switching", ("model", "placement"), memory))
    if reload_figs:
        tables.append(
            figure_table("E4B's reload, the cost of a level change back", (), reload_figs)
        )
    if chained:
        tables.append(
            figure_table(
                "Chained turn: Parakeet on the CPU, the model, Pocket TTS on the CPU",
                ("model", "placement", "turns"),
                chained,
                "The measured turn runs from the end of the recording to first audio. The turn "
                "with endpointing adds 200 ms of silence and run 4's median Smart Turn inference "
                "(derived), marked against about 2 to 3 s (2,000 ms target, 3,000 ms floor). "
                f"`{BUSY_DIR}` is `gpu-full` while E4B streams a long answer.",
            )
        )
    return _run_doc(7, "low level, no game", tables, notes, why="no rows under `lowtier/`")


# ---------------------------------------------------------------------------
# Runs 8 and 9
# ---------------------------------------------------------------------------


def _avg_fps(rows: Sequence[Mapping]) -> list:
    return _nums((r.get("frames") or {}).get("avg_fps") for r in rows)


def _game_cost(session: GameSession, placement: str) -> dict | None:
    g0 = _avg_fps([p for p in session.passes if p.get("condition") == "G0"])
    g2_rows = [p for p in session.passes if p.get("condition") == f"G2-{placement}"]
    g2 = _avg_fps(g2_rows)
    if not g0 or not g2:
        return None
    return value_fig(
        "average frame rate, G2 against G0",
        statistics.median(g2) - statistics.median(g0),
        "fps",
        n=len(g2),
        keys={"session": session.session, "placement": placement},
    )


def _session_figs(session: GameSession, inference: Mapping) -> list:
    conditions = list(dict.fromkeys(str(p.get("condition")) for p in session.passes))
    g0 = _avg_fps([p for p in session.passes if p.get("condition") == "G0"])
    figs = []
    for condition in conditions:
        rows = [p for p in session.passes if p.get("condition") == condition]
        keys = {
            "session": session.session,
            "condition": condition,
            "placement": str(rows[0].get("placement") or "none"),
        }
        peak = max(_nums((p.get("gpu") or {}).get("peak_used_mib") for p in rows), default=None)
        figs += [
            summary_fig("average frame rate", _avg_fps(rows), "fps", keys=keys),
            summary_fig(
                "1% low frame rate",
                ((p.get("frames") or {}).get("low_1pct_fps") for p in rows),
                "fps",
                keys=keys,
            ),
            summary_fig(
                "99th-percentile frame time",
                ((p.get("frames") or {}).get("p99_frame_ms") for p in rows),
                "ms",
                keys=keys,
            ),
            value_fig("peak GPU memory used", peak, "MiB", n=len(rows), keys=keys),
            summary_fig(
                "CPU load, mean per pass",
                ((p.get("cpu") or {}).get("mean") for p in rows),
                "%",
                keys=keys,
            ),
        ]
        if condition == "G0":
            spread = max(g0) - min(g0) if g0 else None
            figs.append(
                value_fig(
                    "G0's own spread, highest pass less lowest", spread, "fps", n=len(g0), keys=keys
                )
            )
        elif g0:
            figs.append(
                value_fig(
                    "average frame rate against G0",
                    (
                        statistics.median(_avg_fps(rows)) - statistics.median(g0)
                        if _avg_fps(rows)
                        else None
                    ),
                    "fps",
                    n=len(rows),
                    keys=keys,
                )
            )
        turns = [r for r in session.chain if r.get("condition") == condition]
        if turns:
            ok = [r for r in turns if not r.get("error")]
            figs.append(
                summary_fig(
                    "chained turn, measured (to first audio)",
                    (r.get("turn_ms") for r in ok),
                    "ms",
                    keys=keys,
                )
            )
            figs.append(chained_turn_fig(ok, inference, keys))
        if session.run == 9:
            figs.append(
                summary_fig(
                    "server process CPU, mean per pass",
                    (((p.get("server") or {}).get("cpu") or {}).get("mean") for p in rows),
                    "%",
                    keys=keys,
                )
            )
            figs.append(
                value_fig(
                    "server process memory, highest",
                    max(
                        _nums((p.get("server") or {}).get("memory_max_mib") for p in rows),
                        default=None,
                    ),
                    "MiB",
                    n=len(rows),
                    keys=keys,
                )
            )
    back = (session.e4b_return or {}).get("e4b_return_s")
    figs.append(
        value_fig(
            "E4B's return after the session",
            back,
            "s",
            keys={"session": session.session, "condition": "restore", "placement": "none"},
        )
    )
    return figs


def _session_notes(session: GameSession) -> list:
    meta = session.meta
    notes = [
        f"{session.session}: {meta.get('game')} {meta.get('version')}, settings "
        f"{meta.get('settings')}; frames from {meta.get('frames_mode')}; model {meta.get('model')}"
    ]
    for row in session.conditions:
        if row.get("status") == "skipped":
            notes.append(
                f"{session.session}: {', '.join(row.get('conditions') or [])} skipped: "
                f"{row.get('reason')}"
            )
    for row in session.passes:
        if row.get("note"):
            notes.append(
                f"{session.session} {row.get('condition')} pass {row.get('pass')}: "
                f"{row.get('note')}"
            )
        if row.get("tick"):
            notes.append(
                f"{session.session} {row.get('condition')} pass {row.get('pass')} tick: "
                f"{row.get('tick')}"
            )
    return notes


GAME_TITLES = {8: "low level while a light game runs", 9: "Minecraft with its server on this PC"}
GAME_MISSING = {
    8: "no game session under `game/` yet",
    9: "no game session with its server process on this PC under `game/` yet",
}


def _game_run(inp: Inputs, number: int) -> dict:
    inference = _turn_inference(inp)
    sessions = [s for s in inp.games if s.run == number]
    tables, notes = [], []
    for session in sessions:
        tables.append(
            figure_table(
                f"Session {session.session}",
                ("session", "condition", "placement"),
                _session_figs(session, inference),
                "No prior target: each condition's frame rate is set beside G0's own spread.",
            )
        )
        notes += _session_notes(session)
    return _run_doc(
        number,
        GAME_TITLES[number],
        tables,
        notes,
        why=f"{GAME_MISSING[number]}; `report --replace` renders again when one arrives",
    )


# ---------------------------------------------------------------------------
# Blind listening
# ---------------------------------------------------------------------------


def _rating(value: object) -> int | None:
    try:
        number = int(str(value).strip())
    except ValueError:
        return None
    return number if 1 <= number <= 5 else None


def _listening(inp: Inputs) -> dict:
    if inp.ratings is None:
        return {
            "status": NOT_YET_RUN,
            "tables": [],
            "notes": [],
            "why": "no listening pack under `listening/`",
        }
    clips = {str(c.get("label")): c for c in inp.ratings.key.get("clips", [])}
    groups: dict[tuple[str, str, str], list] = {}
    unknown = []
    for row in inp.ratings.rows:
        clip = clips.get(str(row.get("label")))
        if clip is None:
            unknown.append(str(row.get("label")))
            continue
        group = (str(clip.get("engine")), str(clip.get("setting")), str(clip.get("voice")))
        groups.setdefault(group, []).append(row)
    figs = []
    order = sorted(
        groups, key=lambda g: (_index(TTS_ORDER, g[0]), g[0], g[1] != "default", g[1], g[2])
    )
    for engine, setting, voice in order:
        rows = groups[(engine, setting, voice)]
        keys = {"engine": engine, "setting": setting, "voice": voice}
        figs += [
            mean_fig(
                "naturalness, 1 to 5", (_rating(r.get("natural")) for r in rows), "score", keys=keys
            ),
            mean_fig(
                "carries the line's feeling, 1 to 5",
                (_rating(r.get("feeling")) for r in rows),
                "score",
                keys=keys,
            ),
            value_fig(
                "picked as the favourite of its line",
                sum(1 for r in rows if str(r.get("preferred")).strip() == "1"),
                "count",
                n=len(rows),
                keys=keys,
            ),
        ]
    notes = (
        [f"ratings for labels the key does not hold: {', '.join(sorted(unknown))}"]
        if unknown
        else []
    )
    tables = [
        figure_table("Ratings by engine, setting and voice", ("engine", "setting", "voice"), figs)
    ]
    return {"status": "measured", "tables": tables, "notes": notes}


# ---------------------------------------------------------------------------
# The section 8 budget
# ---------------------------------------------------------------------------


def _c0_decode(inp: Inputs) -> dict:
    rows = inp.cache.get("c0", [])
    return summary_fig("decode rate", (r.get("predicted_per_second") for r in rows), "tok/s")


def _budget_options(inp: Inputs, stage: BudgetStage) -> list:
    goal = stage.goal
    if stage.stage == 1:
        if not inp.turn:
            return []
        values = (SILENCE_MS + v for v in _nums(r.get("inference_ms") for r in inp.turn))
        return [
            summary_fig(
                "200 ms of silence plus Smart Turn inference",
                values,
                "ms",
                keys={"option": "smart-turn-v3.2"},
                target=goal,
            )
        ]
    if stage.stage == 2:
        engines = _ordered((str(r.get("engine")) for r in inp.stt), STT_ORDER)
        return [
            summary_fig(
                "final text",
                (r.get("final_ms") for r in inp.stt if r.get("engine") == e),
                "ms",
                keys={"option": e},
                target=goal,
            )
            for e in engines
        ]
    if stage.stage == 3:
        return [
            value_fig(
                "memory fetch, carried in",
                stage.floor,
                "ms",
                keys={"option": "floor"},
                target=goal,
                derived=True,
                inputs=(
                    "not measured here",
                    f"the plan floor in {BUDGET_SOURCE}: {stage.floor:g} ms",
                ),
            )
        ]
    if stage.stage == 4:
        return [
            summary_fig(
                "warm time to first token, turns 2 to 10",
                (r.get("ttft_ms") for r in _later_turns(inp.cache[a])),
                "ms",
                keys={"option": a},
                target=goal,
            )
            for a in _ordered(inp.cache, CACHE_ARMS)
        ]
    if stage.stage == 5:
        rate = _c0_decode(inp)
        tps = basis(rate)
        value = FIRST_CLAUSE_TOKENS / tps * 1000.0 if tps else None
        shown = f"{tps:,.1f} tok/s (n={rate['n']})" if tps else "not yet run"
        return [
            value_fig(
                "first clause at the measured decode rate",
                value,
                "ms",
                keys={"option": "c0"},
                target=goal,
                derived=True,
                inputs=(
                    f"run 2 c0 decode rate, median {shown}",
                    f"a first clause of about {FIRST_CLAUSE_TOKENS} tokens",
                ),
            )
        ]
    if stage.stage == 6:
        engines = _ordered((str(r.get("engine")) for r in inp.tts_samples), TTS_ORDER)
        return [
            summary_fig(
                "first audio, lines, model writing",
                (
                    r.get("first_audio_ms")
                    for r in inp.tts_samples
                    if r.get("engine") == e
                    and r.get("condition") == "decoding"
                    and r.get("kind") == "line"
                ),
                "ms",
                keys={"option": e},
                target=goal,
            )
            for e in engines
        ]
    return [
        value_fig(
            "send and start playback, carried in",
            100,
            "ms",
            keys={"option": "today"},
            target=goal,
            derived=True,
            inputs=("not measured here", f"today's figure in {BUDGET_SOURCE}: about 100 ms"),
        )
    ]


def _budget(inp: Inputs) -> dict:
    stages, chosen = [], []
    for stage in BUDGET_STAGES:
        options = _budget_options(inp, stage)
        with_data = [f for f in options if basis(f) is not None]
        best = min(with_data, key=basis) if with_data else None
        chosen.append((stage, best))
        stages.append(
            {
                "stage": stage.stage,
                "name": stage.name,
                "today": stage.today,
                "floor": stage.floor,
                "target": stage.target,
                "source": f"measured, {stage.run}" if stage.run else "derived",
                "options": options,
                "chosen": None if best is None else best["keys"]["option"],
            }
        )
    complete = all(best is not None for _, best in chosen)
    inputs = [
        (
            f"stage {stage.stage}: {best['label']}, {best['keys']['option']}"
            if best is not None
            else f"stage {stage.stage}: no data"
        )
        for stage, best in chosen
    ]
    median = sum(basis(b) for _, b in chosen) if complete else None
    slow = sum(_slow(b) for _, b in chosen) if complete else None
    total = {
        "median": value_fig(
            "total, median",
            median,
            "ms",
            n=len(chosen),
            target=BUDGET_TOTAL,
            derived=True,
            inputs=["the fastest option of each stage", *inputs],
        ),
        "slow": value_fig(
            "total, slow case",
            slow,
            "ms",
            n=len(chosen),
            target=BUDGET_SLOW,
            derived=True,
            inputs=["each chosen stage's slow case, summed: an upper " "bound", *inputs],
        ),
    }
    return {"source": BUDGET_SOURCE, "stages": stages, "total": total}


# ---------------------------------------------------------------------------
# The levels section
# ---------------------------------------------------------------------------


def _game_placement(placement: str) -> str:
    return "gtx1070-shared" if placement in (SHARED_IDLE, SHARED_BUSY) else placement


def _game_cell(inp: Inputs, number: int, model: str, placement: str) -> object:
    sessions = [s for s in inp.games if s.run == number]
    if not sessions:
        return NOT_YET_RUN
    for session in reversed(sessions):
        if session.meta.get("model") == model:
            cost = _game_cost(session, _game_placement(placement))
            if cost is not None:
                cost["keys"] = {"model": model, "placement": placement, "session": session.session}
                return cost
    return NOT_MEASURED


def _placement_grid(inp: Inputs, prompts: Prompts, inference: Mapping) -> dict:
    pairs = (set(inp.lowtier) | set(inp.chain)) - {p for p in inp.chain if p[1] == BUSY_DIR}
    rows = []
    for model, placement in _ordered_pairs(pairs, inp.models):
        keys = {"model": model, "placement": placement}
        requests = inp.lowtier.get((model, placement), [])
        turns = _chain_values(inp.chain.get((model, placement), []), ("command", "short"))
        cells = {
            "chained turn with endpointing": (
                chained_turn_fig(turns, inference, keys) if turns else None
            ),
            "tool-call validity": _command_valid(requests, prompts, keys) if requests else None,
            "hand-off validity": _hand_off_valid(requests, prompts, keys) if requests else None,
            "memory": memory_cell(inp, model, placement),
            "load": _switch(inp, model, placement, "load_s"),
            "unload": _switch(inp, model, placement, "unload_s"),
            "game cost, run 8": _game_cell(inp, 8, model, placement),
            "game cost, run 9": _game_cell(inp, 9, model, placement),
        }
        rows.append(_grid_row(keys, cells))
    return grid_table(
        "Each placement side by side",
        ("model", "placement"),
        PLACEMENT_COLUMNS,
        rows,
        "Chained turn: commands and short answers with endpointing added (derived), against about "
        "2 to 3 s. Tool-call validity: the command prompts, all calls. Memory: the placement's own "
        "method. Game cost: the average frame rate of G2 less G0's, from the latest session.",
    )


def _model_grid(
    inp: Inputs,
    placement: str,
    columns: Mapping[str, Callable[[list, dict], object]],
    title: str,
    note: str | None = None,
) -> dict | None:
    rows = []
    for model in _ordered((m for m, p in inp.lowtier if p == placement), inp.models):
        requests = inp.lowtier[(model, placement)]
        keys = {"model": model, "placement": placement}
        rows.append(_grid_row(keys, {name: make(requests, keys) for name, make in columns.items()}))
    return grid_table(title, ("model", "placement"), list(columns), rows, note) if rows else None


def _levels(inp: Inputs, prompts: Prompts, runs: Mapping[int, dict]) -> list:
    inference = _turn_inference(inp)
    answers: list[tuple[list, list]] = []

    # 1. Where the low level runs.
    grid = _placement_grid(inp, prompts, inference)
    notes = [] if grid["rows"] else ["run 7 not yet run"]
    for number in (8, 9):
        if runs[number]["status"] == NOT_YET_RUN:
            notes.append(f"run {number} not yet run: the game cost per placement waits for it")
    answers.append(([grid] if grid["rows"] else [], notes))

    # 2. Which small model.
    grid = _model_grid(
        inp,
        "gpu-full",
        {
            "command validity": lambda r, k: _command_valid(r, prompts, k),
            "hand-off validity": lambda r, k: _hand_off_valid(r, prompts, k),
            "warm time to first token": _warm_ttft,
            "decode rate": _decode,
        },
        "Each model on gpu-full, the same prompts",
        "`e4b-ref` is today's E4B, the reference.",
    )
    answers.append(([grid], []) if grid else ([], ["run 7 on gpu-full not yet run"]))

    # 3. Routing requests up.
    grid = _model_grid(
        inp,
        "gpu-full",
        {
            "hand-off validity": lambda r, k: _hand_off_valid(r, prompts, k),
            "false hand-offs": lambda r, k: _false_hand_offs(r, prompts, k),
        },
        "Hand-offs on gpu-full",
    )
    answers.append(([grid], []) if grid else ([], ["run 7 on gpu-full not yet run"]))

    # 4. What each tier costs per request.
    tables = []
    grid = _model_grid(
        inp,
        "gpu-full",
        {
            "warm time to first token": _warm_ttft,
            "decode rate": _decode,
        },
        "Low level on gpu-full",
    )
    if grid:
        tables.append(grid)
    run2 = [
        f
        for t in runs[2]["tables"]
        for f in t["figures"]
        if f["keys"]["arm"] == "c0" and f["label"] != "full prompt re-processing logged"
    ]
    if run2:
        tables.append(
            figure_table("The top tier: E4B as production serves it (run 2, c0)", ("arm",), run2)
        )
    answers.append((tables, [] if tables else ["runs 2 and 7 not yet run"]))

    # 5. Both tiers loaded at full level.
    small = [f for step in inp.fit_steps if step.get("part") == "small" for f in _fit_figs(step)]
    table = figure_table(
        "E4B with each small model, speech and recognition (run 5)", ("step", "part"), small
    )
    answers.append(([table], []) if small else ([], ["run 5's small-model steps not yet run"]))

    # 6. What a level change costs.
    tables = []
    rows = []
    for model, placement in _ordered_pairs(set(inp.lowtier_memory), inp.models):
        keys = {"model": model, "placement": placement}
        mem = _window(inp, model, placement) or inp.lowtier_memory[(model, placement)]
        rows.append(
            _grid_row(
                keys,
                {
                    "load": _switch(inp, model, placement, "load_s"),
                    "unload": _switch(inp, model, placement, "unload_s"),
                    "memory returned": value_fig(
                        "memory returned on unload",
                        mem.get("memory_returned_mib"),
                        "MiB",
                        keys=keys,
                    ),
                },
            )
        )
    if rows:
        tables.append(
            grid_table(
                "Load and unload per model and placement",
                ("model", "placement"),
                ("load", "unload", "memory returned"),
                rows,
            )
        )
    if inp.e4b_reload:
        tables.append(
            figure_table(
                "E4B's reload",
                (),
                [
                    summary_fig(
                        "E4B reload after unserve", (r.get("reload_s") for r in inp.e4b_reload), "s"
                    )
                ],
            )
        )
    notes = []
    returns = [
        value_fig(
            "E4B's return after the session",
            (s.e4b_return or {}).get("e4b_return_s"),
            "s",
            keys={"session": s.session, "run": str(s.run)},
        )
        for s in inp.games
    ]
    if returns:
        tables.append(
            figure_table("E4B's return after a game session", ("session", "run"), returns)
        )
    else:
        notes.append("E4B's return after a game session: runs 8 and 9 not yet run")
    answers.append((tables, notes))

    # 7. A low-level request while the top tier writes.
    tables, rows = [], []
    for model in _ordered((m for m, p in inp.chain if p == BUSY_DIR), inp.models):
        keys = {"model": model, "placement": BUSY_DIR}
        turns = _chain_values(inp.chain[(model, BUSY_DIR)], ("command", "short"))
        rows.append(
            _grid_row(
                keys,
                {
                    "chained turn with endpointing": chained_turn_fig(turns, inference, keys),
                    "E4B decode rate during the turns": summary_fig(
                        "E4B decode rate during the turns",
                        (
                            r.get("busy_decode_tps")
                            for r in inp.chain[(model, BUSY_DIR)]
                            if not r.get("error")
                        ),
                        "tok/s",
                        keys=keys,
                    ),
                },
            )
        )
    if rows:
        tables.append(
            grid_table(
                "Each small model on gpu-full while E4B streams",
                ("model", "placement"),
                ("chained turn with endpointing", "E4B decode rate during the turns"),
                rows,
            )
        )
    if inp.tts_load:
        tables.append(
            figure_table(
                "E4B's decode rate with nothing beside it (run 1 baseline)",
                (),
                [
                    summary_fig(
                        "E4B decode rate, no speech",
                        (r.get("decode_tps_alone") for r in inp.tts_load),
                        "tok/s",
                    )
                ],
            )
        )
    answers.append((tables, [] if tables else ["run 7's busy chain not yet run"]))

    # 8. A busy extraction job on host-1070.
    rows = []
    for model in _ordered(
        (m for m, p in inp.lowtier if p in (SHARED_IDLE, SHARED_BUSY)), inp.models
    ):
        idle = inp.lowtier.get((model, SHARED_IDLE), [])
        busy = inp.lowtier.get((model, SHARED_BUSY), [])
        keys = {"model": model, "placement": "gtx1070-shared"}
        rows.append(
            _grid_row(
                keys,
                {
                    "warm time to first token, idle": _warm_ttft(idle, keys) if idle else None,
                    "warm time to first token, busy": _warm_ttft(busy, keys) if busy else None,
                    "decode rate, idle": _decode(idle, keys) if idle else None,
                    "decode rate, busy": _decode(busy, keys) if busy else None,
                    "command validity, idle": _command_valid(idle, prompts, keys) if idle else None,
                    "command validity, busy": _command_valid(busy, prompts, keys) if busy else None,
                },
            )
        )
    columns = (
        "warm time to first token, idle",
        "warm time to first token, busy",
        "decode rate, idle",
        "decode rate, busy",
        "command validity, idle",
        "command validity, busy",
    )
    answers.append(
        (
            [
                grid_table(
                    "The shared placement, idle and busy", ("model", "placement"), columns, rows
                )
            ],
            [],
        )
        if rows
        else ([], ["host-1070 rows not yet run"])
    )

    # 9. What the low level keeps on the CPU.
    figs = [
        summary_fig(
            "Parakeet on the CPU, final text (run 3)",
            (r.get("final_ms") for r in inp.stt if r.get("engine") == "parakeet-cpu"),
            "ms",
            target=STT_FINAL,
        ),
    ]
    for condition in CONDITIONS:
        figs.append(
            summary_fig(
                f"Pocket TTS first audio, {condition} (run 1)",
                (
                    r.get("first_audio_ms")
                    for r in inp.tts_samples
                    if r.get("engine") == "pocket"
                    and r.get("condition") == condition
                    and r.get("kind") == "line"
                ),
                "ms",
                target=TTS_FIRST_AUDIO,
            )
        )
    tables = [figure_table("Speech on the CPU, alone", (), figs)]
    rows = []
    for model in _ordered((m for m, p in inp.chain if p == "cpu"), inp.models):
        keys = {"model": model, "placement": "cpu"}
        turns = _chain_values(inp.chain[(model, "cpu")], ("command", "short"))
        rows.append(
            _grid_row(
                keys,
                {
                    "speech to text": summary_fig(
                        "speech to text", (r.get("stt_ms") for r in turns), "ms", keys=keys
                    ),
                    "speech to first audio": summary_fig(
                        "speech to first audio",
                        (r.get("tts_first_audio_ms") for r in turns),
                        "ms",
                        keys=keys,
                    ),
                    "chained turn with endpointing": chained_turn_fig(turns, inference, keys),
                },
            )
        )
    if rows:
        tables.append(
            grid_table(
                "Sharing the CPU with the model (run 7, cpu placement)",
                ("model", "placement"),
                ("speech to text", "speech to first audio", "chained turn with endpointing"),
                rows,
            )
        )
    answers.append((tables, []))

    # 10. The larger card.
    answers.append(
        (
            [],
            [
                "See the section on what a 16 to 24 GB card would add, below: every "
                "figure there is derived from run 5's memory and run 2's decode rate."
            ],
        )
    )

    # 11. Extraction on host-1070 beside a low level.
    rows = []
    for model in _ordered((m for m, p in inp.lowtier if p == SHARED_IDLE), inp.models):
        keys = {"model": model, "placement": SHARED_IDLE}
        idle = inp.lowtier[(model, SHARED_IDLE)]
        rows.append(
            _grid_row(
                keys,
                {
                    "decode rate": _decode(idle, keys),
                    "warm time to first token": _warm_ttft(idle, keys),
                    "extraction decode given up": _extraction_cost(inp, model, keys),
                },
            )
        )
    answers.append(
        (
            [
                grid_table(
                    "The small model in the 1070's spare memory",
                    ("model", "placement"),
                    ("decode rate", "warm time to first token", "extraction decode given up"),
                    rows,
                    "Extraction decode given up: what freeing enough of the card to hold "
                    "the model whole would cost extraction (derived).",
                )
            ],
            [],
        )
        if rows
        else ([], ["host-1070 rows not yet run"])
    )

    return [
        {"question": q, "runs": runs_text, "tables": t, "notes": n}
        for q, runs_text, (t, n) in zip(LEVEL_QUESTIONS, LEVEL_RUNS, answers, strict=True)
    ]


def _extraction_cost(inp: Inputs, model: str, keys: Mapping) -> dict:
    alone = _window(inp, model, "gtx1070") or {}
    shared = _window(inp, model, SHARED_IDLE) or {}
    whole, part = _num(alone.get("gpu_load_mib")), _num(shared.get("gpu_load_mib"))
    extra = None if whole is None or part is None else max(0.0, whole - part)
    steps = None if extra is None else math.ceil(extra / NCMOE_STEP_MIB)
    shown = "not measured" if extra is None else f"{extra:,.0f} MiB"
    return value_fig(
        "extraction decode given up",
        None if steps is None else steps * NCMOE_STEP_TPS,
        "tok/s",
        keys=keys,
        derived=True,
        inputs=(
            f"memory to free: {shown}, the model's load alone on the GTX 1070 less its load beside "
            "extraction (H6 summary)",
            f"each ncmoe step frees about {NCMOE_STEP_MIB} MiB and costs about {NCMOE_STEP_TPS} "
            "tok/s of extraction decode (the host-1070 runbook's own sweep)",
        ),
    )


# ---------------------------------------------------------------------------
# What a 16 to 24 GB card would add
# ---------------------------------------------------------------------------


def _heaviest_full_level(inp: Inputs) -> dict | None:
    steps = [
        s
        for s in inp.fit_steps
        if s.get("part") == "small"
        and s.get("fits")
        and (s.get("components") or {}).get("tts")
        and (s.get("components") or {}).get("stt")
        and _num(s.get("max_mib")) is not None
    ]
    return max(steps, key=lambda s: s["max_mib"]) if steps else None


def _larger_card(inp: Inputs) -> dict:
    tables, notes = [], []
    rate = _c0_decode(inp)
    tps = basis(rate)
    decode = []
    for card in CARDS[1:]:
        decode.append(
            value_fig(
                "E4B decode rate, scaled by bandwidth",
                None if tps is None else tps * card.bandwidth_gbs / TODAY_BANDWIDTH_GBS,
                "tok/s",
                keys={"card": card.name},
                derived=True,
                inputs=(
                    "run 2 c0 decode rate on the RTX 4070 SUPER, median "
                    + ("not yet run" if tps is None else f"{tps:,.1f} tok/s (n={rate['n']})"),
                    f"bandwidth {card.bandwidth} against about 504 GB/s",
                ),
            )
        )
    tables.append(figure_table("Decode scaled by bandwidth", ("card",), decode))

    heavy = _heaviest_full_level(inp)
    memory = []
    for gb in LARGER_MEMORY_GB:
        total = gb * 1024
        peak = None if heavy is None else _num(heavy.get("max_mib"))
        head = None if peak is None else total - peak
        step = "not yet run" if heavy is None else f"{heavy.get('label')}, peak {peak:,.0f} MiB"
        memory.append(
            value_fig(
                "headroom beside E4B, a small model, speech and recognition",
                head,
                "MiB",
                keys={"card": f"{gb} GB"},
                mark_text=memory_mark(head),
                derived=True,
                inputs=(
                    f"run 5's heaviest full-level step that fits: {step}",
                    f"{gb} GB taken as {total:,} MiB",
                ),
            )
        )
    tables.append(figure_table("Today's full level on a larger card", ("card",), memory))

    fits = []
    if inp.larger is None:
        notes.append(
            "The larger models' file sizes are not given yet: H9 reads them from the Hugging Face "
            f"API into `host/{LARGER_MODELS_FILE}` (or `--larger-models <file>`), and the fit of a "
            "bigger thinking model is computed then."
        )
    else:
        fits = _bigger_model_fits(inp, heavy)
        tables.append(
            figure_table(
                "A bigger thinking model in E4B's place",
                ("model", "card"),
                fits,
                "Need: the desktop, the small model, speech and recognition as run 5 measured them, "
                "the larger model's file, and E4B's cache and buffers taken as the larger model's, "
                "which understates a larger model's cache.",
            )
        )
    notes.append(
        "Docker Desktop passes only NVIDIA GPUs to WSL 2 containers, so an AMD card would not run "
        "MIST's stack as built."
    )
    return {
        "cards": [
            {"card": c.name, "memory_gb": c.memory_gb, "bandwidth": c.bandwidth, "price": c.price}
            for c in CARDS
        ],
        "larger_models": inp.larger,
        "tables": tables,
        "notes": notes,
    }


def _bigger_model_fits(inp: Inputs, heavy: Mapping | None) -> list:
    e4b = next((s for s in inp.fit_steps if s.get("label") == "e4b"), None)
    e4b_mib = None if e4b is None else _num(e4b.get("by_difference_mib"))
    heavy_mib = None if heavy is None else _num(heavy.get("by_difference_mib"))
    desktop = None if heavy is None else _num(heavy.get("desktop_mib"))
    beside = None if heavy_mib is None or e4b_mib is None else heavy_mib - e4b_mib
    reference = (inp.larger or {}).get("reference")
    overhead = None
    if reference is not None and e4b_mib is not None:
        overhead = e4b_mib - reference["size_bytes"] / 2**20
    source = (inp.larger or {}).get("source", "the lead's file")
    figs = []
    for model in (inp.larger or {}).get("models", []):
        file_mib = model["size_bytes"] / 2**20
        parts = (desktop, beside, overhead)
        need = None if any(p is None for p in parts) else file_mib + sum(parts)
        for gb in LARGER_MEMORY_GB:
            total = gb * 1024
            head = None if need is None else total - need
            figs.append(
                value_fig(
                    "headroom with the larger model in E4B's place",
                    head,
                    "MiB",
                    keys={"model": str(model["id"]), "card": f"{gb} GB"},
                    mark_text=memory_mark(head),
                    derived=True,
                    inputs=(
                        f"{model['id']} file size {file_mib:,.0f} MiB ({source})",
                        "the desktop, run 5: " + _mib(desktop),
                        "the small model, speech and recognition beside E4B, run 5 heaviest step "
                        "less step e4b: " + _mib(beside),
                        "E4B's cache and buffers, run 5 step e4b less E4B's file size: "
                        + _mib(overhead),
                        f"{gb} GB taken as {total:,} MiB",
                    ),
                )
            )
    return figs


def _mib(value: float | None) -> str:
    return "not known" if value is None else f"{value:,.0f} MiB"


# ---------------------------------------------------------------------------
# The numbers file
# ---------------------------------------------------------------------------


def build_numbers(inp: Inputs, *, run: str, source_shas: Sequence[str], git_sha: str) -> dict:
    """Every figure the report shows, one object per run plus the budget, levels and cards."""
    prompts = Prompts.load()
    runs = [
        _run1(inp),
        _run2(inp),
        _run3(inp),
        _run4(inp),
        _run5(inp),
        _run7(inp, prompts),
        _game_run(inp, 8),
        _game_run(inp, 9),
    ]
    by_number = {r["run"]: r for r in runs}
    return {
        "schema": 1,
        "run": run,
        "report_git_sha": git_sha,
        "source_git_shas": list(source_shas),
        "targets": [
            t.as_dict() for t in (TTS_FIRST_AUDIO, STT_FINAL, PREFILL, TURN_INFERENCE, CHAINED_TURN)
        ]
        + [{"name": "memory headroom", "margin_mib": MARGIN_MIB, "gate_mib": GATE_MIB}],
        "runs": runs,
        "listening": _listening(inp),
        "budget": _budget(inp),
        "levels": _levels(inp, prompts, by_number),
        "larger_card": _larger_card(inp),
    }


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------


def fmt(value: float | None, unit: str) -> str:
    """A value with its unit, rounded for reading."""
    if value is None:
        return "-"
    formats = {
        "ms": "{:,.0f} ms",
        "s": "{:,.2f} s",
        "MiB": "{:,.0f} MiB",
        "tok/s": "{:,.1f} tok/s",
        "fps": "{:,.1f} fps",
        "%": "{:.1f}%",
        "score": "{:.2f}",
        "count": "{:,.0f}",
        "x": "{:.3f}",
    }
    return formats.get(unit, "{:g} " + unit).format(value)


def _value_text(fig: Mapping) -> str:
    if fig["stat"] == "share":
        if not fig["n"]:
            return "-"
        return f"{fig['count']} of {fig['n']} ({100 * fig['value']:.0f}%)"
    return fmt(basis(fig), str(fig["unit"]))


def _mark_text(fig: Mapping) -> str:
    return f"{fig['mark']}, derived" if fig.get("derived") else str(fig["mark"])


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _md_table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(_cell(h) for h in header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _derived_notes(figures: Iterable[Mapping]) -> list[str]:
    seen: dict[tuple, None] = {}
    for fig in figures:
        if fig.get("derived"):
            seen.setdefault((fig["label"], tuple(fig["inputs"])), None)
    if not seen:
        return []
    lines = ["", "Derived figures and their inputs:", ""]
    lines += [f"- {label}: {'; '.join(inputs)}." for label, inputs in seen]
    return lines


def _render_figures(table: Mapping) -> list[str]:
    keys = list(table["keys"])
    header = [k.capitalize() for k in keys] + [
        "Figure",
        "Median or value",
        "Slow case",
        "n",
        "Mark",
    ]
    rows = []
    for fig in table["figures"]:
        slow = fmt(fig.get("p90"), str(fig["unit"])) if fig["stat"] == "summary" else "-"
        rows.append(
            [fig["keys"].get(k, "") for k in keys]
            + [fig["label"], _value_text(fig), slow, fig["n"], _mark_text(fig)]
        )
    return _md_table(header, rows) + _derived_notes(table["figures"])


def _grid_cell(cell: object) -> str:
    if cell is None:
        return "-"
    if isinstance(cell, str):
        return cell
    text = _value_text(cell)
    if cell["stat"] == "summary" and cell["n"]:
        text += f", slow case {fmt(cell.get('p90'), str(cell['unit']))}"
    if cell["stat"] != "share" and cell["n"]:
        text += f" (n={cell['n']})"
    marks = [m for m in (cell["mark"],) if m not in (NO_TARGET,)]
    if cell.get("derived"):
        marks.append("derived")
    return text + (f"; {', '.join(marks)}" if marks else "")


def _render_grid(table: Mapping) -> list[str]:
    keys = list(table["keys"])
    header = [k.capitalize() for k in keys] + list(table["columns"])
    rows = [
        [row["keys"].get(k, "") for k in keys]
        + [_grid_cell(row["cells"].get(c)) for c in table["columns"]]
        for row in table["rows"]
    ]
    figures = [c for row in table["rows"] for c in row["cells"].values() if isinstance(c, Mapping)]
    return _md_table(header, rows) + _derived_notes(figures)


def _render_table(table: Mapping, level: str = "###") -> list[str]:
    lines = [f"{level} {table['title']}", ""]
    if table.get("note"):
        lines += [table["note"], ""]
    lines += _render_figures(table) if table["type"] == "figures" else _render_grid(table)
    return lines + [""]


def _render_notes(notes: Sequence[str]) -> list[str]:
    return [f"- {_cell(n)}" for n in notes] + ([""] if notes else [])


def _render_run(run: Mapping) -> list[str]:
    lines = [f"## Run {run['run']}: {run['title']}", ""]
    if run["status"] == NOT_YET_RUN:
        return lines + [f"Not yet run: {run['why']}.", ""]
    for table in run["tables"]:
        lines += _render_table(table)
    return lines + _render_notes(run["notes"])


def _render_budget(budget: Mapping) -> list[str]:
    lines = [
        f"## Budget: {budget['source']}",
        "",
        "From the end of speech to first audio, medians in ms. Today, the plan floor and the "
        f"target are {budget['source']}'s. Each stage this bench measures is put in once per "
        "option; the total adds the fastest option of each stage. Stages 3, 5 and 7 are not "
        "measured here and are derived.",
        "",
    ]
    header = [
        "Stage",
        "Today (derived)",
        "Plan floor",
        "Target",
        "Option",
        "This run",
        "Slow case",
        "n",
        "Source",
        "Mark",
    ]
    rows, figures = [], []
    for stage in budget["stages"]:
        name = f"{stage['stage']}. {stage['name']}"
        options = stage["options"] or [None]
        for fig in options:
            if fig is None:
                rows.append(
                    [
                        name,
                        stage["today"],
                        fmt(stage["floor"], "ms"),
                        fmt(stage["target"], "ms"),
                        "-",
                        "-",
                        "-",
                        0,
                        stage["source"],
                        NOT_YET_RUN,
                    ]
                )
                continue
            figures.append(fig)
            slow = fmt(fig.get("p90"), "ms") if fig["stat"] == "summary" else "-"
            option = fig["keys"]["option"]
            if option == stage["chosen"] and len(options) > 1:
                option += " (in the total)"
            rows.append(
                [
                    name,
                    stage["today"],
                    fmt(stage["floor"], "ms"),
                    fmt(stage["target"], "ms"),
                    option,
                    _value_text(fig),
                    slow,
                    fig["n"],
                    stage["source"],
                    _mark_text(fig),
                ]
            )
    total, slow = budget["total"]["median"], budget["total"]["slow"]
    rows.append(
        [
            "Total, median",
            BUDGET_TOTAL_TODAY,
            fmt(BUDGET_TOTAL.floor, "ms"),
            f"about {fmt(BUDGET_TOTAL.target, 'ms')}",
            "-",
            _value_text(total),
            "-",
            total["n"],
            "derived",
            _mark_text(total),
        ]
    )
    rows.append(
        [
            "Total, slow case",
            BUDGET_SLOW_TODAY,
            fmt(BUDGET_SLOW.floor, "ms"),
            f"{fmt(BUDGET_SLOW.target, 'ms')} or less",
            "-",
            _value_text(slow),
            "-",
            slow["n"],
            "derived",
            _mark_text(slow),
        ]
    )
    return lines + _md_table(header, rows) + _derived_notes([*figures, total, slow]) + [""]


def _render_levels(levels: Sequence[Mapping]) -> list[str]:
    lines = [
        "## Levels: what the tier design needs",
        "",
        "Each question the tier router and power-level design asks, with the figures that answer "
        "it.",
        "",
    ]
    for level in levels:
        lines += [f"### {level['question']}", "", f"Runs: {level['runs']}.", ""]
        for table in level["tables"]:
            lines += _render_table(table, "####")
        lines += _render_notes(level["notes"])
    return lines


def _render_listening(doc: Mapping) -> list[str]:
    lines = ["## Blind listening", ""]
    if doc["status"] == NOT_YET_RUN:
        return lines + [f"Not yet run: {doc['why']}.", ""]
    lines += [
        "Unblinded from the key after the ratings were saved; every clip is a stock voice.",
        "",
    ]
    for table in doc["tables"]:
        lines += _render_table(table)
    return lines + _render_notes(doc["notes"])


def _render_larger(doc: Mapping) -> list[str]:
    lines = [
        "## What a 16 to 24 GB card would add (derived)",
        "",
        "Every figure in this section is derived: run 5's measured memory and run 2's measured "
        "decode rate, carried to cards this PC does not have. The card figures are published "
        "specifications and November 2026 street prices from price aggregators, plausible only "
        "and re-checked before any purchase.",
        "",
    ]
    lines += _md_table(
        ["Card", "Memory", "Bandwidth", "Street price, November 2026"],
        [[c["card"], f"{c['memory_gb']} GB", c["bandwidth"], c["price"]] for c in doc["cards"]],
    )
    lines.append("")
    for table in doc["tables"]:
        lines += _render_table(table)
    return lines + _render_notes(doc["notes"])


def render_markdown(numbers: Mapping) -> str:
    """`REPORT.md` from the numbers file, so the two never disagree."""
    shas = ", ".join(f"`{s}`" for s in numbers["source_git_shas"]) or "none recorded"
    lines = [
        f"# Speech bench report, run {numbers['run']}",
        "",
        f"Rendered by `speech_bench report` from the rows under `<results-root>/{numbers['run']}/"
        f"speech/`, written at git {shas}.",
        "",
        "Each figure gives its median and slow case (the nearest-rank 90th percentile) with its "
        "sample count n, or a single value, count or share. Marks set a figure against its target "
        f"and floor from {BUDGET_SOURCE}; they mark, they do not rule. A figure marked derived "
        "is computed from the inputs listed under its table, not measured by this bench.",
        "",
        "## Targets",
        "",
    ]
    lines += _md_table(
        ["Figure", "Target", "Floor"],
        [
            ["speech first audio", "250 ms", "500 ms"],
            ["final text", "80 ms", "250 ms"],
            ["prefill to first token", "120 ms", "250 ms"],
            ["end-of-turn inference", "under 15 ms, on top of 200 ms of silence", "-"],
            ["memory headroom", "2 GiB gate", "512 MiB margin"],
            ["chained turn, commands and short answers", "2,000 ms", "3,000 ms"],
        ],
    )
    lines.append("")
    lines += _render_budget(numbers["budget"])
    lines += _render_levels(numbers["levels"])
    for run in numbers["runs"]:
        lines += _render_run(run)
    lines += _render_listening(numbers["listening"])
    lines += _render_larger(numbers["larger_card"])
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Building and writing
# ---------------------------------------------------------------------------


def build_report(
    speech: Path, *, run: str, git_sha: str, larger_models: Path | None = None
) -> tuple[dict, str]:
    """The numbers file and `REPORT.md` for one speech results tree; writes nothing."""
    inputs = load_inputs(speech, larger_models=larger_models)
    numbers = build_numbers(inputs, run=run, source_shas=_git_shas(speech), git_sha=git_sha)
    return numbers, render_markdown(numbers)


def write_outputs(speech: Path, numbers: Mapping, text: str, *, replace: bool) -> tuple[Path, Path]:
    """Write both files; an existing pair is replaced only with `replace`, one step per file."""
    targets = (speech / NUMBERS_FILE, speech / REPORT_FILE)
    existing = [p for p in targets if p.exists()]
    if existing and not replace:
        raise core.OutputExistsError(
            f"{existing[0]} already exists; run report --replace to render it again"
        )
    temps = tuple(p.with_name(p.name + ".tmp") for p in targets)
    for temp in temps:
        temp.unlink(missing_ok=True)
    core.write_json_exclusive(temps[0], numbers)
    core.write_text_exclusive(temps[1], text)
    for temp, target in zip(temps, targets, strict=True):
        os.replace(temp, target)
    return targets


def cmd_report(args: argparse.Namespace) -> int:
    speech = core.speech_root(args.results_root, args.run, repo_root=core.REPO_ROOT)
    if not speech.is_dir():
        raise ReportError(f"no speech results under {speech}")
    numbers, text = build_report(
        speech, run=args.run, git_sha=core.current_git_sha(), larger_models=args.larger_models
    )
    for path in write_outputs(speech, numbers, text, replace=args.replace):
        print(f"[ok] wrote {path}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "report", help="write numbers.json and REPORT.md from every suite's rows"
    )
    core.add_results_args(parser)
    parser.add_argument(
        "--larger-models",
        type=Path,
        help=f"the larger models' file sizes (default <speech>/host/{LARGER_MODELS_FILE})",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="render again over an existing report, as when runs 8 and 9 arrive",
    )
    parser.set_defaults(func=cmd_report)


def container_argv(ctx: core.PlanContext) -> list[tuple[str, list[str]]]:
    """`report` runs on the host and starts no container."""
    return []


def validate_inputs(repo_root: Path) -> list[str]:
    """The report reads only results; its committed inputs are the tool prompts, checked by T7."""
    return []


# ---------------------------------------------------------------------------
# The fixture tree and selftest
# ---------------------------------------------------------------------------


def write_fixture_tree(
    rows_path: Path, speech: Path, *, skip: Callable[[str], bool] | None = None
) -> None:
    """Lay out a results tree from `rows.jsonl`: each line names a file under `speech/` and holds
    a JSONL `row`, a whole `json` document or a `text` file.
    """
    for entry in read_jsonl(rows_path):
        rel = str(entry["file"])
        parts = Path(rel).parts
        if Path(rel).is_absolute() or ".." in parts:
            raise core.InputError(f"{rows_path}: {rel!r} is not a path under the speech root")
        if skip is not None and skip(rel):
            continue
        target = speech.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if "row" in entry:
            with target.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry["row"], ensure_ascii=False) + "\n")
        elif "json" in entry:
            target.write_text(json.dumps(entry["json"], indent=2) + "\n", encoding="utf-8")
        else:
            target.write_text(str(entry["text"]), encoding="utf-8", newline="\n")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def selftest() -> None:
    _check(mark(250, TTS_FIRST_AUDIO) == "target met", "250 ms meets the 250 ms target")
    _check(mark(501, TTS_FIRST_AUDIO) == "over floor", "501 ms is over the 500 ms floor")
    _check(mark(15, TURN_INFERENCE) == "15 ms or over", "15 ms is not under 15 ms")
    _check(memory_mark(511) == "under 512 MiB margin", "511 MiB is under the margin")
    with tempfile.TemporaryDirectory() as tmp:
        speech = Path(tmp) / FIXTURE_RUN / "speech"
        write_fixture_tree(FIXTURE_ROWS, speech)
        _, text = build_report(speech, run=FIXTURE_RUN, git_sha=FIXTURE_GIT_SHA)
        golden = GOLDEN_REPORT.read_text(encoding="utf-8")
        _check(text == golden, f"the fixture report differs from {GOLDEN_REPORT.name}")
        (speech / listening.SUITE / listening.RATINGS_FILE).unlink()
        try:
            build_report(speech, run=FIXTURE_RUN, git_sha=FIXTURE_GIT_SHA)
        except listening.KeySealedError:
            pass
        else:
            raise AssertionError("the report unblinded before ratings.csv existed")
