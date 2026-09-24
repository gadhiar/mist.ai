"""Analysis driver for the mist-model-bench goal (T3b).

Turns a `bench_host.py` results directory into `REPORT.md`, `summary.json`
and per-arm grades under `grades/`, evaluating the rules pre-registered in
`decision_rules.json`. Stdlib only.

Invocation: `python -m scripts.model_bench.analyse --results <dir> --out <dir>`
from the repo root. `--check` regenerates in memory and diffs byte-for-byte
against `--out`; with no `--results` it validates `decision_rules.json`
against `arms.json` only.

Decision 8 (mist.ai is public): no model text ever reaches an output file.
Every writer in this module either emits numbers/ids/labels it computed
itself, or copies a field through an explicit allowlist -- never a raw
record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
ARMS_JSON_PATH = PACKAGE_DIR / "arms.json"
DECISION_RULES_PATH = PACKAGE_DIR / "decision_rules.json"

if __package__ in (None, ""):
    # Allow `python scripts/model_bench/analyse.py` as well as the
    # documented `python -m scripts.model_bench.analyse` invocation.
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_harness import run as harness_run  # noqa: E402
from scripts.eval_harness import scorers as harness_scorers  # noqa: E402

LAYOUT_PASSES: tuple[str, ...] = ("screen", "finalist")
GRADES_LAYOUT_ALLOWLIST: tuple[str, ...] = (
    "id",
    "layout_id",
    "task",
    "correct",
    "parse_ok",
    "status",
    "finish_reason",
    "wall_ms",
    "phase",
    "model",
)
GRADES_USAGE_ALLOWLIST: tuple[str, ...] = ("completion_tokens", "prompt_tokens", "total_tokens")


# ---------------------------------------------------------------------------
# Small, deterministic JSON serialization helpers
# ---------------------------------------------------------------------------


def _round_floats(obj: Any) -> Any:
    """Recursively round every float to 6 decimals for byte-stable output."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    if isinstance(obj, tuple):
        return [_round_floats(v) for v in obj]
    return obj


def dumps_stable(obj: Any) -> str:
    """Deterministic JSON: sorted keys, fixed float precision, LF only."""
    return json.dumps(_round_floats(obj), sort_keys=True, ensure_ascii=True, indent=2) + "\n"


def dumps_line(obj: Any) -> str:
    """Deterministic single-line JSON for JSONL bodies."""
    return json.dumps(_round_floats(obj), sort_keys=True, ensure_ascii=True)


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Force LF regardless of platform line-ending conventions.
    normalized = text.replace("\r\n", "\n")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(normalized)


# ---------------------------------------------------------------------------
# Statistics: Wilson interval, nearest-rank percentile, cluster bootstrap
# ---------------------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float) -> tuple[float, float] | None:
    """Wilson score interval for a simple (unclustered) proportion k/n."""
    if n <= 0:
        return None
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = phat + (z * z) / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    lo = max(0.0, (center - margin) / denom)
    hi = min(1.0, (center + margin) / denom)
    return (lo, hi)


def nearest_rank_percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: rank = ceil(p/100 * N), 1-indexed on sorted values."""
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    rank = max(1, min(n, math.ceil((p / 100.0) * n)))
    return sorted_vals[rank - 1]


def cluster_bootstrap_ci(
    values_by_cluster: dict[str, list[float]], *, B: int, seed: int, confidence: float
) -> tuple[float, float] | None:
    """Percentile cluster bootstrap CI: resample cluster ids with replacement.

    Each bootstrap replicate resamples the same number of clusters as the
    original data (with replacement), keeps every row-level value in a
    resampled cluster, and takes the mean over the pooled values. The CI
    bounds are the nearest-rank (100*alpha/2) and (100*(1-alpha/2))
    percentiles of the B replicate means, so this is deterministic for a
    fixed seed and a fixed input dict.
    """
    clusters = sorted(values_by_cluster)
    if not clusters:
        return None
    rng = random.Random(seed)
    n_clusters = len(clusters)
    replicate_means: list[float] = []
    for _ in range(B):
        pooled: list[float] = []
        for _ in range(n_clusters):
            cluster = clusters[rng.randrange(n_clusters)]
            pooled.extend(values_by_cluster[cluster])
        replicate_means.append(statistics.mean(pooled) if pooled else 0.0)
    alpha = 1.0 - confidence
    lo = nearest_rank_percentile(replicate_means, 100.0 * (alpha / 2.0))
    hi = nearest_rank_percentile(replicate_means, 100.0 * (1.0 - alpha / 2.0))
    assert lo is not None and hi is not None
    return (lo, hi)


def margin_for(value: float | None, ci: tuple[float, float] | None, threshold: float, op: str) -> str:
    """Clause margin: clear if the bootstrap CI lies entirely on one side of the threshold."""
    if ci is None:
        return "n/a"
    lo, hi = ci
    if op in (">=", ">"):
        if lo > threshold or (op == ">=" and lo >= threshold):
            return "clear"
        if hi < threshold or (op == ">" and hi <= threshold):
            return "clear"
        return "within-noise"
    if op in ("<=", "<"):
        if hi < threshold or (op == "<=" and hi <= threshold):
            return "clear"
        if lo > threshold or (op == "<" and lo >= threshold):
            return "clear"
        return "within-noise"
    return "n/a"


# ---------------------------------------------------------------------------
# MetricResult: value + coverage + CIs, missing/incomplete tracked explicitly
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    value: float | None
    n: int | None = None
    n_expected: int | None = None
    k: int | None = None  # numerator, for proportion-shaped metrics
    wilson: tuple[float, float] | None = None
    bootstrap: tuple[float, float] | None = None
    missing: bool = False
    complete: bool = True
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def usable(self) -> bool:
        """True only when there is a value AND coverage is not incomplete."""
        return not self.missing and self.complete and self.value is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "n": self.n,
            "n_expected": self.n_expected,
            "k": self.k,
            "wilson": list(self.wilson) if self.wilson else None,
            "bootstrap": list(self.bootstrap) if self.bootstrap else None,
            "missing": self.missing,
            "complete": self.complete,
            "note": self.note,
            "extra": self.extra,
        }


def missing_metric(note: str) -> MetricResult:
    return MetricResult(value=None, missing=True, complete=False, note=note)


# ---------------------------------------------------------------------------
# decision_rules.json / arms.json loading
# ---------------------------------------------------------------------------


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_decision_rules(path: Path = DECISION_RULES_PATH) -> dict[str, Any]:
    return load_json_file(path)


def load_arms(path: Path = ARMS_JSON_PATH) -> dict[str, Any]:
    doc = load_json_file(path)
    return doc["arms"]


def decision_rules_sha256(path: Path = DECISION_RULES_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_decision_rules(rules: dict[str, Any], arms: dict[str, Any]) -> list[str]:
    """Cross-check decision_rules.json against arms.json and its own metrics table.

    Returns a list of problem strings; empty means valid.
    """
    problems: list[str] = []
    metric_names = set(rules.get("metrics", {}))
    known_metric_refs = {
        "layout_acc",
        "layout_mean_completion_tokens",
        "layout_p95_wall_ms",
        "harness_score",
        "truncation_rate",
        "decode_tps",
        "ttft_ms",
        "arm_peak_mib",
        "voice_vram_lower_mib",
        "voice_vram_upper_mib",
        "total_mib",
    }
    if known_metric_refs - metric_names:
        problems.append(f"metrics missing definitions: {sorted(known_metric_refs - metric_names)}")

    def _check_arm(arm_id: str, where: str) -> None:
        if arm_id not in arms:
            problems.append(f"{where} references unknown arm {arm_id!r}")

    constants = rules.get("constants", {})
    for rule_key, cfg in constants.get("r2_candidates", {}).items():
        _check_arm(cfg["arm"], f"constants.r2_candidates.{rule_key}.arm")
        _check_arm(cfg["thinking_arm"], f"constants.r2_candidates.{rule_key}.thinking_arm")
    _check_arm(constants.get("r2_anchor_arm", ""), "constants.r2_anchor_arm")
    for arm_id, cfg in constants.get("r6_arms", {}).items():
        _check_arm(arm_id, "constants.r6_arms")
        _check_arm(cfg["base"], f"constants.r6_arms.{arm_id}.base")
    pair = constants.get("r7_build_pair", {})
    if pair:
        _check_arm(pair.get("old", ""), "constants.r7_build_pair.old")
        _check_arm(pair.get("new", ""), "constants.r7_build_pair.new")
    _check_arm("c1-512", "R1")
    _check_arm("c1-256", "R1 (informational)")

    thresholds = rules.get("thresholds", {})
    required_thresholds = {
        "r1_keep_e4b_layout_acc_min",
        "r2_l_off_layout_acc_min",
        "r2_l_on_layout_acc_min",
        "r2_l_on_mean_completion_tokens_max",
        "r2_s1_schema_conformance_json_object_min",
        "r2_s2_tool_selection_min",
        "r2_d_decode_tps_min",
        "r2_p_layout_p95_wall_ms_max",
        "r3_drop_moe_truncation_rate_min",
        "r4_thinking_lever_layout_acc_max",
        "r5_voice_vram_lower_trigger_mib",
        "r5_voice_vram_upper_needs_review_mib",
    }
    missing_thresholds = required_thresholds - set(thresholds)
    if missing_thresholds:
        problems.append(f"thresholds missing keys: {sorted(missing_thresholds)}")

    stats = rules.get("statistics", {})
    bootstrap = stats.get("bootstrap", {})
    if bootstrap.get("B") != 10000 or bootstrap.get("seed") != 20260924:
        problems.append("statistics.bootstrap.B/seed do not match the pre-registered B=10000, seed=20260924")

    return problems


# ---------------------------------------------------------------------------
# Results-directory loading
# ---------------------------------------------------------------------------


@dataclass
class ArmInputs:
    arm_id: str
    dir: Path
    meta: dict[str, Any] | None
    harness_records: list[dict[str, Any]]  # raw CaseResult dicts, all test_names in the file
    layout_rows: dict[str, list[dict[str, Any]]]  # pass -> raw graded rows
    ttft_rows: list[dict[str, Any]]
    vram_rows: list[dict[str, Any]]
    correctness: dict[int, list[dict[str, Any]]]  # rep -> raw rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def load_arm_inputs(results_dir: Path, arm_id: str) -> ArmInputs:
    arm_dir = results_dir / arm_id
    meta_path = arm_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None

    harness_records: list[dict[str, Any]] = []
    if meta is not None and meta.get("harness"):
        candidate = meta["harness"]["candidate"]
        harness_path = arm_dir / "harness" / "harness" / f"{candidate}.jsonl"
        harness_records = _read_jsonl(harness_path)

    layout_rows: dict[str, list[dict[str, Any]]] = {}
    for layout_pass in LAYOUT_PASSES:
        graded_path = arm_dir / "layout" / layout_pass / "graded.jsonl"
        if graded_path.exists():
            layout_rows[layout_pass] = _read_jsonl(graded_path)

    ttft_rows = _read_jsonl(arm_dir / "ttft.jsonl")
    vram_rows = _read_csv_rows(arm_dir / "vram.csv")

    correctness: dict[int, list[dict[str, Any]]] = {}
    if arm_dir.exists():
        for path in sorted(arm_dir.glob("correctness.r*.jsonl")):
            suffix = path.stem.split(".r", 1)[1]
            try:
                rep = int(suffix)
            except ValueError:
                continue
            correctness[rep] = _read_jsonl(path)

    return ArmInputs(
        arm_id=arm_id,
        dir=arm_dir,
        meta=meta,
        harness_records=harness_records,
        layout_rows=layout_rows,
        ttft_rows=ttft_rows,
        vram_rows=vram_rows,
        correctness=correctness,
    )


@dataclass
class SessionInputs:
    vram_steps: dict[str, dict[str, Any]]  # label -> step dict
    manual: dict[str, dict[str, Any]]  # arm_id -> {memtest_errors, whea_events}
    total_mib: float | None


def load_session_inputs(results_dir: Path) -> SessionInputs:
    session_dir = results_dir / "session"
    steps: dict[str, dict[str, Any]] = {}
    total_mib: float | None = None
    steps_path = session_dir / "vram_steps.json"
    if steps_path.exists():
        doc = json.loads(steps_path.read_text(encoding="utf-8"))
        for step in doc.get("steps", []):
            label = step.get("label")
            if label:
                steps[label] = step
                if total_mib is None and step.get("total_mib") is not None:
                    total_mib = float(step["total_mib"])
    manual: dict[str, dict[str, Any]] = {}
    manual_path = session_dir / "manual.json"
    if manual_path.exists():
        manual = json.loads(manual_path.read_text(encoding="utf-8"))
    return SessionInputs(vram_steps=steps, manual=manual, total_mib=total_mib)


# ---------------------------------------------------------------------------
# Per-metric computation
# ---------------------------------------------------------------------------

BOOTSTRAP_B_DEFAULT = 10000
BOOTSTRAP_SEED_DEFAULT = 20260924
BOOTSTRAP_CONFIDENCE_DEFAULT = 0.95
WILSON_Z_DEFAULT = 1.9599639845400545


def compute_layout_acc(rows: list[dict[str, Any]] | None, n_expected: int, stats_cfg: dict[str, Any]) -> MetricResult:
    if not rows:
        return missing_metric("no layout rows for this pass")
    accuracy_rows = [r for r in rows if r.get("phase") == "accuracy"]
    n = len(accuracy_rows)
    if n == 0:
        return missing_metric("no phase=='accuracy' rows for this pass")
    k = sum(1 for r in accuracy_rows if r.get("correct") is True)
    value = k / n
    complete = n == n_expected
    wilson = wilson_interval(k, n, stats_cfg["wilson_z"])
    values_by_cluster: dict[str, list[float]] = {}
    for r in accuracy_rows:
        cluster = str(r.get("layout_id"))
        values_by_cluster.setdefault(cluster, []).append(1.0 if r.get("correct") is True else 0.0)
    bootstrap = cluster_bootstrap_ci(
        values_by_cluster,
        B=stats_cfg["B"],
        seed=stats_cfg["seed"],
        confidence=stats_cfg["confidence"],
    )
    note = None if complete else f"incomplete coverage: {n}/{n_expected}"
    return MetricResult(
        value=value,
        n=n,
        n_expected=n_expected,
        k=k,
        wilson=wilson,
        bootstrap=bootstrap,
        complete=complete,
        note=note,
    )


def compute_layout_completion_tokens(
    rows: list[dict[str, Any]] | None,
) -> tuple[MetricResult, float | None]:
    """Returns (mean_completion_tokens over accuracy rows, tokens-per-correct info)."""
    if not rows:
        return missing_metric("no layout rows for this pass"), None
    accuracy_rows = [r for r in rows if r.get("phase") == "accuracy"]
    if not accuracy_rows:
        return missing_metric("no phase=='accuracy' rows for this pass"), None
    tokens = [float((r.get("usage") or {}).get("completion_tokens", 0)) for r in accuracy_rows]
    n_correct = sum(1 for r in accuracy_rows if r.get("correct") is True)
    mean_tokens = statistics.mean(tokens)
    per_correct = (sum(tokens) / n_correct) if n_correct > 0 else None
    return MetricResult(value=mean_tokens, n=len(accuracy_rows)), per_correct


def compute_layout_p95_wall_ms(rows: list[dict[str, Any]] | None) -> MetricResult:
    if not rows:
        return missing_metric("no layout rows for this pass")
    accuracy_rows = [r for r in rows if r.get("phase") == "accuracy"]
    if not accuracy_rows:
        return missing_metric("no phase=='accuracy' rows for this pass")
    values = [float(r["wall_ms"]) for r in accuracy_rows if r.get("wall_ms") is not None]
    if not values:
        return missing_metric("no wall_ms values on accuracy rows")
    p95 = nearest_rank_percentile(values, 95.0)
    return MetricResult(value=p95, n=len(values))


def compute_harness_scores_for_arm(
    inputs: ArmInputs, stats_cfg: dict[str, Any]
) -> tuple[dict[str, MetricResult], dict[str, harness_scorers.TestScores]]:
    """Returns (harness_score per test_name, raw TestScores per test_name)."""
    if inputs.meta is None or not inputs.meta.get("harness"):
        return {}, {}
    harness_cfg = inputs.meta["harness"]
    test_names: list[str] = list(harness_cfg["tests"])
    iterations_expected = int(harness_cfg["iterations"])
    candidate_id = harness_cfg["candidate"]

    test_files = harness_run.load_test_files(test_names, harness_run.DEFAULT_TESTS_DIR)
    cases_expected_by_name = {tf.name: len(tf.cases) for tf in test_files}

    # score_run reads from a path; write the already-loaded records to a
    # temp-free path is unnecessary -- we already have the on-disk file.
    candidate = inputs.meta["harness"]["candidate"]
    harness_path = inputs.dir / "harness" / "harness" / f"{candidate}.jsonl"

    run_scores = harness_scorers.score_run([harness_path], test_files)
    candidate_scores = run_scores.per_candidate.get(candidate_id)

    results: dict[str, MetricResult] = {}
    raw: dict[str, harness_scorers.TestScores] = {}
    for name in test_names:
        cases_expected = cases_expected_by_name.get(name, 0)
        expected_n = cases_expected * iterations_expected
        if candidate_scores is None or name not in candidate_scores.per_test:
            results[name] = missing_metric(f"no harness records for test {name!r}")
            continue
        test_scores = candidate_scores.per_test[name]
        raw[name] = test_scores
        n = len(test_scores.case_scores)
        complete = n == expected_n
        value = test_scores.mean_score
        k = test_scores.pass_count
        wilson = wilson_interval(k, n, stats_cfg["wilson_z"]) if n > 0 else None
        values_by_cluster: dict[str, list[float]] = {}
        for cs in test_scores.case_scores:
            values_by_cluster.setdefault(cs.case_id, []).append(cs.score)
        bootstrap = (
            cluster_bootstrap_ci(
                values_by_cluster,
                B=stats_cfg["B"],
                seed=stats_cfg["seed"],
                confidence=stats_cfg["confidence"],
            )
            if values_by_cluster
            else None
        )
        note = None if complete else f"incomplete coverage: {n}/{expected_n}"
        results[name] = MetricResult(
            value=value,
            n=n,
            n_expected=expected_n,
            k=k,
            wilson=wilson,
            bootstrap=bootstrap,
            complete=complete,
            note=note,
        )
    return results, raw


def compute_truncation_rate(
    inputs: ArmInputs, test_name: str, stats_cfg: dict[str, Any]
) -> MetricResult:
    if inputs.meta is None or not inputs.meta.get("harness"):
        return missing_metric("no harness suite recorded for this arm")
    harness_cfg = inputs.meta["harness"]
    if test_name not in harness_cfg["tests"]:
        return missing_metric(f"test {test_name!r} not run for this arm")
    iterations_expected = int(harness_cfg["iterations"])
    test_files = harness_run.load_test_files([test_name], harness_run.DEFAULT_TESTS_DIR)
    cases_expected = len(test_files[0].cases) * iterations_expected

    records = [r for r in inputs.harness_records if r.get("test_name") == test_name]
    n = len(records)
    if n == 0:
        return missing_metric(f"no records for test {test_name!r}")
    k = sum(1 for r in records if r.get("finish_reason") == "length")
    value = k / n
    complete = n == cases_expected
    wilson = wilson_interval(k, n, stats_cfg["wilson_z"])
    values_by_cluster: dict[str, list[float]] = {}
    for r in records:
        cluster = str(r.get("case_id"))
        values_by_cluster.setdefault(cluster, []).append(1.0 if r.get("finish_reason") == "length" else 0.0)
    bootstrap = cluster_bootstrap_ci(
        values_by_cluster, B=stats_cfg["B"], seed=stats_cfg["seed"], confidence=stats_cfg["confidence"]
    )
    note = None if complete else f"incomplete coverage: {n}/{cases_expected}"
    return MetricResult(
        value=value, n=n, n_expected=cases_expected, k=k, wilson=wilson, bootstrap=bootstrap,
        complete=complete, note=note,
    )


def compute_decode_tps(rows: list[dict[str, Any]], ctx_target: int, min_rows: int) -> MetricResult:
    filtered = [
        r for r in rows
        if r.get("ctx_target") == ctx_target and r.get("warmup") is False and r.get("error") is None
    ]
    if len(filtered) < min_rows:
        return missing_metric(f"only {len(filtered)} ttft rows at ctx={ctx_target} (need >= {min_rows})")
    values = [float(r["predicted_per_second"]) for r in filtered]
    return MetricResult(value=statistics.median(values), n=len(values))


def compute_ttft_ms_by_ctx(rows: list[dict[str, Any]], min_rows: int) -> dict[int, MetricResult]:
    by_ctx: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("warmup") is False and r.get("error") is None and r.get("ctx_target") is not None:
            by_ctx.setdefault(int(r["ctx_target"]), []).append(r)
    out: dict[int, MetricResult] = {}
    for ctx, group in by_ctx.items():
        if len(group) < min_rows:
            out[ctx] = missing_metric(f"only {len(group)} ttft rows at ctx={ctx} (need >= {min_rows})")
            continue
        values = [float(r["ttft_ms"]) for r in group]
        out[ctx] = MetricResult(value=statistics.median(values), n=len(values))
    return out


def compute_arm_peak_mib(vram_rows: list[dict[str, Any]]) -> MetricResult:
    if not vram_rows:
        return missing_metric("no vram.csv rows for this arm")
    values = [float(r["memory_used_mib"]) for r in vram_rows if r.get("memory_used_mib") not in (None, "")]
    if not values:
        return missing_metric("no memory_used_mib values in vram.csv")
    return MetricResult(value=max(values), n=len(values))


def compute_voice_metrics(session: SessionInputs) -> tuple[MetricResult, MetricResult, float | None]:
    backend_idle = session.vram_steps.get("backend_idle")
    mist_llm = session.vram_steps.get("mist_llm")
    voice_peak = session.vram_steps.get("voice_peak")

    if backend_idle is None or mist_llm is None:
        lower = missing_metric("session/vram_steps.json missing backend_idle or mist_llm step")
    else:
        lower = MetricResult(value=float(backend_idle["median_mib"]) - float(mist_llm["median_mib"]))

    if voice_peak is None or mist_llm is None:
        upper = missing_metric("session/vram_steps.json missing voice_peak or mist_llm step")
    else:
        upper = MetricResult(value=float(voice_peak["max_mib"]) - float(mist_llm["median_mib"]))

    return lower, upper, session.total_mib


# ---------------------------------------------------------------------------
# Per-arm metric bundle
# ---------------------------------------------------------------------------


@dataclass
class ArmMetrics:
    arm_id: str
    layout_acc: dict[str, MetricResult] = field(default_factory=dict)  # pass -> result
    layout_mean_completion_tokens: dict[str, MetricResult] = field(default_factory=dict)
    layout_completion_tokens_per_correct: dict[str, float | None] = field(default_factory=dict)
    layout_p95_wall_ms: dict[str, MetricResult] = field(default_factory=dict)
    harness_score: dict[str, MetricResult] = field(default_factory=dict)  # test_name -> result
    truncation_rate: dict[str, MetricResult] = field(default_factory=dict)
    decode_tps: MetricResult = field(default_factory=lambda: missing_metric("no ttft.jsonl for this arm"))
    ttft_ms: dict[int, MetricResult] = field(default_factory=dict)
    arm_peak_mib: MetricResult = field(default_factory=lambda: missing_metric("no vram.csv for this arm"))
    suites_completed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    decision_rules_sha256: str | None = None
    present: bool = True


def pick_layout_pass(acc_by_pass: dict[str, MetricResult]) -> tuple[str | None, MetricResult | None]:
    """Finalist supersedes screen when complete; never pooled."""
    finalist = acc_by_pass.get("finalist")
    screen = acc_by_pass.get("screen")
    if finalist is not None and finalist.complete:
        return "finalist", finalist
    if screen is not None and screen.complete:
        return "screen", screen
    if finalist is not None:
        return "finalist", finalist
    if screen is not None:
        return "screen", screen
    return None, None


@dataclass
class RunMetrics:
    arms: dict[str, ArmMetrics]
    raw_test_scores: dict[str, dict[str, harness_scorers.TestScores]]  # arm -> test_name -> TestScores
    raw_layout_rows: dict[str, dict[str, list[dict[str, Any]]]]  # arm -> pass -> rows
    raw_correctness: dict[str, dict[int, list[dict[str, Any]]]]  # arm -> rep -> rows
    voice_vram_lower_mib: MetricResult
    voice_vram_upper_mib: MetricResult
    total_mib: float | None
    manual: dict[str, dict[str, Any]]
    arm_order: list[str]


def compute_all_metrics(results_dir: Path, arm_ids: list[str], rules: dict[str, Any]) -> RunMetrics:
    stats_cfg = {
        "wilson_z": rules["statistics"]["wilson"]["z"],
        "B": rules["statistics"]["bootstrap"]["B"],
        "seed": rules["statistics"]["bootstrap"]["seed"],
        "confidence": rules["statistics"]["bootstrap"]["confidence"],
    }
    n_expected = rules["constants"]["layout_n_expected"]
    ctx_target = rules["constants"]["ttft_ctx_target"]
    min_ttft_rows = rules["constants"]["min_ttft_rows"]

    arms: dict[str, ArmMetrics] = {}
    raw_test_scores: dict[str, dict[str, harness_scorers.TestScores]] = {}
    raw_layout_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    raw_correctness: dict[str, dict[int, list[dict[str, Any]]]] = {}

    for arm_id in arm_ids:
        inputs = load_arm_inputs(results_dir, arm_id)
        am = ArmMetrics(arm_id=arm_id)
        if inputs.meta is None:
            am.present = False
            arms[arm_id] = am
            raw_test_scores[arm_id] = {}
            raw_layout_rows[arm_id] = {}
            raw_correctness[arm_id] = {}
            continue

        am.suites_completed = list(inputs.meta.get("suites_completed", []))
        am.errors = list(inputs.meta.get("errors", []))
        am.decision_rules_sha256 = inputs.meta.get("decision_rules_sha256")

        for layout_pass, rows in inputs.layout_rows.items():
            am.layout_acc[layout_pass] = compute_layout_acc(rows, n_expected[layout_pass], stats_cfg)
            tokens_result, per_correct = compute_layout_completion_tokens(rows)
            am.layout_mean_completion_tokens[layout_pass] = tokens_result
            am.layout_completion_tokens_per_correct[layout_pass] = per_correct
            am.layout_p95_wall_ms[layout_pass] = compute_layout_p95_wall_ms(rows)

        harness_scores, raw_scores = compute_harness_scores_for_arm(inputs, stats_cfg)
        am.harness_score = harness_scores
        raw_test_scores[arm_id] = raw_scores
        for test_name in harness_scores:
            am.truncation_rate[test_name] = compute_truncation_rate(inputs, test_name, stats_cfg)

        am.decode_tps = compute_decode_tps(inputs.ttft_rows, ctx_target, min_ttft_rows)
        am.ttft_ms = compute_ttft_ms_by_ctx(inputs.ttft_rows, min_ttft_rows)
        am.arm_peak_mib = compute_arm_peak_mib(inputs.vram_rows)

        arms[arm_id] = am
        raw_layout_rows[arm_id] = inputs.layout_rows
        raw_correctness[arm_id] = inputs.correctness

    session = load_session_inputs(results_dir)
    lower, upper, total_mib = compute_voice_metrics(session)

    return RunMetrics(
        arms=arms,
        raw_test_scores=raw_test_scores,
        raw_layout_rows=raw_layout_rows,
        raw_correctness=raw_correctness,
        voice_vram_lower_mib=lower,
        voice_vram_upper_mib=upper,
        total_mib=total_mib,
        manual=session.manual,
        arm_order=arm_ids,
    )


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

VERDICT_PRECEDENCE = {"fail": 0, "missing": 1, "needs-review": 2, "pass": 3}
TRIGGER_PRECEDENCE = {"triggered": 0, "missing": 1, "needs-review": 2, "not-triggered": 3}


def combine_gate_verdicts(verdicts: list[str]) -> str:
    return min(verdicts, key=lambda v: VERDICT_PRECEDENCE[v])


@dataclass
class ClauseResult:
    id: str
    metric: str
    arm: str | None
    value: float | None
    threshold: float | None
    op: str | None
    verdict: str
    margin: str
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "metric": self.metric,
            "arm": self.arm,
            "value": self.value,
            "threshold": self.threshold,
            "op": self.op,
            "verdict": self.verdict,
            "margin": self.margin,
            "note": self.note,
            "extra": self.extra,
        }


@dataclass
class RuleResult:
    id: str
    kind: str
    question: str
    label: str
    verdict: str
    clauses: list[ClauseResult]
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "question": self.question,
            "label": self.label,
            "verdict": self.verdict,
            "clauses": [c.to_dict() for c in self.clauses],
            "info": self.info,
        }


def _op_compare(value: float, threshold: float, op: str) -> bool:
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    raise ValueError(f"unknown op {op!r}")


def evaluate_simple_clause(
    clause_id: str, metric_name: str, arm: str | None, result: MetricResult, op: str, threshold: float
) -> ClauseResult:
    if not result.usable():
        return ClauseResult(
            id=clause_id, metric=metric_name, arm=arm, value=result.value, threshold=threshold, op=op,
            verdict="missing", margin="n/a", note=result.note or "missing or incomplete coverage",
        )
    ok = _op_compare(result.value, threshold, op)
    margin = margin_for(result.value, result.bootstrap, threshold, op)
    return ClauseResult(
        id=clause_id, metric=metric_name, arm=arm, value=result.value, threshold=threshold, op=op,
        verdict="pass" if ok else "fail", margin=margin,
    )


def evaluate_r1(metrics: RunMetrics, rules: dict[str, Any]) -> RuleResult:
    thresholds = rules["thresholds"]
    c1_512 = metrics.arms.get("c1-512")
    pass_used, result = (None, None)
    if c1_512 is not None:
        pass_used, result = pick_layout_pass(c1_512.layout_acc)
    if result is None:
        clause = ClauseResult(
            id="layout_acc_c1_512", metric="layout_acc", arm="c1-512", value=None,
            threshold=thresholds["r1_keep_e4b_layout_acc_min"], op=">=", verdict="missing", margin="n/a",
            note="no layout data for c1-512",
        )
    else:
        clause = evaluate_simple_clause(
            "layout_acc_c1_512", "layout_acc", "c1-512", result, ">=", thresholds["r1_keep_e4b_layout_acc_min"]
        )
        clause.extra["pass_used"] = pass_used

    info: dict[str, Any] = {}
    c1_256 = metrics.arms.get("c1-256")
    if c1_256 is not None:
        pass_256, result_256 = pick_layout_pass(c1_256.layout_acc)
        if result_256 is not None:
            info["c1_256_layout_acc"] = {
                "pass_used": pass_256,
                "value": result_256.value,
                "n": result_256.n,
                "n_expected": result_256.n_expected,
                "complete": result_256.complete,
            }
    return RuleResult(
        id="R1", kind="gate", label="keep_e4b_budget",
        question="Does Gemma 4 E4B with a thinking budget clear the layout accuracy bar?",
        verdict=combine_gate_verdicts([clause.verdict]), clauses=[clause], info=info,
    )


def _l_clause(
    candidate: str, thinking_candidate: str, metrics: RunMetrics, thresholds: dict[str, Any]
) -> tuple[ClauseResult, str | None, str | None]:
    """Returns (L clause, arm used for P, pass used for P)."""
    off_arm = metrics.arms.get(candidate)
    on_arm = metrics.arms.get(thinking_candidate)

    off_pass, off_result = (None, None)
    if off_arm is not None:
        off_pass, off_result = pick_layout_pass(off_arm.layout_acc)
    on_pass, on_result = (None, None)
    on_tokens_result = None
    if on_arm is not None:
        on_pass, on_result = pick_layout_pass(on_arm.layout_acc)
        if on_pass is not None:
            on_tokens_result = on_arm.layout_mean_completion_tokens.get(on_pass)

    off_holds = off_result is not None and off_result.usable() and off_result.value >= thresholds["r2_l_off_layout_acc_min"]
    on_holds = (
        on_result is not None and on_result.usable()
        and on_tokens_result is not None and on_tokens_result.usable()
        and on_result.value >= thresholds["r2_l_on_layout_acc_min"]
        and on_tokens_result.value <= thresholds["r2_l_on_mean_completion_tokens_max"]
    )

    if off_holds:
        margin = margin_for(off_result.value, off_result.bootstrap, thresholds["r2_l_off_layout_acc_min"], ">=")
        clause = ClauseResult(
            id="L", metric="layout_acc", arm=candidate, value=off_result.value,
            threshold=thresholds["r2_l_off_layout_acc_min"], op=">=", verdict="pass", margin=margin,
            extra={"branch": "off", "pass_used": off_pass},
        )
        return clause, candidate, off_pass

    if on_holds:
        margin = margin_for(on_result.value, on_result.bootstrap, thresholds["r2_l_on_layout_acc_min"], ">=")
        clause = ClauseResult(
            id="L", metric="layout_acc", arm=thinking_candidate, value=on_result.value,
            threshold=thresholds["r2_l_on_layout_acc_min"], op=">=", verdict="pass", margin=margin,
            extra={
                "branch": "on", "pass_used": on_pass,
                "mean_completion_tokens": on_tokens_result.value,
                "mean_completion_tokens_threshold": thresholds["r2_l_on_mean_completion_tokens_max"],
            },
        )
        return clause, thinking_candidate, on_pass

    off_conclusive = off_result is not None and off_result.usable()
    on_conclusive = (
        on_result is not None and on_result.usable()
        and on_tokens_result is not None and on_tokens_result.usable()
    )
    if off_conclusive and on_conclusive:
        verdict = "fail"
        note = "neither the off-branch nor the on-branch layout clause held"
    else:
        verdict = "missing"
        note = "insufficient data to evaluate L on either branch"
    clause = ClauseResult(
        id="L", metric="layout_acc", arm=candidate, value=off_result.value if off_result else None,
        threshold=thresholds["r2_l_off_layout_acc_min"], op=">=", verdict=verdict, margin="n/a", note=note,
        extra={"branch": None},
    )
    return clause, None, None


def _f_clause(
    candidate: str, metrics: RunMetrics, margin_mib: float
) -> ClauseResult:
    am = metrics.arms.get(candidate)
    peak = am.arm_peak_mib if am is not None else missing_metric("arm absent")
    lower = metrics.voice_vram_lower_mib
    upper = metrics.voice_vram_upper_mib
    total = metrics.total_mib

    if not peak.usable() or not lower.usable() or not upper.usable() or total is None:
        missing_bits = []
        if not peak.usable():
            missing_bits.append(f"arm_peak_mib({candidate})")
        if not lower.usable():
            missing_bits.append("voice_vram_lower_mib")
        if not upper.usable():
            missing_bits.append("voice_vram_upper_mib")
        if total is None:
            missing_bits.append("total_mib")
        return ClauseResult(
            id="F", metric="arm_peak_mib", arm=candidate, value=peak.value, threshold=None, op=None,
            verdict="missing", margin="n/a", note=f"missing inputs: {missing_bits}",
        )

    fits_upper = peak.value + upper.value + margin_mib <= total
    fails_lower = peak.value + lower.value + margin_mib > total
    if fits_upper:
        verdict = "pass"
    elif fails_lower:
        verdict = "fail"
    else:
        verdict = "needs-review"
    return ClauseResult(
        id="F", metric="arm_peak_mib", arm=candidate, value=peak.value, threshold=total, op="<=",
        verdict=verdict, margin="n/a",
        extra={
            "voice_vram_lower_mib": lower.value, "voice_vram_upper_mib": upper.value,
            "total_mib": total, "margin_mib": margin_mib,
        },
    )


def evaluate_r2(
    rule_key: str, candidate: str, thinking_candidate: str, metrics: RunMetrics, rules: dict[str, Any]
) -> RuleResult:
    thresholds = rules["thresholds"]
    constants = rules["constants"]
    anchor_arm = constants["r2_anchor_arm"]
    margin_mib = constants["margin_mib"]

    l_clause, arm_for_p, pass_for_p = _l_clause(candidate, thinking_candidate, metrics, thresholds)

    def _anchored_clause(clause_id: str, test_name: str, threshold_key: str) -> ClauseResult:
        candidate_result = metrics.arms.get(candidate)
        cand_metric = (
            candidate_result.harness_score.get(test_name) if candidate_result is not None else None
        )
        anchor_arm_metrics = metrics.arms.get(anchor_arm)
        anchor_metric = anchor_arm_metrics.harness_score.get(test_name) if anchor_arm_metrics is not None else None
        threshold = thresholds[threshold_key]

        if anchor_arm_metrics is None or anchor_metric is None or not anchor_metric.usable():
            return ClauseResult(
                id=clause_id, metric="harness_score", arm=candidate, value=(cand_metric.value if cand_metric else None),
                threshold=threshold, op=">=", verdict="missing", margin="n/a",
                note=f"anchor arm {anchor_arm!r} harness_score for {test_name!r} is missing",
            )
        if cand_metric is None or not cand_metric.usable():
            return ClauseResult(
                id=clause_id, metric="harness_score", arm=candidate, value=None, threshold=threshold, op=">=",
                verdict="missing", margin="n/a", note=f"no usable harness_score for {candidate}/{test_name}",
            )
        if anchor_metric.value < threshold:
            return ClauseResult(
                id=clause_id, metric="harness_score", arm=candidate, value=cand_metric.value, threshold=threshold,
                op=">=", verdict="needs-review", margin="n/a",
                note="anchor_below_threshold",
                extra={"anchor_arm": anchor_arm, "anchor_value": anchor_metric.value},
            )
        ok = cand_metric.value >= threshold
        margin = margin_for(cand_metric.value, cand_metric.bootstrap, threshold, ">=")
        return ClauseResult(
            id=clause_id, metric="harness_score", arm=candidate, value=cand_metric.value, threshold=threshold,
            op=">=", verdict="pass" if ok else "fail", margin=margin,
        )

    s1 = _anchored_clause("S1", "schema_conformance_json_object", "r2_s1_schema_conformance_json_object_min")
    s2 = _anchored_clause("S2", "tool_selection", "r2_s2_tool_selection_min")

    candidate_am = metrics.arms.get(candidate)
    d_result = candidate_am.decode_tps if candidate_am is not None else missing_metric("arm absent")
    d_clause = evaluate_simple_clause("D", "decode_tps", candidate, d_result, ">=", thresholds["r2_d_decode_tps_min"])
    if d_clause.margin == "n/a" and d_clause.verdict in ("pass", "fail"):
        d_clause.margin = "n/a"  # decode_tps carries no bootstrap CI by design

    if arm_for_p is None:
        arm_for_p = candidate
        p_am = metrics.arms.get(candidate)
        pass_for_p, p_result = (None, None)
        if p_am is not None:
            pass_for_p, acc_result = pick_layout_pass(p_am.layout_acc)
            p_result = p_am.layout_p95_wall_ms.get(pass_for_p) if pass_for_p else None
    else:
        p_am = metrics.arms.get(arm_for_p)
        p_result = p_am.layout_p95_wall_ms.get(pass_for_p) if p_am is not None and pass_for_p else None

    if p_result is None:
        p_clause = ClauseResult(
            id="P", metric="layout_p95_wall_ms", arm=arm_for_p, value=None,
            threshold=thresholds["r2_p_layout_p95_wall_ms_max"], op="<=", verdict="missing", margin="n/a",
            note="no layout_p95_wall_ms available on the arm used for P",
        )
    else:
        p_clause = evaluate_simple_clause(
            "P", "layout_p95_wall_ms", arm_for_p, p_result, "<=", thresholds["r2_p_layout_p95_wall_ms_max"]
        )
        p_clause.margin = "n/a"
        p_clause.extra["pass_used"] = pass_for_p

    f_clause = _f_clause(candidate, metrics, margin_mib)

    clauses = [l_clause, s1, s2, d_clause, p_clause, f_clause]
    verdict = combine_gate_verdicts([c.verdict for c in clauses])
    return RuleResult(
        id="R2", kind="gate", label=rule_key,
        question=f"Should MIST switch from Gemma 4 E4B to {candidate}?",
        verdict=verdict, clauses=clauses,
        info={"candidate": candidate, "thinking_candidate": thinking_candidate},
    )


def evaluate_r3(metrics: RunMetrics, rules: dict[str, Any]) -> RuleResult:
    thresholds = rules["thresholds"]
    c3 = metrics.arms.get("c3")
    result = c3.truncation_rate.get("schema_conformance_json_object") if c3 is not None else None
    if result is None or not result.usable():
        clause = ClauseResult(
            id="truncation_json_object", metric="truncation_rate", arm="c3", value=(result.value if result else None),
            threshold=thresholds["r3_drop_moe_truncation_rate_min"], op=">", verdict="missing", margin="n/a",
            note="no usable truncation_rate for c3/schema_conformance_json_object",
        )
        verdict = "missing"
    else:
        triggered = result.value > thresholds["r3_drop_moe_truncation_rate_min"]
        margin = margin_for(result.value, result.bootstrap, thresholds["r3_drop_moe_truncation_rate_min"], ">")
        clause = ClauseResult(
            id="truncation_json_object", metric="truncation_rate", arm="c3", value=result.value,
            threshold=thresholds["r3_drop_moe_truncation_rate_min"], op=">",
            verdict="triggered" if triggered else "not-triggered", margin=margin,
        )
        verdict = clause.verdict

    info: dict[str, Any] = {}
    grammar = c3.truncation_rate.get("schema_conformance") if c3 is not None else None
    if grammar is not None and grammar.usable():
        info["truncation_grammar_schema_conformance"] = {
            "value": grammar.value, "n": grammar.n, "n_expected": grammar.n_expected, "complete": grammar.complete,
        }
    return RuleResult(
        id="R3", kind="trigger", label="drop_moe",
        question="Is c3's MoE output truncated often enough to drop the MoE candidates?",
        verdict=verdict, clauses=[clause], info=info,
    )


def evaluate_r4(metrics: RunMetrics, rules: dict[str, Any]) -> RuleResult:
    thresholds = rules["thresholds"]
    c2 = metrics.arms.get("c2")
    pass_used, result = (None, None)
    if c2 is not None:
        pass_used, result = pick_layout_pass(c2.layout_acc)
    if result is None or not result.usable():
        clause = ClauseResult(
            id="layout_acc_c2_off", metric="layout_acc", arm="c2", value=(result.value if result else None),
            threshold=thresholds["r4_thinking_lever_layout_acc_max"], op="<", verdict="missing", margin="n/a",
            note="no usable layout_acc for c2",
        )
        verdict = "missing"
    else:
        triggered = result.value < thresholds["r4_thinking_lever_layout_acc_max"]
        margin = margin_for(result.value, result.bootstrap, thresholds["r4_thinking_lever_layout_acc_max"], "<")
        clause = ClauseResult(
            id="layout_acc_c2_off", metric="layout_acc", arm="c2", value=result.value,
            threshold=thresholds["r4_thinking_lever_layout_acc_max"], op="<",
            verdict="triggered" if triggered else "not-triggered", margin=margin,
            extra={"pass_used": pass_used},
        )
        verdict = clause.verdict
    return RuleResult(
        id="R4", kind="trigger", label="thinking_is_lever",
        question="Is thinking mode a large enough lever on layout accuracy at c2 to be load-bearing?",
        verdict=verdict, clauses=[clause],
    )


def evaluate_r5(metrics: RunMetrics, rules: dict[str, Any], r2_results: list[RuleResult]) -> RuleResult:
    thresholds = rules["thresholds"]
    lower = metrics.voice_vram_lower_mib
    upper = metrics.voice_vram_upper_mib

    if not lower.usable() and not upper.usable():
        clause = ClauseResult(
            id="voice_vram_lower_bound", metric="voice_vram_lower_mib", arm=None, value=None,
            threshold=thresholds["r5_voice_vram_lower_trigger_mib"], op=">=", verdict="missing", margin="n/a",
            note="voice_vram_lower_mib and voice_vram_upper_mib both missing",
        )
        return RuleResult(
            id="R5", kind="trigger", label="gtx1070_moves_up",
            question="Does the voice VRAM footprint justify moving the GTX 1070 up in priority?",
            verdict="missing", clauses=[clause],
        )

    def _r2_all_pass_except_f(r: RuleResult) -> bool:
        by_id = {c.id: c for c in r.clauses}
        f = by_id.get("F")
        others_pass = all(c.verdict == "pass" for c in r.clauses if c.id != "F")
        f_bad = f is not None and f.verdict in ("fail", "needs-review")
        return others_pass and f_bad

    def _r2_has_missing_clause(r: RuleResult) -> bool:
        return any(c.verdict == "missing" for c in r.clauses)

    matching = [r for r in r2_results if _r2_all_pass_except_f(r)]
    any_r2_inconclusive = any(_r2_has_missing_clause(r) for r in r2_results)

    lower_trigger = lower.usable() and lower.value >= thresholds["r5_voice_vram_lower_trigger_mib"]
    upper_trigger = upper.usable() and upper.value >= thresholds["r5_voice_vram_upper_needs_review_mib"]

    note: str | None = None
    if lower_trigger and matching:
        verdict = "triggered"
    elif lower_trigger and not matching and any_r2_inconclusive:
        # The lower bound clears the trigger threshold, but not one R2 candidate could be
        # confirmed as "every other clause pass, F bad" because at least one R2 rule has a
        # clause reading "missing" -- that candidate might have matched had its data been
        # complete, so this cannot be reported as a clean not-triggered.
        verdict = "missing"
        note = "lower bound triggers, but at least one R2 rule has an incomplete (missing) clause"
    elif (not lower_trigger) and upper_trigger:
        verdict = "needs-review"
    else:
        verdict = "not-triggered"

    clause = ClauseResult(
        id="voice_vram_lower_bound", metric="voice_vram_lower_mib", arm=None,
        value=lower.value if lower.usable() else None,
        threshold=thresholds["r5_voice_vram_lower_trigger_mib"], op=">=", verdict=verdict, margin="n/a",
        note=note,
        extra={
            "voice_vram_upper_mib": upper.value if upper.usable() else None,
            "matching_r2_candidates": [r.info.get("candidate") for r in matching],
        },
    )
    return RuleResult(
        id="R5", kind="trigger", label="gtx1070_moves_up",
        question="Does the voice VRAM footprint justify moving the GTX 1070 up in priority?",
        verdict=verdict, clauses=[clause],
    )


def _correctness_file_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce one correctness.r<K>.jsonl's raw rows to what a determinism clause needs.

    Unlike the old `_tokens_by_prompt`, an errored row is not silently dropped: it is
    recorded via `has_error` so the caller can refuse to compare rather than comparing a
    smaller, error-free subset that happens to still agree (an errored prompt must never
    make a determinism clause read as `pass`).
    """
    tokens_by_prompt: dict[str, list[int]] = {}
    prompt_ids: set[str] = set()
    has_error = False
    for r in rows:
        pid = r.get("prompt_id")
        if pid is not None:
            prompt_ids.add(pid)
        if r.get("error"):
            has_error = True
            continue
        tokens_by_prompt[pid] = list(r.get("tokens") or [])
    return {
        "tokens_by_prompt": tokens_by_prompt,
        "has_error": has_error,
        "prompt_ids": prompt_ids,
        "n": len(rows),
    }


def _determinism_clause(
    clause_id: str,
    rows_a: list[dict[str, Any]] | None,
    rows_b: list[dict[str, Any]] | None,
    note_prefix: str,
    expected_prompts: int,
) -> ClauseResult:
    """Compare two correctness runs' token ids per prompt.

    `missing` (not `pass`, not `fail`) whenever either file has an errored row, a prompt
    count other than `expected_prompts`, or a prompt-id set that differs from the other
    file's -- an errored or incomplete file cannot license either a pass or a fail
    verdict, and it must never coincidentally read as `pass` just because the errored
    rows were excluded from the comparison.
    """
    if rows_a is None or rows_b is None or not rows_a or not rows_b:
        return ClauseResult(
            id=clause_id, metric="correctness_tokens", arm=None, value=None, threshold=None, op=None,
            verdict="missing", margin="n/a", note=f"{note_prefix}: missing correctness data",
        )
    sa = _correctness_file_summary(rows_a)
    sb = _correctness_file_summary(rows_b)
    if sa["has_error"] or sb["has_error"]:
        return ClauseResult(
            id=clause_id, metric="correctness_tokens", arm=None, value=None, threshold=None, op=None,
            verdict="missing", margin="n/a", note=f"{note_prefix}: an errored row is present",
        )
    if sa["n"] != expected_prompts or sb["n"] != expected_prompts:
        return ClauseResult(
            id=clause_id, metric="correctness_tokens", arm=None, value=None, threshold=None, op=None,
            verdict="missing", margin="n/a",
            note=f"{note_prefix}: prompt count {sa['n']}/{sb['n']} != expected {expected_prompts}",
        )
    if sa["prompt_ids"] != sb["prompt_ids"]:
        return ClauseResult(
            id=clause_id, metric="correctness_tokens", arm=None, value=None, threshold=None, op=None,
            verdict="missing", margin="n/a", note=f"{note_prefix}: prompt_id sets differ",
        )
    a = sa["tokens_by_prompt"]
    b = sb["tokens_by_prompt"]
    mismatched = [pid for pid in a if a[pid] != b[pid]]
    if mismatched:
        return ClauseResult(
            id=clause_id, metric="correctness_tokens", arm=None, value=None, threshold=None, op=None,
            verdict="fail", margin="n/a", note=f"{note_prefix}: token mismatch on {sorted(mismatched)}",
        )
    return ClauseResult(
        id=clause_id, metric="correctness_tokens", arm=None, value=None, threshold=None, op=None,
        verdict="pass", margin="n/a", note=f"{note_prefix}: identical for all {len(a)} prompts",
    )


def evaluate_r6(arm: str, base: str, metrics: RunMetrics, rules: dict[str, Any]) -> RuleResult:
    expected_prompts = rules["constants"]["correctness_expected_prompts"]
    base_rows_r1 = metrics.raw_correctness.get(base, {}).get(1, [])
    base_rows_r2 = metrics.raw_correctness.get(base, {}).get(2, [])
    tuned_rows_r1 = metrics.raw_correctness.get(arm, {}).get(1, [])

    base_det = _determinism_clause(
        "base_determinism", base_rows_r1 or None, base_rows_r2 or None, f"base {base} r1 vs r2", expected_prompts
    )
    tuned_match = _determinism_clause(
        "tuned_matches_base", tuned_rows_r1 or None, base_rows_r1 or None,
        f"{arm} r1 vs base {base} r1", expected_prompts,
    )

    base_am = metrics.arms.get(base)
    tuned_am = metrics.arms.get(arm)
    base_score = base_am.harness_score.get("schema_conformance") if base_am is not None else None
    tuned_score = tuned_am.harness_score.get("schema_conformance") if tuned_am is not None else None
    if base_score is None or not base_score.usable() or tuned_score is None or not tuned_score.usable():
        ci_clause = ClauseResult(
            id="harness_ci_containment", metric="harness_score", arm=arm, value=(tuned_score.value if tuned_score else None),
            threshold=None, op=None, verdict="missing", margin="n/a",
            note="missing or incomplete schema_conformance harness_score for base or tuned arm",
        )
    elif base_score.bootstrap is None:
        ci_clause = ClauseResult(
            id="harness_ci_containment", metric="harness_score", arm=arm, value=tuned_score.value,
            threshold=None, op=None, verdict="missing", margin="n/a", note="base has no bootstrap CI (no clusters)",
        )
    else:
        lo, hi = base_score.bootstrap
        within = lo <= tuned_score.value <= hi
        ci_clause = ClauseResult(
            id="harness_ci_containment", metric="harness_score", arm=arm, value=tuned_score.value,
            threshold=None, op=None, verdict="pass" if within else "fail", margin="n/a",
            extra={"base_ci": [lo, hi], "base_arm": base},
        )

    manual_entry = metrics.manual.get(arm)
    if manual_entry is None:
        manual_clause = ClauseResult(
            id="manual_clean", metric="manual", arm=arm, value=None, threshold=0, op="==",
            verdict="missing", margin="n/a", note=f"session/manual.json has no entry for {arm!r}",
        )
    else:
        memtest = manual_entry.get("memtest_errors")
        whea = manual_entry.get("whea_events")
        ok = memtest == 0 and whea == 0
        manual_clause = ClauseResult(
            id="manual_clean", metric="manual", arm=arm, value=memtest, threshold=0, op="==",
            verdict="pass" if ok else "fail", margin="n/a",
            extra={"memtest_errors": memtest, "whea_events": whea},
        )

    clauses = [base_det, tuned_match, ci_clause, manual_clause]
    verdict = combine_gate_verdicts([c.verdict for c in clauses])
    return RuleResult(
        id="R6", kind="gate", label=f"tuning_gate_{arm}",
        question=f"Does GPU tuning arm {arm} (base {base}) preserve correctness and stability?",
        verdict=verdict, clauses=clauses, info={"arm": arm, "base": base},
    )


def evaluate_r7(metrics: RunMetrics, rules: dict[str, Any]) -> RuleResult:
    constants = rules["constants"]
    old_arm = constants["r7_build_pair"]["old"]
    new_arm = constants["r7_build_pair"]["new"]
    old_am = metrics.arms.get(old_arm)
    new_am = metrics.arms.get(new_arm)

    deltas: dict[str, Any] = {}

    def _acc_delta() -> None:
        if old_am is None or new_am is None:
            deltas["layout_acc"] = {"missing": True}
            return
        _, old_result = pick_layout_pass(old_am.layout_acc)
        _, new_result = pick_layout_pass(new_am.layout_acc)
        if old_result is None or not old_result.usable() or new_result is None or not new_result.usable():
            deltas["layout_acc"] = {"missing": True}
            return
        deltas["layout_acc"] = {
            "old": old_result.value, "new": new_result.value, "delta": new_result.value - old_result.value,
        }

    def _harness_delta(test_name: str) -> None:
        key = f"harness_score_{test_name}"
        if old_am is None or new_am is None:
            deltas[key] = {"missing": True}
            return
        old_result = old_am.harness_score.get(test_name)
        new_result = new_am.harness_score.get(test_name)
        if old_result is None or not old_result.usable() or new_result is None or not new_result.usable():
            deltas[key] = {"missing": True}
            return
        deltas[key] = {
            "old": old_result.value, "new": new_result.value, "delta": new_result.value - old_result.value,
        }

    def _decode_delta() -> None:
        if old_am is None or new_am is None or not old_am.decode_tps.usable() or not new_am.decode_tps.usable():
            deltas["decode_tps"] = {"missing": True}
            return
        deltas["decode_tps"] = {
            "old": old_am.decode_tps.value, "new": new_am.decode_tps.value,
            "delta": new_am.decode_tps.value - old_am.decode_tps.value,
        }

    _acc_delta()
    for test_name in constants["r7_harness_tests"]:
        _harness_delta(test_name)
    _decode_delta()

    return RuleResult(
        id="R7", kind="informational", label="build_effect",
        question=f"What did the pinned build change ({new_arm} vs {old_arm})?",
        verdict="n/a", clauses=[], info={"old_arm": old_arm, "new_arm": new_arm, "deltas": deltas},
    )


def evaluate_all_rules(metrics: RunMetrics, rules: dict[str, Any]) -> list[RuleResult]:
    results: list[RuleResult] = []
    results.append(evaluate_r1(metrics, rules))

    r2_results: list[RuleResult] = []
    for rule_key, cfg in rules["constants"]["r2_candidates"].items():
        r2_results.append(evaluate_r2(rule_key, cfg["arm"], cfg["thinking_arm"], metrics, rules))
    results.extend(r2_results)

    results.append(evaluate_r3(metrics, rules))
    results.append(evaluate_r4(metrics, rules))
    results.append(evaluate_r5(metrics, rules, r2_results))

    for arm, cfg in rules["constants"]["r6_arms"].items():
        results.append(evaluate_r6(arm, cfg["base"], metrics, rules))

    results.append(evaluate_r7(metrics, rules))
    return results


# ---------------------------------------------------------------------------
# Finalist candidates
# ---------------------------------------------------------------------------


def compute_finalist_candidates(metrics: RunMetrics, rules: dict[str, Any]) -> list[str]:
    thresholds = rules["thresholds"]
    candidates: list[str] = []
    seen: set[str] = set()
    for cfg in rules["constants"]["r2_candidates"].values():
        for arm_id, threshold_key, op in (
            (cfg["arm"], "r2_l_off_layout_acc_min", ">="),
            (cfg["thinking_arm"], "r2_l_on_layout_acc_min", ">="),
        ):
            if arm_id in seen:
                continue
            am = metrics.arms.get(arm_id)
            if am is None:
                continue
            screen_result = am.layout_acc.get("screen")
            if screen_result is None or not screen_result.usable():
                continue
            margin = margin_for(screen_result.value, screen_result.bootstrap, thresholds[threshold_key], op)
            if margin == "within-noise":
                candidates.append(arm_id)
                seen.add(arm_id)
    return sorted(candidates)


# ---------------------------------------------------------------------------
# Coverage / missing-inputs summary
# ---------------------------------------------------------------------------


def compute_coverage(metrics: RunMetrics, rules: dict[str, Any]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for arm_id in metrics.arm_order:
        am = metrics.arms.get(arm_id)
        entry: dict[str, Any] = {"present": am.present if am is not None else False}
        if am is None or not am.present:
            coverage[arm_id] = entry
            continue
        entry["suites_completed"] = am.suites_completed
        # Decision 8 / finding 5: meta.json's `errors` are free-text (may embed absolute
        # host paths under --results-root or --layout-dir), so only a count crosses into
        # a public output; the texts stay in meta.json, which lives outside the repo.
        entry["error_count"] = len(am.errors)
        entry["layout"] = {
            p: {"n": r.n, "n_expected": r.n_expected, "complete": r.complete}
            for p, r in am.layout_acc.items()
        }
        entry["harness"] = {
            t: {"n": r.n, "n_expected": r.n_expected, "complete": r.complete}
            for t, r in am.harness_score.items()
        }
        coverage[arm_id] = entry
    return coverage


def compute_missing_inputs(metrics: RunMetrics, rule_results: list[RuleResult]) -> list[str]:
    missing: list[str] = []
    for r in rule_results:
        for c in r.clauses:
            if c.verdict == "missing":
                missing.append(f"{r.id}/{r.label}/{c.id}: {c.note or 'missing'}")
    return sorted(set(missing))


def compute_sha_warnings(metrics: RunMetrics, expected_sha: str) -> list[str]:
    warnings: list[str] = []
    for arm_id in metrics.arm_order:
        am = metrics.arms.get(arm_id)
        if am is None or not am.present or am.decision_rules_sha256 is None:
            continue
        if am.decision_rules_sha256 != expected_sha:
            warnings.append(
                f"arm {arm_id!r} meta.decision_rules_sha256={am.decision_rules_sha256!r} "
                f"differs from the analysed decision_rules.json={expected_sha!r}"
            )
    return warnings


# ---------------------------------------------------------------------------
# grades/ writers
# ---------------------------------------------------------------------------


def _finish_reason_lookup(results_dir: Path, arm_id: str, meta: dict[str, Any] | None) -> dict[tuple[str, str, int], str | None]:
    if meta is None or not meta.get("harness"):
        return {}
    candidate = meta["harness"]["candidate"]
    path = results_dir / arm_id / "harness" / "harness" / f"{candidate}.jsonl"
    out: dict[tuple[str, str, int], str | None] = {}
    for r in _read_jsonl(path):
        key = (r.get("test_name"), r.get("case_id"), int(r.get("iteration", 1)))
        out[key] = r.get("finish_reason")
    return out


def build_grades_harness_with_finish_reason(
    results_dir: Path, metrics: RunMetrics
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for arm_id, per_test in metrics.raw_test_scores.items():
        if not per_test:
            continue
        am = metrics.arms.get(arm_id)
        meta = None
        inputs_meta_path = results_dir / arm_id / "meta.json"
        if inputs_meta_path.exists():
            meta = json.loads(inputs_meta_path.read_text(encoding="utf-8"))
        lookup = _finish_reason_lookup(results_dir, arm_id, meta)
        rows: list[dict[str, Any]] = []
        for test_name in sorted(per_test):
            test_scores = per_test[test_name]
            for cs in sorted(test_scores.case_scores, key=lambda c: (c.case_id, c.iteration)):
                finish_reason = lookup.get((test_name, cs.case_id, cs.iteration))
                rows.append(
                    {
                        "test_name": test_name,
                        "case_id": cs.case_id,
                        "iteration": cs.iteration,
                        "score": cs.score,
                        "passed": cs.passed,
                        "finish_reason": finish_reason,
                        "errored": cs.error is not None,
                    }
                )
        out[arm_id] = rows
    return out


def build_grades_layout(metrics: RunMetrics) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for arm_id, by_pass in metrics.raw_layout_rows.items():
        for layout_pass, rows in by_pass.items():
            allowed_rows: list[dict[str, Any]] = []
            for row in rows:
                allowed = {k: row[k] for k in GRADES_LAYOUT_ALLOWLIST if k in row}
                usage = row.get("usage")
                if isinstance(usage, dict):
                    allowed["usage"] = {k: usage[k] for k in GRADES_USAGE_ALLOWLIST if k in usage}
                allowed_rows.append(allowed)
            out.setdefault(arm_id, {})[layout_pass] = allowed_rows
    return out


# ---------------------------------------------------------------------------
# REPORT.md rendering
# ---------------------------------------------------------------------------


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _fmt_ci(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def render_report(
    *,
    metrics: RunMetrics,
    rule_results: list[RuleResult],
    finalist_candidates: list[str],
    coverage: dict[str, Any],
    missing_inputs: list[str],
    sha_warnings: list[str],
    decision_rules_sha: str,
) -> str:
    lines: list[str] = []
    lines.append("# mist-model-bench analysis report")
    lines.append("")
    lines.append(f"decision_rules.json sha256: `{decision_rules_sha}`")
    lines.append("")
    lines.append("## Measurement notes")
    lines.append("")
    lines.append(
        "- Layout sampling is fixed by the command-center layout runner (`run_host.py`) at "
        "temperature 0.7 and top_p 0.9 for every arm, not at each vendor's recommended "
        "settings. This matches how run1 measured layout, so layout accuracy here is "
        "comparable to run1. Vendor sampling applies to the harness and the server defaults "
        "only (`arms.json`)."
    )
    lines.append("")
    if sha_warnings:
        lines.append("## decision_rules.json mismatch warnings")
        lines.append("")
        for w in sha_warnings:
            lines.append(f"- [WARN] {w}")
        lines.append("")

    lines.append("## Per-arm metrics")
    lines.append("")
    for arm_id in metrics.arm_order:
        am = metrics.arms.get(arm_id)
        lines.append(f"### {arm_id}")
        lines.append("")
        if am is None or not am.present:
            lines.append("no meta.json / arm absent")
            lines.append("")
            continue
        lines.append("| metric | value | n | n_expected | complete | wilson | bootstrap |")
        lines.append("|---|---|---|---|---|---|---|")
        for layout_pass, r in sorted(am.layout_acc.items()):
            lines.append(
                f"| layout_acc[{layout_pass}] | {_fmt(r.value)} | {_fmt(r.n)} | {_fmt(r.n_expected)} | "
                f"{r.complete} | {_fmt_ci(r.wilson)} | {_fmt_ci(r.bootstrap)} |"
            )
        for layout_pass, r in sorted(am.layout_mean_completion_tokens.items()):
            per_correct = am.layout_completion_tokens_per_correct.get(layout_pass)
            lines.append(
                f"| layout_mean_completion_tokens[{layout_pass}] | {_fmt(r.value)} | {_fmt(r.n)} | n/a | "
                f"n/a | n/a | n/a | (per_correct={_fmt(per_correct)}) |"
            )
        for layout_pass, r in sorted(am.layout_p95_wall_ms.items()):
            lines.append(
                f"| layout_p95_wall_ms[{layout_pass}] | {_fmt(r.value)} | {_fmt(r.n)} | n/a | n/a | n/a | n/a |"
            )
        for test_name, r in sorted(am.harness_score.items()):
            lines.append(
                f"| harness_score[{test_name}] | {_fmt(r.value)} | {_fmt(r.n)} | {_fmt(r.n_expected)} | "
                f"{r.complete} | {_fmt_ci(r.wilson)} | {_fmt_ci(r.bootstrap)} |"
            )
        for test_name, r in sorted(am.truncation_rate.items()):
            lines.append(
                f"| truncation_rate[{test_name}] | {_fmt(r.value)} | {_fmt(r.n)} | {_fmt(r.n_expected)} | "
                f"{r.complete} | {_fmt_ci(r.wilson)} | {_fmt_ci(r.bootstrap)} |"
            )
        lines.append(f"| decode_tps | {_fmt(am.decode_tps.value)} | {_fmt(am.decode_tps.n)} | n/a | n/a | n/a | n/a |")
        for ctx, r in sorted(am.ttft_ms.items()):
            lines.append(f"| ttft_ms[ctx={ctx}] | {_fmt(r.value)} | {_fmt(r.n)} | n/a | n/a | n/a | n/a |")
        lines.append(f"| arm_peak_mib | {_fmt(am.arm_peak_mib.value)} | {_fmt(am.arm_peak_mib.n)} | n/a | n/a | n/a | n/a |")
        lines.append("")

    lines.append("## Session (voice VRAM)")
    lines.append("")
    lines.append(f"- voice_vram_lower_mib: {_fmt(metrics.voice_vram_lower_mib.value)}")
    lines.append(f"- voice_vram_upper_mib: {_fmt(metrics.voice_vram_upper_mib.value)}")
    lines.append(f"- total_mib: {_fmt(metrics.total_mib)}")
    lines.append("")

    lines.append("## Rules")
    lines.append("")
    for r in rule_results:
        lines.append(f"### {r.id} `{r.label}` -- {r.kind}")
        lines.append("")
        lines.append(f"{r.question}")
        lines.append("")
        lines.append(f"verdict: **{r.verdict}**")
        lines.append("")
        if r.clauses:
            lines.append("| clause | metric | arm | value | threshold | op | verdict | margin | note |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for c in r.clauses:
                lines.append(
                    f"| {c.id} | {c.metric} | {_fmt(c.arm)} | {_fmt(c.value)} | {_fmt(c.threshold)} | "
                    f"{_fmt(c.op)} | {c.verdict} | {c.margin} | {_fmt(c.note)} |"
                )
            lines.append("")
        if r.info:
            lines.append(f"info: `{dumps_line(r.info)}`")
            lines.append("")

    lines.append("## Finalist candidates")
    lines.append("")
    if finalist_candidates:
        for arm_id in finalist_candidates:
            lines.append(f"- {arm_id}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Coverage per arm")
    lines.append("")
    lines.append(
        "Each arm's `error_count` is the number of entries in its `meta.json`'s `errors` "
        "list; the free-text of those errors is not reproduced here (it may embed host "
        "paths under `--results-root` or `--layout-dir`) and stays in `meta.json`, which "
        "lives outside this repository."
    )
    lines.append("")
    lines.append(f"```\n{dumps_stable(coverage).rstrip()}\n```")
    lines.append("")

    lines.append("## Missing inputs")
    lines.append("")
    if missing_inputs:
        for m in missing_inputs:
            lines.append(f"- {m}")
    else:
        lines.append("(none)")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# summary.json
# ---------------------------------------------------------------------------


def build_summary(
    *,
    metrics: RunMetrics,
    rule_results: list[RuleResult],
    finalist_candidates: list[str],
    coverage: dict[str, Any],
    missing_inputs: list[str],
    sha_warnings: list[str],
    decision_rules_sha: str,
) -> dict[str, Any]:
    arms_out: dict[str, Any] = {}
    for arm_id in metrics.arm_order:
        am = metrics.arms.get(arm_id)
        if am is None or not am.present:
            arms_out[arm_id] = {"present": False}
            continue
        arms_out[arm_id] = {
            "present": True,
            "layout_acc": {p: r.to_dict() for p, r in am.layout_acc.items()},
            "layout_mean_completion_tokens": {p: r.to_dict() for p, r in am.layout_mean_completion_tokens.items()},
            "layout_completion_tokens_per_correct": dict(am.layout_completion_tokens_per_correct),
            "layout_p95_wall_ms": {p: r.to_dict() for p, r in am.layout_p95_wall_ms.items()},
            "harness_score": {t: r.to_dict() for t, r in am.harness_score.items()},
            "truncation_rate": {t: r.to_dict() for t, r in am.truncation_rate.items()},
            "decode_tps": am.decode_tps.to_dict(),
            "ttft_ms": {str(ctx): r.to_dict() for ctx, r in am.ttft_ms.items()},
            "arm_peak_mib": am.arm_peak_mib.to_dict(),
            "suites_completed": am.suites_completed,
            # Free-text error strings stay in meta.json (outside the repo, under
            # --results-root); only the count crosses into this public output (finding 5).
            "error_count": len(am.errors),
        }
    return {
        "schema": 1,
        "decision_rules_sha256": decision_rules_sha,
        "decision_rules_sha256_warnings": sha_warnings,
        "arms": arms_out,
        "session": {
            "voice_vram_lower_mib": metrics.voice_vram_lower_mib.to_dict(),
            "voice_vram_upper_mib": metrics.voice_vram_upper_mib.to_dict(),
            "total_mib": metrics.total_mib,
        },
        "rules": [r.to_dict() for r in rule_results],
        "finalist_candidates": finalist_candidates,
        "coverage": coverage,
        "missing_inputs": missing_inputs,
    }


# ---------------------------------------------------------------------------
# Top-level analysis + output generation
# ---------------------------------------------------------------------------


@dataclass
class GeneratedOutputs:
    files: dict[str, str]  # relative path (posix, forward slashes) -> file content


def generate_outputs(results_dir: Path, rules_path: Path = DECISION_RULES_PATH) -> GeneratedOutputs:
    rules = load_decision_rules(rules_path)
    arms_doc = load_arms()
    problems = validate_decision_rules(rules, arms_doc)
    if problems:
        raise ValueError(f"decision_rules.json failed validation: {problems}")

    arm_ids = list(arms_doc)
    metrics = compute_all_metrics(results_dir, arm_ids, rules)
    rule_results = evaluate_all_rules(metrics, rules)
    finalist_candidates = compute_finalist_candidates(metrics, rules)
    coverage = compute_coverage(metrics, rules)
    missing_inputs = compute_missing_inputs(metrics, rule_results)
    sha = decision_rules_sha256(rules_path)
    sha_warnings = compute_sha_warnings(metrics, sha)

    report = render_report(
        metrics=metrics, rule_results=rule_results, finalist_candidates=finalist_candidates,
        coverage=coverage, missing_inputs=missing_inputs, sha_warnings=sha_warnings,
        decision_rules_sha=sha,
    )
    summary = build_summary(
        metrics=metrics, rule_results=rule_results, finalist_candidates=finalist_candidates,
        coverage=coverage, missing_inputs=missing_inputs, sha_warnings=sha_warnings,
        decision_rules_sha=sha,
    )

    files: dict[str, str] = {"REPORT.md": report, "summary.json": dumps_stable(summary)}

    grades_harness = build_grades_harness_with_finish_reason(results_dir, metrics)
    for arm_id, rows in grades_harness.items():
        content = "".join(dumps_line(row) + "\n" for row in rows)
        files[f"grades/{arm_id}/harness.jsonl"] = content

    grades_layout = build_grades_layout(metrics)
    for arm_id, by_pass in grades_layout.items():
        for layout_pass, rows in by_pass.items():
            content = "".join(dumps_line(row) + "\n" for row in rows)
            files[f"grades/{arm_id}/layout_{layout_pass}.jsonl"] = content

    return GeneratedOutputs(files=files)


def write_outputs(outputs: GeneratedOutputs, out_dir: Path) -> None:
    for rel_path, content in outputs.files.items():
        write_text_lf(out_dir / rel_path, content)


def diff_outputs(outputs: GeneratedOutputs, out_dir: Path) -> list[str]:
    """Byte-for-byte diff of generated outputs against files already on disk."""
    problems: list[str] = []
    for rel_path, content in outputs.files.items():
        disk_path = out_dir / rel_path
        if not disk_path.exists():
            problems.append(f"missing on disk: {rel_path}")
            continue
        disk_content = disk_path.read_text(encoding="utf-8")
        if disk_content != content:
            problems.append(f"content differs: {rel_path}")
    generated_rel = set(outputs.files)
    if out_dir.exists():
        for path in out_dir.rglob("*"):
            if path.is_file():
                rel = path.relative_to(out_dir).as_posix()
                if rel not in generated_rel:
                    problems.append(f"extra file on disk (not generated): {rel}")
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.model_bench.analyse")
    parser.add_argument("--results", type=Path, default=None, help="bench_host.py results directory for one run")
    parser.add_argument("--out", type=Path, default=None, help="output directory for REPORT.md/summary.json/grades")
    parser.add_argument("--check", action="store_true", help="regenerate in memory and diff against --out; exits non-zero on any difference")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check and args.results is None:
        rules = load_decision_rules()
        arms_doc = load_arms()
        problems = validate_decision_rules(rules, arms_doc)
        if problems:
            for p in problems:
                print(f"[FAIL] {p}")
            return 1
        print("decision_rules.json ok")
        return 0

    if args.results is None or args.out is None:
        print("[FAIL] --results and --out are required (unless --check is given with no --results)")
        return 1

    try:
        outputs = generate_outputs(args.results)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1

    if args.check:
        problems = diff_outputs(outputs, args.out)
        if problems:
            for p in problems:
                print(f"[FAIL] {p}")
            return 1
        print("analyse --check ok: byte-identical")
        return 0

    write_outputs(outputs, args.out)
    print(f"wrote outputs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
