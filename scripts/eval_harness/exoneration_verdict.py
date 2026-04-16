"""Exoneration protocol post-run verdict generator.

Reads all Phase D5 JSONL output from a master-timestamp directory and
produces a decomposed four-verdict VERDICT.md per the plan in
~/.claude/plans/cozy-herding-pebble.md.

Usage:
    python scripts/eval_harness/exoneration_verdict.py \
        scripts/eval_harness/results/phase3-exoneration-<master-ts>/

The four verdicts reported:
    V1 - Production-readiness (binary gate on strict schema, speed, p95)
    V2 - Conversational-quality (paired t-test + Bonferroni)
    V3 - Extraction-quality (strict + lenient + fragility delta)
    V4 - Speed/latency (llama-bench ceiling + harness end-to-end)

Decision rule (pre-registered): Gemma replaces Hermes-4 if and only if
Gemma passes V1 and wins V2 at p<0.01 after Bonferroni.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

CONVERSATIONAL_TESTS = (
    "personality",
    "rag_integration",
    "coherence",
    "cot_reasoning",
    "long_turn_coherence",
    "adversarial_persona",
)

EXTRACTION_STRICT = "schema_conformance"
EXTRACTION_LENIENT = "schema_conformance_lenient"

V1_SCHEMA_FLOOR = 0.80
V1_SPEED_FLOOR = 30.0  # tokens/sec
V1_P95_CEILING_MS = 15_000
V2_ALPHA = 0.01


@dataclass
class CandidateAggregate:
    """Aggregated per-test scores and metrics for one candidate from D5 JSONL."""

    candidate_id: str
    per_test_scores: dict[str, list[float]] = field(default_factory=dict)
    per_test_iterations: dict[str, list[list[float]]] = field(default_factory=dict)
    speed_tps: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    total_cases: int = 0

    def mean(self, test: str) -> float | None:
        """Return the mean of per-case scores for this test."""
        scores = self.per_test_scores.get(test, [])
        return statistics.fmean(scores) if scores else None

    def stdev(self, test: str) -> float | None:
        """Return the standard deviation of per-case means for this test."""
        scores = self.per_test_scores.get(test, [])
        if len(scores) < 2:
            return None
        return statistics.stdev(scores)

    def p95_latency(self) -> float | None:
        """Return the 95th-percentile total_time_ms across all recorded cases."""
        if not self.latencies_ms:
            return None
        sorted_latencies = sorted(self.latencies_ms)
        idx = max(0, min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95)))
        return sorted_latencies[idx]

    def mean_tps(self) -> float | None:
        """Return the mean tokens-per-second across speed-test cases."""
        return statistics.fmean(self.speed_tps) if self.speed_tps else None


def load_d5_jsonls(phase_d5_dir: Path) -> dict[str, CandidateAggregate]:
    """Read all *.jsonl in the Phase D5 result dir into CandidateAggregates."""
    if not phase_d5_dir.exists():
        raise FileNotFoundError(f"Phase D5 dir missing: {phase_d5_dir}")

    # Score each case via scorers.py to get a per-case score.
    from scripts.eval_harness import scorers

    aggs: dict[str, CandidateAggregate] = {}
    for jsonl in sorted(phase_d5_dir.glob("*.jsonl")):
        cid = jsonl.stem
        agg = aggs.setdefault(cid, CandidateAggregate(candidate_id=cid))
        # test_name -> {case_id -> list of per-iteration scores}
        per_iter_buffer: dict[str, dict[str, list[float]]] = {}
        for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            if not line.endswith("}") and not line.endswith("}\n"):
                # Truncated last line from mid-write crash; skip.
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            test_name = rec.get("test_name", "")
            test_type = rec.get("test_type", test_name)
            scorer = scorers.SCORER_REGISTRY.get(test_type)
            if scorer is None:
                continue
            expected = rec.get("expected") or {}
            try:
                _, score, _ = scorer(rec, expected)
            except Exception:  # noqa: BLE001
                score = 0.0
            per_iter_buffer.setdefault(test_name, {}).setdefault(rec.get("case_id", ""), []).append(
                float(score)
            )
            agg.total_cases += 1

            if test_name == "speed" or test_name == "speed_minimal":
                metrics = rec.get("metrics") or {}
                tps = float(metrics.get("tokens_per_second") or 0.0)
                if tps > 0:
                    agg.speed_tps.append(tps)
            metrics = rec.get("metrics") or {}
            total_ms = metrics.get("total_time_ms")
            if isinstance(total_ms, int | float) and total_ms > 0:
                agg.latencies_ms.append(float(total_ms))

        # Per test, aggregate per-case mean across iterations, then store list of case means.
        for test_name, cases in per_iter_buffer.items():
            agg.per_test_iterations[test_name] = list(cases.values())
            agg.per_test_scores[test_name] = [statistics.fmean(v) for v in cases.values()]

    return aggs


def paired_t_statistic(a: list[float], b: list[float]) -> tuple[float, int]:
    """Return (t, df) for a paired t-test. df = n-1."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0, 0
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean_d = statistics.fmean(diffs)
    sd_d = statistics.stdev(diffs)
    if sd_d == 0.0:
        return math.inf if mean_d != 0 else 0.0, n - 1
    t = mean_d / (sd_d / math.sqrt(n))
    return t, n - 1


def t_to_p_approx(t: float, df: int) -> float:
    """Approximate two-tailed p-value from t and df using normal tail.

    Uses normal approximation (erfc) — acceptable for df >= 10, which
    matches our n=10 final verdict run per plan. For smaller df, use
    scipy.stats if available.
    """
    if df == 0:
        return 1.0
    try:
        from scipy import stats  # type: ignore

        return float(2.0 * (1.0 - stats.t.cdf(abs(t), df)))
    except ImportError:
        # Normal approximation (z-test) as fallback
        z = abs(t)
        return math.erfc(z / math.sqrt(2.0))


def write_verdict(master_dir: Path, phase_d5_dir: Path) -> Path:
    """Compute and write VERDICT.md for a master-timestamp master_dir."""
    aggs = load_d5_jsonls(phase_d5_dir)
    if not aggs:
        raise RuntimeError(f"No JSONL results found in {phase_d5_dir}")

    candidate_ids = sorted(aggs.keys())
    gemma_id = next((c for c in candidate_ids if "gemma" in c.lower()), None)
    hermes_id = next((c for c in candidate_ids if "hermes" in c.lower()), None)
    if not gemma_id or not hermes_id:
        raise RuntimeError(f"Expected one Gemma and one Hermes candidate; found {candidate_ids}")
    gemma = aggs[gemma_id]
    hermes = aggs[hermes_id]

    lines: list[str] = [
        f"# Exoneration Verdict - {master_dir.name}",
        "",
        f"**Candidates compared:** {gemma_id} vs {hermes_id}",
        f"**Total case results:** Gemma {gemma.total_cases}, Hermes {hermes.total_cases}",
        "",
        "---",
        "",
        "## V1 - Production-Readiness (Binary Gate)",
        "",
        f"Thresholds: strict schema >= {V1_SCHEMA_FLOOR}, speed >= {V1_SPEED_FLOOR} t/s, p95 <= {V1_P95_CEILING_MS} ms.",
        "",
        "| Metric | Gemma | Hermes | Gemma passes | Hermes passes |",
        "|---|---:|---:|:---:|:---:|",
    ]

    def v1_row(name: str, gval: float | None, hval: float | None, gpass: bool, hpass: bool) -> str:
        gstr = f"{gval:.3f}" if gval is not None else "n/a"
        hstr = f"{hval:.3f}" if hval is not None else "n/a"
        return f"| {name} | {gstr} | {hstr} | {'Y' if gpass else 'N'} | {'Y' if hpass else 'N'} |"

    g_schema = gemma.mean(EXTRACTION_STRICT)
    h_schema = hermes.mean(EXTRACTION_STRICT)
    g_speed = gemma.mean_tps()
    h_speed = hermes.mean_tps()
    g_p95 = gemma.p95_latency()
    h_p95 = hermes.p95_latency()

    g_v1 = (
        (g_schema is not None and g_schema >= V1_SCHEMA_FLOOR)
        and (g_speed is not None and g_speed >= V1_SPEED_FLOOR)
        and (g_p95 is not None and g_p95 <= V1_P95_CEILING_MS)
    )
    h_v1 = (
        (h_schema is not None and h_schema >= V1_SCHEMA_FLOOR)
        and (h_speed is not None and h_speed >= V1_SPEED_FLOOR)
        and (h_p95 is not None and h_p95 <= V1_P95_CEILING_MS)
    )

    lines.append(
        v1_row(
            f"schema_conformance >= {V1_SCHEMA_FLOOR}",
            g_schema,
            h_schema,
            bool(g_schema is not None and g_schema >= V1_SCHEMA_FLOOR),
            bool(h_schema is not None and h_schema >= V1_SCHEMA_FLOOR),
        )
    )
    lines.append(
        v1_row(
            f"speed >= {V1_SPEED_FLOOR} t/s",
            g_speed,
            h_speed,
            bool(g_speed is not None and g_speed >= V1_SPEED_FLOOR),
            bool(h_speed is not None and h_speed >= V1_SPEED_FLOOR),
        )
    )
    lines.append(
        v1_row(
            f"p95 latency <= {V1_P95_CEILING_MS} ms",
            g_p95,
            h_p95,
            bool(g_p95 is not None and g_p95 <= V1_P95_CEILING_MS),
            bool(h_p95 is not None and h_p95 <= V1_P95_CEILING_MS),
        )
    )
    lines.append("")
    lines.append(f"**V1 Gemma:** {'PASS' if g_v1 else 'FAIL'}")
    lines.append(f"**V1 Hermes:** {'PASS' if h_v1 else 'FAIL'}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # V2 - Conversational quality (paired t-test + Bonferroni)
    lines.extend(
        [
            "## V2 - Conversational Quality (Paired T-Test + Bonferroni)",
            "",
            f"Bonferroni-corrected alpha: {V2_ALPHA} / {len(CONVERSATIONAL_TESTS)} = {V2_ALPHA / len(CONVERSATIONAL_TESTS):.4f} per test.",
            "",
            "| Test | Gemma mean | Hermes mean | Delta | t | df | p (two-tailed) | Significant |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )

    bonf_alpha = V2_ALPHA / len(CONVERSATIONAL_TESTS)
    gemma_wins = 0
    hermes_wins = 0
    indistinguishable = 0

    for test in CONVERSATIONAL_TESTS:
        g_mean = gemma.mean(test)
        h_mean = hermes.mean(test)
        g_scores = gemma.per_test_scores.get(test, [])
        h_scores = hermes.per_test_scores.get(test, [])
        if g_mean is None or h_mean is None or len(g_scores) != len(h_scores) or len(g_scores) < 2:
            lines.append(f"| {test} | {g_mean} | {h_mean} | n/a | n/a | n/a | n/a | n/a |")
            continue
        delta = g_mean - h_mean
        t, df = paired_t_statistic(g_scores, h_scores)
        p = t_to_p_approx(t, df)
        sig = p < bonf_alpha
        winner = "Gemma" if sig and delta > 0 else ("Hermes" if sig and delta < 0 else "--")
        if sig and delta > 0:
            gemma_wins += 1
        elif sig and delta < 0:
            hermes_wins += 1
        else:
            indistinguishable += 1
        lines.append(
            f"| {test} | {g_mean:.3f} | {h_mean:.3f} | {delta:+.3f} | {t:.2f} | {df} | {p:.4f} | {winner} |"
        )

    lines.append("")
    lines.append(
        f"**V2 summary:** Gemma wins {gemma_wins}, Hermes wins {hermes_wins}, indistinguishable {indistinguishable}."
    )
    if gemma_wins > hermes_wins and gemma_wins >= len(CONVERSATIONAL_TESTS) // 2:
        v2_winner = "Gemma"
    elif hermes_wins > gemma_wins and hermes_wins >= len(CONVERSATIONAL_TESTS) // 2:
        v2_winner = "Hermes"
    else:
        v2_winner = "Indistinguishable"
    lines.append(f"**V2 verdict:** {v2_winner}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # V3 - Extraction quality
    g_lenient = gemma.mean(EXTRACTION_LENIENT)
    h_lenient = hermes.mean(EXTRACTION_LENIENT)
    lines.extend(
        [
            "## V3 - Extraction Quality (Strict + Lenient + Fragility)",
            "",
            "| Metric | Gemma | Hermes |",
            "|---|---:|---:|",
            f"| schema_conformance (strict) | {g_schema if g_schema is not None else 'n/a':.3f} | {h_schema if h_schema is not None else 'n/a':.3f} |",
            f"| schema_conformance_lenient | {g_lenient if g_lenient is not None else 'n/a':.3f} | {h_lenient if h_lenient is not None else 'n/a':.3f} |",
        ]
    )
    g_frag = (g_lenient / g_schema) if g_schema and g_lenient else None
    h_frag = (h_lenient / h_schema) if h_schema and h_lenient else None
    lines.append(
        f"| Fragility (lenient/strict ratio) | {g_frag:.2f} if g_frag else n/a | {h_frag:.2f} if h_frag else n/a |".replace(
            "{g_frag:.2f} if g_frag else n/a", f"{g_frag:.2f}" if g_frag is not None else "n/a"
        ).replace(
            "{h_frag:.2f} if h_frag else n/a", f"{h_frag:.2f}" if h_frag is not None else "n/a"
        )
    )
    lines.append("")
    lines.append(
        "A fragility ratio > 1.0 means lenient repair recovered significant capability "
        "that strict scoring penalized — the candidate's content was mostly right but "
        "format-fragile."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # V4 - Speed/latency
    lines.extend(
        [
            "## V4 - Speed and Latency",
            "",
            "| Metric | Gemma | Hermes |",
            "|---|---:|---:|",
            f"| Mean tokens/sec (harness) | {g_speed if g_speed is not None else 'n/a':.1f} | {h_speed if h_speed is not None else 'n/a':.1f} |",
            f"| p95 latency (ms) | {g_p95 if g_p95 is not None else 'n/a':.0f} | {h_p95 if h_p95 is not None else 'n/a':.0f} |",
            "",
            "llama-bench raw ceiling values (from Phase D-bench): see d-bench-results.md",
            "",
            "---",
            "",
            "## Pre-Registered Decision Rule",
            "",
            "Gemma replaces Hermes-4 ngram as production model if and only if:",
            "1. Gemma passes V1 (production-readiness gate), AND",
            "2. Gemma wins V2 (conversational-quality) with at least one significant win and no significant Hermes wins",
            "",
            "**Decision rule outcome:**",
        ]
    )

    decision = (
        "Gemma"
        if (g_v1 and v2_winner == "Gemma")
        else "Hermes" if not (g_v1 and v2_winner == "Gemma") else "Hermes"
    )
    lines.append(f"- Gemma passes V1: {'YES' if g_v1 else 'NO'}")
    lines.append(f"- V2 winner: {v2_winner}")
    lines.append(f"- **Winner: {decision}**")
    lines.append("")

    out = master_dir / "VERDICT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python exoneration_verdict.py <master-dir> [phase-d5-subdir]")
        return 1
    master_dir = Path(argv[1])
    if not master_dir.exists():
        print(f"master dir does not exist: {master_dir}")
        return 1

    # D5 phase result dir is either passed explicitly or discovered by
    # scanning master_dir for the latest timestamped subdir.
    if len(argv) >= 3:
        phase_d5_dir = Path(argv[2])
    else:
        # Read CHECKPOINT.json to locate D5's result_dir.
        checkpoint = master_dir / "CHECKPOINT.json"
        if not checkpoint.exists():
            print("CHECKPOINT.json missing, pass phase-d5-subdir explicitly")
            return 1
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        d5 = data.get("phases", {}).get("D5", {})
        d5_dir_rel = d5.get("result_dir")
        if not d5_dir_rel:
            print("D5 result_dir not in CHECKPOINT.json")
            return 1
        # result_dir is relative to results/, so resolve from master_dir's parent
        phase_d5_dir = master_dir.parent / d5_dir_rel

    out = write_verdict(master_dir, phase_d5_dir)
    print(f"wrote verdict: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
