"""`rehearsal score`'s measures: an extraction run against the rehearsal's ground truth.

Host side and standard library only. Segments name their episode by source id; the inventory
gives each source's sha256, and the ground truth names each episode by its file's sha256, so the
episodes can sit anywhere under `seed/raw/`. A segment is the target's (or the impostor's) when
that speaker's lines cover at least half of it.

Precision is over the kept primary-cluster segments (the seed before Raj's labels): the share
that are the target's. Recall is over the target's speech that no other speaker overlaps (the
extraction drops overlapped speech by design): the share covered by kept segments, primary or
flagged, since flagged segments come back through the labels pack. The effect measures come from
the `effects` job step (`mix.py`); everything else here needs no audio.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from ..records import SegmentRecord, SourceRecord
from ..states import SPEAKING_STATES
from . import TruthLine, majority, overlap_s

Interval = tuple[float, float]
PRIMARY = "primary"
FLAGGED = "flagged"
DOWNMIX = "downmix"
NOT_ADMITTED = "not admitted"
NOT_RATED = "a rating by Raj; the rehearsal holds no ratings to inform it"
# Why each [E] gate the rehearsal cannot inform is "not measured".
NOT_MEASURED = {
    "retrieval.margin": "segments.jsonl records no per-turn margin to the rival clusters",
    "segment.window_percentile": (
        "a percentile of the target's own window distribution; the ground truth marks lines, "
        "not windows"
    ),
    "ingest.bandwidth_cutoff_db": (
        "the rehearsal's encodes are full band, so they hold no band limit to calibrate against"
    ),
    "bakeoff.weak_rating_below": NOT_RATED,
    "text.reject_max": NOT_RATED,
    "spotcheck.reject_max": NOT_RATED,
}


def percentile(values: Sequence[float], q: float) -> float:
    """Return the `q`th percentile with linear interpolation, as numpy's default does."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("a percentile needs at least one value")
    rank = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def union(intervals: Sequence[Interval]) -> list[Interval]:
    """Return the intervals merged into disjoint ones, in order."""
    merged: list[Interval] = []
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def length(intervals: Sequence[Interval]) -> float:
    """Return the total length of disjoint intervals."""
    return sum(end - start for start, end in intervals)


def intersect(a: Sequence[Interval], b: Sequence[Interval]) -> list[Interval]:
    """Return the intersection of two sets of disjoint intervals."""
    found = []
    for a0, a1 in a:
        for b0, b1 in b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                found.append((lo, hi))
    return union(found)


def subtract(a: Sequence[Interval], b: Sequence[Interval]) -> list[Interval]:
    """Return the parts of `a` that `b` does not cover."""
    return union(_gaps(a, union(b)))


def _gaps(a: Sequence[Interval], b: Sequence[Interval]) -> list[Interval]:
    out = []
    for start, end in a:
        cursor = start
        for b0, b1 in b:
            if b1 <= cursor or b0 >= end:
                continue
            if b0 > cursor:
                out.append((cursor, b0))
            cursor = max(cursor, b1)
        if cursor < end:
            out.append((cursor, end))
    return out


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den > 0 else None


class Episodes:
    """The ground truth by source id, through the inventory's sha256s."""

    def __init__(self, truth: Sequence[TruthLine], inventory: Sequence[SourceRecord]) -> None:
        by_sha: dict[str, list[TruthLine]] = {}
        for line in truth:
            by_sha.setdefault(line.episode_sha256, []).append(line)
        self.lines: dict[str, list[TruthLine]] = {}
        self.episode: dict[str, str] = {}
        self.unmatched = 0
        for record in inventory:
            lines = by_sha.get(record.sha256)
            if lines is None:
                self.unmatched += 1
                continue
            self.lines[record.source_id] = lines
            self.episode[record.source_id] = lines[0].episode

    def of(self, source_id: str, speaker: str) -> list[TruthLine]:
        """Return the source's lines of one speaker kind."""
        return [ln for ln in self.lines.get(source_id, []) if ln.speaker == speaker]

    def who(self, seg: SegmentRecord) -> str | None:
        """Return `target`, `impostor` or `other` when that kind covers half the segment."""
        span = seg.end_s - seg.start_s
        if seg.source_id not in self.lines or span <= 0:
            return None
        for speaker in ("target", "impostor", "other"):
            if overlap_s(seg.start_s, seg.end_s, self.of(seg.source_id, speaker)) >= 0.5 * span:
                return speaker
        return None

    def target_line(self, seg: SegmentRecord) -> TruthLine | None:
        """Return the target line covering most of the segment, when it covers half of it."""
        return majority(seg.start_s, seg.end_s, self.of(seg.source_id, "target"))


def extraction(eps: Episodes, segments: Sequence[SegmentRecord]) -> dict[str, object]:
    """Return precision, recall and the per-cluster counts of the kept segments."""
    own = [s for s in segments if s.source_id in eps.lines]
    primary = [s for s in own if s.kept and s.cluster == PRIMARY]
    kept = [s for s in own if s.kept]
    true_primary = [s for s in primary if eps.who(s) == "target"]
    true_kept = [s for s in kept if eps.who(s) == "target"]
    target_s = covered_s = 0.0
    for sid, lines in eps.lines.items():
        targets = union([(ln.start_s, ln.end_s) for ln in lines if ln.speaker == "target"])
        others = union([(ln.start_s, ln.end_s) for ln in lines if ln.speaker != "target"])
        eligible = subtract(targets, others)
        spans = union([(s.start_s, s.end_s) for s in kept if s.source_id == sid])
        target_s += length(eligible)
        covered_s += length(intersect(eligible, spans))
    primary_s = sum(s.end_s - s.start_s for s in primary)
    true_s = sum(overlap_s(s.start_s, s.end_s, eps.of(s.source_id, "target")) for s in primary)
    return {
        "precision": _ratio(len(true_primary), len(primary)),
        "precision_time": _ratio(true_s, primary_s),
        "precision_with_flagged": _ratio(len(true_kept), len(kept)),
        "recall": _ratio(covered_s, target_s),
        "kept_primary": len(primary),
        "kept_flagged": len(kept) - len(primary),
        "true_primary": len(true_primary),
        "target_s": round(target_s, 3),
        "covered_s": round(covered_s, 3),
    }


def state_accuracy(eps: Episodes, segments: Sequence[SegmentRecord]) -> dict[str, object]:
    """Return how many kept target segments propose their line's true state, overall and per state."""
    per_state = {s: {"n": 0, "correct": 0} for s in SPEAKING_STATES}
    for seg in segments:
        line = eps.target_line(seg) if seg.kept else None
        if line is None or line.state not in per_state:
            continue
        per_state[line.state]["n"] += 1
        per_state[line.state]["correct"] += int(seg.state_proposed == line.state)
    n = sum(v["n"] for v in per_state.values())
    correct = sum(v["correct"] for v in per_state.values())
    return {"accuracy": _ratio(correct, n), "n": n, "per_state": per_state}


def impostor_turns(
    eps: Episodes, segments: Sequence[SegmentRecord], labels: Mapping[str, str] | None
) -> dict[str, object]:
    """Return the impostor's kept segments as primary (must be 0) and flagged, and in the labels."""
    imp = [s for s in segments if s.kept and eps.who(s) == "impostor"]
    lines = sum(len(eps.of(sid, "impostor")) for sid in eps.lines)
    return {
        "lines": lines,
        "kept_primary": sum(s.cluster == PRIMARY for s in imp),
        "kept_flagged": sum(s.cluster == FLAGGED for s in imp),
        "in_labels": None if labels is None else sum(s.segment_id in labels for s in imp),
    }


def downmix_recovery(
    eps: Episodes, segments: Sequence[SegmentRecord], layouts: Mapping[str, str]
) -> dict[str, object]:
    """Return, per geometry, the 5.1 episodes' off-centre and fallback lines kept from the downmix.

    A line is recovered when a kept downmix segment (primary or flagged) lies mostly on it; the
    recovering segments are counted by their `channel_reason`.
    """
    found: dict[str, dict[str, object]] = {}
    for geometry in ("off-centre", "fallback"):
        recovered: set[tuple[str, int]] = set()
        reasons: Counter[str] = Counter()
        total = 0
        for sid in eps.lines:
            if layouts.get(sid) != "5.1":
                continue
            total += sum(ln.geometry == geometry for ln in eps.of(sid, "target"))
            for seg in segments:
                if seg.source_id != sid or not seg.kept or seg.channel != DOWNMIX:
                    continue
                line = eps.target_line(seg)
                if line is not None and line.geometry == geometry:
                    recovered.add((sid, line.line))
                    reasons[str(seg.channel_reason)] += 1
        found[geometry] = {
            "lines": total,
            "recovered": len(recovered),
            "by_reason": dict(sorted(reasons.items())),
        }
    return found


BY_SEGMENTS = "segments"
UPPER_BOUND = "upper bound: highest non-admitted cluster"


def impostor_clusters(
    eps: Episodes, segments: Sequence[SegmentRecord], report: Mapping[str, object]
) -> list[dict[str, object]]:
    """Return each episode's impostor cluster: its status and median SIM to the anchor centroid.

    Extract writes segment records for admitted clusters only (primary or flagged), kept or
    dropped. An admitted cluster is the impostor's when more than half of its records in the
    episode are the impostor's (`Episodes.who`); a few impostor turns the diarizer put in another
    admitted cluster, the target's primary one most likely, do not make that cluster the
    impostor's, and are counted per cluster in `stray_records`. When several admitted clusters
    are the impostor's, the row names the worst: primary before flagged, then the higher median.

    When no admitted cluster is the impostor's, its cluster was not admitted and left no records,
    so which cluster it was is unknown: the row reports the highest median among the episode's
    non-admitted clusters, an upper bound on the impostor's (`identified_by` says so).
    """
    clusters_by_source: dict[str, list[Mapping[str, object]]] = {}
    for source in report.get("sources") or []:  # type: ignore[union-attr]
        if isinstance(source, Mapping) and isinstance(source.get("clusters"), list):
            clusters_by_source[str(source.get("source_id"))] = source["clusters"]  # type: ignore[assignment]
    rows = []
    for sid in sorted(eps.lines):
        clusters = {str(c.get("diar_cluster")): c for c in clusters_by_source.get(sid, [])}
        records: Counter[str] = Counter()
        impostor: Counter[str] = Counter()
        for seg in segments:
            if seg.source_id == sid and seg.diar_cluster:
                records[seg.diar_cluster] += 1
                impostor[seg.diar_cluster] += eps.who(seg) == "impostor"
        own, stray = [], []
        for label in sorted(impostor):
            role = clusters.get(label, {}).get("role")
            if role not in (PRIMARY, FLAGGED) or not impostor[label]:
                continue
            if 2 * impostor[label] > records[label]:
                own.append(clusters[label])
            else:
                stray.append({"diar_cluster": label, "role": role, "records": impostor[label]})
        row: dict[str, object] = {"episode": eps.episode[sid], "source_id": sid}
        if own:
            worst = max(own, key=lambda c: (c.get("role") == PRIMARY, _mean(c.get("median_sim"))))
            row.update(
                diar_cluster=worst.get("diar_cluster"),
                status=worst.get("role"),
                median_sim=worst.get("median_sim"),
                identified_by=BY_SEGMENTS,
            )
        else:
            rest = [c for c in clusters.values() if c.get("role") not in (PRIMARY, FLAGGED)]
            best = max(rest, key=lambda c: _mean(c.get("median_sim")), default=None)
            row.update(
                diar_cluster=None if best is None else best.get("diar_cluster"),
                status=NOT_ADMITTED,
                median_sim=None if best is None else best.get("median_sim"),
                identified_by="none" if best is None else UPPER_BOUND,
            )
        row["stray_records"] = stray
        rows.append(row)
    return rows


def impostor_cluster_lines(rows: Sequence[Mapping[str, object]]) -> list[str]:
    """Return each episode's impostor cluster row as printed."""
    lines = []
    for row in rows:
        by_role: Counter[str] = Counter()
        for entry in row.get("stray_records") or []:  # type: ignore[union-attr]
            by_role[str(entry["role"])] += int(entry["records"])
        lines.append(
            f"  {row['episode']} ({row['source_id']}) impostor cluster: {row['status']}, median "
            f"{_table(row['median_sim'])} ({row['identified_by']}); stray impostor records in "
            f"admitted clusters: {_table(dict(sorted(by_role.items())))}"
        )
    return lines


def _mean(table: object) -> float:
    if not isinstance(table, Mapping) or not table:
        return -math.inf
    values = [float(v) for v in table.values() if isinstance(v, int | float)]
    return sum(values) / len(values) if values else -math.inf


def _median_max(rows: Sequence[Mapping[str, object]], identified_by: str) -> float | None:
    """Return the highest mean median SIM of the rows identified one way, or `None`.

    `impostor_median_max` takes the rows whose cluster the impostor's records identified; the
    upper-bound rows go to their own entry, so a non-admitted episode's bound is never read as
    the impostor's median.
    """
    return max(
        (
            round(_mean(row.get("median_sim")), 3)
            for row in rows
            if row.get("identified_by") == identified_by and row.get("median_sim")
        ),
        default=None,
    )


def _sims(segs: Sequence[SegmentRecord]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    for seg in segs:
        for name, value in seg.sim.items():
            if isinstance(value, int | float) and math.isfinite(value):
                found.setdefault(name, []).append(float(value))
    return found


def _p02_floor(values: Sequence[float], below: Sequence[float]) -> dict[str, object]:
    entry: dict[str, object] = {"value": round(percentile(values, 2), 3), "n": len(values)}
    entry["fallback_fc_n"] = len(below)
    if below:
        entry["fallback_fc_max"] = round(max(below), 3)
        entry["separates"] = max(below) < entry["value"]  # type: ignore[operator]
    return entry


def fallback_fc(build: object, eps: Episodes) -> dict[str, list[float]]:
    """Return the centre-cut SNR and C50 the build measured on the scored episodes' fallback lines.

    Extract writes no centre-cut record for a fallback line whose downmix passes, so these come
    from `build.json` (`mix.measure_fallback`), for the episodes the inventory matched only.
    """
    scored = {lines[0].episode_sha256 for lines in eps.lines.values() if lines}
    found: dict[str, list[float]] = {"snr_db": [], "c50_db": []}
    episodes = build.get("episodes") if isinstance(build, Mapping) else None
    for ep in episodes if isinstance(episodes, list) else []:
        if not isinstance(ep, Mapping) or ep.get("sha256") not in scored:
            continue
        for row in ep.get("fallback_fc") or []:
            for field, values in found.items():
                value = row.get(field) if isinstance(row, Mapping) else None
                if isinstance(value, int | float) and math.isfinite(value):
                    values.append(float(value))
    return found


def gate_values(
    eps: Episodes,
    segments: Sequence[SegmentRecord],
    clusters: Sequence[Mapping[str, object]],
    effects: Mapping[str, object] | None,
    h6_gates: Mapping[str, float],
    fallback: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, object]]:
    """Return, for each H6-calibrated gate, the value the ground truth supports, or why not.

    Each entry holds `default` (the gate's value in force), and either `value` with its `basis`
    or `value` null with `reason`. `fallback` is `fallback_fc`'s result: the SNR and C50 floors
    are compared against it, since the calibrated floors must still fail the fallback lines'
    centre cut for the downmix path to run.
    """
    true_kept = [s for s in segments if s.kept and eps.who(s) == "target"]
    true_fc = [s for s in true_kept if s.channel == "FC"]
    sims = _sims(true_kept)
    out: dict[str, dict[str, object]] = {}
    for key, default in sorted(h6_gates.items()):
        entry: dict[str, object] = {"default": default, "value": None}
        if key in NOT_MEASURED:
            entry["reason"] = NOT_MEASURED[key]
        elif key == "anchors.sim_min":
            if sims:
                entry["value"] = round(min(percentile(v, 2) for v in sims.values()), 3)
                entry["basis"] = (
                    "p02 of the true target segments' SIM to the target centroid, lower embedder"
                )
            else:
                entry["reason"] = "no kept segment is the target's"
        elif key == "retrieval.anchor_floor":
            entry["reason"] = (
                "not calibrated: H6 keeps the anchor floor at its recorded value and lowers it "
                "only to exercise the flagged path"
            )
            entry["impostor_median_max"] = _median_max(clusters, BY_SEGMENTS)
            entry["non_admitted_median_upper_bound_max"] = _median_max(clusters, UPPER_BOUND)
            entry["impostor_primary_episodes"] = sum(c.get("status") == PRIMARY for c in clusters)
            entry["impostor_stray_records"] = sum(
                int(s["records"]) for c in clusters for s in c.get("stray_records") or []  # type: ignore[union-attr]
            )
        elif key in ("segment.snr_min_db", "segment.c50_min_db"):
            field = "snr_db" if key == "segment.snr_min_db" else "c50_db"
            values = [getattr(s, field) for s in true_fc if math.isfinite(getattr(s, field))]
            if values:
                entry.update(_p02_floor(values, fallback.get(field, [])))
                entry["basis"] = (
                    f"p02 of the true target centre-channel segments' {field}; "
                    "fallback_fc_max is the highest centre-cut value the build measured on a "
                    "fallback line (speech band against burst band, and C50 deconvolved from "
                    "the known line: the build's estimators, not extract's), which must fall "
                    "below it"
                )
            else:
                entry["reason"] = "no kept centre-channel segment is the target's"
        elif key == "cleaning.spectrum_max_db":
            eq = (effects or {}).get("eq")
            if isinstance(eq, Mapping) and isinstance(eq.get("eq_own_distance_db"), int | float):
                entry["value"] = round(float(eq["eq_own_distance_db"]) / 2.0, 3)
                entry["basis"] = (
                    "half the EQ's own spectrum distance, so a step that strips the EQ moves "
                    "the spectrum past the gate"
                )
            else:
                entry["reason"] = "no clean run was scored, so the EQ was not measured"
        elif key.startswith("state_floor.offset."):
            state = key.rsplit(".", 1)[1]
            in_state = [s for s in true_kept if (ln := eps.target_line(s)) and ln.state == state]
            own = _sims(in_state)
            if own and sims:
                gap = max(percentile(sims[n], 10) - percentile(v, 10) for n, v in own.items())
                entry["value"] = round(max(0.0, gap), 3)
                entry["n"] = len(in_state)
                entry["basis"] = (
                    "pooled p10 less the state's p10 of the true target segments' SIM, larger "
                    "embedder gap, at least 0"
                )
            else:
                entry["reason"] = f"no kept target segment of {state}"
        elif key == "state_floor.rehearsal_identity_window_floor":
            windows = (effects or {}).get("identity_windows")
            p10 = windows.get("p10") if isinstance(windows, Mapping) else None
            if isinstance(p10, Mapping) and p10:
                entry["value"] = round(min(float(v) for v in p10.values()), 3)
                entry["n"] = windows.get("n")  # type: ignore[union-attr]
                entry["basis"] = (
                    "p10 of the stand-in target's 3 s identity windows against the other "
                    "clips' windows, lower embedder"
                )
            else:
                entry["reason"] = "the effects step found too few identity windows"
        else:
            entry["reason"] = "the rehearsal has no measure for this gate"
        out[key] = entry
    return out


def _num(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return "?"


def _table(table: object) -> str:
    if not isinstance(table, Mapping) or not table:
        return "none"
    return ", ".join(f"{k} {_num(v)}" for k, v in table.items())


def score_lines(score: Mapping[str, object]) -> list[str]:
    """Return the score as printed: ids, gate and state names, and numbers only."""
    ex = score["extraction"]
    lines = [
        f"rehearsal score of run {score['run']} (clean run {score.get('clean_run') or 'none'}), "
        f"{_num(score['episodes'])} episodes:",
        f"  precision {_num(ex['precision'])} ({_num(ex['true_primary'])} of "  # type: ignore[index]
        f"{_num(ex['kept_primary'])} primary segments; time {_num(ex['precision_time'])}; "  # type: ignore[index]
        f"with flagged {_num(ex['precision_with_flagged'])})",  # type: ignore[index]
        f"  recall {_num(ex['recall'])} ({_num(ex['covered_s'])} of {_num(ex['target_s'])} s)",  # type: ignore[index]
    ]
    acc = score["state_accuracy"]
    lines.append(f"  state accuracy {_num(acc['accuracy'])} over {_num(acc['n'])} segments")  # type: ignore[index]
    for state, row in acc["per_state"].items():  # type: ignore[index]
        lines.append(f"    {state}: {_num(row['correct'])} of {_num(row['n'])}")
    imp = score["impostor"]
    lines.append(
        f"  impostor turns kept: primary {_num(imp['kept_primary'])} (must be 0), flagged "  # type: ignore[index]
        f"{_num(imp['kept_flagged'])}, in labels {_num(imp['in_labels'])}, of "  # type: ignore[index]
        f"{_num(imp['lines'])} lines"  # type: ignore[index]
    )
    for geometry, row in score["downmix"].items():  # type: ignore[union-attr]
        lines.append(
            f"  {geometry} lines recovered from the downmix: {_num(row['recovered'])} of "
            f"{_num(row['lines'])}; by reason: {_table(row['by_reason'])}"
        )
    lines += impostor_cluster_lines(score["impostor_clusters"])  # type: ignore[arg-type]
    effects = score.get("effects") or {}
    comb = effects.get("comb") if isinstance(effects, Mapping) else None
    if isinstance(comb, Mapping):
        lines.append(
            f"  comb filter survived: {_num(comb.get('survived'))}; SIM to processed "
            f"{_table(comb.get('sim_processed'))}, to unprocessed "
            f"{_table(comb.get('sim_unprocessed'))}"
        )
    else:
        lines.append("  comb filter: not measured (no clean run)")
    eq = effects.get("eq") if isinstance(effects, Mapping) else None
    if isinstance(eq, Mapping):
        lines.append(
            f"  EQ survived: {_num(eq.get('survived'))}; spectrum distance to processed "
            f"{_num(eq.get('distance_processed_db'))} dB, to unprocessed "
            f"{_num(eq.get('distance_unprocessed_db'))} dB"
        )
    else:
        lines.append("  EQ: not measured (no clean run)")
    lines.append("  [E] gates, the value the ground truth supports:")
    for key, entry in score["gates"].items():  # type: ignore[union-attr]
        if entry.get("value") is None:
            lines.append(
                f"    {key}: not measured ({entry.get('reason')}); in force {_num(entry['default'])}"
            )
        else:
            lines.append(f"    {key}: {_num(entry['value'])}; in force {_num(entry['default'])}")
    return lines
