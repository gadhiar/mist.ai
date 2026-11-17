"""`rehearsal score`'s measures: an extraction run against the rehearsal's ground truth.

Host side and standard library only. Segments name their episode by source id; the inventory
gives each source's sha256, and the ground truth names each episode by its file's sha256, so the
episodes can sit anywhere under `seed/raw/`. A segment is the target's (or the impostor's) when
that speaker's lines cover at least half of it.

Precision is over the kept primary-cluster segments (the seed before Raj's labels): the share
that are the target's. Recall, H6's figure, is over the speech extraction can keep by rule: the
target's speech that no other speaker overlaps, in spans of at least `segment.min_s` (extract
drops overlapped speech and shorter segments by design). It is the share covered by kept
segments, primary or flagged, since flagged segments come back through the labels pack.
All-target recall, over all the target's speech, is reported beside it. The effect measures come
from the `effects` job step (`mix.py`); everything else here needs no audio.

The impostor's cluster in each episode is named from the run's turn record (`turns.jsonl`,
`commands/extract.py`'s `TurnRecord`), which holds every diarized turn whatever its cluster's
role, by the impostor seconds each cluster's turns hold.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from ..commands.extract import TurnRecord
from ..records import SegmentRecord, SourceRecord
from ..states import SPEAKING_STATES
from . import TruthLine, majority, overlap_s

Interval = tuple[float, float]
PRIMARY = "primary"
FLAGGED = "flagged"
OTHER = "other"
NONE = "none"
DOWNMIX = "downmix"
# How an episode's impostor cluster was found: by the turn record, or not at all.
BY_TURNS = "turns"
STATUSES = (PRIMARY, FLAGGED, OTHER, NONE)
# When two clusters hold the same impostor seconds, the one retrieval admitted further is named.
_CONCERN = {PRIMARY: 2, FLAGGED: 1, OTHER: 0}
# The six per-embedder retrieval values a run used, as `retrieval.thresholds` reports them.
RETRIEVAL_PREFIXES = ("anchors.sim_min.", "retrieval.anchor_floor.", "retrieval.margin.")
MARGIN_PREFIX = "retrieval.margin."
# The flag floor sits this far under the impostor cluster's median before rounding down.
FLAG_STEP = 0.001
# A gap equal to a margin up to rounding meets it, as in `stages/retrieve.py`.
EPS = 1e-9
NOT_RATED = "a rating by Raj; the rehearsal holds no ratings to inform it"
# Why each [E] gate the rehearsal cannot inform is "not measured".
NOT_MEASURED = {
    "segment.window_percentile": (
        "a percentile of the target's own window distribution; the ground truth marks lines, "
        "not windows"
    ),
    "ingest.bandwidth_cutoff_db": (
        "the rehearsal's encodes are band-limited only by their codecs (the T12 review measured "
        "E-AC-3 at 640 kb/s to about 20.5 kHz and AAC to about 18 kHz), and the build sets and "
        "records no known band limit, so the ground truth holds no cutoff to calibrate against"
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

    def who(self, seg: SegmentRecord | TurnRecord) -> str | None:
        """Return `target`, `impostor` or `other` when that kind covers half the segment or turn."""
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


def keepable(lines: Sequence[TruthLine], min_s: float) -> list[Interval]:
    """Return the target speech extraction can keep by rule: unoverlapped, `min_s` or longer.

    Extract drops any segment that touches overlap (`segment.overlap_max_ms` is 0) and any
    shorter than `segment.min_s`, so the target's speech less every other speaker's lines, in
    spans of at least `min_s`, is all it can keep. Splits at VAD pauses and the diarizer's own
    turn edges are not in the ground truth, so they are not taken out.
    """
    targets = union([(ln.start_s, ln.end_s) for ln in lines if ln.speaker == "target"])
    others = union([(ln.start_s, ln.end_s) for ln in lines if ln.speaker != "target"])
    return [(a, b) for a, b in subtract(targets, others) if b - a >= min_s]


def extraction(eps: Episodes, segments: Sequence[SegmentRecord], min_s: float) -> dict[str, object]:
    """Return precision, recall and the per-cluster counts of the kept segments.

    Two recalls, each the share of a span of target speech that kept segments (primary or
    flagged) cover. `recall`, H6's figure, is over the speech extraction can keep (`keepable`,
    with `min_s` the `segment.min_s` gate in force): it does not charge extraction with the
    overlapped speech and short lines it drops by design. `recall_all_target` is over all the
    target's speech, beside it.
    """
    own = [s for s in segments if s.source_id in eps.lines]
    primary = [s for s in own if s.kept and s.cluster == PRIMARY]
    kept = [s for s in own if s.kept]
    true_primary = [s for s in primary if eps.who(s) == "target"]
    true_kept = [s for s in kept if eps.who(s) == "target"]
    keep_s = keep_covered_s = target_s = target_covered_s = 0.0
    for sid, lines in eps.lines.items():
        spans = union([(s.start_s, s.end_s) for s in kept if s.source_id == sid])
        can_keep = keepable(lines, min_s)
        keep_s += length(can_keep)
        keep_covered_s += length(intersect(can_keep, spans))
        targets = union([(ln.start_s, ln.end_s) for ln in lines if ln.speaker == "target"])
        target_s += length(targets)
        target_covered_s += length(intersect(targets, spans))
    primary_s = sum(s.end_s - s.start_s for s in primary)
    true_s = sum(overlap_s(s.start_s, s.end_s, eps.of(s.source_id, "target")) for s in primary)
    return {
        "precision": _ratio(len(true_primary), len(primary)),
        "precision_time": _ratio(true_s, primary_s),
        "precision_with_flagged": _ratio(len(true_kept), len(kept)),
        "recall": _ratio(keep_covered_s, keep_s),
        "recall_all_target": _ratio(target_covered_s, target_s),
        "kept_primary": len(primary),
        "kept_flagged": len(kept) - len(primary),
        "true_primary": len(true_primary),
        "min_s": min_s,
        "keepable_s": round(keep_s, 3),
        "keepable_covered_s": round(keep_covered_s, 3),
        "target_s": round(target_s, 3),
        "target_covered_s": round(target_covered_s, 3),
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


def _medians(report: Mapping[str, object]) -> dict[tuple[str, str], object]:
    """Return each cluster's median SIM to the anchor centroid, by source id and label."""
    found: dict[tuple[str, str], object] = {}
    sources = report.get("sources")
    for source in sources if isinstance(sources, list) else []:
        clusters = source.get("clusters") if isinstance(source, Mapping) else None
        for cluster in clusters if isinstance(clusters, list) else []:
            if isinstance(cluster, Mapping):
                key = (str(source.get("source_id")), str(cluster.get("diar_cluster")))
                found[key] = cluster.get("median_sim")
    return found


def impostor_clusters(
    eps: Episodes, turns: Sequence[TurnRecord], report: Mapping[str, object]
) -> list[dict[str, object]]:
    """Return each episode's impostor cluster, named from the turn record by measured seconds.

    Each turn's seconds on the impostor's ground-truth lines are attributed to the impostor and
    summed per diarizer cluster. The episode's impostor cluster is the one holding the most
    impostor seconds; on a tie, the one retrieval admitted further (primary, then flagged, then
    other), then the first label. Its `status` is its role in the turn record and `median_sim`
    its median SIM to the anchor centroid from `report.json`, so the row reads the impostor's
    own cluster whatever its role, never another cluster's median standing in for it. Every
    other cluster holding impostor seconds is listed in `stray`. `impostor_share` is the
    impostor's share of the named cluster's seconds, which says how far its median is the
    impostor's. An episode in which no turn meets an impostor line has `status` and
    `identified_by` `none`.
    """
    medians = _medians(report)
    rows = []
    for sid in sorted(eps.lines):
        impostor_lines = eps.of(sid, "impostor")
        seconds: dict[str, float] = {}
        held: dict[str, float] = {}
        roles: dict[str, str] = {}
        for turn in turns:
            if turn.source_id != sid:
                continue
            label = turn.diar_cluster
            seconds[label] = seconds.get(label, 0.0) + (turn.end_s - turn.start_s)
            held[label] = held.get(label, 0.0) + overlap_s(turn.start_s, turn.end_s, impostor_lines)
            roles[label] = turn.role
        holding = [label for label in sorted(held) if held[label] > EPS]
        total = round(length(union([(ln.start_s, ln.end_s) for ln in impostor_lines])), 3)
        row: dict[str, object] = {"episode": eps.episode[sid], "source_id": sid}
        if not holding:
            row.update(
                diar_cluster=None, status=NONE, median_sim=None, impostor_s=0.0, cluster_s=None,
                impostor_share=None, impostor_total_s=total, identified_by=NONE, stray=[],
            )  # fmt: skip
            rows.append(row)
            continue
        label = max(holding, key=lambda lb: (round(held[lb], 6), _CONCERN.get(roles[lb], 0)))
        row.update(
            diar_cluster=label,
            status=roles[label],
            median_sim=medians.get((sid, label)),
            impostor_s=round(held[label], 3),
            cluster_s=round(seconds[label], 3),
            impostor_share=_ratio(held[label], seconds[label]),
            impostor_total_s=total,
            identified_by=BY_TURNS,
            stray=[
                {"diar_cluster": lb, "role": roles[lb], "impostor_s": round(held[lb], 3)}
                for lb in holding
                if lb != label
            ],
        )
        rows.append(row)
    return rows


def impostor_cluster_lines(rows: Sequence[Mapping[str, object]]) -> list[str]:
    """Return each episode's impostor cluster as printed: its seconds, share and median."""
    lines = []
    for row in rows:
        where = f"  {row['episode']} ({row['source_id']}) impostor cluster:"
        total = f"impostor {_num(row['impostor_total_s'])} s in all"
        if row["identified_by"] == NONE:
            lines.append(f"{where} none, no turn meets an impostor line; {total}")
            continue
        stray = ", ".join(
            f"{s['diar_cluster']} {s['role']} {_num(s['impostor_s'])} s"
            for s in row.get("stray") or []  # type: ignore[union-attr]
        )
        lines.append(
            f"{where} {row['diar_cluster']} {row['status']}, {_num(row['impostor_s'])} of "
            f"{_num(row['cluster_s'])} s impostor (share {_num(row['impostor_share'])}), median "
            f"{_table(row['median_sim'])}; {total}; stray: {stray or 'none'}"
        )
    return lines


def flag_floor(rows: Sequence[Mapping[str, object]]) -> dict[str, float] | None:
    """Return, per embedder, the anchor floor just under every named impostor cluster's median.

    For each episode whose impostor cluster the turn record named, its median less `FLAG_STEP`,
    rounded down to 3 places; the floor is the minimum over those episodes. An anchor floor at
    this value admits each of those clusters, so a run at it flags the impostor's cluster
    rather than passing over it as another speaker. `None` when no episode's impostor cluster
    was named.
    """
    floors: dict[str, float] = {}
    for row in rows:
        median = row.get("median_sim")
        if row.get("identified_by") != BY_TURNS or not isinstance(median, Mapping):
            continue
        for name, value in median.items():
            # Rounding before the floor keeps float error (0.66 - 0.001 is 0.65899...) out of it.
            floor = math.floor(round((float(value) - FLAG_STEP) * 1000.0, 6)) / 1000.0
            floors[name] = min(floors.get(name, floor), floor)
    return floors or None


def _reaches(turn: TurnRecord, name: str, margin: float) -> bool:
    # A judged turn with no rival had no margin applied, as in `retrieve.judge`.
    if turn.gaps is None:
        return True
    gap = turn.gaps.get(name)
    return gap is not None and gap >= margin - EPS


def margin_share(
    eps: Episodes, turns: Sequence[TurnRecord], speaker: str, margins: Mapping[str, float]
) -> dict[str, float | None]:
    """Return the share of one speaker's primary-cluster turns whose gap reaches the margin.

    The turns counted are the embedded, judged turns of primary clusters that `speaker`'s lines
    cover by half or more (`Episodes.who`). The share is given per embedder against that
    embedder's margin, and as `both`, reaching every embedder's; `None` when no turn counts.
    """
    judged = [
        t
        for t in turns
        if t.role == PRIMARY
        and t.embedded
        and t.sim_target is not None
        and t.source_id in eps.lines
        and eps.who(t) == speaker
    ]
    shares: dict[str, float | None] = {
        name: _ratio(sum(_reaches(t, name, value) for t in judged), len(judged))
        for name, value in margins.items()
    }
    shares["both"] = _ratio(
        sum(all(_reaches(t, n, v) for n, v in margins.items()) for t in judged), len(judged)
    )
    return shares


def retrieval_summary(
    clusters: Sequence[Mapping[str, object]],
    eps: Episodes,
    turns: Sequence[TurnRecord],
    report: Mapping[str, object],
) -> dict[str, object]:
    """Return where the impostor's cluster fell at the retrieval values the run used.

    The anchor floor, the anchor check and the margin are set per embedder by H20 on public
    speakers, not by H6, so these are no gate entry. `thresholds` holds the run's six values
    from its `report.json`, so a reader can check which values the rest was measured at;
    `flag_floor` is what H6's floor-lowering step reads.
    """
    found = report.get("thresholds")
    thresholds = {
        str(k): float(v)
        for k, v in sorted(found.items() if isinstance(found, Mapping) else [])
        if str(k).startswith(RETRIEVAL_PREFIXES) and isinstance(v, int | float)
    }
    margins = {
        k[len(MARGIN_PREFIX) :]: v for k, v in thresholds.items() if k.startswith(MARGIN_PREFIX)
    }
    statuses = Counter(str(c.get("status")) for c in clusters)
    return {
        "thresholds": thresholds,
        "impostor_episodes": {status: statuses.get(status, 0) for status in STATUSES},
        "flag_floor": flag_floor(clusters),
        "target_margin_share": margin_share(eps, turns, "target", margins),
        "impostor_margin_share": margin_share(eps, turns, "impostor", margins),
    }


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
        f"  recall {_num(ex['recall'])} over the speech extraction can keep "  # type: ignore[index]
        f"({_num(ex['keepable_covered_s'])} of {_num(ex['keepable_s'])} s: unoverlapped, "  # type: ignore[index]
        f"spans of {_num(ex['min_s'])} s or more); all-target recall "  # type: ignore[index]
        f"{_num(ex['recall_all_target'])} ({_num(ex['target_covered_s'])} of "  # type: ignore[index]
        f"{_num(ex['target_s'])} s)",  # type: ignore[index]
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
    retrieval = score.get("retrieval")
    if isinstance(retrieval, Mapping):
        floor = retrieval.get("flag_floor")
        lines.append(
            f"  impostor clusters by status: {_table(retrieval.get('impostor_episodes'))}; flag "
            f"floor: {_table(floor) if floor else 'none (no impostor cluster named)'}"
        )
        lines.append(
            f"  primary-cluster turns reaching the margin: target "
            f"{_table(retrieval.get('target_margin_share'))}; impostor "
            f"{_table(retrieval.get('impostor_margin_share'))}"
        )
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
