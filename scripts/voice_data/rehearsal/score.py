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


BY_SEGMENTS = "segments"
UPPER_BOUND = "upper bound: highest non-admitted cluster"
MERGED_MEDIAN = (
    "merged: admitted clusters cover over half of the impostor's speech and this one the most; "
    "the impostor holds at most half its records"
)


def impostor_clusters(
    eps: Episodes, segments: Sequence[SegmentRecord], report: Mapping[str, object]
) -> list[dict[str, object]]:
    """Return each episode's impostor cluster: its status and median SIM to the anchor centroid.

    Extract writes segment records for admitted clusters only (primary or flagged), kept or
    dropped. An admitted cluster is the impostor's when more than half of its records in the
    episode are the impostor's (`Episodes.who`). Impostor records in an admitted cluster that is
    not the impostor's are counted per cluster in `stray_records`. When several admitted clusters
    are the impostor's, the row names the worst: primary before flagged, then the higher median.

    When no admitted cluster is the impostor's, the row measures how much of the impostor's
    speech in the episode (the union of its ground-truth lines) the records of admitted clusters
    cover: every record of those clusters, whichever speaker `Episodes.who` gives it to, or none.
    A record of a diarizer turn that joins a target line and an impostor line can lie on neither
    speaker's lines by half, so `Episodes.who` gives it to no one, yet the impostor speech in it
    went into that turn's cluster. A record split between both speakers adds to the coverage the
    part of it on the impostor's lines; it counts toward the impostor's majority and
    `stray_records` only when `Episodes.who` gives it to the impostor. The denominator is all of
    the impostor's speech, not only the speech extraction could keep, because extract drafts a
    record for each turn of an admitted cluster (`stages/extract.py:782-797`) whether or not it is
    kept: a turn too short to embed (:534-538, :848-849), a turn that fails retrieval (:864-880), a
    cut of the wrong length (:912-915, from `segments.cut_line`) and a cut that touches overlap
    (:917-920) are each drafted, and `_records` writes every draft (:967-1016). A passing turn's
    records leave out what `cut_line` trims (`stages/segments.py:87-103`): the turn's edges outside
    VAD speech, the pauses between segments when a long turn is cut, and the whole turn when VAD
    finds no speech in it.

    - When the admitted clusters together cover more than half of the impostor's speech, the
      impostor is merged into the admitted cluster covering the most of it (primary before
      flagged on a tie). `status` is that cluster's role, so the statuses stay primary, flagged
      and not admitted, and `impostor_primary_episodes` counts a merge into primary.
      `diar_cluster` names the cluster and `median_sim` is its median, of a cluster in which the
      impostor holds at most half the records; `identified_by` is `MERGED_MEDIAN`. Impostor
      records in it stay counted in `stray_records`.
    - Otherwise the row reports the highest median among the episode's non-admitted clusters,
      an upper bound on the impostor's own cluster's median, with the status `not admitted`;
      which non-admitted cluster was the impostor's is unknown, as non-admitted clusters leave
      no records (`stages/extract.py:783-787`).
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
        spans: dict[str, list[Interval]] = {}
        for seg in segments:
            if seg.source_id == sid and seg.diar_cluster:
                records[seg.diar_cluster] += 1
                spans.setdefault(seg.diar_cluster, []).append((seg.start_s, seg.end_s))
                if eps.who(seg) == "impostor":
                    impostor[seg.diar_cluster] += 1
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
        speech = union([(ln.start_s, ln.end_s) for ln in eps.of(sid, "impostor")])
        admitted = [
            (label, str(clusters[label].get("role")))
            for label in sorted(records)
            if clusters.get(label, {}).get("role") in (PRIMARY, FLAGGED)
        ]
        merged = _merged_into(speech, admitted, spans)
        if own:
            worst = max(own, key=lambda c: (c.get("role") == PRIMARY, _mean(c.get("median_sim"))))
            row.update(
                diar_cluster=worst.get("diar_cluster"),
                status=worst.get("role"),
                median_sim=worst.get("median_sim"),
                identified_by=BY_SEGMENTS,
            )
        elif merged is not None:
            label, role = merged
            row.update(
                diar_cluster=label,
                status=role,
                median_sim=clusters[label].get("median_sim"),
                identified_by=MERGED_MEDIAN,
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


def _merged_into(
    speech: Sequence[Interval],
    admitted: Sequence[tuple[str, str]],
    spans: Mapping[str, Sequence[Interval]],
) -> tuple[str, str] | None:
    """Return the label and role of the admitted cluster the impostor is merged into, or `None`.

    `speech` is the impostor's speech in the episode, `admitted` the (label, role) of each
    admitted cluster with records in it, and `spans` the spans of every record by label,
    whichever speaker the record is given to. The impostor is merged when the clusters' records
    together cover more than half of `speech`; the cluster named is the one covering the most,
    primary before flagged on a tie.
    """
    every = [span for label, _ in admitted for span in spans[label]]
    if not admitted or 2 * length(intersect(speech, union(every))) <= length(speech):
        return None
    covered = {label: length(intersect(speech, union(spans[label]))) for label, _ in admitted}
    return max(admitted, key=lambda entry: (covered[entry[0]], entry[1] == PRIMARY))


def impostor_cluster_lines(rows: Sequence[Mapping[str, object]]) -> list[str]:
    """Return each episode's impostor cluster row as printed; a merged row shows `MERGED_MEDIAN`."""
    lines = []
    for row in rows:
        by_role: Counter[str] = Counter()
        for entry in row.get("stray_records") or []:  # type: ignore[union-attr]
            by_role[str(entry["role"])] += int(entry["records"])
        median = _table(row["median_sim"])
        if row["identified_by"] == MERGED_MEDIAN:
            found = f"{row['status']} ({MERGED_MEDIAN}), median {median}"
        else:
            found = f"{row['status']}, median {median} ({row['identified_by']})"
        lines.append(
            f"  {row['episode']} ({row['source_id']}) impostor cluster: {found}; stray impostor "
            f"records in admitted clusters: {_table(dict(sorted(by_role.items())))}"
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
    the impostor's median. Merged rows (`MERGED_MEDIAN`) go to neither: they are counted by role
    in `impostor_merged_episodes` instead.
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


def retrieval_summary(clusters: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return the impostor's cluster statistics over the episodes, for retrieval's values.

    The anchor floor, the anchor check and the margin are set per embedder by H20 on public
    speakers, not by H6, so these are no gate entry; they show where the impostor's cluster
    fell at the values the run used.
    """
    merged = Counter(
        str(c.get("status")) for c in clusters if c.get("identified_by") == MERGED_MEDIAN
    )
    return {
        "impostor_median_max": _median_max(clusters, BY_SEGMENTS),
        "non_admitted_median_upper_bound_max": _median_max(clusters, UPPER_BOUND),
        "impostor_primary_episodes": sum(c.get("status") == PRIMARY for c in clusters),
        "impostor_merged_episodes": dict(sorted(merged.items())),
        "impostor_stray_records": sum(
            int(s["records"]) for c in clusters for s in c.get("stray_records") or []  # type: ignore[union-attr]
        ),
    }


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
