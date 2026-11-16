"""`ingest [--language L]`: probe the sources, copy each one's track, write the inventory.

Sources are the audio and video files anywhere under `seed/raw/`, numbered in sorted path order;
a path already in the inventory keeps its id, so a rerun with a file added renumbers nothing.
Each file's size, modification time and sha256 are taken before the jobs and checked after them,
so a source changed while ingest ran is a failure (exit 4), never a silent new inventory.

Two local jobs run the stage in the base image: `probe` lists every stream of each file, this
command chooses the original-language track, and `copy` copies it and measures the copy. Seed
stages stay on this host: there is no executor option. Output names sources by id only; the
inventory is the one record that holds a file name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .. import gates as gates_mod
from .. import jobs
from .. import root as root_mod
from ..errors import InputError, JobError
from ..records import (
    JobInput,
    SourceRecord,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from ..stages import ingest as stage

FAMILY = "score"
MODULE = "scripts.voice_data.stages.ingest"
INVENTORY_REL = "seed/inventory.jsonl"
# This module sits at `<checkout>/scripts/voice_data/commands/ingest.py`.
CHECKOUT = Path(__file__).resolve().parents[3]
MEDIA_SUFFIXES = frozenset(
    {
        ".mkv",
        ".mka",
        ".mk3d",
        ".webm",
        ".mp4",
        ".m4v",
        ".m4a",
        ".mov",
        ".avi",
        ".wmv",
        ".wma",
        ".ts",
        ".m2ts",
        ".mts",
        ".vob",
        ".flv",
        ".wav",
        ".flac",
        ".aif",
        ".aiff",
        ".mp3",
        ".aac",
        ".ac3",
        ".eac3",
        ".ec3",
        ".dts",
        ".thd",
        ".ogg",
        ".oga",
        ".opus",
    }
)

Fingerprint = tuple[int, int, str]


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `ingest` command."""
    parser = subparsers.add_parser(
        "ingest",
        help="probe sources, copy tracks, write the inventory",
        description=(
            "Probe every audio and video file under seed/raw, copy each one's original-language "
            "audio track without re-encoding into seed/tracks, and write seed/inventory.jsonl."
        ),
    )
    parser.add_argument(
        "--language",
        help="the original-language tag to choose among several audio tracks (e.g. eng)",
    )
    parser.set_defaults(func=run)


def make_docker() -> jobs.DockerOps:
    """Return the docker CLI; tests replace this function."""
    return jobs.HostDocker()


def image_tag(family: str) -> str:
    """Return the local image of `family` at HEAD, as `images build` tags it.

    Raises:
        InputError: git cannot name HEAD in the checkout.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "HEAD"],
            cwd=CHECKOUT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise InputError("refused: git cannot name HEAD, so the image tag is unknown") from exc
    head = proc.stdout.strip()
    if proc.returncode != 0 or not head:
        raise InputError("refused: git cannot name HEAD, so the image tag is unknown")
    image = jobs.load_families()[family]["image"]
    return f"{image}:{head[:12]}"


def list_sources(root: root_mod.Root) -> list[str]:
    """Return the root-relative paths of the media files under `seed/raw/`, sorted.

    Hidden files are skipped (a copy tool's `._` sidecars), and so is anything that is not a
    regular file; symlinked folders are not followed.
    """
    raw = root.path / stage.RAW_REL
    found: list[str] = []
    for dirpath, dirs, files in os.walk(raw):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            path = Path(dirpath, name)
            if name.startswith(".") or path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            if path.is_file() and not path.is_symlink():
                found.append(path.relative_to(root.path).as_posix())
    return sorted(found)


def assign_ids(paths: Sequence[str], existing: Mapping[str, str]) -> dict[str, str]:
    """Return each path's source id: its existing one, else the next number after every id used.

    New paths are numbered in sorted path order after the highest existing id, so an id never
    moves to another file.
    """
    used = [int(sid.removeprefix("src-")) for sid in existing.values()]
    next_number = max(used, default=0) + 1
    ids: dict[str, str] = {}
    for path in sorted(paths):
        if path in existing:
            ids[path] = existing[path]
        else:
            ids[path] = root_mod.source_id(next_number)
            next_number += 1
    return ids


def choose_audio(
    source_id: str, streams: Sequence[Mapping[str, object]], language: str | None
) -> tuple[Mapping[str, object], int] | None:
    """Return the original-language audio stream and its index among audio streams.

    One audio stream is taken whatever its tag. With several, the candidates are those whose
    language tag matches `language` (all of them when none does or none is asked for); one
    candidate is taken, else the one default among them.

    Returns:
        The stream and its audio index, or `None` when the file has no audio stream.

    Raises:
        InputError: Two or more candidates and no single default; the message lists their
            indexes and tags, never the file name.
    """
    audio = [s for s in streams if s.get("type") == "audio"]
    if not audio:
        return None
    candidates = audio
    if language and len(audio) > 1:
        wanted = language.lower()
        matched = [s for s in audio if str(s.get("language") or "").lower() == wanted]
        if matched:
            candidates = matched
    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        defaults = [s for s in candidates if s.get("default") is True]
        if len(defaults) != 1:
            listed = "; ".join(
                f"stream {s.get('index')} (audio {audio.index(s)}, language "
                f"{s.get('language') or 'none'}{', default' if s.get('default') else ''})"
                for s in candidates
            )
            raise InputError(
                f"refused: {source_id} has {len(candidates)} candidate audio streams and no "
                f"single default; pass --language: {listed}"
            )
        chosen = defaults[0]
    return chosen, audio.index(chosen)


def _fingerprint(path: Path) -> Fingerprint:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, sha256_file(path)


def _job_id(root: root_mod.Root) -> str:
    now = datetime.now(UTC)
    number = 1
    while (root.path / "runs" / root_mod.job_id(now, number)).exists():
        number += 1
    return root_mod.job_id(now, number)


def _run_step(
    root: root_mod.Root, executor: jobs.Executor, image: str, step: str, raw_input: JobInput
) -> None:
    job = jobs.JobSpec(
        job_id=_job_id(root),
        family=FAMILY,
        image=image,
        module=MODULE,
        args=("--step", step),
        inputs=(raw_input,),
        outputs=stage.TRACKS_REL,
        gpu=False,
        network=False,
    )
    executor.run(job)


def _rows(root: root_mod.Root, rel: str) -> dict[str, Mapping[str, object]]:
    data = read_json(root.path / rel)
    rows = data.get("sources") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise JobError(f"{rel} is not a stage result")
    return {str(row.get("source_id")): row for row in rows}


def run(args: argparse.Namespace) -> int:
    """Ingest every source under `seed/raw/` and write `seed/inventory.jsonl`."""
    root = root_mod.open_root(args.root)
    cutoff_db = gates_mod.load(root.path).value("ingest.bandwidth_cutoff_db")
    paths = list_sources(root)
    if not paths:
        raise InputError(f"refused: no audio or video files under {stage.RAW_REL}")
    inventory = root.path / INVENTORY_REL
    existing = (
        {r.path: r.source_id for r in read_jsonl(inventory, SourceRecord)}
        if inventory.exists()
        else {}
    )
    ids = assign_ids(paths, existing)
    before = {path: _fingerprint(root.path / path) for path in paths}
    digests = {path: stage.path_digest(path) for path in paths}
    # The read-only `seed/raw` mount's sha256 is a digest of what it held: path digests and
    # file sha256s, so job.json pins the sources without naming them.
    manifest = json.dumps(sorted([digests[p], before[p][2]] for p in paths)).encode("utf-8")
    raw_input = JobInput(rel=stage.RAW_REL, sha256=hashlib.sha256(manifest).hexdigest())
    executor = jobs.executor_for("local", root=root, docker=make_docker())
    image = image_tag(FAMILY)

    plan = [{"source_id": ids[p], "path_sha256": digests[p]} for p in paths]
    write_json(root, stage.PLAN_REL, {"schema": stage.SCHEMA, "sources": plan})
    _run_step(root, executor, image, "probe", raw_input)
    probes = _rows(root, stage.PROBE_REL)

    chosen: dict[str, tuple[Mapping[str, object], int]] = {}
    for path in paths:
        sid = ids[path]
        probe = probes.get(sid)
        if probe is None or not isinstance(probe.get("streams"), list):
            raise JobError(f"{sid}: the probe step returned no streams")
        pick = choose_audio(sid, probe["streams"], args.language)  # type: ignore[arg-type]
        if pick is None:
            print(f"{sid}: no audio stream; skipped")
            continue
        chosen[path] = pick
    if not chosen:
        raise InputError("refused: no source holds an audio stream")

    plan = [
        {"source_id": ids[p], "path_sha256": digests[p], "audio_index": chosen[p][1]}
        for p in chosen
    ]
    write_json(
        root,
        stage.PLAN_REL,
        {"schema": stage.SCHEMA, "bandwidth_cutoff_db": cutoff_db, "sources": plan},
    )
    _run_step(root, executor, image, "copy", raw_input)
    copies = _rows(root, stage.COPY_REL)

    changed = [ids[p] for p in paths if _fingerprint(root.path / p) != before[p]]
    if changed:
        raise JobError(
            f"source {', '.join(changed)} changed during ingest (size, mtime or sha256); "
            "the inventory is not written"
        )

    records = [_record(path, ids[path], before[path], probes, chosen, copies) for path in chosen]
    records.sort(key=lambda r: r.source_id)
    write_jsonl(root, INVENTORY_REL, records)
    for record in records:
        print(_summary(record))
    print(
        f"inventory: {len(records)} sources in {INVENTORY_REL}; all {len(paths)} source files "
        "unchanged (size, mtime, sha256)"
    )
    return 0


def _record(
    path: str,
    sid: str,
    fingerprint: Fingerprint,
    probes: Mapping[str, Mapping[str, object]],
    chosen: Mapping[str, tuple[Mapping[str, object], int]],
    copies: Mapping[str, Mapping[str, object]],
) -> SourceRecord:
    stream, audio_index = chosen[path]
    copy = copies.get(sid)
    if copy is None:
        raise JobError(f"{sid}: the copy step returned no result")
    size, mtime_ns, sha256 = fingerprint
    try:
        return SourceRecord(
            source_id=sid,
            path=path,
            container=str(probes[sid]["container"]),
            size_bytes=size,
            mtime_ns=mtime_ns,
            streams=tuple(dict(s) for s in probes[sid]["streams"]),  # type: ignore[attr-defined]
            stream_index=int(stream["index"]),  # type: ignore[call-overload]
            audio_index=audio_index,
            language=stream.get("language"),  # type: ignore[arg-type]
            codec=str(stream["codec"]),
            profile=stream.get("profile"),  # type: ignore[arg-type]
            bitrate_bps=int(copy["bitrate_bps"]),  # type: ignore[call-overload]
            bitrate_from=str(copy["bitrate_from"]),
            sample_rate=int(stream["sample_rate"]),  # type: ignore[call-overload]
            channels=int(stream["channels"]),  # type: ignore[call-overload]
            layout=str(stream["layout"]),
            duration_s=float(copy["duration_s"]),  # type: ignore[arg-type]
            lossless=bool(copy["lossless"]),
            effective_bandwidth_hz=int(copy["effective_bandwidth_hz"]),  # type: ignore[call-overload]
            sha256=sha256,
            track=str(copy["track"]),
            primary_channel=str(copy["primary_channel"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise JobError(f"{sid}: the stage result is incomplete") from exc


def _summary(record: SourceRecord) -> str:
    profile = f" ({record.profile})" if record.profile else ""
    language = record.language or "no language tag"
    return (
        f"{record.source_id}: {record.codec}{profile}, {language}, {record.channels} ch "
        f"{record.layout}, {record.sample_rate} Hz, {record.bitrate_bps // 1000} kbps "
        f"(packets), {record.duration_s:.1f} s, bandwidth {record.effective_bandwidth_hz} Hz, "
        f"primary {record.primary_channel}, track {record.track}"
    )
