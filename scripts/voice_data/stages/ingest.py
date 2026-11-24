"""Ingest stage: probe source files, copy the chosen audio track without re-encoding, measure it.

Runs in the base image (family `score`, which holds ffmpeg and numpy) as
`/opt/venv-score/bin/python -I -u -m scripts.voice_data.stages.ingest --data /data
--step probe|copy`, with `seed/raw/` mounted read-only and `seed/tracks/` writable. `probe` lists every stream of each
source; the host chooses the track; `copy` copies it with `-c copy` into Matroska audio and
measures the copy's packet bitrate and effective bandwidth. The video stream is never copied or
decoded.

The host names each source in the plan by its id and the sha256 of its root-relative path, so no
file name passes through an argv, the plan, a result or a log, and the host's
`seed/inventory.jsonl` stays the one record that names a source file. ffmpeg's stderr names its
input, so it is discarded and a failure is reported by source id.

The host imports this module for the plan's file names and `path_digest`, so its top level uses
the standard library only; numpy comes in with `bandwidth`, imported where it is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

RAW_REL = "seed/raw"
TRACKS_REL = "seed/tracks"
PLAN_REL = f"{TRACKS_REL}/ingest-plan.json"
PROBE_REL = f"{TRACKS_REL}/ingest-probe.json"
COPY_REL = f"{TRACKS_REL}/ingest-copy.json"
OUTPUTS_REL = f"{TRACKS_REL}/outputs.json"
STEPS = ("probe", "copy")
SCHEMA = 1
SOURCE_ID = re.compile(r"src-[0-9]{4,}")

# ffmpeg's AC-3 and E-AC-3 decoders apply dynamic range compression unless told not to. Other
# decoders lack the option, and ffmpeg's warning about an unused option would name the input.
DRC_CODECS = frozenset({"ac3", "eac3"})
LOSSLESS_CODECS = frozenset({"flac", "alac", "truehd", "mlp", "wavpack", "tta", "ape"})
# ffmpeg 4.4's named layouts that hold a front centre channel. `mono` is FC to ffmpeg, but it is
# the whole mix, so it is decoded as the downmix, as stereo is.
CENTRE_LAYOUTS = frozenset(
    {
        "3.0",
        "3.1",
        "4.0",
        "4.1",
        "5.0",
        "5.0(side)",
        "5.1",
        "5.1(side)",
        "6.0",
        "6.1",
        "6.1(back)",
        "hexagonal",
        "7.0",
        "7.0(front)",
        "7.1",
        "7.1(wide)",
        "7.1(wide-side)",
        "octagonal",
        "hexadecagonal",
        "22.2",
    }
)
# ffprobe names no layout for some streams (AAC 5.1 in Matroska); ffmpeg's decoder then takes its
# default layout for the channel count, as `av_get_default_channel_layout` gives it, so the
# inventory records the same one.
_DEFAULT_LAYOUTS = {
    1: "mono",
    2: "stereo",
    3: "2.1",
    4: "4.0",
    5: "5.0",
    6: "5.1",
    7: "6.1",
    8: "7.1",
}


class StageError(RuntimeError):
    """A source could not be probed, copied or measured; the message names its id only."""


def path_digest(rel: str) -> str:
    """Return the sha256 of a root-relative posix path: how the plan names a source."""
    return hashlib.sha256(rel.encode("utf-8")).hexdigest()


def primary_channel(layout: str) -> str:
    """Return `FC` when the layout has a front centre channel, else `downmix`."""
    if layout in CENTRE_LAYOUTS or "FC" in layout.split("+"):
        return "FC"
    return "downmix"


def decode_argv(
    track: str, codec: str, channel: str, *, output: str = "-", fmt: str = "f32le"
) -> list[str]:
    """Return the ffmpeg argv that decodes a copied track to mono.

    DRC is off for the AC-3 family (`-drc_scale 0`, a decoder option, so before `-i`). `FC`
    takes the centre channel alone; `downmix` is ffmpeg's default matrix to mono, LFE excluded.

    Args:
        track: The copied track (one audio stream).
        codec: The track's codec, as ffprobe names it.
        channel: `FC` or `downmix`.
        output: Where ffmpeg writes; `-` is stdout.
        fmt: The output format; `f32le` is raw float32 for a pipe.
    """
    argv = ["ffmpeg", "-nostdin", "-v", "error"]
    if codec in DRC_CODECS:
        argv += ["-drc_scale", "0"]
    argv += ["-i", track, "-map", "0:a:0"]
    if channel == "FC":
        argv += ["-af", "pan=mono|c0=FC"]
    elif channel == "downmix":
        argv += ["-ac", "1"]
    else:
        raise ValueError(f"unknown channel {channel!r}; use FC or downmix")
    return [*argv, "-f", fmt, output]


def _ffprobe(path: Path, what: str, *args: str) -> dict[str, object]:
    argv = ["ffprobe", "-v", "error", *args, "-of", "json", str(path)]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise StageError(f"{what}: ffprobe exited {proc.returncode}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise StageError(f"{what}: ffprobe output is not JSON") from exc
    if not isinstance(data, dict):
        raise StageError(f"{what}: ffprobe output is not an object")
    return data


def _layout(stream: Mapping[str, object]) -> str:
    layout = stream.get("channel_layout")
    if isinstance(layout, str) and layout and layout != "unknown":
        return layout
    channels = int(stream.get("channels") or 0)  # type: ignore[call-overload]
    return _DEFAULT_LAYOUTS.get(channels, f"{channels} channels")


def stream_facts(stream: Mapping[str, object]) -> dict[str, object]:
    """Return one stream's entry of the inventory's `streams`, from ffprobe's stream object.

    Only codec facts are kept; a stream's title tag, which can name the episode, is not.
    """
    kind = stream.get("codec_type")
    facts: dict[str, object] = {
        "index": int(stream["index"]),  # type: ignore[call-overload]
        "type": kind,
        "codec": stream.get("codec_name"),
    }
    profile = stream.get("profile")
    if isinstance(profile, str) and profile and profile != "unknown":
        facts["profile"] = profile
    tags = stream.get("tags")
    if isinstance(tags, Mapping):
        language = next((v for k, v in tags.items() if str(k).lower() == "language"), None)
        if isinstance(language, str) and language:
            facts["language"] = language
    disposition = stream.get("disposition")
    default = isinstance(disposition, Mapping) and disposition.get("default") == 1
    if kind == "video":
        facts["width"] = stream.get("width")
        facts["height"] = stream.get("height")
    elif kind == "audio":
        facts["channels"] = int(stream.get("channels") or 0)  # type: ignore[call-overload]
        facts["layout"] = _layout(stream)
        facts["sample_rate"] = int(stream.get("sample_rate") or 0)  # type: ignore[call-overload]
        facts["default"] = default
    else:
        facts["default"] = default
    return facts


def probe_file(path: Path, what: str) -> dict[str, object]:
    """Return a file's container and every stream it holds, in index order."""
    data = _ffprobe(path, what, "-show_format", "-show_streams")
    fmt = data.get("format")
    name = fmt.get("format_name") if isinstance(fmt, Mapping) else None
    streams = data.get("streams")
    if not isinstance(name, str) or not isinstance(streams, list):
        raise StageError(f"{what}: ffprobe found no container or streams")
    facts = sorted((stream_facts(s) for s in streams), key=lambda s: s["index"])  # type: ignore[arg-type,return-value]
    return {"container": name.split(",")[0], "streams": facts}


def packet_bitrate(path: Path, audio_index: int, what: str = "track") -> tuple[int, float]:
    """Return an audio stream's bitrate from its packet bytes over its duration, and the duration.

    Matroska reports no per-stream bitrate for AAC, so the packets are the one measure every
    container shares. The duration runs from the first packet's start to the last packet's end.
    """
    data = _ffprobe(
        path,
        what,
        "-select_streams",
        f"a:{audio_index}",
        "-show_entries",
        "packet=size,pts_time,duration_time",
    )
    packets = data.get("packets")
    if not isinstance(packets, list) or not packets:
        raise StageError(f"{what}: no audio packets")
    total = 0
    starts: list[float] = []
    ends: list[float] = []
    for packet in packets:
        total += int(packet.get("size", 0))
        pts = packet.get("pts_time")
        if pts in (None, "N/A"):
            continue
        start = float(pts)
        duration = packet.get("duration_time")
        starts.append(start)
        ends.append(start + (float(duration) if duration not in (None, "N/A") else 0.0))
    span = max(ends) - min(starts) if starts else 0.0
    if span <= 0:
        raise StageError(f"{what}: audio packets carry no duration")
    return round(total * 8 / span), span


def copy_track(source: Path, dest: Path, audio_index: int, what: str) -> None:
    """Copy one audio stream into Matroska audio without re-encoding, atomically.

    Container and chapter metadata are not carried over: a title tag can name the episode, and
    the inventory is the one record that names a source.
    """
    part = dest.with_name(dest.name + ".part")
    argv = [
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
        "-map", f"0:a:{audio_index}", "-c", "copy", "-map_metadata", "-1",
        "-map_chapters", "-1", "-f", "matroska", str(part),
    ]  # fmt: skip
    proc = subprocess.run(argv, capture_output=True, check=False)
    if proc.returncode != 0:
        if part.exists():
            part.unlink()
        raise StageError(f"{what}: track copy exited {proc.returncode}")
    os.replace(part, dest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(part, path)


def _read_plan(data: Path) -> dict[str, object]:
    try:
        plan = json.loads((data / PLAN_REL).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("the ingest plan is missing or not JSON") from exc
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != SCHEMA
        or not isinstance(plan.get("sources"), list)
    ):
        raise StageError("the ingest plan is not a schema 1 plan")
    for entry in plan["sources"]:
        if not isinstance(entry, dict) or not SOURCE_ID.fullmatch(str(entry.get("source_id"))):
            raise StageError("the ingest plan names a source without a valid id")
    return plan


def _locate(data: Path, sources: Sequence[Mapping[str, object]]) -> dict[str, Path]:
    by_digest: dict[str, Path] = {}
    for dirpath, _dirs, files in os.walk(data / RAW_REL):
        for name in files:
            path = Path(dirpath, name)
            by_digest[path_digest(path.relative_to(data).as_posix())] = path
    found: dict[str, Path] = {}
    for entry in sources:
        sid = str(entry["source_id"])
        path = by_digest.get(str(entry.get("path_sha256")))
        if path is None:
            raise StageError(f"{sid}: source not found under {RAW_REL}")
        found[sid] = path
    return found


def _copy(data: Path, entry: Mapping[str, object], source: Path, cutoff_db: float) -> dict:
    from . import bandwidth

    sid = str(entry["source_id"])
    audio_index = entry.get("audio_index")
    if not isinstance(audio_index, int) or isinstance(audio_index, bool) or audio_index < 0:
        raise StageError(f"{sid}: the plan gives no audio index")
    track_rel = f"{TRACKS_REL}/{sid}.mka"
    track = data / track_rel
    copy_track(source, track, audio_index, sid)
    streams: list[dict[str, object]] = probe_file(track, sid)["streams"]  # type: ignore[assignment]
    if [s["type"] for s in streams] != ["audio"]:
        raise StageError(f"{sid}: the copy holds {len(streams)} streams, not one audio stream")
    stream = streams[0]
    bitrate_bps, duration_s = packet_bitrate(track, 0, sid)
    channel = primary_channel(str(stream["layout"]))
    codec = str(stream["codec"])
    try:
        bandwidth_hz = bandwidth.stream_bandwidth(
            decode_argv(str(track), codec, channel), int(stream["sample_rate"]), cutoff_db
        )
    except bandwidth.DecodeError as exc:
        raise StageError(f"{sid}: decode of the copy failed") from exc
    return {
        "source_id": sid,
        "track": track_rel,
        "bitrate_bps": bitrate_bps,
        "bitrate_from": "packets",
        "duration_s": round(duration_s, 3),
        "lossless": codec in LOSSLESS_CODECS or codec.startswith("pcm_"),
        "effective_bandwidth_hz": bandwidth_hz,
        "primary_channel": channel,
    }


def run(data: Path, step: str) -> list[str]:
    """Run one step over the plan's sources; return the root-relative files written."""
    plan = _read_plan(data)
    sources: list[dict[str, object]] = plan["sources"]  # type: ignore[assignment]
    located = _locate(data, sources)
    if step == "probe":
        rows = [{"source_id": sid, **probe_file(path, sid)} for sid, path in located.items()]
        _write_json(data / PROBE_REL, {"schema": SCHEMA, "sources": rows})
        print(f"ingest probe: {len(rows)} sources")
        return [PROBE_REL]
    cutoff = plan.get("bandwidth_cutoff_db")
    if not isinstance(cutoff, int | float) or isinstance(cutoff, bool):
        raise StageError("the ingest plan gives no bandwidth cutoff")
    rows = []
    for entry in sources:
        row = _copy(data, entry, located[str(entry["source_id"])], float(cutoff))
        print(f"ingest copy: {row['source_id']} copied and measured")
        rows.append(row)
    _write_json(data / COPY_REL, {"schema": SCHEMA, "sources": rows})
    return [COPY_REL, *(str(row["track"]) for row in rows)]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stage; write `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.ingest")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--step", required=True, choices=STEPS)
    args = parser.parse_args(argv)
    try:
        written = run(args.data, args.step)
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    files = [{"rel": rel, "sha256": _sha256(args.data / rel)} for rel in written]
    _write_json(args.data / OUTPUTS_REL, {"schema": SCHEMA, "files": files})
    return 0


if __name__ == "__main__":
    sys.exit(main())
