"""Stage the AMI train meetings the verifier mines and calibrates on, kept apart from dev and eval.

    python -m scripts.wake.verifier.stage ami --annotations <zip> --dev <dir> --eval <dir> \
        --out <dir>

The bar is measured on AMI dev (`spotter_eval --ambient ami/dev`), so no dev or eval audio, and
no voice heard in it, may reach training or the calibration of the verifier's threshold table.
This keeps the full-corpus ASR train meetings (`AMI_TRAIN`) less every meeting that shares a
participant (`global_name` in the public annotations' `corpusResources/meetings.xml`) with a dev
or eval meeting, and downloads only the kept meetings' `Mix-Headset` WAV, the channel v14 H28
staged for dev and eval, so training, calibration and the bar hear one channel.

The kept meetings split into three shares by speaker group, deterministically. A group is a
connected component of meetings that share a `global_name`: the non-scenario meetings share
speakers across series, so a split by series would put one voice in two shares. Groups, each in
meeting-id order and taken in the order of their first meeting, go to calibration until it
holds 40% of the kept hours, then to check until it holds 15%, and the rest to mining. Exit 1
when calibration is under 20 h or check under 6 h: the hours are printed and go to Raj, and
`split.json` is not written.

Each download is over https with its status checked, written beside its final name and renamed
only after it is checked as 16 kHz mono PCM16, its sha256 in neither held-out `SHA256SUMS`, and
listed in `<out>/SHA256SUMS`. A rerun downloads only what is missing and refuses, exit 2, a
file whose sha256 differs from `SHA256SUMS`, a file `SHA256SUMS` does not list, or a file of a
meeting that is not kept. Exit 0 staged, 1 a share under its minimum, 2 refused.
"""

import argparse
import json
import os
import sys
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree  # nosec B405 - the AMI public annotations, sha256 recorded

from .dataset import (
    SAMPLE_RATE,
    DatasetError,
    is_pcm16_16k_mono,
    read_sums,
    sha256_bytes,
    sha256_file,
    sums_bytes,
    wav_info,
)

AMI_PARTITION_SOURCE = (
    "The AMI full-corpus ASR partition: https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml "
    "('Full-corpus-ASR', the full-corpus partition with ES2003, ES2014, TS3006 and TS3007 moved "
    "into training), as meeting ids in Kaldi's AMI recipe, "
    "https://github.com/kaldi-asr/kaldi/tree/master/egs/ami/s5/local split_train.orig (137), "
    "split_dev.orig (18) and split_eval.orig (16). Kaldi's train list holds IB4005, which the "
    "datasets page's series list does not name; it is kept as Kaldi lists it."
)

AMI_TRAIN: tuple[str, ...] = tuple(
    """
    EN2001a EN2001b EN2001d EN2001e EN2003a EN2004a EN2005a EN2006a EN2006b EN2009b EN2009c
    EN2009d ES2002a ES2002b ES2002c ES2002d ES2003a ES2003b ES2003c ES2003d ES2005a ES2005b
    ES2005c ES2005d ES2006a ES2006b ES2006c ES2006d ES2007a ES2007b ES2007c ES2007d ES2008a
    ES2008b ES2008c ES2008d ES2009a ES2009b ES2009c ES2009d ES2010a ES2010b ES2010c ES2010d
    ES2012a ES2012b ES2012c ES2012d ES2013a ES2013b ES2013c ES2013d ES2014a ES2014b ES2014c
    ES2014d ES2015a ES2015b ES2015c ES2015d ES2016a ES2016b ES2016c ES2016d IB4005 IN1001
    IN1002 IN1005 IN1007 IN1008 IN1009 IN1012 IN1013 IN1014 IN1016 IS1000a IS1000b IS1000c
    IS1000d IS1001a IS1001b IS1001c IS1001d IS1002b IS1002c IS1002d IS1003a IS1003b IS1003c
    IS1003d IS1004a IS1004b IS1004c IS1004d IS1005a IS1005b IS1005c IS1006a IS1006b IS1006c
    IS1006d IS1007a IS1007b IS1007c IS1007d TS3005a TS3005b TS3005c TS3005d TS3006a TS3006b
    TS3006c TS3006d TS3007a TS3007b TS3007c TS3007d TS3008a TS3008b TS3008c TS3008d TS3009a
    TS3009b TS3009c TS3009d TS3010a TS3010b TS3010c TS3010d TS3011a TS3011b TS3011c TS3011d
    TS3012a TS3012b TS3012c TS3012d
    """.split()
)
"""The 137 train meetings (`AMI_PARTITION_SOURCE`)."""

AMI_DEV: tuple[str, ...] = tuple(
    """
    ES2011a ES2011b ES2011c ES2011d IB4001 IB4002 IB4003 IB4004 IB4010 IB4011 IS1008a IS1008b
    IS1008c IS1008d TS3004a TS3004b TS3004c TS3004d
    """.split()
)
"""The 18 dev meetings (`AMI_PARTITION_SOURCE`); the bar's ambient set."""

AMI_EVAL: tuple[str, ...] = tuple(
    """
    EN2002a EN2002b EN2002c EN2002d ES2004a ES2004b ES2004c ES2004d IS1009a IS1009b IS1009c
    IS1009d TS3003a TS3003b TS3003c TS3003d
    """.split()
)
"""The 16 eval meetings (`AMI_PARTITION_SOURCE`)."""

MEETINGS_XML = "corpusResources/meetings.xml"
"""In `ami_public_manual_1.6.2.zip`, read as lhotse's AMI recipe reads it
(https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/ami.py,
`parse_ami_annotations`): the root's children are meetings whose `observation` is the meeting
id, and each meeting's children are speakers whose `global_name` is the participant id
(https://groups.inf.ed.ac.uk/ami/corpus/participantids.shtml). The tag names and any namespace
on them are not read."""
MIRROR = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/"
    "{meeting}.Mix-Headset.wav"
)
WAV_SUFFIX = ".Mix-Headset.wav"
SUMS = "SHA256SUMS"
SPLIT = "split.json"
SHARES = ("calibration", "check", "mining")
CALIBRATION_SHARE = 0.40
CHECK_SHARE = 0.15
FETCH_TIMEOUT_S = 60
_BLOCK = 1 << 20

Fetcher = Callable[[str, Path], None]


class StageError(Exception):
    """A refusal; the command exits 2 with the message."""


@dataclass(frozen=True, slots=True)
class Partition:
    """The train, dev and eval meeting ids."""

    train: tuple[str, ...]
    dev: tuple[str, ...]
    eval: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Minimums:
    """The least calibration and check audio, in hours, the verifier's tables are built on."""

    calibration_hours: float = 20.0
    check_hours: float = 6.0


AMI_PARTITION = Partition(AMI_TRAIN, AMI_DEV, AMI_EVAL)


def https_fetch(
    url: str, dest: Path, *, opener: Callable[..., object] = urllib.request.urlopen
) -> None:
    """Download `url` to `dest` over https, refusing any status but 200.

    Raises:
        StageError: The URL is not https, or the response's status is not 200.
    """
    if not url.startswith("https://"):
        raise StageError(f"{url}: only https downloads are staged")
    with opener(url, timeout=FETCH_TIMEOUT_S) as response:  # nosec B310 - https checked above
        status = getattr(response, "status", None)
        if status != 200:
            raise StageError(f"{url}: the server answered status {status}, not 200")
        with dest.open("wb") as handle:
            for block in iter(lambda: response.read(_BLOCK), b""):
                handle.write(block)


def read_meetings(annotations: Path) -> dict[str, frozenset[str]]:
    """Each meeting's participants (`global_name`), from `meetings.xml` in the annotations zip.

    Raises:
        StageError: The zip holds no single `meetings.xml`, or it is malformed.
    """
    try:
        with zipfile.ZipFile(annotations) as archive:
            names = [name for name in archive.namelist() if name.endswith(MEETINGS_XML)]
            if len(names) != 1:
                raise StageError(f"{annotations.name} must hold one {MEETINGS_XML}, found {names}")
            data = archive.read(names[0])
    except (zipfile.BadZipFile, OSError) as exc:
        raise StageError(f"{annotations.name} could not be read as a zip: {exc}") from exc
    try:
        root = ElementTree.fromstring(data)  # nosec B314 - the public annotations, sha recorded
    except ElementTree.ParseError as exc:
        raise StageError(f"{MEETINGS_XML} is not XML: {exc}") from exc
    meetings: dict[str, frozenset[str]] = {}
    for meeting in root:
        observation = meeting.attrib.get("observation")
        names = [speaker.attrib.get("global_name") for speaker in meeting]
        if not observation or not all(names):
            raise StageError(f"{MEETINGS_XML}: a meeting without observation or global_name")
        meetings[observation] = frozenset(names)
    return meetings


def speaker_filter(
    meetings: Mapping[str, frozenset[str]], partition: Partition
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """The train meetings that share no participant with dev or eval, and those that do.

    Returns the kept meetings in id order and each excluded meeting's shared participants.

    Raises:
        StageError: A meeting of the partition is not in `meetings.xml`, so its speakers are
            unknown.
    """
    missing = sorted(set(partition.train + partition.dev + partition.eval).difference(meetings))
    if missing:
        raise StageError(f"{MEETINGS_XML} does not list meetings {missing}")
    held_out = set().union(*(meetings[m] for m in partition.dev + partition.eval))
    kept, excluded = [], {}
    for meeting in sorted(partition.train):
        shared = sorted(meetings[meeting] & held_out)
        if shared:
            excluded[meeting] = tuple(shared)
        else:
            kept.append(meeting)
    return tuple(kept), excluded


def speaker_groups(
    kept: Sequence[str], meetings: Mapping[str, frozenset[str]]
) -> list[tuple[str, ...]]:
    """Meetings linked by a shared participant, transitively; each in id order, by first id."""
    parent = {meeting: meeting for meeting in kept}

    def find(meeting: str) -> str:
        while parent[meeting] != meeting:
            parent[meeting] = parent[parent[meeting]]
            meeting = parent[meeting]
        return meeting

    by_speaker: dict[str, str] = {}
    for meeting in sorted(kept):
        for speaker in sorted(meetings[meeting]):
            other = by_speaker.setdefault(speaker, meeting)
            low, high = sorted((find(meeting), find(other)))
            parent[high] = low
    groups: dict[str, list[str]] = {}
    for meeting in sorted(kept):
        groups.setdefault(find(meeting), []).append(meeting)
    return sorted((tuple(group) for group in groups.values()), key=lambda group: group[0])


def assign_shares(
    groups: Sequence[tuple[str, ...]], hours: Mapping[str, float]
) -> dict[str, tuple[str, ...]]:
    """Each share's meetings: groups in order to calibration, then check, then mining."""
    total = sum(hours[meeting] for group in groups for meeting in group)
    targets = {"calibration": CALIBRATION_SHARE * total, "check": CHECK_SHARE * total}
    held = dict.fromkeys(SHARES, 0.0)
    members: dict[str, list[str]] = {share: [] for share in SHARES}
    for group in groups:
        share = next(
            (name for name in ("calibration", "check") if held[name] < targets[name]), "mining"
        )
        held[share] += sum(hours[meeting] for meeting in group)
        members[share].extend(group)
    return {share: tuple(sorted(found)) for share, found in members.items()}


def _held_out(directory: Path, name: str) -> tuple[dict[str, str], str]:
    path = directory / SUMS
    if not path.is_file():
        raise StageError(f"--{name} {directory} has no {SUMS}; stage {name} before train")
    try:
        return read_sums(path), sha256_file(path)
    except DatasetError as exc:
        raise StageError(str(exc)) from exc


def _write_atomic(path: Path, data: bytes) -> None:
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(data)
    os.replace(partial, path)


def _check_wav(path: Path, name: str) -> int:
    """The WAV's sample count, once it is checked as 16 kHz mono PCM16."""
    try:
        info = wav_info(path)
    except DatasetError as exc:
        raise StageError(f"{name}: {exc}") from exc
    if not is_pcm16_16k_mono(info):
        raise StageError(
            f"{name} is {info.rate} Hz, {info.channels} channels, {8 * info.width}-bit; "
            "expected 16 kHz mono PCM16"
        )
    return info.frames


def _stage_files(
    kept: Sequence[str], out: Path, fetch: Fetcher, held_out: Mapping[str, str]
) -> dict[str, int]:
    """Download what is missing, check every file, and return each meeting's sample count."""
    out.mkdir(parents=True, exist_ok=True)
    sums_path = out / SUMS
    sums = read_sums(sums_path) if sums_path.exists() else {}
    expected = {f"{meeting}{WAV_SUFFIX}": meeting for meeting in kept}
    strays = sorted(
        path.name for path in out.iterdir() if path.suffix == ".wav" and path.name not in expected
    )
    strays += sorted(name for name in sums if name not in expected)
    if strays:
        raise StageError(f"{out} holds files of meetings that are not kept: {strays}")
    samples: dict[str, int] = {}
    for name, meeting in expected.items():
        path = out / name
        if path.exists():
            if name not in sums:
                raise StageError(f"{name} is not listed in {SUMS}; remove it to download again")
            digest = sha256_file(path)
            if digest != sums[name]:
                raise StageError(f"{name} has sha256 {digest}, but {SUMS} lists {sums[name]}")
            samples[meeting] = _check_wav(path, name)
        else:
            partial = out / f"{name}.part"
            partial.unlink(missing_ok=True)
            try:
                fetch(MIRROR.format(meeting=meeting), partial)
                samples[meeting] = _check_wav(partial, name)
                digest = sha256_file(partial)
                if name in sums and digest != sums[name]:
                    raise StageError(
                        f"{name} downloaded with sha256 {digest}, but {SUMS} lists {sums[name]}"
                    )
            except StageError:
                partial.unlink(missing_ok=True)
                raise
            sums[name] = digest
            _write_atomic(sums_path, sums_bytes(sums))
            os.replace(partial, path)
        if sums[name] in held_out:
            raise StageError(f"{name} has the sha256 of {held_out[sums[name]]}")
    return samples


def stage(
    annotations: Path,
    dev: Path,
    eval_dir: Path,
    out: Path,
    *,
    fetch: Fetcher,
    partition: Partition = AMI_PARTITION,
    minimums: Minimums = Minimums(),
) -> int:
    """Stage the kept train meetings in `out`; 0 staged, 1 a share under its minimum, 2 refused."""
    try:
        dev_sums, dev_sha = _held_out(dev, "dev")
        eval_sums, eval_sha = _held_out(eval_dir, "eval")
        held_out = {digest: f"ami/dev/{name}" for name, digest in dev_sums.items()}
        held_out |= {digest: f"ami/eval/{name}" for name, digest in eval_sums.items()}
        meetings = read_meetings(annotations)
        kept, excluded = speaker_filter(meetings, partition)
        samples = _stage_files(kept, out, fetch, held_out)
    except (StageError, DatasetError, OSError) as exc:
        print(f"stage: {exc}", file=sys.stderr)
        return 2
    hours = {meeting: count / SAMPLE_RATE / 3600 for meeting, count in samples.items()}
    groups = speaker_groups(kept, meetings)
    shares = assign_shares(groups, hours)
    held = {share: sum(hours[m] for m in members) for share, members in shares.items()}
    print(f"kept {len(kept)} of {len(partition.train)} train meetings; excluded {len(excluded)}")
    for share in SHARES:
        print(f"{share}: {len(shares[share])} meetings, {held[share]:.2f} h")
    short = [
        f"{share} {held[share]:.2f} h is under {minimum:g} h"
        for share, minimum in (
            ("calibration", minimums.calibration_hours),
            ("check", minimums.check_hours),
        )
        if held[share] < minimum
    ]
    if short:
        print("stage: " + "; ".join(short) + "; split.json is not written")
        return 1
    payload = {
        "version": 1,
        "partition": AMI_PARTITION_SOURCE,
        "annotations_sha256": sha256_file(annotations),
        "dev_sha256sums": dev_sha,
        "eval_sha256sums": eval_sha,
        "excluded": {meeting: list(shared) for meeting, shared in excluded.items()},
        "groups": [list(group) for group in groups],
        "shares": {share: list(members) for share, members in shares.items()},
        "hours": held,
    }
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_atomic(out / SPLIT, data)
    print(f"split.json sha256 {sha256_bytes(data)}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.verifier.stage", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ami = commands.add_parser("ami", help="stage the AMI train meetings")
    ami.add_argument("--annotations", required=True, type=Path)
    ami.add_argument("--dev", required=True, type=Path)
    ami.add_argument("--eval", required=True, type=Path)
    ami.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None, *, fetch: Fetcher = https_fetch) -> int:
    """Run `ami`; see the module docstring for the exit codes."""
    args = _parser().parse_args(argv)
    return stage(args.annotations, args.dev, args.eval, args.out, fetch=fetch)


if __name__ == "__main__":
    sys.exit(main())
