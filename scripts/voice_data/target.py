"""The target voice (Design 6): its plan, the mode rule, references with transcripts, `target.json`.

`target build` runs in two halves. The job (`stages/target_build.py`) embeds audio and computes
the numbers: centroids, the leave-one-out and window distributions, per-state floors, each
identity clip's SIM against its state's floor, the mode and the profiles. This host half reads
the seed run and the identity, writes the job's plan, and afterwards places the references and
writes `target.json`. Transcripts stay on this side: each reference's `<n>.private.txt` is
copied and hashed as bytes, never decoded, so no job log or printed line can carry the text.

A reference's `transcript_sha256` is the sha256 of the `.private.txt` beside it. A seed
reference's file is T8's; an identity clip that becomes a state's reference has its own
transcript copied beside it. A missing transcript is an error, never an empty hash.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from . import identity as identity_mod
from .commands import clean as clean_cmd
from .errors import InputError, JobError
from .records import SegmentRecord, TargetRecord, read_json, read_jsonl, sha256_file
from .root import Root
from .states import SPEAKING_STATES

SCHEMA = 1
TARGET_NAME = "target.json"
PLAN_NAME = "target-plan.json"
BUILD_NAME = "build.json"
FAMILY = "score"
MODULE = "scripts.voice_data.stages.target_build"
EMBEDDERS = ("wavlm-sv", "ecapa")
SEED, IDENTITY, SEED_IDENTITY = "seed", "identity", "seed+identity"
LOO_SEGMENTS = "loo-segments"
LOO_IDENTITY_WINDOWS = "loo-identity-windows"
REHEARSAL_FLOOR = "rehearsal-floor"
IDENTITY_WINDOW_S = 3.0
FROM_SEED, FROM_IDENTITY = "seed", "identity"
EPS = 1e-9
_REF_NUMBER = re.compile(r"(?P<n>[1-9][0-9]*)\.wav")


def target_rel(target_id: str) -> str:
    """Return a target's folder, relative to the root."""
    return clean_cmd.target_rel(target_id)


def clip_passes(check: Mapping[str, object]) -> bool:
    """Return whether an identity clip reaches its state's floor on every embedder.

    SIMs equal up to rounding pass, as in retrieval: the last bit of a cosine must not decide.
    """
    sim: Mapping[str, float] = check["sim"]  # type: ignore[assignment]
    floor: Mapping[str, float] = check["floor"]  # type: ignore[assignment]
    return bool(floor) and all(sim.get(name, float("-inf")) >= f - EPS for name, f in floor.items())


def decide_mode(*, seed: bool, checks: Sequence[Mapping[str, object]]) -> str:
    """Return the target's mode by Design 6's rule.

    With no seed the identity is the voice. With a seed and no identity clip to check, the mode
    is `seed`. With both, every clip reaching its state's floor on both embedders makes the
    identity the seed voice (`seed+identity`); any clip below makes it a different voice
    (`identity`).
    """
    if not seed:
        return IDENTITY
    if not checks:
        return SEED
    return SEED_IDENTITY if all(clip_passes(c) for c in checks) else IDENTITY


def embedder_pins(families: Mapping[str, Mapping[str, object]]) -> dict[str, str]:
    """Return `<repository>@<revision>` for each embedder, from the score family's pins.

    Raises:
        InputError: `families.json` does not pin an embedder.
    """
    models = families.get(FAMILY, {}).get("models", {})
    pins: dict[str, str] = {}
    for name in EMBEDDERS:
        model = models.get(name) if isinstance(models, Mapping) else None
        if not isinstance(model, Mapping) or "repository" not in model or "revision" not in model:
            raise InputError(f"refused: families.json pins no {name} in family {FAMILY}")
        pins[name] = f"{model['repository']}@{model['revision']}"
    return pins


# The seed run.


@dataclass(frozen=True, slots=True)
class SeedRef:
    """A reference T8 chose and placed under the target, with its transcript's sha256."""

    state: str
    audio: str
    transcript: str
    transcript_sha256: str
    duration_s: float
    sim: dict[str, float]


@dataclass(frozen=True, slots=True)
class Seed:
    """A clean run as the target reads it: confirmed segments, their states, and references."""

    run_id: str
    segments: tuple[SegmentRecord, ...]
    refs: Mapping[str, tuple[SeedRef, ...]]


def _read_labels(root: Root, run_dir: str) -> dict[str, str] | None:
    plan = read_json(root.path / run_dir / clean_cmd.PLAN_NAME)
    if not isinstance(plan, dict):
        raise InputError(f"refused: {run_dir}/{clean_cmd.PLAN_NAME} is not a JSON object")
    labels = plan.get("labels")
    if labels is not None and not isinstance(labels, dict):
        raise InputError(f"refused: {run_dir}/{clean_cmd.PLAN_NAME} has malformed labels")
    return labels


def _state(record: SegmentRecord) -> str:
    return record.state or record.state_proposed


def read_seed(root: Root, run_id: str, target_id: str) -> Seed:
    """Read clean run `run_id`: its confirmed segments and the references it placed.

    Confirmed segments are the ones `clean` counted: labelled ones, else primary-cluster ones.
    Each reference must lie under `targets/<target_id>/refs/` with its transcript beside it.

    Raises:
        InputError: The run is not a finished clean run, it placed its references for another
            target, a reference or its transcript is missing, or fewer than two segments are
            confirmed.
    """
    run_dir = clean_cmd.run_rel(run_id)
    for name in (clean_cmd.SEGMENTS_NAME, clean_cmd.REFS_NAME, clean_cmd.PLAN_NAME):
        if not (root.path / run_dir / name).is_file():
            raise InputError(f"refused: {run_dir}/{name} is missing; run clean first")
    records = read_jsonl(root.path / run_dir / clean_cmd.SEGMENTS_NAME, SegmentRecord)
    labels = _read_labels(root, run_dir)
    ids = set(clean_cmd.confirmed_ids(records, labels))
    confirmed = tuple(r for r in records if r.segment_id in ids)
    if len(confirmed) < 2:
        raise InputError(f"refused: run {run_id} has fewer than two confirmed segments")
    for record in confirmed:
        if _state(record) not in SPEAKING_STATES:
            raise InputError(f"refused: {record.segment_id} has no speaking state")
        if not record.audio.startswith(f"{run_dir}/") or not (root.path / record.audio).is_file():
            raise InputError(f"refused: {record.segment_id}'s audio is missing from {run_dir}")
    refs_doc = read_json(root.path / run_dir / clean_cmd.REFS_NAME)
    if not isinstance(refs_doc, dict):
        raise InputError(f"refused: {run_dir}/{clean_cmd.REFS_NAME} is not a JSON object")
    if refs_doc.get("target") != target_id:
        raise InputError(
            f"refused: run {run_id} placed its references for target {refs_doc.get('target')}"
        )
    return Seed(run_id, confirmed, _seed_refs(root, run_dir, target_id, refs_doc))


def _seed_refs(
    root: Root, run_dir: str, target_id: str, refs_doc: Mapping[str, object]
) -> dict[str, tuple[SeedRef, ...]]:
    target_dir = target_rel(target_id)
    states = refs_doc.get("states")
    found: dict[str, tuple[SeedRef, ...]] = {}
    for state, entry in (states if isinstance(states, Mapping) else {}).items():
        refs = []
        for ref in entry.get("refs", []) if isinstance(entry, Mapping) else []:
            rel = ref.get("audio") if isinstance(ref, Mapping) else None
            match = clean_cmd.REF_REL.fullmatch(rel) if isinstance(rel, str) else None
            if match is None or match["state"] != state or state not in SPEAKING_STATES:
                raise InputError(f"refused: {run_dir}/{clean_cmd.REFS_NAME} names a bad reference")
            text_rel = clean_cmd.transcript_rel(state, match["n"])
            if not (root.path / target_dir / rel).is_file():
                raise InputError(f"refused: reference {target_dir}/{rel} is missing")
            text = root.path / target_dir / text_rel
            if ref.get("transcript") != text_rel or not text.is_file():
                raise InputError(
                    f"refused: reference {target_dir}/{rel} has no transcript beside it"
                )
            refs.append(
                SeedRef(
                    state=state,
                    audio=rel,
                    transcript=text_rel,
                    transcript_sha256=sha256_file(text),
                    duration_s=float(ref.get("duration_s", 0.0)),
                    sim={k: float(v) for k, v in dict(ref.get("sim", {})).items()},
                )
            )
        found[state] = tuple(refs)
    return found


# The job's plan.


def plan(
    *,
    target_id: str,
    seed: Seed | None,
    identity: identity_mod.Identity | None,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    """Return the job's plan: ids, root-relative audio paths, prosody and thresholds, no text."""
    segments = [
        {
            "segment_id": r.segment_id,
            "source_id": r.source_id,
            "state": _state(r),
            "audio": r.audio,
            "prosody": r.prosody.to_json(),
        }
        for r in (seed.segments if seed is not None else ())
    ]
    clips = [
        {"state": c.state, "audio": f"{identity_mod.voice_rel(identity.voice_id)}/{c.audio}"}
        for c in (identity.clips if identity is not None else ())
    ]
    return {
        "schema": SCHEMA,
        "target": target_id,
        "embedders": list(EMBEDDERS),
        "seed_run": seed.run_id if seed is not None else None,
        "segments": segments,
        "identity": identity.voice_id if identity is not None else None,
        "clips": clips,
        "thresholds": dict(thresholds),
    }


# After the job: references and `target.json`.


def _next_number(root: Root, target_dir: str, state: str) -> int:
    folder = root.path / target_dir / clean_cmd.REFS_DIR / state
    numbers = [
        int(m["n"])
        for p in (folder.iterdir() if folder.is_dir() else ())
        if (m := _REF_NUMBER.fullmatch(p.name))
    ]
    return max(numbers, default=0) + 1


def place_identity_ref(
    root: Root,
    target_id: str,
    identity: identity_mod.Identity,
    clip: identity_mod.IdentityClip,
    sim: Mapping[str, float],
) -> dict[str, object]:
    """Copy an identity clip and its transcript in as the state's next reference.

    It is numbered after T8's references of that state, which stay where they were. Both files
    are copied as bytes, and the copied transcript is hashed again to check the copy.

    Raises:
        InputError: The identity's transcript is missing or no longer matches its sha256.
    """
    target_dir = target_rel(target_id)
    number = _next_number(root, target_dir, clip.state)
    source_text = identity.path / clip.transcript
    if not source_text.is_file():
        raise InputError(f"refused: the identity's {clip.state} transcript is missing")
    text_rel = clean_cmd.transcript_rel(clip.state, number)
    audio_rel = f"{clean_cmd.REFS_DIR}/{clip.state}/{number}.wav"
    root.write_bytes(f"{target_dir}/{audio_rel}", (identity.path / clip.audio).read_bytes())
    written = root.write_bytes(f"{target_dir}/{text_rel}", source_text.read_bytes())
    digest = sha256_file(written)
    if digest != clip.transcript_sha256:
        raise InputError(f"refused: the identity's {clip.state} transcript changed during build")
    return {
        "audio": audio_rel,
        "transcript_sha256": digest,
        "duration_s": round(clip.duration_s, 3),
        "sim": {k: round(float(v), 6) for k, v in sim.items()},
        "from": FROM_IDENTITY,
    }


def _seed_ref_json(ref: SeedRef) -> dict[str, object]:
    return {
        "audio": ref.audio,
        "transcript_sha256": ref.transcript_sha256,
        "duration_s": round(ref.duration_s, 3),
        "sim": dict(ref.sim),
        "from": FROM_SEED,
    }


def read_build(root: Root, target_id: str) -> dict[str, object]:
    """Read the job's `build.json`.

    Raises:
        JobError: It is missing, malformed, or for another target.
    """
    path = root.path / target_rel(target_id) / BUILD_NAME
    try:
        doc = read_json(path)
    except InputError as exc:
        raise JobError(f"target {target_id}: {BUILD_NAME} is missing") from exc
    if (
        not isinstance(doc, dict)
        or doc.get("schema") != SCHEMA
        or doc.get("target") != target_id
        or doc.get("mode") not in (SEED, IDENTITY, SEED_IDENTITY)
        or not isinstance(doc.get("states"), dict)
    ):
        raise JobError(f"target {target_id}: {BUILD_NAME} is not this target's schema 1 build")
    return doc


def assemble(
    root: Root,
    *,
    target_id: str,
    build: Mapping[str, object],
    seed: Seed | None,
    identity: identity_mod.Identity | None,
    pins: Mapping[str, str],
) -> TargetRecord:
    """Place the references the mode calls for and return the target's record.

    In `seed` mode every state keeps T8's references. In `seed+identity` the identity's clip
    becomes its state's only reference and the other states keep T8's. In `identity` mode the
    identity's clips are the only references.
    """
    mode = str(build["mode"])
    sims = build.get("identity_sims")
    identity_sims: Mapping[str, Mapping[str, float]] = sims if isinstance(sims, Mapping) else {}
    states: dict[str, object] = {}
    for state, entry in build["states"].items():  # type: ignore[union-attr]
        clip = identity.clip(state) if identity is not None and mode != SEED else None
        if identity is not None and clip is not None:
            sim = identity_sims.get(state, {})
            refs = [place_identity_ref(root, target_id, identity, clip, sim)]
        elif seed is not None and mode != IDENTITY:
            refs = [_seed_ref_json(r) for r in seed.refs.get(state, ())]
        else:
            refs = []
        states[state] = {"refs": refs, **entry}
    # A seed target's id is its voice id, which `identity export` writes; otherwise the voice is
    # the identity's.
    voice_id = target_id if mode == SEED or identity is None else identity.voice_id
    return TargetRecord(
        schema=SCHEMA,
        target_id=target_id,
        voice_id=voice_id,
        mode=mode,
        seed_run=seed.run_id if seed is not None else None,
        identity=identity.voice_id if identity is not None else None,
        embedders=dict(pins),
        centroids=dict(build["centroids"]),  # type: ignore[arg-type]
        distribution=dict(build["distribution"]),  # type: ignore[arg-type]
        windows=dict(build["windows"]),  # type: ignore[arg-type]
        mode_check=tuple(build["mode_check"]),  # type: ignore[arg-type]
        states=states,
    )


# Output: numbers, ids and modes only.


def _num(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return "none" if value is None else str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return "?"


def _pairs(values: object, keys: Sequence[str]) -> str:
    table = values if isinstance(values, Mapping) else {}
    return ", ".join(f"{k} {_num(table.get(k))}" for k in keys)


def report_lines(record: TargetRecord) -> list[str]:
    """Return what `target build` prints: each clip's SIMs and floors, the mode, the numbers."""
    lines = [f"target {record.target_id}: mode {record.mode}, voice {record.voice_id}"]
    for check in record.mode_check:
        sim, floor = check.get("sim", {}), check.get("floor", {})
        parts = ", ".join(
            f"{name} {_num(sim.get(name))} floor {_num(floor.get(name))}" for name in floor
        )
        verdict = "pass" if check.get("pass") is True else "fail"
        lines.append(f"  clip {check.get('state')}: {parts}: {verdict}")
    dist = record.distribution
    lines.append(f"  distribution {dist.get('kind')}:")
    for name in record.embedders:
        lines.append(f"    {name}: {_pairs(dist.get(name), ('n', 'median', 'p02', 'p10'))}")
    for name in record.embedders:
        lines.append(
            f"  windows {name}: {_pairs(record.windows.get(name), ('n', 'median', 'p10'))}"
        )
    for state, entry in record.states.items():
        if not isinstance(entry, Mapping):
            continue
        floors = _pairs(entry.get("floor"), tuple(record.embedders))
        lines.append(
            f"  state {state}: n {_num(entry.get('n'))}, floor {entry.get('floor_from')} "
            f"({floors}), references {len(entry.get('refs', []))}"
        )
    lines.append(f"  wrote {target_rel(record.target_id)}/{TARGET_NAME}")
    return lines


# `identity export`: each state's best reference.


def _margin(ref: Mapping[str, object], medians: Mapping[str, float]) -> float:
    sim = ref.get("sim")
    if not isinstance(sim, Mapping) or not medians:
        return float("-inf")
    return min(float(sim.get(name, float("-inf"))) - m for name, m in medians.items())


def best_reference(
    refs: Sequence[Mapping[str, object]], distribution: Mapping[str, object]
) -> Mapping[str, object] | None:
    """Return the reference with the highest leave-one-out SIM, or `None` with no reference.

    The two embedders' SIMs sit on different scales, so each is taken as its margin over that
    embedder's leave-one-out median, and a reference ranks by its weaker embedder's margin.
    Ties go to the earlier reference (T8's order puts the prosody-nearest first).
    """
    medians = {
        name: float(entry["median"])
        for name, entry in distribution.items()
        if isinstance(entry, Mapping) and isinstance(entry.get("median"), int | float)
    }
    best: Mapping[str, object] | None = None
    for ref in refs:
        if best is None or _margin(ref, medians) > _margin(best, medians):
            best = ref
    return best


def read_target(root: Root, target_id: str) -> TargetRecord:
    """Read `targets/<target_id>/target.json`.

    Raises:
        InputError: It is missing or malformed.
    """
    path = root.path / target_rel(target_id) / TARGET_NAME
    if not path.is_file():
        raise InputError(f"refused: {target_rel(target_id)}/{TARGET_NAME} is missing")
    return TargetRecord.from_json(read_json(path))  # type: ignore[arg-type]


def export_clips(root: Root, record: TargetRecord) -> list[identity_mod.ExportClip]:
    """Return each state's best reference as an identity clip, in `SPEAKING_STATES` order."""
    target_dir = root.path / target_rel(record.target_id)
    clips: list[identity_mod.ExportClip] = []
    for state in SPEAKING_STATES:
        entry = record.states.get(state)
        refs = entry.get("refs", []) if isinstance(entry, Mapping) else []
        ref = best_reference(refs, record.distribution)
        if ref is None:
            continue
        match = clean_cmd.REF_REL.fullmatch(str(ref.get("audio")))
        if match is None or match["state"] != state:
            raise InputError(f"refused: target {record.target_id} names a bad {state} reference")
        clips.append(
            identity_mod.ExportClip(
                state=state,
                audio=target_dir / str(ref["audio"]),
                transcript=target_dir / clean_cmd.transcript_rel(state, match["n"]),
                transcript_sha256=str(ref.get("transcript_sha256")),
            )
        )
    return clips
