"""Run 1's blind listening pack: the `listen-pack` subcommand (mist-model-host plan v2, task T5).

Without `--from-dir`, the pack is built only from clips `tts` already saved, in each engine's
stock voices; it synthesizes nothing. From each engine's timed pass with the model idle it takes repeat 1 of every
line and acknowledgement in the engine's timing voice (`qwen3`, `turbo`, `original` at its 0.5
settings, `pocket`), and from each engine's `listening.jsonl` every untimed clip: the
listening-only voices, any `--extra-voice` `tts` was given, and `original`'s `expressive` setting
on the emotive lines. `pocket-clone` (in its `identity` voice) and `qwen3-design` join the pack
when the results root holds their timed clips; without them the pack says it skips each and
builds from the four run-1 engines, which stay required.

Blinding. Each clip gets a label from a seeded permutation (`c001` to `cNNN`) and is copied,
loudness matched by the container probe, to `listening/clips/<label>.wav`. The page,
`listening/index.html`, groups clips by line in a seeded shuffled order and shows each line's
plain text (never Turbo's tagged text) and intended feeling. Everything that ties a label to an
engine, setting or voice (the key, the probe's job list, its log and the loudness measurements of
the source clips) goes to `listening-key/`, outside the page folder, and `read_key` refuses to
open the key until `listening/ratings.csv` exists. Before anything is written, the page text is
searched for every engine id, voice, setting, image, package and model repository name in
`speech_arms.json`, and the pack refuses to write a page that names one.

The default seed is fresh for every pack and recorded only in the key, so nobody can recompute
the labels from a committed constant.

`--from-dir <dir>` builds the same kind of pack from every WAV directly inside `<dir>` instead,
such as `voice-design`'s candidates: seeded labels, the same loudness probe, the key (label to
file name) in `listening-key/` and the same refusal of a page that names an arm. The directory
must lie outside the repository, and the probe's container sees only it, read-only, as its one
source directory. The page shows the clips as one block with no line text, since the pack knows
nothing about what they say.

`reply-ab --a <dir> --b <dir>` (mist-realtime-voice plan v6, T43) builds a blind page comparing
two `chain --reply-pack` folders, such as two models in the low-level role: one item per
utterance either pack answered, in `utterances.json` order, showing the line's words and both
replies under seeded labels in a seeded order. Both folders must lie outside the checkout. The
key (label to pack, repetition and text, and what each pack's meta says made it) goes to
`reply-ab-key/`, and `read_key` refuses it until `reply-ab/ratings.csv` exists. The page is
searched for every small model id, repository part, family, alias and placement, every speech
engine id, and each pack's own model, placement and engines, and refused if it names one. With
`--voice-clip`, the chain's speech service runs `pocket-clone` once, with that clip, and voices
both replies of every item; the voicings stay in the key folder and the listening probe
loudness-matches them onto the page, so the voice cannot tell the packs apart. Only the clip's
sha256 prefix is recorded. Stdlib only: the host runs it under a plain Python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import secrets
import tempfile
import wave
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TypeVar

from . import chain, core, docker_cmds, tts_suite
from .probes import listen_pack, tts_probe

SUITE = "listening"
KEY_SUITE = "listening-key"
PROBE_MODULE = "scripts.model_bench.speech.probes.listen_pack"
IMAGE = "mist-speech-bench-base"
TEMPLATE_PATH = core.SPEECH_DIR / "listening_template.html"
TEMPLATE_REL = core.SPEECH_DIR_REL / "listening_template.html"
PLACEHOLDER = "__PACK_DATA__"
PAGE_FILE = "index.html"
CLIPS_DIR = "clips"
KEY_FILE = "key.json"
JOBS_FILE = "jobs.json"
LOUDNESS_FILE = "loudness.jsonl"
LOG_FILE = "pack.log"
RATINGS_FILE = "ratings.csv"
RATINGS_COLUMNS = ("label", "line_id", "natural", "feeling", "preferred")
SOURCE_CONDITION = "idle"
ACK_FEELING = "short acknowledgement"
CONTAINER_SPEECH = "/speech"
CONTAINER_JOBS = "/jobs.json"
CONTAINER_OUT = "/out"

# `--from-dir`: where the directory appears under the probe's `tts/`, and the page's one block.
FROM_DIR_MOUNT = "from-dir"
FROM_DIR_GROUP = "all"
FROM_DIR_TEXT = "Every clip in this folder"
FROM_DIR_FEELING = "none given for these clips"

# Names that identify an engine family without being an engine id, voice or repository name.
FAMILY_TERMS = ("qwen", "chatterbox", "kyutai", "resemble", "resembleai")

# Arms a results root may lack: `pocket-clone` runs only with a clip and both came after run 1
# (mist-realtime-voice plan v6, T41). Every other arm's timed clips are required.
OPTIONAL_ARMS = frozenset({core.VOICE_CLIP_ENGINE, "qwen3-design"})

_DATA_RE = re.compile(
    r'<script id="pack-data" type="application/json">(?P<data>.*?)</script>', re.DOTALL
)


class KeySealedError(core.SpeechBenchError):
    """The key was asked for before `listening/ratings.csv` exists."""


# ---------------------------------------------------------------------------
# Collecting the saved clips
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Clip:
    """One saved clip: what made it, and where `tts` saved it relative to the speech root."""

    engine: str
    setting: str
    voice: str
    line_id: str
    kind: str
    rep: int
    condition: str | None
    src: str


def _rows(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise core.InputError(f"cannot read {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise core.InputError(f"{path} holds a row that is not a JSON object")
    return rows


def _checked_src(speech: Path, engine: str, raw: object) -> str:
    """`raw` as a path under `tts/<engine>/`, refusing anything else or a file that is missing."""
    rel = PurePosixPath(str(raw))
    if rel.is_absolute() or ".." in rel.parts or rel.parts[:2] != ("tts", engine):
        raise core.SpeechBenchError(f"{engine}: clip {raw!r} is outside tts/{engine}/")
    if not speech.joinpath(*rel.parts).is_file():
        raise core.SpeechBenchError(f"{engine}: saved clip {rel.as_posix()} is missing")
    return rel.as_posix()


def _clip(speech, engine, cfg, row, *, kind, condition, items, where) -> Clip:
    try:
        voice, setting, line_id = str(row["voice"]), str(row["setting"]), str(row["line_id"])
        wav = row["wav"]
    except KeyError as exc:
        raise core.InputError(f"{where}: a row lacks {exc}") from exc
    if voice not in [str(v) for v in cfg["voices"]]:
        raise core.SpeechBenchError(f"{where}: {engine} lists no stock voice {voice!r}")
    if setting not in (cfg.get("settings") or {}):
        raise core.SpeechBenchError(f"{where}: {engine} has no setting {setting!r}")
    if line_id not in items:
        raise core.SpeechBenchError(f"{where}: unknown line id {line_id!r}")
    src = _checked_src(speech, engine, wav)
    return Clip(engine, setting, voice, line_id, kind, 1, condition, src)


def collect_clips(
    speech: Path,
    tts: Mapping[str, Mapping[str, object]],
    lines_doc: Mapping[str, object],
    *,
    emit: Callable[[str], None] = lambda line: None,
) -> list[Clip]:
    """Every clip the pack plays, in `speech_arms.json` engine order, timed clips first.

    Refuses a run-1 engine with no timed clips saved, an engine whose listening clips are due but
    not saved, a clip that is missing or outside its engine's directory, a voice or setting the
    engine does not list, and the same engine, setting, voice and line saved twice. An arm in
    `OPTIONAL_ARMS` with no timed clips is left out with one `emit` line, so a results root from
    before those arms packs as it did.
    """
    acks = {str(a["id"]) for a in lines_doc["acknowledgements"]}
    items = [str(i["id"]) for i in [*lines_doc["lines"], *lines_doc["acknowledgements"]]]
    clips: list[Clip] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(clip: Clip) -> None:
        identity = (clip.engine, clip.setting, clip.voice, clip.line_id)
        if identity in seen:
            raise core.SpeechBenchError(
                f"{clip.engine}: {clip.setting} {clip.voice} {clip.line_id} is saved twice"
            )
        seen.add(identity)
        clips.append(clip)

    for engine, cfg in tts.items():
        base = speech / "tts" / engine
        timed = base / tts_probe.probe_file(SOURCE_CONDITION)
        if not timed.is_file() and engine in OPTIONAL_ARMS:
            emit(f"[skip] {engine}: tts saved no timed clips for it; the pack leaves it out")
            continue
        if not timed.is_file():
            raise core.SpeechBenchError(
                f"tts has saved no timed clips for {engine}: {timed} is missing; run tts first"
            )
        for row in _rows(timed):
            if row.get("row_type") != "sample" or row.get("rep") != 1:
                continue
            clip = _clip(
                speech,
                engine,
                cfg,
                row,
                kind="ack" if str(row.get("line_id")) in acks else "line",
                condition=SOURCE_CONDITION,
                items=items,
                where=str(timed),
            )
            if clip.voice != cfg["timing_voice"]:
                raise core.SpeechBenchError(
                    f"{timed}: timed in {clip.voice!r}, not the timing voice {cfg['timing_voice']!r}"
                )
            add(clip)

        untimed = base / tts_probe.LISTENING_FILE
        voices, expressive = tts_suite.render_plan(engine, cfg, ())
        if not untimed.is_file():
            if voices or expressive:
                raise core.SpeechBenchError(
                    f"tts has saved no listening clips for {engine}: {untimed} is missing"
                )
            continue
        for row in _rows(untimed):
            add(
                _clip(
                    speech,
                    engine,
                    cfg,
                    row,
                    kind="line",
                    condition=None,
                    items=items,
                    where=str(untimed),
                )
            )
    return clips


@dataclass(frozen=True, slots=True)
class DirClip:
    """One WAV of a `--from-dir` pack: its file name and its path as the probe sees it."""

    name: str
    src: str


def collect_dir_clips(from_dir: Path) -> list[DirClip]:
    """Every `.wav` directly inside `from_dir`, by name; refuses a directory holding none."""
    if not from_dir.is_dir():
        raise core.InputError(f"--from-dir {from_dir} is not a directory")
    names = sorted(p.name for p in from_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if not names:
        raise core.SpeechBenchError(f"--from-dir {from_dir} holds no WAV file")
    return [DirClip(name, f"tts/{FROM_DIR_MOUNT}/{name}") for name in names]


@dataclass(frozen=True, slots=True)
class ReplyClip:
    """One reply on a `reply-ab` page: its utterance, the pack it came from, its repetition."""

    item_id: str
    side: str
    rep: int
    text: str


_Labelled = TypeVar("_Labelled", Clip, DirClip, ReplyClip)


# ---------------------------------------------------------------------------
# Labels, groups and the page
# ---------------------------------------------------------------------------


def assign_labels(clips: Sequence[_Labelled], *, seed: int) -> dict[str, _Labelled]:
    """Label each clip from a seeded permutation of `c1` to `cN`; returned sorted by label.

    The labels are numbered for easy reading, and the permutation makes the number unrelated to
    the clip's engine, setting or voice.
    """
    width = len(str(len(clips)))
    numbers = list(range(1, len(clips) + 1))
    random.Random(f"{seed}:labels").shuffle(numbers)
    labelled = {f"c{n:0{width}d}": clip for n, clip in zip(numbers, clips, strict=True)}
    return dict(sorted(labelled.items()))


def page_groups(
    labelled: Mapping[str, Clip], lines_doc: Mapping[str, object], *, seed: int
) -> list[dict]:
    """One group per line that has clips, in `lines.json` order, its clips in a seeded order.

    Each group shows the plain line text, never Turbo's tagged text, and the line's intended
    feeling; an acknowledgement has none of its own, so it shows `ACK_FEELING`.
    """
    rng = random.Random(f"{seed}:order")
    groups = []
    for item in [*lines_doc["lines"], *lines_doc["acknowledgements"]]:
        line_id = str(item["id"])
        labels = [label for label, clip in labelled.items() if clip.line_id == line_id]
        if not labels:
            continue
        rng.shuffle(labels)
        groups.append(
            {
                "line_id": line_id,
                "text": str(item["text"]),
                "feeling": str(item.get("feeling") or ACK_FEELING),
                "clips": labels,
            }
        )
    return groups


def dir_groups(labelled: Mapping[str, DirClip], *, seed: int) -> list[dict]:
    """A `--from-dir` page's one block: every clip, in a seeded order, with no line text."""
    labels = list(labelled)
    random.Random(f"{seed}:order").shuffle(labels)
    return [
        {
            "line_id": FROM_DIR_GROUP,
            "text": FROM_DIR_TEXT,
            "feeling": FROM_DIR_FEELING,
            "clips": labels,
        }
    ]


def pack_id(seed: int) -> str:
    """The page's id for keeping ratings between sittings; derived from the seed, so stable."""
    return f"{random.Random(f'{seed}:pack').getrandbits(64):016x}"


def render_page(template: str, data: Mapping[str, object]) -> str:
    """The template with the page data in its JSON block; `</` is escaped so it cannot close it."""
    payload = json.dumps(data, ensure_ascii=True, indent=1).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, payload)


def page_data(html: str) -> dict:
    match = _DATA_RE.search(html)
    if match is None:
        raise core.InputError("the page holds no pack-data block")
    return json.loads(match.group("data"))


def forbidden_terms(tts: Mapping[str, Mapping[str, object]]) -> set[str]:
    """Every name that would tell a listener which engine, setting or voice made a clip."""
    terms = set(FAMILY_TERMS)
    for engine, cfg in tts.items():
        terms.add(engine)
        terms.update(str(v) for v in cfg.get("voices") or [])
        terms.update(str(s) for s in cfg.get("settings") or {})
        terms.update(str(p) for p in cfg.get("packages") or {})
        if cfg.get("image"):
            terms.add(str(cfg["image"]))
        model = cfg.get("model") or {}
        for field in ("repository", "gated_fallback"):
            terms.update(p for p in str(model.get(field, "")).split("/") if p)
    return {t.lower() for t in terms if t}


def term_hits(text: str, terms: Iterable[str]) -> list[str]:
    """The terms `text` contains as whole words, case-insensitively."""
    return sorted(
        term
        for term in set(terms)
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text, re.IGNORECASE)
    )


def blind_hits(text: str, tts: Mapping[str, Mapping[str, object]]) -> list[str]:
    """The forbidden names `text` contains as whole words, case-insensitively."""
    return term_hits(text, forbidden_terms(tts))


def load_template(path: Path = TEMPLATE_PATH) -> str:
    try:
        template = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise core.InputError(f"cannot read the page template {path}: {exc}") from exc
    if template.count(PLACEHOLDER) != 1:
        raise core.InputError(f"{path} must hold {PLACEHOLDER} exactly once")
    return template


# ---------------------------------------------------------------------------
# The container argv
# ---------------------------------------------------------------------------


def pack_argv(
    *,
    repo_dir: str,
    speech_dir: str,
    from_dir: str | None = None,
    suite: str = SUITE,
    key_suite: str = KEY_SUITE,
) -> list[str]:
    """`docker run` of the probe: base image, no network, no GPU, only `clips/` writable.

    The container sees the code, the saved `tts/` clips (or, with `from_dir`, only that
    directory, as `tts/from-dir/`) and the job list read-only, and the page's `clips/`
    directory; it is never given the key or a models directory. `suite` and `key_suite` name the
    page and key folders: `listening` by default, `reply-ab` for the reply page.
    """
    argv = ["docker", "run", "--rm", "--network", "none"]
    argv += docker_cmds.env_args({"PYTHONDONTWRITEBYTECODE": "1"})
    argv += docker_cmds.ro_mount(repo_dir, docker_cmds.REPO_MOUNT)
    if from_dir is None:
        argv += docker_cmds.ro_mount(f"{speech_dir}/tts", f"{CONTAINER_SPEECH}/tts")
    else:
        argv += docker_cmds.ro_mount(from_dir, f"{CONTAINER_SPEECH}/tts/{FROM_DIR_MOUNT}")
    argv += docker_cmds.ro_mount(f"{speech_dir}/{key_suite}/{JOBS_FILE}", CONTAINER_JOBS)
    argv += ["-v", f"{speech_dir}/{suite}/{CLIPS_DIR}:{CONTAINER_OUT}"]
    argv += ["-w", docker_cmds.REPO_MOUNT, IMAGE, "python", "-m", PROBE_MODULE]
    argv += ["--jobs", CONTAINER_JOBS, "--speech", CONTAINER_SPEECH, "--out", CONTAINER_OUT]
    return argv


# ---------------------------------------------------------------------------
# Building the pack and reading the key
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PackRequest:
    """One `listen-pack` invocation."""

    results_root: Path
    run: str
    seed: int
    from_dir: Path | None = None


def default_runner() -> tts_suite.ContainerRunner:
    return tts_suite.DockerRunner()


def _refuse_existing(speech: Path, suite: str = SUITE, key_suite: str = KEY_SUITE) -> None:
    existing = [p for p in (speech / suite, speech / key_suite) if p.exists()]
    if existing:
        raise core.OutputExistsError(
            f"{', '.join(str(p) for p in existing)} already exists; a pack is built once per run"
        )


def _key_doc(
    request: PackRequest,
    labelled: Mapping[str, Clip] | Mapping[str, DirClip],
    git_sha: str,
    *,
    from_dir: Path | None = None,
) -> dict:
    if from_dir is None:
        source = {"source_condition": SOURCE_CONDITION}
    else:
        source = {"from_dir": str(from_dir)}
    return {
        "schema": 1,
        "run": request.run,
        "git_sha": git_sha,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": request.seed,
        "pack": pack_id(request.seed),
        **source,
        "target": {
            "integrated_lufs": listen_pack.TARGET_I,
            "true_peak_dbtp": listen_pack.TARGET_TP,
            "lra_lu": listen_pack.TARGET_LRA,
            "tolerance_lu": listen_pack.TOLERANCE_LU,
            "sample_rate": listen_pack.OUTPUT_RATE,
        },
        "clips": [{"label": label, **asdict(clip)} for label, clip in labelled.items()],
    }


def _reports(lines: Sequence[str]) -> dict[str, dict]:
    reports = {}
    for line in lines:
        if line.startswith(listen_pack.REPORT_PREFIX):
            row = json.loads(line[len(listen_pack.REPORT_PREFIX) :])
            reports[str(row["label"])] = row
    return reports


def _match_loudness(
    labels: Sequence[str],
    argv: list[str],
    *,
    runner: tts_suite.ContainerRunner,
    key_claim: core.OutputClaim,
    page_dir: Path,
    what: str,
) -> dict[str, dict]:
    """Run the probe, check it made a clip and a report per label, and keep the reports.

    The reports (the source clips' loudness) go to the key folder, never the page's.
    """
    output: list[str] = []
    log_path = key_claim.path / LOG_FILE
    code = runner.run(argv, log_path=log_path, on_line=output.append)
    if code != 0:
        raise core.SpeechBenchError(f"{what}: the container exited {code}; log at {log_path}")
    reports = _reports(output)
    missing = [
        label
        for label in labels
        if label not in reports or not (page_dir / CLIPS_DIR / f"{label}.wav").is_file()
    ]
    if missing:
        raise core.SpeechBenchError(
            f"{what}: the container produced no clip or report for {missing}; log at {log_path}"
        )
    with key_claim.jsonl(LOUDNESS_FILE) as writer:
        for label in labels:
            writer.write(reports[label])
    return {label: reports[label] for label in labels}


def _loudness_warnings(
    reports: Mapping[str, Mapping[str, object]], key_dir: Path, emit: Callable[[str], None]
) -> None:
    off = [label for label, report in reports.items() if not report.get("on_target")]
    if off:
        emit(
            f"[warn] {len(off)} clips are off the {listen_pack.TARGET_I:g} LUFS target by more "
            f"than {listen_pack.TOLERANCE_LU:g} LU: {off}; see {key_dir / LOUDNESS_FILE}"
        )
    dynamic = [
        label for label, report in reports.items() if report.get("normalization_type") != "linear"
    ]
    if dynamic:
        emit(f"[warn] ffmpeg fell back to dynamic loudnorm for {dynamic}")


def build_pack(
    request: PackRequest,
    *,
    runner: tts_suite.ContainerRunner,
    arms: Mapping[str, object] | None = None,
    lines_doc: Mapping[str, object] | None = None,
    git_sha: str | None = None,
    repo_root: Path = core.REPO_ROOT,
    template_path: Path = TEMPLATE_PATH,
    emit: Callable[[str], None] = print,
) -> Path:
    """Build the page and its sealed key; return the page's path.

    Refuses before any write: a results root inside the repository, an existing `listening/` or
    `listening-key/`, a saved clip the collection refuses (with `from_dir`: a directory inside the
    repository or one holding no WAV), and a page that would name an engine, setting or voice.
    The page is written last, so a failed container leaves no page to open.
    """
    tts = dict((arms if arms is not None else core.load_json(core.SPEECH_ARMS_PATH))["tts"])
    speech = core.speech_root(request.results_root, request.run, repo_root=repo_root)
    from_dir = None
    if request.from_dir is not None:
        from_dir = core.refuse_path_inside_repo(
            request.from_dir, what="--from-dir", repo_root=repo_root
        )
    _refuse_existing(speech)
    if from_dir is None:
        lines = lines_doc if lines_doc is not None else core.load_json(core.LINES_PATH)
        clips = collect_clips(speech, tts, lines, emit=emit)
        if not clips:
            raise core.SpeechBenchError(f"tts saved no clips under {speech / 'tts'}")
        labelled = assign_labels(clips, seed=request.seed)
        groups = page_groups(labelled, lines, seed=request.seed)
    else:
        labelled = assign_labels(collect_dir_clips(from_dir), seed=request.seed)
        groups = dir_groups(labelled, seed=request.seed)
    data = {
        "pack": pack_id(request.seed),
        "columns": list(RATINGS_COLUMNS),
        "ratings_file": RATINGS_FILE,
        "groups": groups,
    }
    page = render_page(load_template(template_path), data)
    hits = blind_hits(page, tts)
    if hits:
        raise core.SpeechBenchError(f"the page would name {hits}; refusing to write it")

    sha = git_sha or core.current_git_sha(repo_root)
    key_claim = core.claim_output(
        request.results_root, request.run, KEY_SUITE, git_sha=sha, repo_root=repo_root
    )
    page_claim = core.claim_output(
        request.results_root, request.run, SUITE, git_sha=sha, repo_root=repo_root
    )
    key_dir, page_dir = key_claim.path, page_claim.path
    key = _key_doc(request, labelled, sha, from_dir=from_dir)
    core.write_json_exclusive(key_dir / KEY_FILE, key)
    jobs = [{"label": label, "src": clip.src} for label, clip in labelled.items()]
    core.write_json_exclusive(key_dir / JOBS_FILE, {"clips": jobs})
    (page_dir / CLIPS_DIR).mkdir()

    argv = pack_argv(
        repo_dir=repo_root.resolve().as_posix(),
        speech_dir=speech.as_posix(),
        from_dir=from_dir.as_posix() if from_dir is not None else None,
    )
    reports = _match_loudness(
        list(labelled),
        argv,
        runner=runner,
        key_claim=key_claim,
        page_dir=page_dir,
        what="listen-pack",
    )

    path = page_dir / PAGE_FILE
    core.write_text_exclusive(path, page)
    _loudness_warnings(reports, key_dir, emit)
    emit(
        f"listen-pack ok: {len(labelled)} clips over {len(groups)} lines; open {path}; the key "
        f"stays sealed in {key_dir} until {page_dir / RATINGS_FILE} exists"
    )
    return path


def read_key(speech: Path, *, suite: str = SUITE, key_suite: str = KEY_SUITE) -> dict:
    """The label-to-clip key, refused until Raj's `<suite>/ratings.csv` exists.

    `suite` and `key_suite` default to the listening pack's; `reply-ab` passes its own, so only
    the reply page's ratings open the reply key.
    """
    ratings = speech / suite / RATINGS_FILE
    if not ratings.is_file():
        raise KeySealedError(
            f"the key stays sealed until {ratings} exists; rate the clips on the page and save "
            f"its export as ratings.csv there first"
        )
    return core.load_json(speech / key_suite / KEY_FILE)


def cmd_listen_pack(args: argparse.Namespace) -> int:
    if args.show_key:
        speech = core.speech_root(args.results_root, args.run, repo_root=core.REPO_ROOT)
        print(json.dumps(read_key(speech), indent=2, ensure_ascii=False))
        return 0
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    request = PackRequest(
        results_root=args.results_root, run=args.run, seed=seed, from_dir=args.from_dir
    )
    build_pack(request, runner=default_runner())
    return 0


# ---------------------------------------------------------------------------
# reply-ab: a blind A/B page of two `chain --reply-pack` directories
# ---------------------------------------------------------------------------

REPLY_SUITE = "reply-ab"
REPLY_KEY_SUITE = "reply-ab-key"
REPLY_RATINGS_COLUMNS = ("label", "item_id", "quality", "preferred")
VOICED_DIR = "voiced"
SIDES = ("a", "b")
# Model families a reply could name itself by, beyond each model's id and repository parts.
MODEL_FAMILY_TERMS = ("gemma", "minicpm", "qwen")
# What a pack's meta says made it; each value is kept off the page.
PACK_META_FIELDS = ("model", "placement", "alias", "stt_engine", "tts_engine", "busy_arm")

# The reply page. Each item shows what was said and two replies under their labels, with a
# player per reply when the replies are voiced; ratings stay in the browser until exported.
REPLY_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind reply comparison</title>
<style>
  body {
    font-family: system-ui, sans-serif;
    line-height: 1.45;
    color: #1d1d1f;
    background: #f6f6f4;
    max-width: 62rem;
    margin: 0 auto;
    padding: 0 1rem 4rem;
  }
  header {
    position: sticky;
    top: 0;
    z-index: 1;
    background: #f6f6f4;
    border-bottom: 1px solid #d8d8d4;
    padding: 0.75rem 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.5rem;
    align-items: center;
  }
  header h1 { font-size: 1.3rem; margin: 0; }
  #progress { margin: 0; color: #444; }
  button { font: inherit; padding: 0.35rem 0.9rem; cursor: pointer; }
  .intro { max-width: 46rem; }
  section.item {
    background: #fff;
    border: 1px solid #d8d8d4;
    border-radius: 6px;
    padding: 0.75rem 1rem 1rem;
    margin: 1.25rem 0;
  }
  section.item h2 { font-size: 0.95rem; font-weight: 600; color: #666; margin: 0; }
  .said { font-size: 1.2rem; margin: 0.35rem 0 0.75rem; }
  .reply { border-top: 1px solid #eee; padding: 0.5rem 0; }
  .label { font-family: ui-monospace, monospace; font-weight: 600; }
  .reply.complete .label::after { content: " \2713"; color: #2a7d4f; }
  .text { margin: 0.25rem 0; white-space: pre-wrap; }
  .scale label { white-space: nowrap; margin-right: 0.5rem; cursor: pointer; }
  audio { height: 2rem; max-width: 16rem; display: block; margin: 0.25rem 0; }
</style>
</head>
<body>
<header>
  <h1>Blind reply comparison</h1>
  <p id="progress"></p>
  <button id="export" type="button">Export ratings.csv</button>
</header>
<main>
  <div class="intro">
    <p>Each block shows one thing a person said, then two replies to it, under random labels
    and in a shuffled order. Where the replies have a player, both are read in the same voice
    at the same loudness, so judge what each reply says rather than how it sounds.</p>
    <p>Rate each reply's <strong>quality</strong>: how good a reply it is to what was said,
    taking in whether it is right, helpful and natural to hear (1 poor, 5 excellent). Then pick
    the better reply in each block.</p>
    <p>Ratings are kept in this browser between sittings. When every reply is rated, export and
    save the file as <code>ratings.csv</code> in the folder that holds this page.</p>
  </div>
  <div id="items"></div>
</main>
<script id="pack-data" type="application/json">__PACK_DATA__</script>
<script>
"use strict";

const DATA = JSON.parse(document.getElementById("pack-data").textContent);
const STORE = "reply-pack-" + DATA.pack;
const TOTAL = DATA.items.reduce((n, item) => n + item.replies.length, 0);
const state = restore();

function restore() {
  let saved = {};
  try {
    saved = JSON.parse(window.localStorage.getItem(STORE) || "{}");
  } catch (err) {
    saved = {};
  }
  return { quality: saved.quality || {}, preferred: saved.preferred || {} };
}

function persist() {
  try {
    window.localStorage.setItem(STORE, JSON.stringify(state));
  } catch (err) {
    // No storage on this page: ratings last until the tab closes, so export before closing.
  }
}

function el(tag, attrs, text) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs || {})) {
    node.setAttribute(name, value);
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

function scale(label) {
  const box = el("div", { class: "scale" });
  box.append("Quality: ");
  for (let value = 1; value <= 5; value++) {
    const id = "quality-" + label + "-" + value;
    const input = el("input", { type: "radio", name: "quality-" + label, id: id });
    input.checked = state.quality[label] === value;
    input.addEventListener("change", () => {
      state.quality[label] = value;
      persist();
      refresh();
    });
    const tag = el("label", { for: id });
    tag.append(input, " " + value);
    box.append(tag);
  }
  return box;
}

function better(item, label) {
  const id = "preferred-" + label;
  const input = el("input", { type: "radio", name: "preferred-" + item.item_id, id: id });
  input.checked = state.preferred[item.item_id] === label;
  input.addEventListener("change", () => {
    state.preferred[item.item_id] = label;
    persist();
    refresh();
  });
  const tag = el("label", { for: id });
  tag.append(input, " the better reply");
  return tag;
}

function render() {
  const container = document.getElementById("items");
  DATA.items.forEach((item, index) => {
    const section = el("section", { class: "item" });
    section.append(el("h2", {}, "Block " + (index + 1) + " of " + DATA.items.length));
    section.append(el("p", { class: "said" }, "“" + item.text + "”"));
    for (const reply of item.replies) {
      const box = el("div", { class: "reply", id: "reply-" + reply.label });
      box.append(el("span", { class: "label" }, reply.label));
      box.append(el("p", { class: "text" }, reply.text));
      if (DATA.voiced) {
        const src = "clips/" + reply.label + ".wav";
        box.append(el("audio", { controls: "", preload: "none", src: src }));
      }
      box.append(scale(reply.label), better(item, reply.label));
      section.append(box);
    }
    container.append(section);
  });
}

function refresh() {
  let done = 0;
  for (const item of DATA.items) {
    for (const reply of item.replies) {
      const complete = Boolean(state.quality[reply.label]);
      done += complete ? 1 : 0;
      document.getElementById("reply-" + reply.label).classList.toggle("complete", complete);
    }
  }
  const picked = DATA.items.filter((item) => state.preferred[item.item_id]).length;
  document.getElementById("progress").textContent =
    done + " of " + TOTAL + " replies rated; the better reply picked in " + picked + " of " +
    DATA.items.length + " blocks";
}

function cell(value) {
  const text = value === undefined || value === null ? "" : String(value);
  return /[",\r\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}

function ratingsCsv() {
  const rows = [DATA.columns.join(",")];
  for (const item of DATA.items) {
    for (const reply of item.replies) {
      const row = {
        label: reply.label,
        item_id: item.item_id,
        quality: state.quality[reply.label],
        preferred: state.preferred[item.item_id] === reply.label ? 1 : 0,
      };
      rows.push(DATA.columns.map((column) => cell(row[column])).join(","));
    }
  }
  return rows.join("\n") + "\n";
}

function exportRatings() {
  const done = DATA.items.reduce(
    (n, item) => n + item.replies.filter((reply) => state.quality[reply.label]).length, 0);
  if (done < TOTAL && !window.confirm((TOTAL - done) + " replies are not rated. Export anyway?")) {
    return;
  }
  const url = URL.createObjectURL(new Blob([ratingsCsv()], { type: "text/csv" }));
  const link = el("a", { href: url, download: DATA.ratings_file });
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// One reply at a time: starting one pauses any other that is playing.
document.addEventListener("play", (event) => {
  for (const audio of document.querySelectorAll("audio")) {
    if (audio !== event.target) {
      audio.pause();
    }
  }
}, true);

document.getElementById("export").addEventListener("click", exportRatings);
render();
refresh();
</script>
</body>
</html>
"""


@dataclass(frozen=True, slots=True)
class ReplyPack:
    """One `chain --reply-pack` directory: where it is, what made it, its rows by utterance."""

    path: Path
    meta: dict
    rows: dict[str, list[dict]]


@dataclass(frozen=True, slots=True)
class ReplyRequest:
    """One `reply-ab` invocation: the two packs, compared blind."""

    results_root: Path
    run: str
    seed: int
    a: Path
    b: Path


@dataclass(frozen=True, slots=True)
class ReplyVoice:
    """The speech service that voices every reply in one cloned voice, and its client.

    The clip's path lives only in the service's container argv; the key keeps its sha256 prefix.
    """

    services: chain.Services
    client: chain.TtsClient
    sha256_prefix: str


def load_reply_pack(
    path: Path, *, what: str, utterance_ids: Iterable[str], repo_root: Path = core.REPO_ROOT
) -> ReplyPack:
    """Read a reply pack, refusing one inside the checkout, a directory that is not one, and a
    row for an utterance `utterances.json` does not hold.
    """
    resolved = core.refuse_path_inside_repo(path, what=what, repo_root=repo_root)
    meta_path = resolved / chain.REPLY_PACK_META
    replies_path = resolved / chain.REPLY_PACK_FILE
    if not meta_path.is_file() or not replies_path.is_file():
        raise core.InputError(
            f"{what} {resolved} is not a reply pack: it needs {chain.REPLY_PACK_META} and "
            f"{chain.REPLY_PACK_FILE}, as chain --reply-pack writes them"
        )
    meta = core.load_json(meta_path)
    if not isinstance(meta, dict):
        raise core.InputError(f"{meta_path} is not a JSON object")
    known = set(utterance_ids)
    rows: dict[str, list[dict]] = {}
    for row in _rows(replies_path):
        uid = str(row.get("utt_id"))
        if uid not in known:
            raise core.InputError(f"{replies_path}: {uid!r} is not in utterances.json")
        rows.setdefault(uid, []).append(row)
    for uid_rows in rows.values():
        uid_rows.sort(key=lambda r: int(r.get("rep") or 0))
    return ReplyPack(resolved, meta, rows)


def first_reply(rows: Sequence[Mapping[str, object]]) -> tuple[int, str] | None:
    """The first repetition with a reply to show, and its text; None when none has one."""
    for row in rows:
        text = row.get("reply")
        if isinstance(text, str) and text.strip():
            return int(row.get("rep") or 0), text
    return None


def collect_replies(
    packs: Mapping[str, ReplyPack], utterance_ids: Iterable[str]
) -> tuple[list[ReplyClip], list[dict]]:
    """Both packs' replies to each utterance either pack holds, in `utterances.json` order.

    An utterance one pack has no reply for (every repetition failed, or it was not run) is left
    out and listed as skipped, so each item compares exactly two replies.
    """
    clips: list[ReplyClip] = []
    skipped: list[dict] = []
    for uid in utterance_ids:
        if not any(uid in pack.rows for pack in packs.values()):
            continue
        found = {side: first_reply(pack.rows.get(uid, [])) for side, pack in packs.items()}
        missing = [side for side, got in found.items() if got is None]
        if missing:
            skipped.append({"item_id": uid, "missing": missing})
            continue
        clips += [ReplyClip(uid, side, *got) for side, got in found.items() if got is not None]
    return clips, skipped


def reply_items(
    labelled: Mapping[str, ReplyClip], texts: Mapping[str, str], *, seed: int
) -> list[dict]:
    """One item per utterance, in `utterances.json` order: what was said and both replies in a
    seeded order, each under its label only.
    """
    rng = random.Random(f"{seed}:order")
    items = []
    for uid, said in texts.items():
        labels = [label for label, clip in labelled.items() if clip.item_id == uid]
        if not labels:
            continue
        rng.shuffle(labels)
        replies = [{"label": label, "text": labelled[label].text} for label in labels]
        items.append({"item_id": uid, "text": said, "replies": replies})
    return items


def reply_forbidden_terms(
    packs: Iterable[ReplyPack],
    *,
    small_arms: Mapping[str, object],
    speech_arms: Mapping[str, object],
) -> set[str]:
    """Every name that would tell a reader which model, placement or engine made a reply.

    The small models' ids, repository parts, families and aliases, every placement, every speech
    engine id, and whatever each pack's meta says made it.
    """
    terms = set(MODEL_FAMILY_TERMS)
    placements = list(small_arms.get("placements") or {})
    terms.update(placements)
    for model_id, model in (small_arms.get("models") or {}).items():
        terms.add(model_id)
        terms.update(core.small_alias(model_id, placement) for placement in placements)
        terms.update(p for p in str(model.get("repository", "")).split("/") if p)
    for section in ("stt", "tts"):
        terms.update(speech_arms.get(section) or {})
    for pack in packs:
        terms.update(str(pack.meta[f]) for f in PACK_META_FIELDS if pack.meta.get(f))
    return {t.lower() for t in terms if t}


def write_pcm_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    """16-bit mono PCM as a new WAV file, refusing to overwrite one."""
    try:
        handle = path.open("xb")
    except FileExistsError as exc:
        raise core.OutputExistsError(f"{path} already exists; refusing to overwrite it") from exc
    with handle, wave.open(handle, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm)


def reply_voice(clip: core.VoiceClip, *, speech_models_dir: Path) -> ReplyVoice:
    """`pocket-clone` behind the chain's speech service, with `clip` mounted read-only."""
    services = chain.ChainServices.speech_only(
        repo_dir=core.REPO_ROOT.resolve().as_posix(),
        speech_models_dir=speech_models_dir.resolve().as_posix(),
        engine=core.VOICE_CLIP_ENGINE,
        voice_clip=clip.path.as_posix(),
    )
    return ReplyVoice(services, chain.TtsClient(chain.TTS_URL), clip.sha256_prefix)


def _reply_key_doc(
    request: ReplyRequest,
    packs: Mapping[str, ReplyPack],
    labelled: Mapping[str, ReplyClip],
    skipped: list[dict],
    voice: ReplyVoice | None,
    git_sha: str,
) -> dict:
    sides = {side: {"dir": str(pack.path), "meta": pack.meta} for side, pack in packs.items()}
    return {
        "schema": 1,
        "run": request.run,
        "git_sha": git_sha,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": request.seed,
        "pack": pack_id(request.seed),
        **sides,
        "voice": (
            None
            if voice is None
            else {"engine": core.VOICE_CLIP_ENGINE, "voice_clip_sha256": voice.sha256_prefix}
        ),
        "replies": [{"label": label, **asdict(clip)} for label, clip in labelled.items()],
        "skipped": skipped,
    }


def _voice_replies(
    labelled: Mapping[str, ReplyClip],
    *,
    voice: ReplyVoice,
    runner: tts_suite.ContainerRunner,
    speech: Path,
    key_claim: core.OutputClaim,
    page_dir: Path,
    repo_root: Path,
) -> dict[str, dict]:
    """Voice every reply with the one service, then loudness-match them all onto the page.

    The raw voicings stay in the key folder; the probe sees only them, read-only, and writes the
    matched clips under the page's `clips/`.
    """
    key_dir = key_claim.path
    voiced = key_dir / VOICED_DIR
    voiced.mkdir()
    try:
        voice.services.start()
        for label, clip in labelled.items():
            pcm, rate = voice.client.render(clip.text)
            write_pcm_wav(voiced / f"{label}.wav", pcm, rate)
    finally:
        voice.services.stop(key_dir)
    jobs = [{"label": label, "src": f"tts/{FROM_DIR_MOUNT}/{label}.wav"} for label in labelled]
    core.write_json_exclusive(key_dir / JOBS_FILE, {"clips": jobs})
    (page_dir / CLIPS_DIR).mkdir()
    argv = pack_argv(
        repo_dir=repo_root.resolve().as_posix(),
        speech_dir=speech.as_posix(),
        from_dir=voiced.as_posix(),
        suite=REPLY_SUITE,
        key_suite=REPLY_KEY_SUITE,
    )
    return _match_loudness(
        list(labelled),
        argv,
        runner=runner,
        key_claim=key_claim,
        page_dir=page_dir,
        what="reply-ab",
    )


def build_reply_ab(
    request: ReplyRequest,
    *,
    runner: tts_suite.ContainerRunner,
    voice: ReplyVoice | None = None,
    utterances_doc: Mapping[str, object] | None = None,
    small_arms: Mapping[str, object] | None = None,
    speech_arms: Mapping[str, object] | None = None,
    git_sha: str | None = None,
    repo_root: Path = core.REPO_ROOT,
    emit: Callable[[str], None] = print,
) -> Path:
    """Build the blind reply page and its sealed key; return the page's path.

    Refuses before any write: a results root or pack inside the checkout, a directory that is not
    a reply pack, an existing `reply-ab/` or `reply-ab-key/`, packs with no utterance answered in
    both, and a page that would name a model, placement or engine. With `voice`, both replies of
    every item are voiced by the one service and loudness-matched by the listening probe; the
    page is written last, so a failed voicing or container leaves no page to open.
    """
    speech = core.speech_root(request.results_root, request.run, repo_root=repo_root)
    doc = utterances_doc if utterances_doc is not None else core.load_json(core.UTTERANCES_PATH)
    texts = {str(u["id"]): str(u["text"]) for u in doc["utterances"]}
    packs = {
        side: load_reply_pack(path, what=f"--{side}", utterance_ids=texts, repo_root=repo_root)
        for side, path in zip(SIDES, (request.a, request.b), strict=True)
    }
    _refuse_existing(speech, REPLY_SUITE, REPLY_KEY_SUITE)
    clips, skipped = collect_replies(packs, texts)
    for entry in skipped:
        emit(
            f"[skip] {entry['item_id']}: pack {', '.join(entry['missing'])} holds no reply to "
            f"it; the page leaves it out"
        )
    if not clips:
        raise core.SpeechBenchError("the two reply packs answer no utterance in common")
    labelled = assign_labels(clips, seed=request.seed)
    items = reply_items(labelled, texts, seed=request.seed)
    data = {
        "pack": pack_id(request.seed),
        "columns": list(REPLY_RATINGS_COLUMNS),
        "ratings_file": RATINGS_FILE,
        "voiced": voice is not None,
        "items": items,
    }
    page = render_page(REPLY_TEMPLATE, data)
    if small_arms is None:
        small_arms = core.load_small_arms()
    if speech_arms is None:
        speech_arms = core.load_json(core.SPEECH_ARMS_PATH)
    terms = reply_forbidden_terms(packs.values(), small_arms=small_arms, speech_arms=speech_arms)
    hits = term_hits(page, terms)
    if hits:
        raise core.SpeechBenchError(f"the page would name {hits}; refusing to write it")

    sha = git_sha or core.current_git_sha(repo_root)
    key_claim = core.claim_output(
        request.results_root, request.run, REPLY_KEY_SUITE, git_sha=sha, repo_root=repo_root
    )
    page_claim = core.claim_output(
        request.results_root, request.run, REPLY_SUITE, git_sha=sha, repo_root=repo_root
    )
    key_dir, page_dir = key_claim.path, page_claim.path
    key = _reply_key_doc(request, packs, labelled, skipped, voice, sha)
    core.write_json_exclusive(key_dir / KEY_FILE, key)
    reports = None
    if voice is not None:
        reports = _voice_replies(
            labelled,
            voice=voice,
            runner=runner,
            speech=speech,
            key_claim=key_claim,
            page_dir=page_dir,
            repo_root=repo_root,
        )

    path = page_dir / PAGE_FILE
    core.write_text_exclusive(path, page)
    if reports is not None:
        _loudness_warnings(reports, key_dir, emit)
    voiced = ", voiced in one cloned voice" if voice is not None else ""
    emit(
        f"reply-ab ok: {len(items)} items, {len(labelled)} replies{voiced}; open {path}; the key "
        f"stays sealed in {key_dir} until {page_dir / RATINGS_FILE} exists"
    )
    return path


def cmd_reply_ab(args: argparse.Namespace) -> int:
    speech = core.speech_root(args.results_root, args.run, repo_root=core.REPO_ROOT)
    if args.show_key:
        key = read_key(speech, suite=REPLY_SUITE, key_suite=REPLY_KEY_SUITE)
        print(json.dumps(key, indent=2, ensure_ascii=False))
        return 0
    clip = None
    if args.voice_clip is not None:
        try:
            clip = core.check_voice_clip(args.voice_clip)
        except core.VoiceClipError as exc:
            print(f"[REFUSED] {exc}")
            return 2
        if args.speech_models_dir is None:
            raise core.SpeechBenchError(
                f"--voice-clip needs --speech-models-dir (or {tts_suite.SPEECH_MODELS_DIR_ENV}) "
                f"for {core.VOICE_CLIP_ENGINE}'s weights"
            )
    if args.a is None or args.b is None:
        raise core.SpeechBenchError("reply-ab needs --a and --b, two chain --reply-pack folders")
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    request = ReplyRequest(args.results_root, args.run, seed, args.a, args.b)
    voice = None
    if clip is not None:
        voice = reply_voice(clip, speech_models_dir=args.speech_models_dir)
    build_reply_ab(request, runner=default_runner(), voice=voice)
    return 0


# ---------------------------------------------------------------------------
# Registry contract: register, container_argv, validate_inputs, selftest
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "listen-pack", help="build the blind listening page from run 1's saved clips"
    )
    core.add_results_args(parser)
    parser.add_argument(
        "--seed",
        type=int,
        help="seed of the labels and the clip order (default: fresh, recorded only in the key)",
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        help="build the pack from every WAV in this directory (outside the repository) instead",
    )
    parser.add_argument(
        "--show-key",
        action="store_true",
        help="print the key instead of building; refused until listening/ratings.csv exists",
    )
    parser.set_defaults(func=cmd_listen_pack)

    reply = subparsers.add_parser(
        "reply-ab", help="build a blind A/B page of two chain --reply-pack folders"
    )
    core.add_results_args(reply)
    reply.add_argument(
        "--a", type=Path, help="one chain --reply-pack folder (outside the checkout)"
    )
    reply.add_argument("--b", type=Path, help="the other chain --reply-pack folder")
    reply.add_argument(
        "--seed",
        type=int,
        help="seed of the labels and the reply order (default: fresh, recorded only in the key)",
    )
    reply.add_argument(
        "--voice-clip",
        type=Path,
        help=f"voice both replies of every item with {core.VOICE_CLIP_ENGINE} cloning this "
        "identity WAV, outside the checkout",
    )
    env_models = os.environ.get(tts_suite.SPEECH_MODELS_DIR_ENV)
    reply.add_argument(
        "--speech-models-dir",
        type=Path,
        default=Path(env_models) if env_models else None,
        help="host speech models directory, for --voice-clip "
        f"(env {tts_suite.SPEECH_MODELS_DIR_ENV})",
    )
    reply.add_argument(
        "--show-key",
        action="store_true",
        help="print the key instead of building; refused until reply-ab/ratings.csv exists",
    )
    reply.set_defaults(func=cmd_reply_ab)


def container_argv(ctx: core.PlanContext) -> list[tuple[str, list[str]]]:
    voiced = f"{ctx.speech_dir}/{REPLY_KEY_SUITE}/{VOICED_DIR}"
    reply_probe = pack_argv(
        repo_dir=ctx.repo_dir,
        speech_dir=ctx.speech_dir,
        from_dir=voiced,
        suite=REPLY_SUITE,
        key_suite=REPLY_KEY_SUITE,
    )
    speech_service = chain.tts_service_argv(
        repo_dir=ctx.repo_dir,
        speech_models_dir=ctx.speech_models_dir,
        engine=core.VOICE_CLIP_ENGINE,
        voice_clip=ctx.voice_clip,
    )
    return [
        ("listen-pack", pack_argv(repo_dir=ctx.repo_dir, speech_dir=ctx.speech_dir)),
        ("reply-ab", reply_probe),
        ("reply-ab speech-service", speech_service),
    ]


def validate_inputs(repo_root: Path) -> list[str]:
    """Check the page template and what the page shows; an empty list means usable."""
    speech = repo_root / core.SPEECH_DIR_REL
    try:
        tts = dict(core.load_json(speech / "speech_arms.json")["tts"])
        lines_doc = core.load_json(speech / "lines.json")
        template = load_template(repo_root / TEMPLATE_REL)
    except (core.InputError, KeyError, TypeError) as exc:
        return [f"listening inputs unreadable: {exc}"]
    problems = []
    hits = blind_hits(template, tts)
    if hits:
        problems.append(f"{TEMPLATE_REL.as_posix()} names {hits}")
    for needed in ("DATA.columns.join(", "DATA.columns.map(", 'id="pack-data"'):
        if needed not in template:
            problems.append(f"{TEMPLATE_REL.as_posix()} lacks {needed!r}")
    for item in [*lines_doc.get("lines", []), *lines_doc.get("acknowledgements", [])]:
        shown = f"{item.get('text', '')} {item.get('feeling', '')}"
        hits = blind_hits(shown, tts)
        if hits:
            problems.append(f"lines.json: {item.get('id')} would show {hits} on the page")
    return problems + _reply_problems(repo_root, tts)


def _reply_problems(repo_root: Path, tts: Mapping[str, object]) -> list[str]:
    """The reply page's template and every chain utterance it shows name no model or engine."""
    speech = repo_root / core.SPEECH_DIR_REL
    try:
        arms = core.load_json(speech / "speech_arms.json")
        small_arms = core.load_json(speech / "small_arms.json")
        utterances = core.load_json(speech / "utterances.json")["utterances"]
    except (core.InputError, KeyError, TypeError) as exc:
        return [f"reply-ab inputs unreadable: {exc}"]
    terms = reply_forbidden_terms([], small_arms=small_arms, speech_arms=arms)
    problems = []
    hits = term_hits(REPLY_TEMPLATE, terms) + blind_hits(REPLY_TEMPLATE, tts)
    if hits:
        problems.append(f"the reply page template names {hits}")
    if REPLY_TEMPLATE.count(PLACEHOLDER) != 1:
        problems.append(f"the reply page template must hold {PLACEHOLDER} exactly once")
    for needed in ("DATA.columns.join(", "DATA.columns.map(", 'id="pack-data"'):
        if needed not in REPLY_TEMPLATE:
            problems.append(f"the reply page template lacks {needed!r}")
    for item in utterances:
        if str(item.get("id")) in chain.CHAIN_UTTERANCES:
            hits = term_hits(str(item.get("text", "")), terms)
            if hits:
                problems.append(f"utterances.json: {item.get('id')} would show {hits}")
    return problems


def _check(condition: bool, message: str) -> None:
    # Explicit raise rather than `assert`, which `python -O` strips.
    if not condition:
        raise AssertionError(message)


def selftest() -> None:
    """Check labels, grouping, the loudnorm argv and report, the meter, blinding and the seal."""
    clips = [
        Clip(e, "default", "v", line, "line", 1, "idle", f"tts/{e}/a/{line}.wav")
        for e in ("x", "y")
        for line in ("L01", "L02", "L03")
    ]
    first = assign_labels(clips, seed=1)
    _check(set(first) == {f"c{i}" for i in range(1, 7)}, f"labels {sorted(first)}")
    _check(first == assign_labels(clips, seed=1), "labels differ for one seed")
    _check(set(first.values()) == set(clips), "labels lose a clip")
    doc = {"lines": [{"id": "L01", "text": "t", "feeling": "f"}], "acknowledgements": []}
    groups = page_groups(first, doc, seed=1)
    _check([g["line_id"] for g in groups] == ["L01"] and len(groups[0]["clips"]) == 2, "groups")

    measured = {
        "input_i": -27.6,
        "input_tp": -4.5,
        "input_lra": 6.0,
        "input_thresh": -38.2,
        "target_offset": 0.1,
    }
    first_pass = listen_pack.measure_argv("in.wav")
    second_pass = listen_pack.apply_argv("in.wav", "out.wav", measured)
    _check(first_pass[-3:] == ["-f", "null", "-"], "pass 1 writes audio")
    _check("measured_I=-27.60" in " ".join(second_pass), "pass 2 lacks pass 1's measurement")
    _check("linear=true" in " ".join(second_pass), "pass 2 is not linear")
    report = '[Parsed_loudnorm_0 @ 0x1] \n{"input_i" : "-27.6", "normalization_type" : "linear"'
    report += "".join(f', "{k}" : "1.0"' for k in listen_pack._REPORT_FIELDS if k != "input_i")
    _check(listen_pack.parse_loudnorm(report + "}")["input_i"] == -27.6, "report parse")

    rate = 16000
    sine = [0.1 * math.sin(2 * math.pi * 997 * i / rate) for i in range(rate)]
    lufs = listen_pack.integrated_loudness(sine, rate)
    _check(abs(lufs - (-23.01)) < 0.15, f"a -20 dBFS 997 Hz tone measured {lufs:.2f} LUFS")

    tts = core.load_json(core.SPEECH_ARMS_PATH)["tts"]
    _check(blind_hits("heard Aiden here", tts) == ["aiden"], "the blind check missed a voice")
    _check(blind_hits(load_template(), tts) == [], "the page template names an engine")

    with tempfile.TemporaryDirectory() as scratch:
        speech = Path(scratch)
        (speech / KEY_SUITE).mkdir()
        (speech / SUITE).mkdir()
        (speech / KEY_SUITE / KEY_FILE).write_text("{}", encoding="utf-8")
        try:
            read_key(speech)
        except KeySealedError:
            pass
        else:
            raise AssertionError("the key was read before ratings.csv existed")
        (speech / SUITE / RATINGS_FILE).write_text(",".join(RATINGS_COLUMNS), encoding="utf-8")
        _check(read_key(speech) == {}, "the key was not readable once ratings.csv existed")

    argv = pack_argv(repo_dir="/repo", speech_dir="/s")
    _check("--gpus" not in argv and argv[argv.index("--network") + 1] == "none", "pack argv")

    packs = {
        side: ReplyPack(Path(side), {}, {"U01": [{"rep": 1, "reply": f"reply {side}"}]})
        for side in SIDES
    }
    replies, skipped = collect_replies(packs, ["U01", "U02"])
    _check([(r.item_id, r.side) for r in replies] == [("U01", "a"), ("U01", "b")], "reply items")
    _check(skipped == [], "a reply both packs hold was skipped")
    _check(REPLY_TEMPLATE.count(PLACEHOLDER) == 1, "the reply template's data block")
    with tempfile.TemporaryDirectory() as scratch:
        speech = Path(scratch)
        (speech / REPLY_KEY_SUITE).mkdir()
        (speech / SUITE).mkdir()
        (speech / REPLY_KEY_SUITE / KEY_FILE).write_text("{}", encoding="utf-8")
        (speech / SUITE / RATINGS_FILE).write_text("label\n", encoding="utf-8")
        try:
            read_key(speech, suite=REPLY_SUITE, key_suite=REPLY_KEY_SUITE)
        except KeySealedError:
            pass
        else:
            raise AssertionError("the listening ratings opened the reply key")
