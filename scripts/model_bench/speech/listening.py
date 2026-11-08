"""Run 1's blind listening pack: the `listen-pack` subcommand (mist-model-host plan v2, task T5).

The pack is built only from clips `tts` already saved, in each engine's stock voices; it
synthesizes nothing. From each engine's timed pass with the model idle it takes repeat 1 of every
line and acknowledgement in the engine's timing voice (`qwen3`, `turbo`, `original` at its 0.5
settings, `pocket`), and from each engine's `listening.jsonl` every untimed clip: the
listening-only voices, any `--extra-voice` `tts` was given, and `original`'s `expressive` setting
on the emotive lines.

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
the labels from a committed constant. Stdlib only: the host runs it under a plain Python.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import secrets
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from . import core, docker_cmds, tts_suite
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

# Names that identify an engine family without being an engine id, voice or repository name.
FAMILY_TERMS = ("qwen", "chatterbox", "kyutai", "resemble", "resembleai")

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
    speech: Path, tts: Mapping[str, Mapping[str, object]], lines_doc: Mapping[str, object]
) -> list[Clip]:
    """Every clip the pack plays, in `speech_arms.json` engine order, timed clips first.

    Refuses an engine with no timed clips saved, an engine whose listening clips are due but not
    saved, a clip that is missing or outside its engine's directory, a voice or setting the
    engine does not list, and the same engine, setting, voice and line saved twice.
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


# ---------------------------------------------------------------------------
# Labels, groups and the page
# ---------------------------------------------------------------------------


def assign_labels(clips: Sequence[Clip], *, seed: int) -> dict[str, Clip]:
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


def blind_hits(text: str, tts: Mapping[str, Mapping[str, object]]) -> list[str]:
    """The forbidden names `text` contains as whole words, case-insensitively."""
    return sorted(
        term
        for term in forbidden_terms(tts)
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text, re.IGNORECASE)
    )


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


def pack_argv(*, repo_dir: str, speech_dir: str) -> list[str]:
    """`docker run` of the probe: base image, no network, no GPU, only `clips/` writable.

    The container sees the code, the saved `tts/` clips and the job list read-only, and the
    page's `clips/` directory; it is never given the key or a models directory.
    """
    argv = ["docker", "run", "--rm", "--network", "none"]
    argv += docker_cmds.env_args({"PYTHONDONTWRITEBYTECODE": "1"})
    argv += docker_cmds.ro_mount(repo_dir, docker_cmds.REPO_MOUNT)
    argv += docker_cmds.ro_mount(f"{speech_dir}/tts", f"{CONTAINER_SPEECH}/tts")
    argv += docker_cmds.ro_mount(f"{speech_dir}/{KEY_SUITE}/{JOBS_FILE}", CONTAINER_JOBS)
    argv += ["-v", f"{speech_dir}/{SUITE}/{CLIPS_DIR}:{CONTAINER_OUT}"]
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


def default_runner() -> tts_suite.ContainerRunner:
    return tts_suite.DockerRunner()


def _refuse_existing(speech: Path) -> None:
    existing = [p for p in (speech / SUITE, speech / KEY_SUITE) if p.exists()]
    if existing:
        raise core.OutputExistsError(
            f"{', '.join(str(p) for p in existing)} already exists; a pack is built once per run"
        )


def _key_doc(request: PackRequest, labelled: Mapping[str, Clip], git_sha: str) -> dict:
    return {
        "schema": 1,
        "run": request.run,
        "git_sha": git_sha,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": request.seed,
        "pack": pack_id(request.seed),
        "source_condition": SOURCE_CONDITION,
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
    `listening-key/`, a saved clip the collection refuses, and a page that would name an engine,
    setting or voice. The page is written last, so a failed container leaves no page to open.
    """
    tts = dict((arms if arms is not None else core.load_json(core.SPEECH_ARMS_PATH))["tts"])
    lines = lines_doc if lines_doc is not None else core.load_json(core.LINES_PATH)
    speech = core.speech_root(request.results_root, request.run, repo_root=repo_root)
    _refuse_existing(speech)
    clips = collect_clips(speech, tts, lines)
    if not clips:
        raise core.SpeechBenchError(f"tts saved no clips under {speech / 'tts'}")
    labelled = assign_labels(clips, seed=request.seed)
    groups = page_groups(labelled, lines, seed=request.seed)
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
    core.write_json_exclusive(key_dir / KEY_FILE, _key_doc(request, labelled, sha))
    jobs = [{"label": label, "src": clip.src} for label, clip in labelled.items()]
    core.write_json_exclusive(key_dir / JOBS_FILE, {"clips": jobs})
    (page_dir / CLIPS_DIR).mkdir()

    output: list[str] = []
    argv = pack_argv(repo_dir=repo_root.resolve().as_posix(), speech_dir=speech.as_posix())
    log_path = key_dir / LOG_FILE
    code = runner.run(argv, log_path=log_path, on_line=output.append)
    if code != 0:
        raise core.SpeechBenchError(f"listen-pack: the container exited {code}; log at {log_path}")
    reports = _reports(output)
    missing = [
        label
        for label in labelled
        if label not in reports or not (page_dir / CLIPS_DIR / f"{label}.wav").is_file()
    ]
    if missing:
        raise core.SpeechBenchError(
            f"listen-pack: the container produced no clip or report for {missing}; log at {log_path}"
        )
    with key_claim.jsonl(LOUDNESS_FILE) as writer:
        for label in labelled:
            writer.write(reports[label])

    path = page_dir / PAGE_FILE
    core.write_text_exclusive(path, page)
    off = [label for label in labelled if not reports[label].get("on_target")]
    if off:
        emit(
            f"[warn] {len(off)} clips are off the {listen_pack.TARGET_I:g} LUFS target by more "
            f"than {listen_pack.TOLERANCE_LU:g} LU: {off}; see {key_dir / LOUDNESS_FILE}"
        )
    dynamic = [label for label in labelled if reports[label].get("normalization_type") != "linear"]
    if dynamic:
        emit(f"[warn] ffmpeg fell back to dynamic loudnorm for {dynamic}")
    emit(
        f"listen-pack ok: {len(labelled)} clips over {len(groups)} lines; open {path}; the key "
        f"stays sealed in {key_dir} until {page_dir / RATINGS_FILE} exists"
    )
    return path


def read_key(speech: Path) -> dict:
    """The label-to-clip key, refused until Raj's `listening/ratings.csv` exists."""
    ratings = speech / SUITE / RATINGS_FILE
    if not ratings.is_file():
        raise KeySealedError(
            f"the key stays sealed until {ratings} exists; rate the clips on the page and save "
            f"its export as ratings.csv there first"
        )
    return core.load_json(speech / KEY_SUITE / KEY_FILE)


def cmd_listen_pack(args: argparse.Namespace) -> int:
    if args.show_key:
        speech = core.speech_root(args.results_root, args.run, repo_root=core.REPO_ROOT)
        print(json.dumps(read_key(speech), indent=2, ensure_ascii=False))
        return 0
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    request = PackRequest(results_root=args.results_root, run=args.run, seed=seed)
    build_pack(request, runner=default_runner())
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
        "--show-key",
        action="store_true",
        help="print the key instead of building; refused until listening/ratings.csv exists",
    )
    parser.set_defaults(func=cmd_listen_pack)


def container_argv(ctx: core.PlanContext) -> list[tuple[str, list[str]]]:
    return [("listen-pack", pack_argv(repo_dir=ctx.repo_dir, speech_dir=ctx.speech_dir))]


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
