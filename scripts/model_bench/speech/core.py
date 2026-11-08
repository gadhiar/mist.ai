"""Shared pieces of the speech bench (mist-model-host plan v2, task T1).

Every speech suite builds on these: the error classes, the results-root check (bench_host's),
`<results-root>/<run>/speech/` resolution, output claims and the JSONL writer, percentiles, the
material manifest check, the pin-file check, the deterministic prefix builder of runs 2 and 7, the
bench-run window helpers, a streaming client for llama-server's OpenAI chat endpoint, and
`small-serve`'s argv and lifecycle. Stdlib only: container probes import this module too, with
the repository mounted at `/work:ro`, and the host runs it under a plain Python.

Refusals raise before anything is written, so a refused command leaves no partial output that a
retry would then trip over.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .. import bench_host
from ..probes import help_flags
from . import docker_cmds

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEECH_DIR = Path(__file__).resolve().parent
SPEECH_DIR_REL = Path("scripts") / "model_bench" / "speech"
LINES_PATH = SPEECH_DIR / "lines.json"
UTTERANCES_PATH = SPEECH_DIR / "utterances.json"
SPEECH_ARMS_PATH = SPEECH_DIR / "speech_arms.json"
SMALL_ARMS_PATH = SPEECH_DIR / "small_arms.json"
CACHE_FILLER_PATH = SPEECH_DIR / "cache_filler.txt"
SPEECH_FIXTURES_DIR = REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "speech"
HELP_CAPTURE_PATH = (
    REPO_ROOT
    / "tests"
    / "unit"
    / "model_bench"
    / "fixtures"
    / "host"
    / "llama_server_help_b11151.txt"
)
SYSTEM_PROMPT_REL = Path("config") / "system_prompt.txt"
TEST_VAULT_REL = Path("tests") / "fixtures" / "test-vault"

RESULTS_ROOT_ENV = "MODEL_BENCH_RESULTS_ROOT"
PREFIX_TOLERANCE = 25


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

# Re-exported so suites catch one family: every speech error is a `BenchHostError`, which
# `speech_bench.main` reports as `[FAIL]` with a non-zero exit.
BenchHostError = bench_host.BenchHostError
ResultsRootError = bench_host.ResultsRootError
DockerError = bench_host.DockerError
check_results_root_outside_repo = bench_host.check_results_root_outside_repo


class SpeechBenchError(bench_host.BenchHostError):
    """Base class for speech bench refusals and failures."""


class InputError(SpeechBenchError):
    """A committed input file is missing or malformed."""


class OutputExistsError(SpeechBenchError):
    """An output directory or file already exists; results are never overwritten."""


class ManifestError(SpeechBenchError):
    """A material file is missing or its sha256 differs from the manifest."""


class PinError(SpeechBenchError):
    """A model file is missing from the pin file, or differs from it in size or sha256."""


class BenchRunError(SpeechBenchError):
    """A `--bench-run` id is reused, or its session snapshot is missing."""


class PlacementError(SpeechBenchError):
    """A small model or placement `small-serve` cannot serve as asked."""


class RepoPathError(SpeechBenchError):
    """An output path that must live outside the repository resolves inside it."""


class StreamError(SpeechBenchError):
    """A chat completion stream failed: an HTTP error, a bad event or a server error event."""


class PrefixError(SpeechBenchError):
    """The prefix builder could not land within tolerance of its token target."""


class ServeFailedError(SpeechBenchError):
    """`mist-bench-small` did not come up; `log_path` holds its saved logs when there are any."""

    def __init__(self, message: str, *, log_path: Path | None = None):
        super().__init__(message)
        self.log_path = log_path


# ---------------------------------------------------------------------------
# JSON inputs
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise InputError(f"{path} does not hold a JSON object")
    return doc


def load_small_arms(path: Path = SMALL_ARMS_PATH) -> dict:
    return load_json(path)


# ---------------------------------------------------------------------------
# Names, results root and output claims
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def validate_name(value: str, what: str = "run") -> str:
    """Refuse anything but one plain path segment, so an id can never climb out of its root."""
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise SpeechBenchError(
            f"{what} {value!r} is not a plain name (letters, digits, '.', '_', '-')"
        )
    return value


def speech_root(results_root: Path, run: str, *, repo_root: Path = REPO_ROOT) -> Path:
    """`<results-root>/<run>/speech/`, after refusing a results root inside the repository."""
    check_results_root_outside_repo(results_root, repo_root)
    validate_name(run, "run")
    return results_root.resolve() / run / "speech"


def refuse_path_inside_repo(path: Path, *, what: str, repo_root: Path = REPO_ROOT) -> Path:
    """Return `path` resolved, refusing one inside the repository (results are never committed)."""
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved
    raise RepoPathError(f"{what} {resolved} is inside the repository {root}; refusing")


def claim_dir(path: Path) -> Path:
    """Create `path` as a new directory, refusing one that already exists.

    `mkdir` without `exist_ok` is the claim itself, so two runs racing for the same directory
    cannot both win.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise OutputExistsError(
            f"{path} already exists; pick a new run or arm instead of overwriting"
        ) from exc
    return path


class JsonlWriter:
    """Writes one JSON object per line, each stamped with the run, suite and git sha.

    The file is opened exclusively unless `append` is set; append mode exists for files several
    arms share (such as `tts/samples.jsonl`), and is reached through `OutputClaim.shared_jsonl`,
    after the arm's own directory has been claimed.
    """

    def __init__(self, path: Path, *, stamp: Mapping[str, str], append: bool = False):
        self.path = path
        self._stamp = dict(stamp)
        try:
            self._handle = path.open("a" if append else "x", encoding="utf-8", newline="\n")
        except FileExistsError as exc:
            raise OutputExistsError(f"{path} already exists; refusing to overwrite it") from exc

    def write(self, row: Mapping[str, object]) -> None:
        """Append `row` with the stamp fields, refusing a row whose own value contradicts one."""
        for key, value in self._stamp.items():
            if key in row and row[key] != value:
                raise SpeechBenchError(
                    f"row sets {key}={row[key]!r}, which contradicts this file's {key}={value!r}"
                )
        record = dict(self._stamp)
        record.update(row)
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the file."""
        self._handle.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class OutputClaim:
    """A claimed suite or suite/arm output directory and the stamp its rows carry."""

    path: Path
    suite_dir: Path
    run: str
    suite: str
    arm: str | None
    git_sha: str

    def stamp(self) -> dict[str, str]:
        """The fields every row written through this claim carries."""
        return {"run": self.run, "suite": self.suite, "git_sha": self.git_sha}

    def jsonl(self, name: str) -> JsonlWriter:
        """An exclusive JSONL file inside the claimed directory."""
        return JsonlWriter(self.path / validate_name(name, "file name"), stamp=self.stamp())

    def shared_jsonl(self, name: str) -> JsonlWriter:
        """An append-mode JSONL file in the suite directory, shared by the suite's arms."""
        target = self.suite_dir / validate_name(name, "file name")
        return JsonlWriter(target, stamp=self.stamp(), append=True)


def claim_output(
    results_root: Path,
    run: str,
    suite: str,
    arm: str | None = None,
    *,
    git_sha: str,
    repo_root: Path = REPO_ROOT,
) -> OutputClaim:
    """Claim `<speech>/<suite>/` or `<speech>/<suite>/<arm>/`; `arm` may hold `/` segments."""
    root = speech_root(results_root, run, repo_root=repo_root)
    suite_dir = root / validate_name(suite, "suite")
    path = suite_dir
    if arm is not None:
        for segment in arm.split("/"):
            path = path / validate_name(segment, "arm")
    claim_dir(path)
    return OutputClaim(path, suite_dir, run, suite, arm, git_sha)


def current_git_sha(repo_root: Path = REPO_ROOT) -> str:
    """HEAD of `repo_root`, or `unknown` where git or its metadata is unavailable."""
    try:
        head = bench_host.git_head(repo_root)
    except OSError:
        return "unknown"
    return head or "unknown"


def write_json_exclusive(path: Path, doc: Mapping[str, object]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(doc, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise OutputExistsError(f"{path} already exists; refusing to overwrite it") from exc


def write_text_exclusive(path: Path, text: str) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise OutputExistsError(f"{path} already exists; refusing to overwrite it") from exc


def add_results_args(parser: argparse.ArgumentParser, *, bench_run: bool = False) -> None:
    """Add `--results-root` and `--run` (and `--bench-run`) to a subcommand that writes.

    `--results-root` falls back to `MODEL_BENCH_RESULTS_ROOT`, as bench_host's does.
    """
    env_root = os.environ.get(RESULTS_ROOT_ENV)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(env_root) if env_root else None,
        required=env_root is None,
        help=f"results root outside the repository (env {RESULTS_ROOT_ENV})",
    )
    parser.add_argument("--run", required=True, help="speech results id, e.g. mh1")
    if bench_run:
        parser.add_argument(
            "--bench-run",
            required=True,
            help="bench_host run id of this stop-and-restore window, e.g. mh1-h4",
        )


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Summary:
    """A sample count with its median and nearest-rank 90th percentile (the slow case)."""

    n: int
    median: float | None
    p90: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        """The summary as a JSON-ready mapping."""
        return {"n": self.n, "median": self.median, "p90": self.p90}


def nearest_rank(values: Iterable[float], pct: int) -> float | None:
    """The nearest-rank percentile: the smallest value with at least `pct`% of values at or below.

    Integer arithmetic for the rank, so 90% of 10 values is exactly rank 9.
    """
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, -(-pct * len(ordered) // 100))
    return ordered[rank - 1]


def summarize(values: Iterable[float]) -> Summary:
    data = list(values)
    if not data:
        return Summary(0, None, None)
    return Summary(len(data), statistics.median(data), nearest_rank(data, 90))


# ---------------------------------------------------------------------------
# Material manifest
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MaterialItem:
    """One recording in the material manifest."""

    id: str
    file: str
    kind: str
    text: str
    duration_s: float
    sha256: str


@dataclass(frozen=True, slots=True)
class Manifest:
    """The material manifest `prepare` writes beside the recordings."""

    path: Path
    items: tuple[MaterialItem, ...]


def load_manifest(path: Path) -> Manifest:
    """Read `manifest.json`: `{"items": [{id, file, kind, text, duration_s, sha256}, ...]}`.

    `file` must be a plain name in the material directory, so a manifest cannot point a suite at
    a file elsewhere on the disk.
    """
    try:
        doc = load_json(path)
    except InputError as exc:
        raise ManifestError(str(exc)) from exc
    raw_items = doc.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ManifestError(f"{path} has no items")
    items = []
    for raw in raw_items:
        try:
            item = MaterialItem(
                id=str(raw["id"]),
                file=str(raw["file"]),
                kind=str(raw["kind"]),
                text=str(raw["text"]),
                duration_s=float(raw["duration_s"]),
                sha256=str(raw["sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"{path}: malformed item {raw!r}: {exc}") from exc
        if not _NAME_RE.fullmatch(item.file):
            raise ManifestError(f"{path}: item {item.id} file {item.file!r} is not a plain name")
        if not _SHA256_RE.fullmatch(item.sha256):
            raise ManifestError(f"{path}: item {item.id} sha256 is not 64 lowercase hex")
        items.append(item)
    return Manifest(path, tuple(items))


def verify_material(manifest_path: Path, material_dir: Path | None = None) -> Manifest:
    """Check every manifest file exists with its sha256; refuse with every mismatch named."""
    manifest = load_manifest(manifest_path)
    directory = material_dir if material_dir is not None else manifest_path.parent
    problems = []
    for item in manifest.items:
        path = directory / item.file
        if not path.is_file():
            problems.append(f"{item.file}: missing")
            continue
        actual = sha256_file(path)
        if actual != item.sha256:
            problems.append(
                f"{item.file}: sha256 {actual} differs from the manifest's {item.sha256}"
            )
    if problems:
        raise ManifestError(f"material does not match {manifest_path}: " + "; ".join(problems))
    return manifest


# ---------------------------------------------------------------------------
# Pin file
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelPin:
    """One model file as host step H2 pinned it."""

    id: str
    repository: str
    revision: str
    file: str
    sha256: str
    size: int


def load_pins(path: Path) -> dict:
    """Read the run's pin file: `{"models": {<id>: {repository, revision, file, sha256, size}}}`."""
    try:
        doc = load_json(path)
    except InputError as exc:
        raise PinError(str(exc)) from exc
    if not isinstance(doc.get("models"), dict):
        raise PinError(f"{path} has no models table")
    return doc


def pinned_model(pins: Mapping[str, object], model_id: str) -> ModelPin:
    models = pins.get("models")
    entry = models.get(model_id) if isinstance(models, dict) else None
    if entry is None:
        raise PinError(f"the pin file lists no model {model_id!r}")
    try:
        pin = ModelPin(
            id=model_id,
            repository=str(entry["repository"]),
            revision=str(entry["revision"]),
            file=str(entry["file"]),
            sha256=str(entry["sha256"]),
            size=int(entry["size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PinError(f"pin for {model_id!r} is malformed: {exc}") from exc
    parts = Path(pin.file).parts
    if not parts or Path(pin.file).is_absolute() or any(p in ("..", ".") for p in parts):
        raise PinError(f"pin for {model_id!r} has file {pin.file!r}, not a relative path")
    if not _SHA256_RE.fullmatch(pin.sha256):
        raise PinError(f"pin for {model_id!r} has a sha256 that is not 64 lowercase hex")
    return pin


def verify_model_file(pin: ModelPin, path: Path) -> None:
    """Refuse a model file that differs from its pin: size first (cheap), then sha256."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PinError(f"model file for {pin.id} at {path} cannot be read: {exc}") from exc
    if size != pin.size:
        raise PinError(f"{path}: size {size} differs from the pin file's {pin.size} for {pin.id}")
    actual = sha256_file(path)
    if actual != pin.sha256:
        raise PinError(f"{path}: sha256 {actual} differs from the pin file's {pin.sha256}")


# ---------------------------------------------------------------------------
# Prefix builder (runs 2 and 7)
# ---------------------------------------------------------------------------


def prefix_source_paths(repo_root: Path = REPO_ROOT) -> list[Path]:
    """The prefix's sources in order: the system prompt, then the test vault by relative path."""
    vault = repo_root / TEST_VAULT_REL
    notes = sorted(vault.rglob("*.md"), key=lambda p: p.relative_to(vault).as_posix())
    return [repo_root / SYSTEM_PROMPT_REL, *notes]


def _read_normalised(path: Path) -> str:
    # A Windows checkout may hold CRLF; the prefix must be the same bytes on every machine.
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()


def build_prefix(
    token_target: int,
    tokenize: Callable[[str], Sequence[int]],
    *,
    repo_root: Path = REPO_ROOT,
    filler_path: Path = CACHE_FILLER_PATH,
    tolerance: int = PREFIX_TOLERANCE,
) -> str:
    """Build the fixed, non-personal prefix of `token_target` tokens, within `tolerance` below.

    The text is the system prompt, every test-vault note in sorted relative-path order, and the
    filler paragraph, then the filler repeated until the target is passed; it is then cut at the
    last word boundary that keeps it at or under the target. `tokenize` is the server's
    `/tokenize` in a run and a fake in tests; the search assumes that a longer text never has
    fewer tokens. The result depends only on the committed files and `tokenize`, so every call
    returns the same bytes.
    """
    if token_target <= 0:
        raise PrefixError(f"token target must be positive, got {token_target}")
    filler = _read_normalised(filler_path)
    parts = [_read_normalised(p) for p in prefix_source_paths(repo_root)]
    text = "\n\n".join([*parts, filler])

    def count(candidate: str) -> int:
        return len(tokenize(candidate))

    filler_tokens = max(1, count(filler))
    n_tokens = count(text)
    while n_tokens < token_target:
        repeats = max(1, -(-(token_target - n_tokens) // filler_tokens))
        text += ("\n\n" + filler) * repeats
        n_tokens = count(text)

    if n_tokens > token_target:
        cuts = [m.start() for m in re.finditer(r"\s+", text)]
        best = -1
        lo, hi = 0, len(cuts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if count(text[: cuts[mid]]) <= token_target:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best < 0:
            raise PrefixError(f"no word boundary keeps the prefix under {token_target} tokens")
        text = text[: cuts[best]]
        n_tokens = count(text)

    if token_target - n_tokens > tolerance:
        raise PrefixError(
            f"prefix lands at {n_tokens} tokens, more than {tolerance} under {token_target}"
        )
    return text


# ---------------------------------------------------------------------------
# Bench-run windows
# ---------------------------------------------------------------------------


def begin_bench_window(results_root: Path, bench_run: str, *, repo_root: Path = REPO_ROOT) -> Path:
    """Refuse a `--bench-run` whose `session/` exists; return that unused session path.

    bench_host's `snapshot` refuses to overwrite a snapshot, and `serve` and `run` read the image
    from it, so each stop-and-restore window needs an id of its own, used once.
    """
    check_results_root_outside_repo(results_root, repo_root)
    validate_name(bench_run, "bench run")
    session = results_root.resolve() / bench_run / "session"
    if session.exists():
        raise BenchRunError(
            f"--bench-run {bench_run}: {session} already exists; a window's id is used once, "
            f"so start this window under a new id"
        )
    return session


def bench_run_snapshot(results_root: Path, bench_run: str, *, repo_root: Path = REPO_ROOT) -> Path:
    """The window's `session/snapshot.json`, which the image check and `vram-step` read."""
    check_results_root_outside_repo(results_root, repo_root)
    validate_name(bench_run, "bench run")
    snapshot = results_root.resolve() / bench_run / "session" / "snapshot.json"
    if not snapshot.is_file():
        raise BenchRunError(
            f"no snapshot at {snapshot}; run `bench_host snapshot --run {bench_run}` first"
        )
    return snapshot


# ---------------------------------------------------------------------------
# Streaming chat client (llama-server's OpenAI /v1/chat/completions)
# ---------------------------------------------------------------------------

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class ChatStreamResult:
    """One streamed chat completion, with every timestamp taken from the injected clock."""

    content: str
    reasoning_content: str
    tool_calls: tuple[dict, ...]
    t_request: float
    t_first_content: float | None
    t_first_tool_call: float | None
    t_end: float
    finish_reason: str | None
    timings: dict | None
    usage: dict | None = None
    model: str | None = None

    @property
    def t_first_delta(self) -> float | None:
        """The first content or tool-call delta; reasoning text does not count as a reply."""
        seen = [t for t in (self.t_first_content, self.t_first_tool_call) if t is not None]
        return min(seen) if seen else None

    @property
    def ttft_s(self) -> float | None:
        """Seconds from sending the request to the first content or tool-call delta."""
        first = self.t_first_delta
        return None if first is None else first - self.t_request


def parse_chat_stream(
    lines: Iterable[str | bytes], *, clock: Clock, t_request: float
) -> ChatStreamResult:
    """Parse server-sent events from `/v1/chat/completions`.

    `clock` is read once per `data:` event, as it arrives; comment and blank lines are skipped.
    Tool calls are assembled by index from their streamed pieces. The last `timings` object seen
    is kept: llama-server sends it on the final chunk, with `cache_n`, `prompt_n` and `prompt_ms`.
    """
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict] = {}
    t_first_content: float | None = None
    t_first_tool: float | None = None
    t_end = t_request
    finish_reason: str | None = None
    timings: dict | None = None
    usage: dict | None = None
    model: str | None = None

    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        now = clock()
        t_end = now
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StreamError(f"malformed stream event {payload[:200]!r}: {exc}") from exc
        if not isinstance(event, dict):
            raise StreamError(f"stream event is not an object: {payload[:200]!r}")
        if "error" in event:
            error = event["error"]
            message = error.get("message", error) if isinstance(error, dict) else error
            raise StreamError(f"server error event: {message}")
        if isinstance(event.get("timings"), dict):
            timings = event["timings"]
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        if event.get("model"):
            model = str(event["model"])
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                content.append(piece)
                if t_first_content is None:
                    t_first_content = now
            thought = delta.get("reasoning_content")
            if thought:
                reasoning.append(thought)
            for call_delta in delta.get("tool_calls") or []:
                if t_first_tool is None:
                    t_first_tool = now
                index = int(call_delta.get("index", len(calls)))
                call = calls.setdefault(
                    index,
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if call_delta.get("id"):
                    call["id"] = call_delta["id"]
                if call_delta.get("type"):
                    call["type"] = call_delta["type"]
                function = call_delta.get("function") or {}
                if function.get("name"):
                    call["function"]["name"] += function["name"]
                if function.get("arguments"):
                    call["function"]["arguments"] += function["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    return ChatStreamResult(
        content="".join(content),
        reasoning_content="".join(reasoning),
        tool_calls=tuple(calls[i] for i in sorted(calls)),
        t_request=t_request,
        t_first_content=t_first_content,
        t_first_tool_call=t_first_tool,
        t_end=t_end,
        finish_reason=finish_reason,
        timings=timings,
        usage=usage,
        model=model,
    )


def stream_chat(
    base_url: str,
    body: Mapping[str, object],
    *,
    clock: Clock = time.perf_counter,
    timeout: float = 300.0,
    opener: Callable[..., object] | None = None,
) -> ChatStreamResult:
    """POST `body` to `<base_url>/v1/chat/completions` with `stream` on and parse the reply.

    The request time is read from `clock` just before sending. `opener` defaults to
    `urllib.request.urlopen`; tests pass a fake. A status other than 200 raises `StreamError`.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {**body, "stream": True}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    open_fn = opener or urllib.request.urlopen
    t_request = clock()
    try:
        with open_fn(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status != 200:
                raise StreamError(f"POST {url} returned HTTP {status}")
            return parse_chat_stream(response, clock=clock, t_request=t_request)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise StreamError(f"POST {url} returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise StreamError(f"POST {url} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# small-serve and small-unserve
# ---------------------------------------------------------------------------

SMALL_CONTAINER = "mist-bench-small"
SMALL_PORT = 8081
SMALL_BASE_URL = f"http://{docker_cmds.LOCALHOST}:{SMALL_PORT}"
LOCAL_PLACEMENTS: tuple[str, ...] = ("gpu-full", "gpu-4g", "gpu-2g", "cpu")
HOST1070_PLACEMENTS: tuple[str, ...] = ("gtx1070-shared", "host1070-cpu", "gtx1070")
WARMUP_BODY: dict = {
    "messages": [{"role": "user", "content": "Say hello in three words."}],
    "max_tokens": 16,
}

_SAMPLING_FLAGS = (
    ("temperature", "--temp"),
    ("top_p", "--top-p"),
    ("top_k", "--top-k"),
    ("min_p", "--min-p"),
)


def small_alias(model_id: str, placement: str) -> str:
    return f"{model_id}@{placement}"


def _small_model(arms_doc: Mapping[str, object], model_id: str) -> dict:
    models = arms_doc["models"]
    if model_id not in models:
        raise PlacementError(f"unknown small model {model_id!r}; known: {list(models)}")
    return models[model_id]


def _local_placement(arms_doc: Mapping[str, object], placement: str) -> dict:
    placements = arms_doc["placements"]
    cfg = placements.get(placement)
    if cfg is None:
        raise PlacementError(f"unknown placement {placement!r}; known: {list(placements)}")
    if cfg.get("where") != "local":
        raise PlacementError(
            f"placement {placement!r} is served on host-1070 by its own script, not by small-serve"
        )
    return cfg


def small_serve_argv(
    arms_doc: Mapping[str, object],
    model_id: str,
    placement: str,
    *,
    models_dir_host: str,
    model_file: str,
    cuda_image: str,
    cpu_image: str = docker_cmds.CPU_LLAMA_IMAGE,
    ngl: int | str | None = None,
) -> list[str]:
    """The `docker run` argv serving `model_id` on a local `placement` as `mist-bench-small`.

    `gpu-full` takes every layer, `gpu-4g` and `gpu-2g` take the layer count the slice search
    chose (`ngl`, required), and `cpu` runs the CPU image with no GPU at all. No `--rm`: logs are
    saved before `docker rm`, as bench_host's `serve` does.
    """
    model = _small_model(arms_doc, model_id)
    cfg = _local_placement(arms_doc, placement)
    rule = cfg.get("ngl")
    if rule == "given":
        if ngl is None:
            raise PlacementError(
                f"placement {placement!r} needs --ngl, the layer count the slice search chose"
            )
        layers: str | None = str(ngl)
    elif ngl is not None:
        raise PlacementError(
            f"placement {placement!r} sets its own layer count; --ngl is for "
            f"{[p for p, c in arms_doc['placements'].items() if c.get('ngl') == 'given']} only"
        )
    else:
        layers = None if rule is None else str(rule)

    argv = ["docker", "run", "-d", "--name", SMALL_CONTAINER]
    if cfg.get("gpus"):
        argv += docker_cmds.gpus_all()
    argv += docker_cmds.publish_local(SMALL_PORT, 8080)
    argv += docker_cmds.ro_mount(models_dir_host, "/models")
    argv.append(cuda_image if cfg.get("image") == "cuda" else cpu_image)
    argv += ["-m", f"/models/{model_file}", *arms_doc["common_args"]]
    if layers is not None:
        argv += ["-ngl", layers]
    sampling = arms_doc["sampling"]
    for key, flag in _SAMPLING_FLAGS:
        argv += [flag, str(sampling[key])]
    argv += list(arms_doc["thinking_args"][model["thinking"]])
    argv += ["--alias", small_alias(model_id, placement)]
    return argv


class DockerOps(Protocol):
    """The docker operations small-serve needs; tests pass a fake."""

    def container_state(self, name: str) -> str:
        """One of `running`, `exited`, `absent`, `unknown`."""
        ...

    def run_detached(self, argv: list[str]) -> str:
        """Run `argv` (a full `docker run -d` argv) and return the container id."""
        ...

    def wait_healthy(self, base_url: str, container: str, timeout: float) -> None:
        """Wait for llama-server's `/health`, failing fast if the container exits."""
        ...

    def logs(self, name: str) -> tuple[str, str]:
        """The container's stdout and stderr."""
        ...

    def stop(self, name: str) -> None:
        """Stop the container."""
        ...

    def rm(self, name: str) -> None:
        """Remove the container."""
        ...


class HostDocker:
    """The docker CLI on the host, through bench_host's helpers (argv lists, no shell)."""

    def container_state(self, name: str) -> str:
        """Classify the container's state with bench_host's probe."""
        return bench_host.probe_container_state(name).state

    def run_detached(self, argv: list[str]) -> str:
        """Run a detached container."""
        return bench_host.docker_run_detached(argv)

    def wait_healthy(self, base_url: str, container: str, timeout: float) -> None:
        """Wait for `/health` while watching the container."""
        bench_host.wait_for_llama_health(base_url, timeout=timeout, container_name=container)

    def logs(self, name: str) -> tuple[str, str]:
        """Read the container's logs."""
        return bench_host.docker_logs(name)

    def stop(self, name: str) -> None:
        """Stop the container."""
        bench_host.docker_stop([name])

    def rm(self, name: str) -> None:
        """Remove the container."""
        bench_host.docker_rm([name])


@dataclass(frozen=True, slots=True)
class ServeResult:
    """A served small model and its load time."""

    model_id: str
    placement: str
    alias: str
    argv: tuple[str, ...]
    image: str
    ngl: str | None
    load_s: float
    warmup_ttft_s: float | None
    base_url: str
    out_dir: Path


def small_serve_dir(speech: Path, model_id: str, placement: str, tag: str | None = None) -> Path:
    """`<speech>/small/<model>/<placement>[-<tag>]/`: one directory per serve, never reused."""
    leaf = placement if tag is None else f"{placement}-{validate_name(tag, 'tag')}"
    return speech / "small" / validate_name(model_id, "model") / validate_name(leaf, "placement")


def _save_logs(docker: DockerOps, path: Path) -> Path | None:
    try:
        stdout, stderr = docker.logs(SMALL_CONTAINER)
    except DockerError:
        return None
    write_text_exclusive(path, f"=== stdout ===\n{stdout}\n=== stderr ===\n{stderr}\n")
    return path


def serve_small(
    model_id: str,
    placement: str,
    *,
    out_dir: Path,
    pins_path: Path,
    models_dirs: Mapping[str, Path | None],
    ngl: int | None = None,
    arms_doc: Mapping[str, object] | None = None,
    cuda_image: str | None = None,
    docker: DockerOps | None = None,
    chat: Callable[..., ChatStreamResult] = stream_chat,
    clock: Clock = time.perf_counter,
    health_timeout: float = 900.0,
) -> ServeResult:
    """Serve a small model as `mist-bench-small` on 127.0.0.1:8081 and time its load.

    Refuses, before any docker call or write, a host-1070 placement, a model file that differs
    from the pin file, an existing `mist-bench-small` and an existing `out_dir`. Load time runs
    from just before `docker run` to the first token of a streamed warm-up request, so it covers
    container start, model load and the first prefill. Writes `serve.json` into `out_dir`.
    """
    arms = arms_doc if arms_doc is not None else load_small_arms()
    model = _small_model(arms, model_id)
    placement_cfg = _local_placement(arms, placement)
    pin = pinned_model(load_pins(pins_path), model_id)
    models_dir = models_dirs.get(model["models_dir"])
    if models_dir is None:
        raise SpeechBenchError(
            f"model {model_id!r} lives in the {model['models_dir']!r} models directory, "
            f"which is not set"
        )
    verify_model_file(pin, models_dir / pin.file)
    if placement_cfg.get("image") == "cuda" and cuda_image is None:
        cuda_image = docker_cmds.cuda_llama_image()
    argv = small_serve_argv(
        arms,
        model_id,
        placement,
        models_dir_host=models_dir.resolve().as_posix(),
        model_file=pin.file,
        cuda_image=cuda_image or "",
        ngl=ngl,
    )
    if out_dir.exists():
        raise OutputExistsError(f"{out_dir} already exists; pass a new --tag or run")
    docker = docker or HostDocker()
    state = docker.container_state(SMALL_CONTAINER)
    if state != "absent":
        raise SpeechBenchError(
            f"{SMALL_CONTAINER} already exists or its state is unknown ({state}); "
            f"run small-unserve first"
        )

    claim_dir(out_dir)
    image = argv[argv.index("-m") - 1]
    t_start = clock()
    docker.run_detached(argv)
    try:
        docker.wait_healthy(SMALL_BASE_URL, SMALL_CONTAINER, health_timeout)
    except bench_host.ContainerExitedError as exc:
        log = _save_logs(docker, out_dir / "serve_failed.log")
        if not exc.absent:
            docker.rm(SMALL_CONTAINER)
        raise ServeFailedError(f"{exc}; log saved to {log}", log_path=log) from exc
    except (bench_host.ContainerStateUnknownError, TimeoutError) as exc:
        log = _save_logs(docker, out_dir / "serve_failed.log")
        raise ServeFailedError(
            f"{exc}; log saved to {log}; {SMALL_CONTAINER} was left in place, "
            f"run small-unserve to remove it",
            log_path=log,
        ) from exc

    body = {**WARMUP_BODY, "model": small_alias(model_id, placement)}
    try:
        warm = chat(SMALL_BASE_URL, body, clock=clock)
    except StreamError as exc:
        log = _save_logs(docker, out_dir / "serve_failed.log")
        raise ServeFailedError(
            f"warm-up request failed: {exc}; log at {log}", log_path=log
        ) from exc
    if warm.t_first_delta is None:
        log = _save_logs(docker, out_dir / "serve_failed.log")
        raise ServeFailedError(f"warm-up request returned no token; log at {log}", log_path=log)

    result = ServeResult(
        model_id=model_id,
        placement=placement,
        alias=small_alias(model_id, placement),
        argv=tuple(argv),
        image=image,
        ngl=argv[argv.index("-ngl") + 1] if "-ngl" in argv else None,
        load_s=warm.t_first_delta - t_start,
        warmup_ttft_s=warm.ttft_s,
        base_url=SMALL_BASE_URL,
        out_dir=out_dir,
    )
    write_json_exclusive(
        out_dir / "serve.json",
        {
            "model": model_id,
            "placement": placement,
            "alias": result.alias,
            "image": image,
            "ngl": result.ngl,
            "model_file": pin.file,
            "model_sha256": pin.sha256,
            "argv": argv,
            "load_s": result.load_s,
            "warmup_ttft_s": result.warmup_ttft_s,
            "served_utc": datetime.now(UTC).isoformat(),
        },
    )
    return result


def unserve_small(
    out_dir: Path, *, docker: DockerOps | None = None, clock: Clock = time.perf_counter
) -> float:
    """Stop and remove `mist-bench-small`, timing the stop; save its log first. Returns seconds.

    Writes `server.log` and `unserve.json` into the serve's `out_dir`.
    """
    if not out_dir.is_dir():
        raise SpeechBenchError(f"{out_dir} does not exist; small-serve writes it")
    docker = docker or HostDocker()
    state = docker.container_state(SMALL_CONTAINER)
    if state == "absent":
        raise SpeechBenchError(f"{SMALL_CONTAINER} does not exist; nothing to unserve")
    t_stop = clock()
    if state in ("running", "unknown"):
        docker.stop(SMALL_CONTAINER)
    unload_s = clock() - t_stop
    stdout, stderr = docker.logs(SMALL_CONTAINER)
    write_text_exclusive(
        out_dir / "server.log", f"=== stdout ===\n{stdout}\n=== stderr ===\n{stderr}\n"
    )
    docker.rm(SMALL_CONTAINER)
    write_json_exclusive(
        out_dir / "unserve.json",
        {
            "unload_s": unload_s,
            "state_before": state,
            "unserved_utc": datetime.now(UTC).isoformat(),
        },
    )
    return unload_s


# ---------------------------------------------------------------------------
# Plan context, input validation, container argv and selftest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlanContext:
    """Placeholder paths and pinned image refs for `container_argv`, which runs no docker."""

    cuda_llama_image: str
    cpu_llama_image: str = docker_cmds.CPU_LLAMA_IMAGE
    repo_dir: str = "<repo>"
    results_root: str = "<results-root>"
    run: str = "<run>"
    speech_dir: str = "<results-root>/<run>/speech"
    bench_run: str = "<bench-run>"
    material_dir: str = "<material-dir>"
    models_dir: str = "<models-dir>"
    small_models_dir: str = "<small-models-dir>"
    speech_models_dir: str = "<speech-models-dir>"
    pins_path: str = "<results-root>/<run>/speech/host/pins.json"


def build_plan_context(repo_root: Path = REPO_ROOT) -> PlanContext:
    return PlanContext(
        cuda_llama_image=docker_cmds.cuda_llama_image(repo_root / "docker-compose.yml")
    )


def container_argv(ctx: PlanContext) -> list[tuple[str, list[str]]]:
    """small-serve's argv for every small model on every local placement."""
    arms = load_small_arms()
    blocks = []
    for model_id, model in arms["models"].items():
        host_dir = ctx.small_models_dir if model["models_dir"] == "small" else ctx.models_dir
        for placement in LOCAL_PLACEMENTS:
            given = arms["placements"][placement].get("ngl") == "given"
            argv = small_serve_argv(
                arms,
                model_id,
                placement,
                models_dir_host=host_dir,
                model_file=f"<{model_id}.gguf>",
                cuda_image=ctx.cuda_llama_image,
                cpu_image=ctx.cpu_llama_image,
                ngl="<ngl>" if given else None,
            )
            blocks.append((f"small-serve {small_alias(model_id, placement)}", argv))
    return blocks


def _validate_lines(doc: Mapping[str, object]) -> list[str]:
    problems = []
    lines = doc.get("lines") or []
    ids = [line.get("id") for line in lines]
    if ids != [f"L{i:02d}" for i in range(1, 21)]:
        problems.append(f"lines.json: ids are {ids}, expected L01 to L20 in order")
    for line in lines:
        words = len(str(line.get("text", "")).split())
        if not 5 <= words <= 25:
            problems.append(f"lines.json: {line.get('id')} has {words} words, outside 5 to 25")
        if not line.get("feeling") or not isinstance(line.get("emotive"), bool):
            problems.append(f"lines.json: {line.get('id')} lacks a feeling or an emotive flag")
    acks = [ack.get("id") for ack in doc.get("acknowledgements") or [] if ack.get("text")]
    if acks != [f"A{i:02d}" for i in range(1, 7)]:
        problems.append(f"lines.json: acknowledgements are {acks}, expected A01 to A06")
    return problems


def _validate_utterances(doc: Mapping[str, object]) -> list[str]:
    problems = []
    items = doc.get("utterances") or []
    expected = [f"U{i:02d}" for i in range(1, 21)] + [f"P{i:02d}" for i in range(1, 11)]
    if [u.get("id") for u in items] != expected:
        problems.append("utterances.json: ids are not U01 to U20 then P01 to P10")
    for item in items:
        pause = str(item.get("id", "")).startswith("P")
        if item.get("kind") != ("pause" if pause else "complete"):
            problems.append(f"utterances.json: {item.get('id')} has kind {item.get('kind')!r}")
        if ("..." in str(item.get("text", ""))) != pause:
            problems.append(f"utterances.json: {item.get('id')} pause mark does not match its kind")
    return problems


def _validate_speech_arms(doc: Mapping[str, object]) -> list[str]:
    problems = []
    for section in ("tts", "stt", "turn"):
        if not isinstance(doc.get(section), dict) or not doc[section]:
            problems.append(f"speech_arms.json: section {section!r} is missing or empty")
    for engine, cfg in (doc.get("tts") or {}).items():
        voices = set(cfg.get("voices") or [])
        timing = cfg.get("timing_voice")
        female = set(cfg.get("female_voices") or [])
        listening = set(cfg.get("listening_voices") or [])
        where = f"speech_arms.json: tts.{engine}"
        if not voices or timing not in voices:
            problems.append(f"{where}: timing voice {timing!r} is not in its voice list")
        if not female <= voices or not listening <= voices:
            problems.append(f"{where}: female or listening voices outside its voice list")
        if timing in listening:
            problems.append(f"{where}: the timing voice is also a listening voice")
        if not female - {timing} <= listening:
            problems.append(f"{where}: a female voice other than the timing voice is not heard")
        if "default" not in (cfg.get("settings") or {}):
            problems.append(f"{where}: no default setting")
    for section in ("tts", "stt", "turn"):
        for engine, cfg in (doc.get(section) or {}).items():
            if not cfg.get("image") or cfg.get("device") not in ("cuda", "cpu"):
                problems.append(f"speech_arms.json: {section}.{engine} lacks an image or device")
    return problems


def _validate_small_arms(doc: Mapping[str, object]) -> list[str]:
    problems = []
    placements = doc.get("placements") or {}
    for placement in LOCAL_PLACEMENTS:
        if (placements.get(placement) or {}).get("where") != "local":
            problems.append(f"small_arms.json: placement {placement!r} is not local")
    for placement in HOST1070_PLACEMENTS:
        if (placements.get(placement) or {}).get("where") != "host-1070":
            problems.append(f"small_arms.json: placement {placement!r} is not on host-1070")
    if set(doc.get("sampling") or {}) != {key for key, _ in _SAMPLING_FLAGS}:
        problems.append("small_arms.json: sampling must set temperature, top_p, top_k and min_p")
    for model_id, model in (doc.get("models") or {}).items():
        if model.get("models_dir") not in ("small", "main"):
            problems.append(f"small_arms.json: {model_id} models_dir must be small or main")
        if model.get("thinking") not in (doc.get("thinking_args") or {}):
            problems.append(f"small_arms.json: {model_id} thinking has no args")
        for placement in LOCAL_PLACEMENTS:
            given = (placements.get(placement) or {}).get("ngl") == "given"
            try:
                small_serve_argv(
                    doc,
                    model_id,
                    placement,
                    models_dir_host="/m",
                    model_file="m.gguf",
                    cuda_image="cuda",
                    ngl=1 if given else None,
                )
            except (PlacementError, KeyError, TypeError) as exc:
                problems.append(f"small_arms.json: {model_id}@{placement} argv fails: {exc}")
    return problems


def validate_inputs(repo_root: Path) -> list[str]:
    """Check core's committed inputs; an empty list means they are usable."""
    speech = repo_root / SPEECH_DIR_REL
    checks = (
        ("lines.json", _validate_lines),
        ("utterances.json", _validate_utterances),
        ("speech_arms.json", _validate_speech_arms),
        ("small_arms.json", _validate_small_arms),
    )
    problems = []
    for name, check in checks:
        try:
            problems += check(load_json(speech / name))
        except (InputError, AttributeError, KeyError, TypeError) as exc:
            problems.append(f"{name}: unreadable or malformed: {exc}")
    filler = speech / "cache_filler.txt"
    if not filler.is_file() or not filler.read_text(encoding="utf-8").strip():
        problems.append("cache_filler.txt: missing or empty")
    sources = prefix_source_paths(repo_root)
    if not sources[0].is_file():
        problems.append(f"prefix source {sources[0]} is missing")
    if len(sources) < 2:
        problems.append(f"no test-vault notes under {repo_root / TEST_VAULT_REL}")
    return problems


def _check(condition: bool, message: str) -> None:
    # Explicit raise rather than `assert`, which `python -O` strips.
    if not condition:
        raise AssertionError(message)


def selftest() -> None:
    """Check core's arithmetic, parsers, refusals and argv without docker or network."""
    summary = summarize([5, 1, 3])
    _check((summary.n, summary.median, summary.p90) == (3, 3, 5), f"summarize gave {summary}")

    ticks = iter([1.1, 1.25, 1.3])
    stream = [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"choices":[{"delta":{"content":"ok"}}],"timings":{"prompt_n":3}}',
        "data: [DONE]",
    ]
    parsed = parse_chat_stream(stream, clock=lambda: next(ticks), t_request=1.0)
    _check(parsed.content == "ok", "stream content")
    _check(abs((parsed.ttft_s or 0) - 0.25) < 1e-9, f"stream ttft {parsed.ttft_s}")
    _check(parsed.timings == {"prompt_n": 3}, "stream timings")

    def words(text: str) -> list[int]:
        return list(range(len(text.split())))

    first = build_prefix(1500, words)
    _check(first == build_prefix(1500, words), "prefix differs between calls")
    _check(abs(len(words(first)) - 1500) <= PREFIX_TOLERANCE, "prefix misses its target")

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        claim_output(root, "st", "tts", "a", git_sha="x")
        try:
            claim_output(root, "st", "tts", "a", git_sha="x")
        except OutputExistsError:
            pass
        else:
            raise AssertionError("an existing arm directory was claimed twice")

        sample = root / "S01.wav"
        sample.write_bytes(b"sample")
        manifest = root / "manifest.json"
        item = {"id": "S01", "file": "S01.wav", "kind": "complete", "text": "t", "duration_s": 1}
        item["sha256"] = hashlib.sha256(b"other").hexdigest()
        manifest.write_text(json.dumps({"items": [item]}), encoding="utf-8")
        try:
            verify_material(manifest)
        except ManifestError:
            pass
        else:
            raise AssertionError("a material file with a different sha256 was accepted")

        pin = ModelPin("m", "r", "v", "m.gguf", hashlib.sha256(b"other").hexdigest(), 6)
        try:
            verify_model_file(pin, sample)
        except PinError:
            pass
        else:
            raise AssertionError("a model file with a different sha256 was accepted")

        (root / "w1" / "session").mkdir(parents=True)
        try:
            begin_bench_window(root, "w1")
        except BenchRunError:
            pass
        else:
            raise AssertionError("a used bench run id was accepted")

    try:
        speech_root(REPO_ROOT / "results", "st")
    except ResultsRootError:
        pass
    else:
        raise AssertionError("a results root inside the repository was accepted")

    _check(HELP_CAPTURE_PATH.is_file(), f"help capture missing at {HELP_CAPTURE_PATH}")
    known = help_flags.parse_help_flags(HELP_CAPTURE_PATH.read_text(encoding="utf-8"))
    arms = load_small_arms()
    for placement in LOCAL_PLACEMENTS:
        given = arms["placements"][placement].get("ngl") == "given"
        argv = small_serve_argv(
            arms,
            "gemma4-e2b",
            placement,
            models_dir_host="/m",
            model_file="m.gguf",
            cuda_image="cuda-image",
            ngl=1 if given else None,
        )
        tail = argv[argv.index("-m") :]
        unknown = help_flags.unknown_flags(tail, known)
        _check(not unknown, f"small-serve {placement} uses flags b11151 does not list: {unknown}")
