"""Answer rounds: put each pending model request of ten probes to an answerer, then store replies.

The answerer sits in the model's seat of the unchanged extraction service. Each bare run
of this tool is one round:

1. The relay starts on 127.0.0.1:18080 over answerer K's store.
2. The unchanged service starts with `uvicorn backend.extraction_service.app:
   build_app_from_env --factory --host 127.0.0.1 --port 18090`, under `SERVICE_ENV`.
3. For each of K's probes (rows 10K-9 to 10K of `data/ceiling/probes.jsonl`) the tool
   posts the tag to `/_relay/probe`, then runs the unchanged bench probe on a one-row
   file into `.local/eval-ceiling-a/K/round-<r>/<tag>`.
4. Each probe's first pending request, in arrival order, is offered. Its later pending
   requests are superseded: they were made only because an earlier request failed, as the
   degraded extract after a pending scope is.
5. The offer goes to `.local/ceiling/pending-K.md`, with each system prompt and schema
   written once under `.local/ceiling/prompts/` and `.local/ceiling/schemas/`, and to
   `.local/ceiling/state-K.json`.

The answerer sees only that pending file, the files it names, and one counts-only stdout
line. The bench probe's own outputs hold service internals and stay in the run directory,
which nothing this tool prints names. `--add` stores the answers in
`.local/ceiling/answers-in-K.jsonl`; `--check` replays K's probes into a temporary
directory and writes nothing in the worktree.

Exit codes: 0 when nothing is pending, 3 when requests are offered (or, for `--add`,
refused), 1 on an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess  # nosec B404 - runs the service and the bench probe, fixed argument lists
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from backend.errors import MistError

from .relay import (
    LOOPBACK,
    LogEntry,
    RelayError,
    RelayLog,
    build_relay,
    offered_request,
    running_relay,
)
from .store import (
    AnswerRefusedError,
    AnswerStore,
    AnswerStoreError,
    OfferedRequest,
    write_state,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROBES = REPO_ROOT / "data" / "ceiling" / "probes.jsonl"
DEFAULT_STORE_DIR = REPO_ROOT / "data" / "ceiling"
DEFAULT_LOCAL_DIR = REPO_ROOT / ".local"
RELAY_PORT = 18080
SERVICE_PORT = 18090
PROBES_PER_ANSWERER = 10
ANSWERERS = (1, 2, 3, 4)
KINDS = ("scope", "extract", "repair", "derive")
HIDDEN_RUN_DIR = "eval-ceiling-a"
SERVICE_APP = "backend.extraction_service.app:build_app_from_env"
BENCH_MODULE = "scripts.model_bench.probes.extraction_service"
EXIT_DONE, EXIT_ERROR, EXIT_OFFERED = 0, 1, 3

# host-1070's r6 service settings, except the four identity stamps, which name the relay
# so no run A result can pass for a gpt-oss one. `EXTRACTION_LLM_BASE_URL` is the relay's.
SERVICE_ENV: dict[str, str] = {
    "EXTRACTION_ADAPTER": "gptoss",
    "EXTRACTION_REASONING_EFFORT": "low",
    "EXTRACTION_LLM_TIMEOUT_SECONDS": "300",
    "EXTRACTION_MAX_ATTEMPTS": "2",
    "EXTRACTION_CONSTRAINED_MODE": "",
    "EXTRACTION_SCOPE_ENABLED": "true",
    "EXTRACTION_TEMPERATURE": "0.0",
    "EXTRACTION_MODEL_FILE": "answerer-relay",
    "EXTRACTION_MODEL_HASH": "answerer-relay",
    "LLAMA_CPP_BUILD": "relay",
    "EXTRACTION_LOCATION_LABEL": "container-relay",
}

_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_BACKTICKS_RE = re.compile(r"`+")


class CeilingRunError(MistError):
    """Base of the round tool's and the replay's own failures."""


class ProbeFileError(CeilingRunError):
    """The probes file cannot be read, or holds no probe for the answerer."""


class ServiceStartError(CeilingRunError):
    """The unchanged service did not start serving."""


class RunError(CeilingRunError):
    """A probe ran in a way no answer can fix, so offering requests would mislead."""


class AnswerInputError(CeilingRunError):
    """The `--add` input file is missing or unreadable."""


@dataclass(frozen=True, slots=True)
class Probe:
    """One probes-file row, with its line kept so a one-row file reproduces it exactly."""

    tag: str
    utterance: str
    line: str


@dataclass(frozen=True, slots=True)
class ProbeRun:
    """What one probe did in a run: the requests it made and the raw rows it got back."""

    tag: str
    entries: tuple[LogEntry, ...]
    raw_rows: int

    @property
    def pending(self) -> tuple[LogEntry, ...]:
        """The requests the relay had no answer for, in arrival order."""
        return tuple(e for e in self.entries if not e.answered)

    @property
    def complete(self) -> bool:
        """True when the probe made no pending request and the service answered it."""
        return not self.pending and self.raw_rows > 0


def load_probes(path: Path) -> list[Probe]:
    """Read the probes file, refusing a row the run directories cannot be named for.

    Raises:
        ProbeFileError: The file is unreadable, a line is not a JSON object with string
            `tag` and `utterance`, a tag is not a plain file name, or a tag repeats.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProbeFileError(f"cannot read the probes file: {exc.strerror}") from exc
    probes: list[Probe] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProbeFileError(f"probes line {line_no} is not JSON: {exc}") from exc
        tag = obj.get("tag") if isinstance(obj, dict) else None
        utterance = obj.get("utterance") if isinstance(obj, dict) else None
        if not (isinstance(tag, str) and isinstance(utterance, str)):
            raise ProbeFileError(f"probes line {line_no} lacks a string tag and utterance")
        if not _TAG_RE.fullmatch(tag):
            raise ProbeFileError(f"probes line {line_no}: tag {tag!r} is not a plain name")
        if any(p.tag == tag for p in probes):
            raise ProbeFileError(f"probes line {line_no} repeats tag {tag}")
        probes.append(Probe(tag=tag, utterance=utterance, line=raw))
    return probes


def answerer_probes(probes: Sequence[Probe], answerer: int) -> list[Probe]:
    """Return answerer K's probes, rows 10K-9 to 10K (fewer when the file is shorter).

    Raises:
        ProbeFileError: The file holds no row for answerer K.
    """
    start = PROBES_PER_ANSWERER * (answerer - 1)
    chosen = list(probes[start : start + PROBES_PER_ANSWERER])
    if not chosen:
        raise ProbeFileError(f"the probes file holds no row for answerer {answerer}")
    return chosen


def service_env(relay_url: str, *, base: Mapping[str, str]) -> dict[str, str]:
    """Return the service's environment: `base` without its own extraction settings, plus ours.

    Every inherited `EXTRACTION_*` variable and `LLAMA_CPP_BUILD` is dropped first, so a
    setting in the caller's shell (a reasoning budget, say) cannot reach the service.
    """
    env = {
        k: v for k, v in base.items() if not k.startswith("EXTRACTION_") and k != "LLAMA_CPP_BUILD"
    }
    env["EXTRACTION_LLM_BASE_URL"] = relay_url
    env.update(SERVICE_ENV)
    return env


def service_command(port: int) -> list[str]:
    """The uvicorn command line that serves the unchanged service on loopback."""
    return [
        sys.executable,
        "-m",
        "uvicorn",
        SERVICE_APP,
        "--factory",
        "--host",
        LOOPBACK,
        "--port",
        str(port),
    ]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK, 0))
        return int(sock.getsockname()[1])


def _wait_until_serving(proc: subprocess.Popen[bytes], url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        code = proc.poll()
        if code is not None:
            raise ServiceStartError(f"the service exited with code {code} before serving")
        try:
            if httpx.get(f"{url}/v1/health", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            raise ServiceStartError(f"the service did not serve within {timeout:.0f} s")
        time.sleep(0.1)


@contextmanager
def running_service(
    relay_url: str, port: int, log_path: Path, *, start_timeout: float = 120.0
) -> Iterator[str]:
    """Run the unchanged service against the relay, yielding its base URL.

    Port 0 picks a free loopback port, which tests use so they never meet a running
    service. Its output goes to `log_path`, never to this tool's own streams.

    Raises:
        ServiceStartError: The service exited or did not serve within `start_timeout`.
    """
    port = port or _free_port()
    url = f"http://{LOOPBACK}:{port}"
    with log_path.open("wb") as log:
        proc = subprocess.Popen(  # nosec B603 - fixed argument list, no shell
            service_command(port),
            cwd=REPO_ROOT,
            env=service_env(relay_url, base=os.environ),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_until_serving(proc, url, start_timeout)
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def run_bench(service_url: str, gold: Path, out_dir: Path, log_path: Path) -> int:
    """Run the unchanged bench probe over `gold` into `out_dir`, its output to `log_path`."""
    with log_path.open("wb") as log:
        return subprocess.run(  # nosec B603 - fixed argument list, no shell
            [
                sys.executable,
                "-m",
                BENCH_MODULE,
                "--endpoint",
                service_url,
                "--gold",
                str(gold),
                "--out",
                str(out_dir),
            ],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def count_raw_rows(out_dir: Path) -> int | None:
    """Return the bench probe's raw row count, or None when it never got past preflight."""
    path = out_dir / "extraction_service_raw.jsonl"
    if not path.is_file():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run_probes(
    probes: Sequence[Probe],
    answers: Mapping[str, Any],
    run_dir: Path,
    *,
    relay_port: int,
    service_port: int,
) -> list[ProbeRun]:
    """Run each probe on its own through the relay and the service, attributing its requests.

    Raises:
        ServiceStartError: The service did not start.
        RunError: The bench probe stopped before a probe ran (a failed preflight).
    """
    log = RelayLog()
    run_dir.mkdir(parents=True, exist_ok=True)
    runs: list[ProbeRun] = []
    with (
        running_relay(build_relay(answers, log), relay_port) as relay_url,
        running_service(relay_url, service_port, run_dir / "service.log") as service_url,
        httpx.Client(timeout=10.0) as client,
    ):
        for probe in probes:
            client.post(f"{relay_url}/_relay/probe", json={"tag": probe.tag}).raise_for_status()
            seen = len(log.entries())
            probe_dir = run_dir / probe.tag
            probe_dir.mkdir()
            gold = probe_dir / "probe.jsonl"
            gold.write_text(probe.line + "\n", encoding="utf-8")
            code = run_bench(service_url, gold, probe_dir, probe_dir / "bench.log")
            rows = count_raw_rows(probe_dir)
            if rows is None:
                raise RunError(f"the bench probe exited {code} before probe {probe.tag} ran")
            entries = log.entries()[seen:]
            runs.append(ProbeRun(tag=probe.tag, entries=entries, raw_rows=rows))
    return runs


def check_runs(runs: Sequence[ProbeRun]) -> None:
    """Refuse a run no answer can move forward.

    Raises:
        RunError: A request matched no stage schema, or a probe failed with nothing
            pending, so its failure is not the answerer's to fix.
    """
    for run in runs:
        if any(e.kind is None for e in run.entries):
            raise RunError(f"probe {run.tag} made a request that matches no stage schema")
        if not run.pending and not run.complete:
            raise RunError(f"probe {run.tag} failed with nothing pending")


def choose_offers(runs: Sequence[ProbeRun]) -> tuple[list[tuple[str, LogEntry]], int]:
    """Pick each probe's first pending request; count its other pending ones as superseded.

    A later pending request was made only because an earlier one failed (a scope failure
    degrades to `unknown` and the turn goes on), so it would change once the earlier one
    is answered. A key a previous probe already offered is superseded too, since one
    answer serves both.

    Returns:
        The offered (tag, entry) pairs in probe order, and the superseded count.
    """
    offers: list[tuple[str, LogEntry]] = []
    keys: set[str] = set()
    superseded = 0
    for run in runs:
        pending = run.pending
        if not pending:
            continue
        superseded += len(pending) - 1
        if pending[0].key in keys:
            superseded += 1
            continue
        keys.add(pending[0].key)
        offers.append((run.tag, pending[0]))
    return offers, superseded


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _write_once(path: Path, text: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _content_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _fenced(text: str) -> str:
    longest = max((len(m) for m in _BACKTICKS_RE.findall(text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def write_pending(
    ceiling_dir: Path,
    answerer: int,
    round_: int,
    offers: Sequence[tuple[str, LogEntry]],
    requests: Sequence[OfferedRequest],
) -> None:
    """Write the pending file and each prompt and schema file it names, once each.

    The file names are relative to the pending file's directory. A request's system
    prompt is long and the same for every request of a kind, so it is named rather than
    repeated; every other message is shown in full.
    """
    answers_in = f"answers-in-{answerer}.jsonl"
    out = [
        f"# Pending requests: answerer {answerer}, round {round_}",
        "",
    ]
    if not offers:
        out += ["Round complete: nothing is pending.", ""]
    else:
        out += [
            "Answer each request below as the model in the seat would, with only the JSON",
            "value its schema asks for. Write one line per answer to "
            f'`{answers_in}` beside this file: `{{"key": "<key>", "answer": <JSON>}}`.',
            "File names below are relative to this file's directory.",
            "",
        ]
    for index, ((tag, entry), request) in enumerate(zip(offers, requests, strict=True), start=1):
        messages = entry.body.get("messages") or []
        systems = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
        prompt_names = []
        for message in systems:
            text = _content_text(message.get("content"))
            name = f"prompts/{_sha12(text)}.txt"
            _write_once(ceiling_dir / name, text)
            prompt_names.append(name)
        schema_compact = json.dumps(request.schema, sort_keys=True, separators=(",", ":"))
        schema_name = f"schemas/{_sha12(schema_compact)}.json"
        _write_once(ceiling_dir / schema_name, json.dumps(request.schema, indent=2) + "\n")
        out += [
            f"## Request {index} of {len(offers)}",
            "",
            f"- key: `{entry.key}`",
            f"- tag: {tag}",
            f"- kind: {request.kind}",
            f"- system prompt: {', '.join(prompt_names) or 'none'}",
            f"- schema: {schema_name}",
            "",
        ]
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                continue
            role = message.get("role") if isinstance(message, dict) else "message"
            content = message.get("content") if isinstance(message, dict) else message
            out += [f"#### {role}", "", _fenced(_content_text(content)), ""]
    pending = ceiling_dir / f"pending-{answerer}.md"
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text("\n".join(out), encoding="utf-8")


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    ceiling = args.local_dir / "ceiling"
    return (
        args.store_dir / f"answers-{args.answerer}.jsonl",
        ceiling / f"state-{args.answerer}.json",
        ceiling,
    )


def do_round(args: argparse.Namespace) -> int:
    """Run one round and print its counts line; see the module docstring for the steps."""
    store_path, state_path, ceiling = _paths(args)
    store = AnswerStore.load(store_path, state_path=state_path)
    probes = answerer_probes(load_probes(args.probes), args.answerer)
    round_ = store.round + 1
    run_dir = args.local_dir / HIDDEN_RUN_DIR / str(args.answerer) / f"round-{round_}"
    if run_dir.exists():
        # Left by a round that stopped before it recorded its state; nothing refers to it.
        shutil.rmtree(run_dir)
    runs = run_probes(
        probes,
        store.answers(),
        run_dir,
        relay_port=args.relay_port,
        service_port=args.service_port,
    )
    check_runs(runs)
    offers, superseded = choose_offers(runs)
    requests = [offered_request(entry) for _tag, entry in offers]
    state = store.offer(round_, requests)
    write_pending(ceiling, args.answerer, round_, offers, requests)
    write_state(state_path, state)
    kinds = Counter(r.kind for r in requests)
    print(
        f"round {round_} answerer {args.answerer}: probes {len(probes)}, "
        f"complete {sum(r.complete for r in runs)}, offered {len(requests)} "
        f"(scope {kinds['scope']}, extract {kinds['extract']}, repair {kinds['repair']}, "
        f"derive {kinds['derive']}), superseded {superseded}, stored {len(store)}"
    )
    return EXIT_OFFERED if requests else EXIT_DONE


def _answer_lines(path: Path) -> Iterator[tuple[int, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnswerInputError(f"cannot read {path.name}: {exc.strerror}") from exc
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if raw.strip():
            try:
                yield line_no, json.loads(raw)
            except json.JSONDecodeError as exc:
                yield line_no, exc


def do_add(args: argparse.Namespace) -> int:
    """Validate and append each answer in the input file, then print the counts and refusals."""
    store_path, state_path, ceiling = _paths(args)
    store = AnswerStore.load(store_path, state_path=state_path)
    lines = list(_answer_lines(ceiling / f"answers-in-{args.answerer}.jsonl"))
    added, refusals = 0, []
    for line_no, obj in lines:
        if not (isinstance(obj, dict) and isinstance(obj.get("key"), str) and "answer" in obj):
            refusals.append(f"refused line {line_no}: not a JSON object with a key and an answer")
            continue
        try:
            store.add(obj["key"], obj["answer"])
        except AnswerRefusedError as exc:
            refusals.append(f"refused {exc}")
            continue
        added += 1
    write_state(state_path, store.state())
    print(f"added {added}, refused {len(refusals)}, stored {len(store)}")
    for line in refusals:
        print(line)
    return EXIT_OFFERED if refusals else EXIT_DONE


def do_check(args: argparse.Namespace) -> int:
    """Replay K's probes into a temporary directory; 0 only when nothing is pending."""
    store_path, state_path, _ceiling = _paths(args)
    store = AnswerStore.load(store_path, state_path=state_path)
    probes = answerer_probes(load_probes(args.probes), args.answerer)
    with tempfile.TemporaryDirectory(prefix="ceiling-check-") as tmp:
        runs = run_probes(
            probes,
            store.answers(),
            Path(tmp) / "run",
            relay_port=args.relay_port,
            service_port=args.service_port,
        )
    check_runs(runs)
    complete = sum(r.complete for r in runs)
    pending = sum(len(r.pending) for r in runs)
    print(
        f"check answerer {args.answerer}: probes {len(probes)}, complete {complete}, "
        f"pending {pending}, stored {len(store)}"
    )
    return EXIT_DONE if pending == 0 and complete == len(probes) else EXIT_ERROR


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line; every optional flag defaults to the plan's run A layout."""
    parser = argparse.ArgumentParser(description="Run an answer round through the service.")
    parser.add_argument("--answerer", required=True, type=int, choices=ANSWERERS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--add", action="store_true", help="Store the answers in answers-in-K.")
    mode.add_argument("--check", action="store_true", help="Replay K's probes into /tmp.")
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help="0 picks a free one.")
    parser.add_argument(
        "--service-port", type=int, default=SERVICE_PORT, help="0 picks a free one."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Errors go to stderr as one line and exit 1."""
    args = parse_args(argv)
    # The service and the bench probe run from the repository root, so a relative path
    # is fixed against the caller's directory first.
    for name in ("probes", "store_dir", "local_dir"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        if args.add:
            return do_add(args)
        if args.check:
            return do_check(args)
        return do_round(args)
    except OSError as exc:
        # Only the reason: the file name could lie in the run directory the answerer
        # never sees.
        print(f"round: {type(exc).__name__}: {exc.strerror or 'I/O error'}", file=sys.stderr)
        return EXIT_ERROR
    except (CeilingRunError, AnswerStoreError, RelayError, httpx.HTTPError) as exc:
        print(f"round: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
