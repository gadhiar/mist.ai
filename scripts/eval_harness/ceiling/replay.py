"""Replay every stored answer through the unchanged service, and compare two replays.

Run A's result is a replay, not the rounds' own output: the rounds ran each probe many
times against partial stores, while a replay runs all 40 probes once against the four
merged stores, the way the bench probe runs a model. Two replays of the same answers must
give the same requests and rows; anything else means run A is not reproducible from its
committed files, which stops the goal.

    python -m scripts.eval_harness.ceiling.replay --out DIR
    python -m scripts.eval_harness.ceiling.replay --compare DIR1 DIR2

`--out` writes the bench probe's outputs into DIR, plus `requests.jsonl`, the relay log
(`seq`, `tag`, `key`, `kind`, `answered` per request), and exits 0 only when nothing is
pending. The bench probe runs once over the whole probes file, so the relay is never told
a tag; a request's tag is the probe whose scope request it follows, since every probe
starts with exactly one scope call and probes run in file order.

`--compare` exits 0 only when the two key sequences are equal and the raw rows are equal
after dropping `job_id`, `response.job_id` and `response.timings_ms`, the fields that vary
by run. Otherwise it prints the first differing tag and the differing field names, never
a value, and exits 1.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from .relay import LogEntry, RelayError, RelayLog, build_relay, running_relay
from .round import (
    ANSWERERS,
    DEFAULT_PROBES,
    DEFAULT_STORE_DIR,
    RELAY_PORT,
    SERVICE_PORT,
    CeilingRunError,
    Probe,
    count_raw_rows,
    load_probes,
    run_bench,
    running_service,
)
from .store import AnswerStoreError, merge_stores

STORE_NAMES = tuple(f"answers-{k}.jsonl" for k in ANSWERERS)
REQUESTS_FILE = "requests.jsonl"
RAW_FILE = "extraction_service_raw.jsonl"
DROPPED_FIELDS: tuple[tuple[str, ...], ...] = (
    ("job_id",),
    ("response", "job_id"),
    ("response", "timings_ms"),
)


class ReplayError(CeilingRunError):
    """Base of the replay's own failures."""


class OutputExistsError(ReplayError):
    """The output directory already holds a replay, which nothing overwrites."""


class CompareInputError(ReplayError):
    """A replay directory lacks a file `--compare` reads, or the file is not JSONL."""


def attribute_tags(entries: Sequence[LogEntry], probes: Sequence[Probe]) -> list[str | None]:
    """Give each request the tag of the probe whose scope request it follows.

    A request before the first scope request, or after the probes run out, gets None, so
    a run that broke the one-scope-per-probe order shows it rather than misattributing.
    """
    tags: list[str | None] = []
    current: str | None = None
    scopes = 0
    for entry in entries:
        if entry.kind == "scope":
            current = probes[scopes].tag if scopes < len(probes) else None
            scopes += 1
        tags.append(current)
    return tags


def replay_out(
    out: Path,
    probes_path: Path,
    store_dir: Path,
    *,
    relay_port: int = RELAY_PORT,
    service_port: int = SERVICE_PORT,
) -> int:
    """Run every probe once over the merged stores into `out`; 0 only when nothing is pending.

    Raises:
        OutputExistsError: `out` already holds a replay.
        StoreIntegrityError, SharedKeyError: A store fails its load, or two share a key.
        ServiceStartError: The service did not start.
    """
    for name in (REQUESTS_FILE, RAW_FILE):
        if (out / name).exists():
            raise OutputExistsError(f"{name} already exists in the output directory")
    probes = load_probes(probes_path)
    answers = merge_stores(store_dir / name for name in STORE_NAMES)
    log = RelayLog()
    out.mkdir(parents=True, exist_ok=True)
    with (
        running_relay(build_relay(answers, log), relay_port) as relay_url,
        running_service(relay_url, service_port, out / "service.log") as service_url,
    ):
        run_bench(service_url, probes_path, out, out / "bench.log")
    entries = log.entries()
    tags = attribute_tags(entries, probes)
    with (out / REQUESTS_FILE).open("x", encoding="utf-8") as fh:
        for entry, tag in zip(entries, tags, strict=True):
            fh.write(json.dumps({**entry.public(), "tag": tag}) + "\n")
    rows = count_raw_rows(out) or 0
    pending = sum(not e.answered for e in entries)
    print(
        f"replay: probes {len(probes)}, complete {rows}, requests {len(entries)}, "
        f"pending {pending}"
    )
    unclassified = any(e.kind is None for e in entries)
    return 0 if pending == 0 and rows == len(probes) and not unclassified else 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CompareInputError(f"cannot read {path.name}: {exc.strerror}") from exc
    try:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise CompareInputError(f"{path.name} is not JSONL: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise CompareInputError(f"{path.name} holds a line that is not a JSON object")
    return rows


def _without_varying_fields(row: dict[str, Any]) -> dict[str, Any]:
    kept = copy.deepcopy(row)
    for path in DROPPED_FIELDS:
        parent: Any = kept
        for key in path[:-1]:
            parent = parent.get(key) if isinstance(parent, dict) else None
        if isinstance(parent, dict):
            parent.pop(path[-1], None)
    return kept


def differing_fields(a: dict[str, Any], b: dict[str, Any], prefix: str = "") -> list[str]:
    """Name every field whose value differs, descending into objects; never the values."""
    names: list[str] = []
    for key in sorted(set(a) | set(b)):
        name = f"{prefix}{key}"
        if key not in a or key not in b:
            names.append(name)
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            names += differing_fields(a[key], b[key], f"{name}.")
        elif a[key] != b[key]:
            names.append(name)
    return names


def compare(first: Path, second: Path) -> int:
    """Compare two replays' key sequences, then their raw rows; print one line, 0 when equal.

    Raises:
        CompareInputError: A replay lacks `requests.jsonl` or the raw rows, or either
            is not JSONL.
    """
    requests = [_read_jsonl(d / REQUESTS_FILE) for d in (first, second)]
    rows = [_read_jsonl(d / RAW_FILE) for d in (first, second)]
    for index in range(max(len(r) for r in requests)):
        a, b = (r[index] if index < len(r) else None for r in requests)
        if a is None or b is None or a.get("key") != b.get("key"):
            tag = (a or b or {}).get("tag")
            what = "key" if a is not None and b is not None else "request missing"
            print(f"compare: requests differ at request {index + 1}, tag {tag}: {what}")
            return 1
    for index in range(max(len(r) for r in rows)):
        a, b = (r[index] if index < len(r) else None for r in rows)
        if a is None or b is None:
            print(f"compare: rows differ at tag {(a or b or {}).get('id')}: row missing")
            return 1
        fields = differing_fields(_without_varying_fields(a), _without_varying_fields(b))
        if fields:
            print(f"compare: rows differ at tag {a.get('id')}: {', '.join(fields)}")
            return 1
    print(f"compare: equal, requests {len(requests[0])}, rows {len(rows[0])}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line; every optional flag defaults to the plan's run A layout."""
    parser = argparse.ArgumentParser(description="Replay stored answers, or compare replays.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path, help="Directory to replay into.")
    mode.add_argument("--compare", type=Path, nargs=2, metavar=("DIR1", "DIR2"))
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help="0 picks a free one.")
    parser.add_argument(
        "--service-port", type=int, default=SERVICE_PORT, help="0 picks a free one."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Errors go to stderr as one line and exit 1."""
    args = parse_args(argv)
    try:
        if args.compare:
            return compare(*args.compare)
        return replay_out(
            args.out.resolve(),
            args.probes.resolve(),
            args.store_dir.resolve(),
            relay_port=args.relay_port,
            service_port=args.service_port,
        )
    except OSError as exc:
        print(f"replay: {type(exc).__name__}: {exc.strerror or 'I/O error'}", file=sys.stderr)
        return 1
    except (CeilingRunError, AnswerStoreError, RelayError, httpx.HTTPError) as exc:
        print(f"replay: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
