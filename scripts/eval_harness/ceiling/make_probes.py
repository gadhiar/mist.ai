"""Write the utterance-only probes file the ceiling runs put to the unchanged service.

An answerer sits in the model's seat, so the service it answers must never hold gold: the
bench probe reads `expected_*` with `.get`, and runs on a file of tags and utterances
alone. This module copies only those two fields from each gold row, in gold order, so
`data/ceiling/probes.jsonl` is reproducible from the gold and carries nothing else.

    python -m scripts.eval_harness.ceiling.make_probes \
        --gold data/gold/extraction/dev-lists-2026-11.jsonl --out data/ceiling/probes.jsonl

It prints only the row count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from backend.errors import MistError


class ProbeSourceError(MistError):
    """A gold row cannot become a probe: not JSON, no string tag or utterance, or a repeat."""


def make_probes(gold_path: Path) -> list[dict[str, str]]:
    """Return one `{"tag", "utterance"}` row per gold row, in gold order.

    A blank line is skipped, as `iter_gold_probes` skips it, so the rows match what the
    bench probe would read from the gold itself.

    Raises:
        ProbeSourceError: A line is not a JSON object, lacks a non-empty string `tag` or
            `utterance`, or repeats a tag; the round tool attributes requests by tag.
    """
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(gold_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProbeSourceError(f"{gold_path} line {line_no} is not JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ProbeSourceError(f"{gold_path} line {line_no} is not a JSON object")
        tag, utterance = obj.get("tag"), obj.get("utterance")
        if not (isinstance(tag, str) and tag and isinstance(utterance, str) and utterance):
            raise ProbeSourceError(
                f"{gold_path} line {line_no} lacks a non-empty string tag and utterance"
            )
        if tag in seen:
            raise ProbeSourceError(f"{gold_path} line {line_no} repeats tag {tag}")
        seen.add(tag)
        rows.append({"tag": tag, "utterance": utterance})
    return rows


def write_probes(rows: list[dict[str, str]], out: Path) -> None:
    """Write `rows` one JSON object per line, replacing `out` whole.

    Non-ASCII text is written as is, so the utterances read in a diff exactly as the gold
    holds them.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    os.replace(tmp, out)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: 0 when the file is written, 1 when a gold row is refused."""
    parser = argparse.ArgumentParser(description="Write utterance-only probes from gold.")
    parser.add_argument("--gold", required=True, type=Path, help="Gold JSONL to read.")
    parser.add_argument("--out", required=True, type=Path, help="Probes JSONL to write.")
    args = parser.parse_args(argv)
    try:
        rows = make_probes(args.gold)
    except (ProbeSourceError, OSError) as exc:
        print(f"make_probes: {exc}", file=sys.stderr)
        return 1
    write_probes(rows, args.out)
    print(f"wrote {len(rows)} probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
