r"""V2 Document Ingestion (single-process, host-side launcher).

Loads the extraction pipeline once, splits an input Markdown file into
paragraph chunks, runs dry-run extraction over each, reports per-chunk
validity, then commits passing chunks above a configurable success-rate gate.

Run inside the backend container to reuse already-loaded GPU/embedding models:

    MSYS_NO_PATHCONV=1 docker exec mist-backend python /app/scripts/v2_ingest.py \
        --input /app/data/ingest/career-context.md --commit-threshold 0.80

Spec: ~/.claude/plans/nimble-forage-cinder.md Part 4 V2.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.factories import build_extraction_pipeline  # noqa: E402
from backend.knowledge.config import get_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("v2_ingest")


_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_HEADING_RE = re.compile(r"^#+\s+", re.MULTILINE)


def split_markdown(text: str, min_chars: int = 40) -> list[str]:
    """Split markdown into extractor-friendly chunks.

    Paragraphs of prose pass through as single chunks. Bullet-list paragraphs
    split into one chunk per bullet item — Gemma 4 E4B produces more reliable
    JSON on shorter, focused inputs (empirical finding from V2 smoke: 38% pass
    rate on paragraph-level; significantly higher on single-bullet input).
    """
    # Strip frontmatter if present.
    text = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)
    raw_chunks = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    for chunk in raw_chunks:
        stripped = chunk.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            continue
        if _HEADING_RE.match(stripped):
            # Headings alone carry little signal — skip.
            continue
        if _BULLET_RE.search(stripped):
            # Split bullets into separate chunks, prepending the non-bullet
            # lead-in text if any (acts as contextual header for each bullet).
            lines = stripped.splitlines()
            header_lines: list[str] = []
            bullets: list[str] = []
            for ln in lines:
                if _BULLET_RE.match(ln):
                    bullets.append(_BULLET_RE.sub("", ln).strip())
                elif not bullets:
                    header_lines.append(ln.strip())
            header = " ".join(header_lines).strip()
            for bullet in bullets:
                fragment = f"{header} {bullet}".strip() if header else bullet
                if len(fragment) >= min_chars:
                    chunks.append(re.sub(r"\s+", " ", fragment))
            continue
        if len(stripped) < min_chars:
            continue
        flat = re.sub(r"\s+", " ", stripped)
        chunks.append(flat)
    return chunks


def extraction_summary(result: Any) -> dict[str, Any]:
    """Extract entity / relationship counts + validity from a ValidationResult.

    `valid` per V2 pass criteria: strict schema conformance AND at least one
    extracted entity. An empty LLM response or truncated JSON parse-failure
    would produce zero entities and is counted as a failure.
    """
    inner = getattr(result, "validation_result", None)
    validation = inner if inner is not None else result
    entities = list(getattr(validation, "entities", []) or [])
    relationships = list(getattr(validation, "relationships", []) or [])
    errors = list(getattr(validation, "errors", []) or [])
    return {
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "error_count": len(errors),
        "valid": (
            bool(getattr(validation, "valid", True)) and len(errors) == 0 and len(entities) > 0
        ),
    }


async def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1
    text = input_path.read_text(encoding="utf-8")
    chunks = split_markdown(text, min_chars=args.min_chars)
    if args.max_chunks and len(chunks) > args.max_chunks:
        chunks = chunks[: args.max_chunks]
    logger.info("Loaded %d chunks from %s", len(chunks), input_path)

    config = get_config()
    dry_pipeline = build_extraction_pipeline(
        config, include_curation=False, include_internal_derivation=False
    )

    session_id = f"v2-ingest-{uuid.uuid4().hex[:8]}"
    dry_results: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, start=1):
        event_id = f"{session_id}-chunk-{idx:03d}"
        preview = chunk[:80].replace("\n", " ")
        logger.info("[%d/%d] extracting: %s...", idx, len(chunks), preview)
        try:
            result = await dry_pipeline.extract_from_utterance(
                utterance=chunk,
                conversation_history=[],
                event_id=event_id,
                session_id=session_id,
                extraction_source="v2-ingest",
            )
        except Exception as e:
            logger.warning("[%d/%d] extraction exception: %s", idx, len(chunks), e)
            dry_results.append(
                {
                    "idx": idx,
                    "event_id": event_id,
                    "chunk": chunk,
                    "error": str(e),
                    "valid": False,
                    "entity_count": 0,
                    "relationship_count": 0,
                }
            )
            continue
        summary = extraction_summary(result)
        dry_results.append(
            {
                "idx": idx,
                "event_id": event_id,
                "chunk": chunk,
                **summary,
            }
        )

    valid_count = sum(1 for r in dry_results if r.get("valid"))
    rate = valid_count / len(dry_results) if dry_results else 0.0
    logger.info(
        "Dry-run complete: %d/%d chunks valid (rate=%.2f)",
        valid_count,
        len(dry_results),
        rate,
    )

    total_entities = sum(r.get("entity_count", 0) for r in dry_results)
    total_rels = sum(r.get("relationship_count", 0) for r in dry_results)
    logger.info("Total dry-run entities=%d relationships=%d", total_entities, total_rels)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(dry_results, indent=2), encoding="utf-8")
        logger.info("Wrote per-chunk report to %s", report_path)

    if not args.commit:
        logger.info("--commit not set; exiting after dry-run.")
        return 0 if rate >= args.commit_threshold else 2

    if rate < args.commit_threshold:
        logger.error(
            "Dry-run success rate %.2f below threshold %.2f; refusing to commit.",
            rate,
            args.commit_threshold,
        )
        return 2

    commit_pipeline = build_extraction_pipeline(
        config, include_curation=True, include_internal_derivation=True
    )
    committed = 0
    skipped = 0
    for row in dry_results:
        if not row.get("valid"):
            skipped += 1
            continue
        logger.info("[commit] chunk %d", row["idx"])
        try:
            await commit_pipeline.extract_from_utterance(
                utterance=row["chunk"],
                conversation_history=[],
                event_id=row["event_id"],
                session_id=session_id,
                extraction_source="v2-ingest-commit",
            )
            committed += 1
        except Exception as e:
            logger.warning("[commit] chunk %d failed: %s", row["idx"], e)
            skipped += 1

    logger.info("Commit phase: %d committed, %d skipped", committed, skipped)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="v2_ingest", description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a markdown file.")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Skip chunks shorter than this character count (default: 40).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Cap total chunks processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="After dry-run, commit chunks that produced valid extraction.",
    )
    parser.add_argument(
        "--commit-threshold",
        type=float,
        default=0.80,
        help="Minimum dry-run success rate required to commit (default: 0.80).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write per-chunk JSON report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
