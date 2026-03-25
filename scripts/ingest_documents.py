"""Batch document ingestion into the MIST.AI knowledge pipeline.

Usage:
    python scripts/ingest_documents.py <path> [--source-type markdown]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.factories import build_graph_store, build_ingestion_pipeline
from backend.knowledge.config import KnowledgeConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst", ".py"}


def ingest_path(path: Path, source_type: str) -> None:
    config = KnowledgeConfig.from_env()
    gs = build_graph_store(config)
    gs.initialize_schema()
    pipeline = build_ingestion_pipeline(config, graph_store=gs)

    files = [path] if path.is_file() else sorted(path.rglob("*"))
    files = [f for f in files if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS]

    if not files:
        logger.warning("No supported files found at %s", path)
        return

    logger.info("Ingesting %d files from %s", len(files), path)

    total_chunks = 0
    for file_path in files:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            continue

        result = pipeline.ingest_document(
            content=content,
            file_path=str(file_path),
            source_type=source_type,
            title=file_path.stem,
        )

        if not result.deduplicated:
            total_chunks += result.chunks_created
            logger.info(
                "[OK] %s: %d chunks (%.1fms)",
                file_path.name,
                result.chunks_created,
                result.duration_ms,
            )
        else:
            logger.info("[SKIP] %s: duplicate", file_path.name)

    logger.info("Ingestion complete: %d total chunks from %d files", total_chunks, len(files))


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into MIST knowledge pipeline")
    parser.add_argument("path", type=Path, help="File or directory to ingest")
    parser.add_argument(
        "--source-type", default="markdown", help="Content type (default: markdown)"
    )
    args = parser.parse_args()

    if not args.path.exists():
        logger.error("Path does not exist: %s", args.path)
        sys.exit(1)

    ingest_path(args.path, args.source_type)


if __name__ == "__main__":
    main()
