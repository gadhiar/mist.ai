"""Document ingestion pipeline.

Chunking, embedding, and storage of documents into the vector store.
"""

from backend.knowledge.ingestion.chunker import FixedSizeChunker, MarkdownChunker
from backend.knowledge.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "FixedSizeChunker",
    "IngestionPipeline",
    "IngestionResult",
    "MarkdownChunker",
]
