"""Build the real embedding provider for the calibration tool.

`backend/factories.py` has no standalone embedding-provider factory: each
builder inlines `EmbeddingGenerator(config.embedding.model_name)` (for example
`build_graph_store`, `backend/factories.py:185-188`). This module does the same
with the same config source (`EmbeddingConfig.from_env`, which reads
`EMBEDDING_MODEL` and defaults to all-MiniLM-L6-v2), so the tool measures the
model the graph runs. It then loads the model eagerly so a missing model fails
here, with a clear error, instead of midway through scoring.
"""

from __future__ import annotations

from backend.errors import EmbeddingError
from backend.interfaces import EmbeddingProvider


class EmbedderUnavailableError(RuntimeError):
    """The embedding model could not be loaded or run."""


def build_real_embedder() -> tuple[EmbeddingProvider, str]:
    """Build and warm the project's EmbeddingGenerator.

    Returns:
        The provider and the model name it was built for.

    Raises:
        EmbedderUnavailableError: The model or its dependencies cannot be
            loaded (no HF cache and no network, missing package, and so on).
    """
    try:
        from backend.knowledge.config import EmbeddingConfig
        from backend.knowledge.embeddings import EmbeddingGenerator

        model_name = EmbeddingConfig.from_env().model_name
        generator = EmbeddingGenerator(model_name)
        generator.warmup()
    except (ImportError, OSError, RuntimeError, ValueError, EmbeddingError) as exc:
        raise EmbedderUnavailableError(
            f"cannot load the embedding model: {type(exc).__name__}: {exc}"
        ) from exc
    return generator, model_name
