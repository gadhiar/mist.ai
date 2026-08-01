"""Embedding generation for semantic search.

`EmbeddingGenerator` is exported lazily (PEP 562 module `__getattr__`)
rather than imported eagerly. `embedding_generator` imports
`sentence_transformers` at module scope, which costs ~2.9s, and I7 Task 1
made this package a dependency of two import paths that never touch a
model: `knowledge.admin` (every `mist_admin.py` subcommand, including
read-only ones like `stats` and `seed-verify`) and `knowledge.seed.gates`,
both of which need only `embedding_text_for` -- a pure-string helper with
no dependencies at all. Measured directly: importing `knowledge.admin`
went 0.59s -> 3.43s when the eager re-export was still in place, for a
module that may never construct a generator.

The lazy form keeps `from backend.knowledge.embeddings import
EmbeddingGenerator` (five call sites in `backend/factories.py`) working
unchanged, and does not affect
`backend.knowledge.embeddings.embedding_generator.EmbeddingGenerator`,
which is patched by name in `tests/unit/test_admin_vault_cli.py` and
imported directly by `scripts/mist_admin.py`.
"""

from typing import TYPE_CHECKING, Any

from backend.knowledge.embeddings.embedding_text import embedding_text_for

if TYPE_CHECKING:
    from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator

__all__ = ["EmbeddingGenerator", "embedding_text_for"]


def __getattr__(name: str) -> Any:
    """Resolve `EmbeddingGenerator` on first access, importing the model layer then."""
    if name == "EmbeddingGenerator":
        from backend.knowledge.embeddings.embedding_generator import (
            EmbeddingGenerator as _EmbeddingGenerator,
        )

        return _EmbeddingGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
