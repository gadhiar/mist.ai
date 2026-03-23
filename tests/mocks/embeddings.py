"""Test double for embedding generation.

FakeEmbeddingGenerator satisfies the EmbeddingProvider protocol.
Produces deterministic vectors by hashing input text.
"""

import hashlib

EMBEDDING_DIMENSION = 384  # Must match production model (all-MiniLM-L6-v2)


class FakeEmbeddingGenerator:
    """Deterministic embeddings for testing. Satisfies EmbeddingProvider protocol.

    Vectors are derived from SHA-256 hash of input text, ensuring:
    - Same input always produces same output (deterministic)
    - Different inputs produce different outputs (distinguishable)
    - Vectors are the correct dimension (384 by default)

    Note: Only the first 32 dimensions have non-zero values (from SHA-256).
    Positions 32-383 are zero-padded. This is sufficient for unit tests
    but should not be used for similarity threshold testing.
    """

    def __init__(self, *, dimension: int = EMBEDDING_DIMENSION):
        self._dimension = dimension
        self.calls: list[str] = []

    def generate_embedding(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text."""
        self.calls.append(text)
        h = hashlib.sha256(text.encode()).digest()
        vec = [float(b) / 255.0 for b in h[: self._dimension]]
        return (vec + [0.0] * self._dimension)[: self._dimension]

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.generate_embedding(t) for t in texts]
