"""Embedding Generation Module.

Generates vector embeddings for text using sentence-transformers.
Enables semantic search on the knowledge graph.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates vector embeddings for text using sentence-transformers.

    Uses all-MiniLM-L6-v2 model by default:
    - 384 dimensions
    - Fast inference (~10ms per text)
    - Good quality for semantic search
    - Small model size (80MB)

    Example:
        generator = EmbeddingGenerator()
        embedding = generator.generate_embedding("Python programming")
        # Returns list of 384 floats
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding generator.

        Args:
            model_name: Sentence-transformers model name
                Options:
                - all-MiniLM-L6-v2 (default): Fast, 384d, good quality
                - all-mpnet-base-v2: Slower, 768d, better quality
        """
        self.model_name = model_name
        self._model = None
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")

    def _get_model(self) -> SentenceTransformer:
        """Lazy load the embedding model.

        Only loads model when first needed to save memory and startup time.
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self._model

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding (384 dimensions)

        Example:
            >>> generator = EmbeddingGenerator()
            >>> embedding = generator.generate_embedding("databases")
            >>> len(embedding)
            384
            >>> isinstance(embedding[0], float)
            True
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * 384

        model = self._get_model()

        # Generate embedding
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)

        # Convert to list and return
        return embedding.tolist()

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (more efficient than one-by-one).

        Args:
            texts: List of input texts

        Returns:
            List of embeddings (one per input text)

        Example:
            >>> texts = ["Python", "JavaScript", "databases"]
            >>> embeddings = generator.generate_embeddings_batch(texts)
            >>> len(embeddings)
            3
            >>> len(embeddings[0])
            384
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t if t and t.strip() else " " for t in texts]

        model = self._get_model()

        # Generate embeddings in batch (much faster)
        embeddings = model.encode(
            valid_texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,  # Show progress for large batches
            batch_size=32,
        )

        # Convert to list of lists
        return embeddings.tolist()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1 (1 = identical, 0 = completely different)

        Example:
            >>> similarity = generator.compute_similarity("database", "PostgreSQL")
            >>> 0.5 < similarity < 0.9  # Semantically related
            True
        """
        emb1 = np.array(self.generate_embedding(text1))
        emb2 = np.array(self.generate_embedding(text2))

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)
