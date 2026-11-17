"""Embedding Generation Module.

Generates vector embeddings for text using sentence-transformers.
Enables semantic search on the knowledge graph.
"""

import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Where the backend image bakes its pinned model copies (Dockerfile step 7,
# `docker/backend/models.lock`). Read when a generator is constructed rather than
# bound at import, so a test can point it elsewhere.
DEFAULT_MODELS_ROOT = "/opt/models"


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

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        *,
        models_root: str | None = None,
        attn_implementation: str | None = None,
    ):
        """Initialize embedding generator.

        Args:
            model_name: Sentence-transformers model name
                Options:
                - all-MiniLM-L6-v2 (default): Fast, 384d, good quality
                - all-mpnet-base-v2: Slower, 768d, better quality
                It stays the model's identity wherever the weights load from;
                `compose_model_hash` stamps this name, not the load path.
            device: Torch device the model loads onto, such as `cpu`, `cuda`
                or `cuda:1`. None lets sentence-transformers choose, which
                means CUDA whenever it is available, so runtime callers pass
                `config.embedding.resolved_device()` to keep the model where
                `EMBEDDING_DEVICE` says.
            models_root: Directory holding baked model copies, one
                subdirectory per model name. None means `DEFAULT_MODELS_ROOT`,
                the path the backend image bakes into.
            attn_implementation: Attention kernel the model loads with, such
                as `sdpa`, passed to `SentenceTransformer` as
                `model_kwargs={"attn_implementation": ...}` on both load
                branches. None passes no `model_kwargs`, so the load is as it
                was before the keyword existed. It exists for
                granite-embedding-small-english-r2, a ModernBERT model the
                need gate runs on CPU. Measured at H20 (15 Nov 2026) in the
                backend image (transformers 4.57.6, sentence-transformers
                3.3.1) with the GPU visible, no network, `local_files_only`,
                no `trust_remote_code` and device `cpu`: the load with `sdpa`
                embedded one sentence to a 384-length vector, and the load
                without `model_kwargs` failed with `ValueError: Pointer
                argument (at 0) cannot be accessed from Triton (cpu tensor?)`,
                because the default load picked a Triton (flash-attention)
                kernel for a CPU forward.
        """
        self.model_name = model_name
        self.device = device
        self.models_root = DEFAULT_MODELS_ROOT if models_root is None else models_root
        self.attn_implementation = attn_implementation
        self._model = None
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}, device: {device}")

    def _get_model(self) -> SentenceTransformer:
        """Lazy load the embedding model.

        Only loads model when first needed to save memory and startup time.
        When `<models_root>/<model_name>` is a directory, the pinned copy the
        image baked loads from it with `local_files_only=True`, so the load
        needs no network. Anywhere else, such as outside the image, the model
        loads by name as before. A `model_name` that is already an absolute
        path also loads as given: `Path` joining would return it unchanged and
        switch that caller to `local_files_only=True` without its asking.
        """
        if self._model is None:
            extra: dict[str, dict[str, str]] = {}
            if self.attn_implementation is not None:
                extra["model_kwargs"] = {"attn_implementation": self.attn_implementation}
            local_path = Path(self.models_root) / self.model_name
            if not Path(self.model_name).is_absolute() and local_path.is_dir():
                self._model = SentenceTransformer(
                    str(local_path), device=self.device, local_files_only=True, **extra
                )
                source = str(local_path)
            else:
                self._model = SentenceTransformer(self.model_name, device=self.device, **extra)
                source = "its name"
            logger.info(f"Embedding model {self.model_name} loaded from {source}")
        return self._model

    def warmup(self) -> None:
        """Eagerly load the embedding model and run a dummy encode.

        Call during startup to avoid cold-start latency on the first
        real embedding request. Safe to call multiple times -- subsequent
        calls are no-ops.
        """
        model = self._get_model()
        # Run a dummy encode so any one-time initialization is completed
        model.encode("warmup", convert_to_numpy=True, show_progress_bar=False)
        logger.info("Embedding model warmup complete")

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

    # Alias to satisfy EmbeddingProvider protocol
    generate_embeddings = generate_embeddings_batch

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
