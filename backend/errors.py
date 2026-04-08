"""Custom exception hierarchy for MIST.AI backend.

All MIST-specific exceptions inherit from MistError, which itself inherits
from the built-in Exception. This allows callers to catch broad categories
(e.g., all Neo4j errors) or the entire MIST tree via MistError.

Hierarchy:

    MistError
    +-- Neo4jConnectionError
    +-- Neo4jQueryError
    +-- LLMConnectionError
    +-- LLMResponseError
    +-- ExtractionError
    +-- ExtractionValidationError
    +-- NormalizationError
    +-- CurationError
    +-- InternalDerivationError
    +-- EmbeddingError
    +-- VectorStoreError
    +-- IngestionError

Usage:
    from backend.errors import Neo4jConnectionError

    try:
        driver.verify_connectivity()
    except Exception as exc:
        raise Neo4jConnectionError("Failed to reach Neo4j") from exc
"""


class MistError(Exception):
    """Base exception for all MIST.AI backend errors."""


# -- Neo4j / graph database ------------------------------------------------


class Neo4jConnectionError(MistError):
    """Raised when a connection to Neo4j cannot be established or is lost."""


class Neo4jQueryError(MistError):
    """Raised when a Cypher query fails or returns unexpected results."""


# -- LLM backend -------------------------------------------------------------


class LLMConnectionError(MistError):
    """Raised when the LLM service is unreachable or refuses a connection."""


class LLMResponseError(MistError):
    """Raised when the LLM service returns an invalid or unparsable response."""


# -- Knowledge pipeline ----------------------------------------------------


class ExtractionError(MistError):
    """Raised when entity or relation extraction from text fails."""


class ExtractionValidationError(MistError):
    """Raised when extracted data fails structural or semantic validation."""


class NormalizationError(MistError):
    """Raised when entity normalization or deduplication fails."""


class CurationError(MistError):
    """Curation pipeline stage failure (dedup, conflict resolution, graph write)."""


class InternalDerivationError(MistError):
    """Internal knowledge derivation failure (signal detection, LLM call, entity creation)."""


# -- Embeddings ------------------------------------------------------------


class EmbeddingError(MistError):
    """Raised when generating or storing a vector embedding fails."""


# -- Vector store --------------------------------------------------------------


class VectorStoreError(MistError):
    """Vector store operation failed."""


# -- Ingestion ----------------------------------------------------------------


class IngestionError(MistError):
    """Document ingestion pipeline failure."""
