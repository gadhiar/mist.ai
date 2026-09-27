"""Guards the PEP 562 lazy `__getattr__` on backend.knowledge.storage and
backend.knowledge.extraction (goal mist-two-loop v2-lazy-imports, MIS-171).

Both package `__init__` modules used to import their public names eagerly,
which meant every one of them pulled in `neo4j` (and pandas/pyarrow behind
it) as a side effect of importing ANY submodule of either package -- see
docker/extraction/requirements.txt's header comment for the full chain.
They are now lazy: every name that used to be importable still is, resolved
on first attribute access via `__getattr__`, and `dir()` still lists all of
them via `__dir__`. This module is the regression guard for that contract --
it does not re-litigate the import-closure measurement itself, which lives
in tests/unit/extraction_service/test_import_closure.py.
"""

import backend.knowledge.extraction as extraction_pkg
import backend.knowledge.storage as storage_pkg
from backend.knowledge.extraction import (
    ConfidenceScorer,
    EntityNormalizer,
    ExtractionPipeline,
    ExtractionResult,
    ExtractionValidator,
    OntologyConstrainedExtractor,
    PreProcessedInput,
    PreProcessor,
    TemporalResolver,
    ToolOutputClassifier,
    ValidationResult,
)
from backend.knowledge.storage import GraphStore, LanceDBVectorStore, Neo4jConnection


class TestStorageLazyExports:
    def test_all_public_names_resolve(self) -> None:
        assert GraphStore.__name__ == "GraphStore"
        assert Neo4jConnection.__name__ == "Neo4jConnection"
        assert LanceDBVectorStore.__name__ == "LanceDBVectorStore"

    def test_dir_lists_all_public_names(self) -> None:
        listed = set(dir(storage_pkg))
        assert set(storage_pkg.__all__) <= listed

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        try:
            storage_pkg.DoesNotExist  # noqa: B018
        except AttributeError as exc:
            assert "DoesNotExist" in str(exc)
        else:
            raise AssertionError("expected AttributeError for an unknown name")


class TestExtractionLazyExports:
    def test_all_public_names_resolve(self) -> None:
        assert ExtractionPipeline.__name__ == "ExtractionPipeline"
        assert OntologyConstrainedExtractor.__name__ == "OntologyConstrainedExtractor"
        assert ExtractionResult.__name__ == "ExtractionResult"
        assert PreProcessor.__name__ == "PreProcessor"
        assert PreProcessedInput.__name__ == "PreProcessedInput"
        assert ConfidenceScorer.__name__ == "ConfidenceScorer"
        assert TemporalResolver.__name__ == "TemporalResolver"
        assert EntityNormalizer.__name__ == "EntityNormalizer"
        assert ExtractionValidator.__name__ == "ExtractionValidator"
        assert ValidationResult.__name__ == "ValidationResult"
        assert ToolOutputClassifier.__name__ == "ToolOutputClassifier"

    def test_dir_lists_all_public_names(self) -> None:
        listed = set(dir(extraction_pkg))
        assert set(extraction_pkg.__all__) <= listed

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        try:
            extraction_pkg.DoesNotExist  # noqa: B018
        except AttributeError as exc:
            assert "DoesNotExist" in str(exc)
        else:
            raise AssertionError("expected AttributeError for an unknown name")

    def test_all_matches_the_documented_names(self) -> None:
        # Guards __all__ itself against silent drift now that nothing else
        # in the module body references these names directly.
        assert extraction_pkg.__all__ == [
            "ExtractionPipeline",
            "OntologyConstrainedExtractor",
            "ExtractionResult",
            "PreProcessor",
            "PreProcessedInput",
            "ConfidenceScorer",
            "TemporalResolver",
            "EntityNormalizer",
            "ExtractionValidator",
            "ValidationResult",
            "ToolOutputClassifier",
        ]
