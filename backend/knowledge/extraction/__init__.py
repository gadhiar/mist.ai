"""Knowledge extraction module.

Ontology-constrained extraction pipeline (stages 1-8):

  ExtractionPipeline  -- orchestrator (stages 1-6, optional 7-8 curation)
  OntologyConstrainedExtractor -- single LLM call with ontology constraints
  PreProcessor -- context assembly (no LLM)
  ConfidenceScorer -- hedge detection, third-party cap
  TemporalResolver -- relative -> absolute dates
  EntityNormalizer -- canonical IDs, alias + embedding dedup
  ExtractionValidator -- schema + constraint checks

PEP 562 lazy exports: `pipeline.py` imports `GraphStore`, so an eager
`from backend.knowledge.extraction.pipeline import ExtractionPipeline` here
pulled `neo4j` (and pandas/pyarrow behind it) into every import of this
package or any of its submodules (importing a submodule always runs the
package `__init__` first). Every name in `__all__` stays importable via
`from backend.knowledge.extraction import X`; only the eager import at
package-init time is removed.
"""

__all__ = [
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


# name -> submodule that defines it, for the lazy __getattr__ below. Several
# names share a submodule (e.g. ExtractionResult/OntologyConstrainedExtractor
# both live in ontology_extractor); each submodule is only imported once
# per process either way, since Python caches it in `sys.modules`.
_SUBMODULE_BY_NAME = {
    "ExtractionPipeline": "pipeline",
    "ExtractionResult": "ontology_extractor",
    "OntologyConstrainedExtractor": "ontology_extractor",
    "PreProcessor": "preprocessor",
    "PreProcessedInput": "preprocessor",
    "ConfidenceScorer": "confidence",
    "TemporalResolver": "temporal",
    "EntityNormalizer": "normalizer",
    "ExtractionValidator": "validator",
    "ValidationResult": "validator",
    "ToolOutputClassifier": "tool_classifier",
}


def __getattr__(name: str):
    """Lazy import of this package's public names."""
    submodule_name = _SUBMODULE_BY_NAME.get(name)
    if submodule_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    submodule = importlib.import_module(f"backend.knowledge.extraction.{submodule_name}")
    return getattr(submodule, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
