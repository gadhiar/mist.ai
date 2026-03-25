"""Knowledge extraction module.

Ontology-constrained extraction pipeline (stages 1-8):

  ExtractionPipeline  -- orchestrator (stages 1-6, optional 7-8 curation)
  OntologyConstrainedExtractor -- single LLM call with ontology constraints
  PreProcessor -- context assembly (no LLM)
  ConfidenceScorer -- hedge detection, third-party cap
  TemporalResolver -- relative -> absolute dates
  EntityNormalizer -- canonical IDs, alias + embedding dedup
  ExtractionValidator -- schema + constraint checks
"""

from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import (
    ExtractionResult,
    OntologyConstrainedExtractor,
)
from backend.knowledge.extraction.pipeline import ExtractionPipeline
from backend.knowledge.extraction.preprocessor import PreProcessedInput, PreProcessor
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.tool_classifier import ToolOutputClassifier
from backend.knowledge.extraction.validator import ExtractionValidator, ValidationResult

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
