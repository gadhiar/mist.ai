"""Per-conversation curation pipeline (Phase 2A).

Stages 7-8 of the knowledge extraction pipeline. Deduplicates entities
against existing graph state, resolves relationship conflicts, and writes
curated knowledge to Neo4j with provenance tracking.
"""

from backend.knowledge.curation.pipeline import CurationPipeline, CurationResult

__all__ = ["CurationPipeline", "CurationResult"]
