"""Versioned seed source: models and loader.

The seed source is authored state applied deterministically to the graph
and vault, rather than passed through LLM extraction. See the R1.4 spec
section 2.0 for the projection identity `graph = f(seed@version, log, epoch)`.
"""

from backend.knowledge.seed.loader import load_seed_documents
from backend.knowledge.seed.models import SeedDocument, SeedFact

__all__ = ["SeedFact", "SeedDocument", "load_seed_documents"]
