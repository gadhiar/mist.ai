"""
Knowledge System Data Models

Core data structures for working with the knowledge graph.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict


@dataclass
class Utterance:
    """
    Represents an immutable utterance from conversation.

    This is the source of truth - the knowledge graph can always
    be regenerated from utterances.
    """
    utterance_id: str
    conversation_id: str
    text: str
    timestamp: datetime
    metadata: Optional[Dict] = None

    def __str__(self):
        return f"Utterance({self.utterance_id[:8]}...): {self.text[:50]}..."


@dataclass
class RegenerationReport:
    """
    Statistics and results from graph regeneration.

    Tracks what happened during regeneration process.
    """
    total_utterances: int
    processed: int
    failed: int
    entities_created: int
    relationships_created: int
    duration_seconds: float
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_utterances == 0:
            return 0.0
        return (self.processed / self.total_utterances) * 100

    def __str__(self):
        return (
            f"RegenerationReport:\n"
            f"  Utterances: {self.processed}/{self.total_utterances} "
            f"({self.success_rate:.1f}% success)\n"
            f"  Entities: {self.entities_created}\n"
            f"  Relationships: {self.relationships_created}\n"
            f"  Duration: {self.duration_seconds:.2f}s\n"
            f"  Errors: {len(self.errors)}"
        )


@dataclass
class SearchResult:
    """
    Result from vector similarity search.

    Contains entity information and similarity score.
    """
    entity_id: str
    entity_type: str
    similarity: float
    properties: Optional[Dict] = None

    def __str__(self):
        return f"{self.entity_id} ({self.entity_type}) - {self.similarity:.3f}"
