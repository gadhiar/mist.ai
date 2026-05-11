"""FakeExtractionPipeline — in-memory test double for ExtractionPipelineProtocol.

Satisfies the surface required by GraphRegenerator:
  - extract_from_file(content, vault_note_path, ontology_version)

Tracks scheduled jobs (deferred async extraction calls) via a counter.
"""

from __future__ import annotations


class FakeExtractionPipeline:
    """In-memory test double for ExtractionPipelineProtocol.

    Counts calls to extract_from_file without performing any real
    extraction. `scheduled_jobs` increments on each call so tests can
    assert that deferred extraction was queued.
    """

    def __init__(self) -> None:
        self.scheduled_jobs: int = 0
        self.extract_from_file_calls: list[dict] = []

    async def extract_from_file(
        self,
        content: str,
        vault_note_path: str,
        ontology_version: str,
    ) -> None:
        """Record the call; do not perform real extraction."""
        self.scheduled_jobs += 1
        self.extract_from_file_calls.append(
            {
                "content": content,
                "vault_note_path": vault_note_path,
                "ontology_version": ontology_version,
            }
        )
