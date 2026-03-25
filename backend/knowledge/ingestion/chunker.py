"""Document chunking strategies.

Provides word-based fixed-size and markdown-aware chunking for the
ingestion pipeline. No external tokenizer dependencies -- chunking
uses `str.split()` for word tokenization.
"""

from __future__ import annotations

import re
import uuid
from typing import Protocol

from backend.knowledge.models import DocumentChunk


class ChunkingStrategy(Protocol):
    """Contract for document chunking implementations."""

    def chunk(self, content: str, source_id: str, source_type: str) -> list[DocumentChunk]:
        """Split content into document chunks."""
        ...


class FixedSizeChunker:
    """Word-based fixed-size sliding window chunker.

    Splits text into words via `str.split()`, then produces chunks of
    `chunk_size` words with `chunk_overlap` words of overlap between
    consecutive chunks.

    Args:
        chunk_size: Target number of words per chunk.
        chunk_overlap: Number of overlapping words between adjacent chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than " f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, content: str, source_id: str, source_type: str) -> list[DocumentChunk]:
        """Split content into fixed-size word chunks with overlap.

        Args:
            content: Raw text to chunk.
            source_id: Parent source document identifier.
            source_type: Source type label for metadata.

        Returns:
            List of DocumentChunk objects with position indices.
        """
        words = content.split()
        if not words:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[DocumentChunk] = []
        position = 0

        for start in range(0, len(words), step):
            chunk_words = words[start : start + self.chunk_size]
            text = " ".join(chunk_words)

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    source_id=source_id,
                    text=text,
                    position=position,
                    embedding=None,
                    metadata={"source_type": source_type},
                )
            )
            position += 1

            # Stop if we've consumed all words
            if start + self.chunk_size >= len(words):
                break

        return chunks


# Regex for splitting on markdown headings (h1-h3).
# Matches newline followed by 1-3 '#' characters and whitespace.
_HEADING_RE = re.compile(r"\n(?=#{1,3}\s)")

# Simple fence detector: tracks whether we're inside a code block.
_FENCE_RE = re.compile(r"^```", re.MULTILINE)


def _strip_code_fences(content: str) -> str:
    """Replace code fence contents with placeholder to avoid false heading splits.

    Lines starting with '#' inside ``` fences are not headings. This
    function masks them so the heading regex does not match them.

    Returns a copy of content with code-block lines replaced by blank lines,
    preserving line count for position tracking.
    """
    lines = content.split("\n")
    result: list[str] = []
    in_fence = False

    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            result.append(line)
        elif in_fence:
            # Replace code-block lines with empty to avoid heading matches
            result.append("")
        else:
            result.append(line)

    return "\n".join(result)


def _extract_heading(section: str) -> str | None:
    """Extract heading text from the first line of a section.

    Returns:
        The heading text (without '#' prefix) or None if no heading found.
    """
    first_line = section.split("\n", 1)[0].strip()
    match = re.match(r"^#{1,3}\s+(.+)$", first_line)
    return match.group(1).strip() if match else None


class MarkdownChunker:
    """Markdown-aware chunker that respects heading boundaries.

    Splits on h1-h3 heading boundaries. Sections smaller than 20 words
    are merged with the next section. Sections exceeding `chunk_size`
    words are sub-chunked using FixedSizeChunker internally.

    Overlap is NOT applied across heading boundaries to preserve
    semantic breaks.

    Args:
        chunk_size: Target number of words per chunk.
        chunk_overlap: Overlap for sub-chunking large sections.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._sub_chunker = FixedSizeChunker(chunk_size, chunk_overlap)

    def chunk(self, content: str, source_id: str, source_type: str) -> list[DocumentChunk]:
        """Split markdown content into heading-aware chunks.

        Args:
            content: Raw markdown text.
            source_id: Parent source document identifier.
            source_type: Source type label for metadata.

        Returns:
            List of DocumentChunk objects preserving heading structure.
        """
        if not content or not content.strip():
            return []

        # Mask code fences to prevent false heading splits
        masked = _strip_code_fences(content)

        # Split on heading boundaries using the masked content for boundary
        # detection, but use the original content for actual text.
        # We split masked content to find split points, then apply to original.
        _raw_sections = _HEADING_RE.split(masked)  # noqa: F841

        # Now split the original content at the same positions
        original_sections = _HEADING_RE.split(content)

        # If no headings found, treat entire content as one section
        if len(original_sections) <= 1:
            original_sections = [content]

        # Merge small sections (< 20 words) with the next section
        sections = self._merge_small_sections(original_sections)

        chunks: list[DocumentChunk] = []
        position = 0

        for section in sections:
            section_text = section.strip()
            if not section_text:
                continue

            section_title = _extract_heading(section_text)
            word_count = len(section_text.split())

            if word_count > self.chunk_size:
                # Sub-chunk large sections
                sub_chunks = self._sub_chunker.chunk(section_text, source_id, source_type)
                for sc in sub_chunks:
                    sc_chunk = DocumentChunk(
                        chunk_id=sc.chunk_id,
                        source_id=source_id,
                        text=sc.text,
                        position=position,
                        embedding=None,
                        section_title=section_title,
                        metadata={"source_type": source_type},
                    )
                    chunks.append(sc_chunk)
                    position += 1
            else:
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        source_id=source_id,
                        text=section_text,
                        position=position,
                        embedding=None,
                        section_title=section_title,
                        metadata={"source_type": source_type},
                    )
                )
                position += 1

        return chunks

    @staticmethod
    def _merge_small_sections(sections: list[str]) -> list[str]:
        """Merge sections with fewer than 20 words into the next section.

        The last section is never merged forward (kept as-is regardless
        of size).
        """
        if len(sections) <= 1:
            return sections

        merged: list[str] = []
        carry = ""

        for i, section in enumerate(sections):
            combined = (carry + "\n" + section).strip() if carry else section
            word_count = len(combined.split())
            is_last = i == len(sections) - 1

            if is_last or word_count >= 20:
                merged.append(combined)
                carry = ""
            else:
                carry = combined

        # If there's leftover carry, append it
        if carry:
            if merged:
                merged[-1] = merged[-1] + "\n" + carry
            else:
                merged.append(carry)

        return merged
