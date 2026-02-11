"""
Seed RAG Corpus from MIST Documentation

This script:
1. Reads all MIST documentation markdown files
2. Breaks them into semantic chunks (100-200 words)
3. Stores them as DocumentChunks in Neo4j
4. Generates embeddings for vector search
5. Creates searchable RAG corpus

Entity extraction happens selectively via LLM tools, not during seeding.
This keeps seeding fast and scalable for large corpora.

Use this after wiping the database to populate with MIST documentation.
"""

import asyncio
import logging
import re
import sys
import uuid
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.storage.graph_store import GraphStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DocumentationSeeder:
    """Seeds RAG corpus from MIST documentation"""

    def __init__(self):
        self.config = KnowledgeConfig.from_env()
        self.graph_store = GraphStore(self.config)

        # Documentation paths
        self.docs_root = Path(__file__).parent.parent.parent.parent / "docs"

    def find_documentation_files(self) -> list[Path]:
        """Find all markdown documentation files"""
        doc_files = []

        # Recursively find all .md files in docs directory
        if self.docs_root.exists():
            doc_files.extend(self.docs_root.rglob("*.md"))

        # Also check root directory for key docs
        root = self.docs_root.parent
        for pattern in [
            "README.md",
            "QUICKSTART*.md",
            "E2E*.md",
            "INTEGRATION*.md",
            "REPOSITORY*.md",
        ]:
            doc_files.extend(root.glob(pattern))

        logger.info(f"Found {len(doc_files)} documentation files")
        return doc_files

    def read_document(self, file_path: Path) -> tuple[str, str]:
        """
        Read a documentation file and extract title and content

        Returns:
            Tuple of (title, content)
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Extract title from first heading or filename
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
            else:
                title = file_path.stem.replace("_", " ").title()

            return title, content

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return "", ""

    def chunk_document(self, title: str, content: str, file_path: Path) -> list[str]:
        """
        Break document into entity-rich chunks

        Strategy:
        - Split by headers (##, ###) to preserve logical sections
        - Keep chunks focused (100-300 words) for better extraction
        - Maintain context with section titles
        - Convert to natural statements for extraction
        """
        chunks = []

        # Split by second-level headers and below
        sections = re.split(r"\n(?=#{2,}\s)", content)

        for section in sections:
            # Extract section title if present
            section_title_match = re.search(r"^#{2,}\s+(.+)$", section, re.MULTILINE)
            section_title_match.group(1).strip() if section_title_match else ""

            # Remove markdown formatting for cleaner extraction
            clean_section = re.sub(r"```[\s\S]*?```", "", section)  # Remove code blocks
            clean_section = re.sub(r"`[^`]+`", "", clean_section)  # Remove inline code
            clean_section = re.sub(r"#{1,6}\s+", "", clean_section)  # Remove headers
            clean_section = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_section)  # Remove links
            clean_section = re.sub(
                r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", clean_section
            )  # Remove bold/italic
            clean_section = re.sub(r"\n+", " ", clean_section)  # Normalize whitespace
            clean_section = clean_section.strip()

            if len(clean_section) < 50:  # Skip tiny sections
                continue

            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", clean_section)

            # Group sentences into chunks of ~150 words
            current_chunk = []
            current_words = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                words = len(sentence.split())

                # Start new chunk if too large
                if current_words + words > 200 and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) > 50:  # Minimum chunk size
                        chunks.append(chunk_text)

                    current_chunk = []
                    current_words = 0

                current_chunk.append(sentence)
                current_words += words

            # Add remaining chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) > 50:
                    chunks.append(chunk_text)

        logger.debug(f"Chunked '{title}' into {len(chunks)} pieces")
        return chunks

    async def seed_from_documentation(self):
        """Main seeding process"""

        print("\n" + "=" * 60)
        print("SEEDING KNOWLEDGE GRAPH FROM MIST DOCUMENTATION")
        print("=" * 60)

        # Initialize schema
        print("\n[1/5] Initializing Neo4j schema...")
        self.graph_store.initialize_schema()

        # No need to create conversation event for documentation
        print("[2/5] Ready to process documents...")

        # Find all documentation files
        print("[3/5] Finding documentation files...")
        doc_files = self.find_documentation_files()

        # Process each document
        print(f"[4/5] Processing {len(doc_files)} documents...")

        total_chunks = 0

        for i, doc_file in enumerate(doc_files, 1):
            try:
                # Read and chunk document
                title, content = self.read_document(doc_file)
                if not content:
                    continue

                chunks = self.chunk_document(title, content, doc_file)
                total_chunks += len(chunks)

                print(f"\n[{i}/{len(doc_files)}] Processing: {title}")
                print(f"  File: {doc_file.relative_to(doc_file.parent.parent.parent)}")
                print(f"  Chunks: {len(chunks)}")

                # Calculate content hash for deduplication
                import hashlib

                content_hash = hashlib.sha256(content.encode()).hexdigest()

                # Store source document
                source_id = str(uuid.uuid4())
                self.graph_store.store_source_document(
                    source_id=source_id,
                    file_path=str(doc_file),
                    source_type="markdown",
                    content_hash=content_hash,
                    title=title,
                    file_size=len(content),
                )

                # Store each chunk (extraction happens on-demand via LLM tools)
                for j, chunk_text in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())

                    # Store document chunk with auto-generated embedding
                    self.graph_store.store_document_chunk(
                        chunk_id=chunk_id, source_id=source_id, text=chunk_text, position=j
                    )

                print(f"  Stored: {len(chunks)} chunks with embeddings")

            except Exception as e:
                logger.error(f"Error processing {doc_file}: {e}")
                continue

        # Summary
        print("\n[5/5] Seeding complete!")
        print("\n" + "=" * 60)
        print("SEEDING SUMMARY")
        print("=" * 60)
        print(f"Source documents: {len(doc_files)}")
        print(f"Document chunks: {total_chunks}")
        print("=" * 60)
        print("\nRAG Corpus Architecture:")
        print("  SourceDocument -> DocumentChunk (with embeddings)")
        print("  Chunks are searchable via vector similarity")
        print("  LLM can optionally extract entities from chunks into KG")
        print("\nThe RAG corpus is now seeded with MIST documentation!")
        print("Use search_documents tool for retrieval")
        print("Use extract_knowledge_from_document for selective extraction")
        print("=" * 60 + "\n")


async def main():
    """Run the seeding process"""
    seeder = DocumentationSeeder()
    await seeder.seed_from_documentation()


if __name__ == "__main__":
    asyncio.run(main())
