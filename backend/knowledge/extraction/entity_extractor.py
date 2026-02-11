"""
Entity Extraction Module

Extracts entities and relationships from conversational text using LLMGraphTransformer.
"""

import logging

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.extraction.property_enricher import PropertyEnricher

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts entities and relationships from conversational text

    Uses LangChain's LLMGraphTransformer with configurable LLM model.
    """

    def __init__(self, config: KnowledgeConfig, enable_property_enrichment: bool = True):
        """
        Initialize entity extractor

        Args:
            config: Knowledge system configuration
            enable_property_enrichment: Whether to enrich with property extraction
        """
        self.config = config
        self.enable_property_enrichment = enable_property_enrichment
        self._llm = None
        self._transformer = None
        self._property_enricher = (
            None if not enable_property_enrichment else PropertyEnricher(config)
        )

    def _get_llm(self) -> ChatOllama:
        """Get or create LLM instance"""
        if self._llm is None:
            logger.info(f"Initializing LLM: {self.config.llm.model}")
            self._llm = ChatOllama(
                model=self.config.llm.model,
                base_url=self.config.llm.base_url,
                temperature=self.config.llm.temperature,
            )
        return self._llm

    def _get_transformer(self) -> LLMGraphTransformer:
        """Get or create graph transformer"""
        if self._transformer is None:
            llm = self._get_llm()

            logger.info("Initializing LLMGraphTransformer")
            self._transformer = LLMGraphTransformer(
                llm=llm,
                node_properties=(
                    ["description"] if self.config.extraction.extract_node_properties else False
                ),
                relationship_properties=(
                    ["description"]
                    if self.config.extraction.extract_relationship_properties
                    else False
                ),
                allowed_nodes=self.config.extraction.allowed_nodes,
                allowed_relationships=self.config.extraction.allowed_relationships,
                additional_instructions=self.config.extraction.additional_instructions,
            )

        return self._transformer

    async def extract_from_utterance(
        self,
        utterance: str,
        conversation_history: list[str] | None = None,
        metadata: dict | None = None,
    ) -> list:
        """
        Extract entities and relationships from a single utterance

        Args:
            utterance: User's statement
            conversation_history: Previous conversation turns for context
            metadata: Additional metadata to attach

        Returns:
            List of GraphDocument objects with extracted nodes and relationships
        """
        # Build context
        if conversation_history:
            # Include last 5 turns for context
            context = "\n".join(conversation_history[-5:]) + "\n" + utterance
        else:
            context = utterance

        # Create document
        doc = Document(page_content=context, metadata=metadata or {"utterance": utterance})

        # Extract
        logger.debug(f"Extracting from: {utterance}")
        transformer = self._get_transformer()

        try:
            graph_docs = await transformer.aconvert_to_graph_documents([doc])

            if graph_docs:
                logger.info(
                    f"Extracted {len(graph_docs[0].nodes)} nodes and {len(graph_docs[0].relationships)} relationships"
                )

                # Enrich with properties if enabled
                if self.enable_property_enrichment and self._property_enricher:
                    for graph_doc in graph_docs:
                        await self._property_enricher.enrich_graph_document(
                            graph_doc, original_text=utterance
                        )
                    logger.info("Property enrichment complete")
            else:
                logger.warning("No entities extracted")

            return graph_docs

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise

    async def extract_batch(
        self, utterances: list[str], metadata_list: list[dict] | None = None
    ) -> list:
        """
        Extract entities from multiple utterances in batch

        Args:
            utterances: List of user statements
            metadata_list: Optional metadata for each utterance

        Returns:
            List of GraphDocument objects
        """
        documents = []

        for i, utterance in enumerate(utterances):
            metadata = metadata_list[i] if metadata_list else {"utterance_id": i}
            doc = Document(page_content=utterance, metadata=metadata)
            documents.append(doc)

        logger.info(f"Batch extracting from {len(documents)} utterances")
        transformer = self._get_transformer()

        try:
            graph_docs = await transformer.aconvert_to_graph_documents(documents)
            logger.info(f"Batch extraction complete: {len(graph_docs)} results")
            return graph_docs

        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            raise
