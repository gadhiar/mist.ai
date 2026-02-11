"""
Property Enrichment Module

Extracts temporal, contextual, and relational properties from text that
LLMGraphTransformer doesn't capture in its structured output.

This is a post-processing step that uses a small, focused LLM call to
extract properties without needing specific prompt patterns.
"""

import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from backend.knowledge.config import KnowledgeConfig

logger = logging.getLogger(__name__)


# Simple schema for property extraction
PROPERTY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a property extractor. Extract temporal, contextual, and descriptive properties from text.

Return ONLY a JSON object with properties. If a property doesn't exist, omit it.

Example input: "I've been working on MIST for 3 months and it's going well"
Example output: {{"duration": "3 months", "status": "going well"}}

Example input: "I plan to learn TypeScript next month"
Example output: {{"when": "next month", "intent": "future"}}

Example input: "I'm an expert in Python"
Example output: {{"proficiency": "expert"}}

Example input: "I use Docker daily in production"
Example output: {{"frequency": "daily", "context": "in production"}}

Available property types:
- duration: time spent (e.g., "3 months", "2 years")
- when: future time (e.g., "next month", "by end of year")
- frequency: how often (e.g., "daily", "weekly", "sometimes")
- context: where/how (e.g., "in production", "for work", "at home")
- proficiency: skill level (e.g., "expert", "beginner", "proficient")
- status: how it's going (e.g., "going well", "struggling", "completed")
- intent: future/past (e.g., "future", "past", "current")

Return ONLY valid JSON. No explanations.""",
        ),
        (
            "user",
            "Text: {text}\n\nRelationship type: {relationship_type}\n\nExtract properties as JSON:",
        ),
    ]
)


class PropertyEnricher:
    """
    Enriches graph relationships with temporal and contextual properties
    by doing a second, focused extraction pass.
    """

    def __init__(self, config: KnowledgeConfig):
        """
        Initialize property enricher

        Args:
            config: Knowledge system configuration
        """
        self.config = config
        self._llm = None

    def _get_llm(self) -> ChatOllama:
        """Get or create LLM instance"""
        if self._llm is None:
            # Use same LLM as extraction, but could use smaller/faster model
            self._llm = ChatOllama(
                model=self.config.llm.model,
                base_url=self.config.llm.base_url,
                temperature=0.0,  # Deterministic for property extraction
            )
        return self._llm

    async def enrich_graph_document(self, graph_document, original_text: str):
        """
        Enrich a GraphDocument with additional properties

        This modifies the graph_document in place, adding properties to
        relationships based on the original text.

        Args:
            graph_document: GraphDocument from LLMGraphTransformer
            original_text: Original utterance text

        Returns:
            The enriched graph_document (modified in place)
        """
        if not graph_document or not graph_document.relationships:
            return graph_document

        logger.info(f"Enriching {len(graph_document.relationships)} relationships")

        # Process each relationship
        for rel in graph_document.relationships:
            try:
                properties = await self._extract_properties(
                    text=original_text, relationship_type=rel.type
                )

                if properties:
                    # Merge with existing properties
                    if not hasattr(rel, "properties") or rel.properties is None:
                        rel.properties = {}

                    # Convert list of {key: value} to dict if needed
                    if isinstance(rel.properties, list):
                        props_dict = {
                            p.get("key"): p.get("value")
                            for p in rel.properties
                            if isinstance(p, dict)
                        }
                        rel.properties = props_dict

                    # Add enriched properties
                    rel.properties.update(properties)

                    logger.debug(f"Enriched {rel.type} with properties: {properties}")

            except Exception as e:
                logger.warning(f"Failed to enrich relationship {rel.type}: {e}")
                continue

        return graph_document

    async def _extract_properties(self, text: str, relationship_type: str) -> dict[str, str]:
        """
        Extract properties from text using LLM

        Args:
            text: Original utterance text
            relationship_type: Type of relationship (USES, WORKS_ON, etc.)

        Returns:
            Dictionary of extracted properties
        """
        llm = self._get_llm()
        chain = PROPERTY_EXTRACTION_PROMPT | llm

        try:
            response = await chain.ainvoke({"text": text, "relationship_type": relationship_type})

            # Parse JSON response
            content = response.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                # Extract JSON from code block
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])  # Remove ``` markers

            properties = json.loads(content)

            # Validate it's a dict
            if not isinstance(properties, dict):
                logger.warning(f"Expected dict, got {type(properties)}")
                return {}

            # Filter out empty values
            return {k: v for k, v in properties.items() if v}

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse properties JSON: {e}")
            logger.debug(f"Response was: {response.content}")
            return {}
        except Exception as e:
            logger.warning(f"Property extraction failed: {e}")
            return {}


class ContextAwarePropertyEnricher(PropertyEnricher):
    """
    Enhanced enricher that uses conversation context to better
    extract properties.

    This is for future enhancement - can use conversation history
    to disambiguate properties.
    """

    async def enrich_with_context(
        self, graph_document, original_text: str, conversation_history: list[str] | None = None
    ):
        """
        Enrich with conversation context

        Future enhancement: Use conversation history to better
        understand temporal references ("still working on it",
        "finished last month", etc.)
        """
        # For now, just call parent
        return await self.enrich_graph_document(graph_document, original_text)
