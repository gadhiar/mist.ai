"""Internal knowledge derivation (Stage 9).

Analyzes conversation turns for self-model signals and creates/updates
internal entities (MistTrait, MistCapability, MistPreference, MistUncertainty).
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from backend.knowledge.extraction.internal_prompts import (
    INTERNAL_DERIVATION_SYSTEM_PROMPT,
    INTERNAL_DERIVATION_USER_TEMPLATE,
)
from backend.knowledge.extraction.signal_detector import SignalDetectionResult, SignalDetector
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

# Valid operation types
VALID_OPS = {
    "CREATE_TRAIT",
    "CREATE_CAPABILITY",
    "CREATE_PREFERENCE",
    "CREATE_UNCERTAINTY",
    "UPDATE",
    "DEPRECATE",
}

# Maps operation type to entity type and relationship type
OP_TO_ENTITY_TYPE = {
    "CREATE_TRAIT": ("MistTrait", "HAS_TRAIT"),
    "CREATE_CAPABILITY": ("MistCapability", "HAS_CAPABILITY"),
    "CREATE_PREFERENCE": ("MistPreference", "HAS_PREFERENCE"),
    "CREATE_UNCERTAINTY": ("MistUncertainty", "IS_UNCERTAIN_ABOUT"),
}

SAFE_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass(frozen=True, slots=True)
class InternalDerivationResult:
    """Result of internal knowledge derivation."""

    operations: tuple[dict, ...] = ()
    derivation_time_ms: float = 0.0
    llm_called: bool = False


class InternalKnowledgeDeriver:
    """Derives internal knowledge from conversation signals.

    Gate: If SignalDetectionResult.has_signals is False, the LLM call
    is skipped entirely. This saves ~1-2s of Ollama inference per turn
    for the majority of turns that have no internal signals.
    """

    def __init__(self, llm, executor: GraphExecutor) -> None:
        """Initialize the deriver.

        Args:
            llm: LLM provider (LLMProvider protocol or ChatOllama).
            executor: Async graph executor for writing internal entities.
        """
        self._llm = llm
        self._executor = executor
        self._signal_detector = SignalDetector()

    async def derive(
        self,
        utterance: str,
        assistant_response: str,
        signals: SignalDetectionResult,
        session_id: str,
        event_id: str,
    ) -> InternalDerivationResult:
        """Analyze a conversation turn and produce self-model operations.

        Args:
            utterance: The user's message.
            assistant_response: MIST's response to the user.
            signals: Pre-detected signals from SignalDetector.
            session_id: Conversation session ID.
            event_id: Event store turn ID.

        Returns:
            InternalDerivationResult with operations applied.
        """
        if not signals.has_signals:
            return InternalDerivationResult()

        start = time.perf_counter()

        # Fetch existing internal entities for context
        existing = await self._fetch_existing_internal_entities()

        # Build prompt
        user_message = INTERNAL_DERIVATION_USER_TEMPLATE.format(
            utterance=utterance,
            assistant_response=assistant_response,
            signal_types=", ".join(signals.signal_types),
            matched_patterns=", ".join(signals.matched_patterns),
            existing_internal_entities=existing,
        )

        messages = [
            {"role": "system", "content": INTERNAL_DERIVATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # LLM call
        try:
            response = await self._llm.ainvoke(messages)
            raw = response.content
        except Exception as e:
            logger.error("Internal derivation LLM call failed: %s", e)
            elapsed = (time.perf_counter() - start) * 1000
            return InternalDerivationResult(derivation_time_ms=elapsed, llm_called=True)

        # Parse response
        try:
            parsed = json.loads(raw)
            operations = parsed.get("operations", [])
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("Internal derivation JSON parse failed: %s", e)
            elapsed = (time.perf_counter() - start) * 1000
            return InternalDerivationResult(derivation_time_ms=elapsed, llm_called=True)

        # Validate and apply operations
        valid_ops = []
        for op in operations:
            op_type = op.get("op", "")
            if op_type not in VALID_OPS:
                logger.warning("Skipping invalid operation type: %s", op_type)
                continue

            try:
                await self._apply_operation(op, session_id, event_id)
                valid_ops.append(op)
            except Exception as e:
                logger.error("Failed to apply operation %s: %s", op_type, e)

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("Internal derivation: %d operations in %.1fms", len(valid_ops), elapsed)

        return InternalDerivationResult(
            operations=tuple(valid_ops),
            derivation_time_ms=elapsed,
            llm_called=True,
        )

    async def _fetch_existing_internal_entities(self) -> str:
        """Fetch existing internal entities for LLM context."""
        try:
            results = await self._executor.execute_query(
                "MATCH (m:MistIdentity)-[r]->(e:__Entity__) "
                "WHERE e.knowledge_domain = 'internal' AND e.status = 'active' "
                "RETURN e.id AS id, e.entity_type AS type, "
                "e.display_name AS name, type(r) AS rel_type "
                "ORDER BY e.entity_type, e.display_name"
            )
            if not results:
                return "No existing internal entities."

            lines = ["Existing internal entities:"]
            for r in results:
                lines.append(f"- [{r['type']}] {r['name']} (id: {r['id']})")
            return "\n".join(lines)
        except Exception:
            return "Could not fetch existing internal entities."

    async def _apply_operation(self, op: dict, session_id: str, event_id: str) -> None:
        """Apply a single internal entity operation to the graph."""
        op_type = op.get("op", "")
        now = datetime.now(UTC).isoformat()

        if op_type in OP_TO_ENTITY_TYPE:
            entity_type, rel_type = OP_TO_ENTITY_TYPE[op_type]
            entity_id = op.get("id", "")
            if not entity_id:
                return

            # Build property params from operation dict
            params: dict = {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "display_name": op.get("display_name", entity_id),
                "description": op.get("description", ""),
                "confidence": op.get("confidence", 0.8),
                "now": now,
                "event_id": event_id,
                "session_id": session_id,
            }

            # Add type-specific properties
            for key in (
                "trait_category",
                "capability_type",
                "preference_type",
                "uncertainty_type",
                "proficiency",
                "resolution_strategy",
                "evidence",
            ):
                if key in op and SAFE_KEY.match(key):
                    params[f"prop_{key}"] = op[key]

            prop_sets = ", ".join(f"e.{k[5:]} = ${k}" for k in params if k.startswith("prop_"))

            create_set = (
                "e.entity_type = $entity_type, "
                "e.display_name = $display_name, "
                "e.knowledge_domain = 'internal', "
                "e.description = $description, "
                "e.confidence = $confidence, "
                "e.source_type = 'self_authored', "
                "e.source_event_id = $event_id, "
                "e.status = 'active', "
                "e.created_at = $now, "
                "e.updated_at = $now, "
                "e.ontology_version = '1.0.0'"
            )
            if prop_sets:
                create_set += ", " + prop_sets

            # MERGE entity + link to MistIdentity
            await self._executor.execute_write(
                f"MERGE (e:__Entity__ {{id: $entity_id}}) "
                f"ON CREATE SET {create_set} "
                "ON MATCH SET e.updated_at = $now, e.confidence = $confidence "
                "WITH e "
                "MATCH (m:MistIdentity {id: 'mist-identity'}) "
                f"MERGE (m)-[:{rel_type}]->(e)",
                params,
            )

        elif op_type == "UPDATE":
            entity_id = op.get("entity_id", "")
            fields = op.get("fields", {})
            if not entity_id or not fields:
                return

            set_clauses = ["e.updated_at = $now"]
            params = {"entity_id": entity_id, "now": now}
            for key, value in fields.items():
                if SAFE_KEY.match(key) and isinstance(value, str | int | float | bool):
                    param_key = f"upd_{key}"
                    set_clauses.append(f"e.{key} = ${param_key}")
                    params[param_key] = value

            await self._executor.execute_write(
                f"MATCH (e:__Entity__ {{id: $entity_id}}) " f"SET {', '.join(set_clauses)}",
                params,
            )

        elif op_type == "DEPRECATE":
            entity_id = op.get("entity_id", "")
            reason = op.get("reason", "")
            if not entity_id:
                return

            await self._executor.execute_write(
                "MATCH (e:__Entity__ {id: $entity_id}) "
                "SET e.status = 'deprecated', e.deprecated_reason = $reason, "
                "e.updated_at = $now",
                {"entity_id": entity_id, "reason": reason, "now": now},
            )
