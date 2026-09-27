"""Internal knowledge derivation (Stage 9).

Analyzes conversation turns for self-model signals and creates/updates
internal entities (MistTrait, MistCapability, MistPreference, MistUncertainty).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from backend.errors import ExtractionError, ExtractionValidationError
from backend.knowledge.extraction.internal_prompts import (
    INTERNAL_DERIVATION_SYSTEM_PROMPT,
    INTERNAL_DERIVATION_USER_TEMPLATE,
)
from backend.knowledge.extraction.signal_detector import SignalDetectionResult, SignalDetector
from backend.knowledge.storage.partitions import SELF_MODEL_LABEL, SELF_MODEL_TYPES
from backend.knowledge.version_stamps import ONTOLOGY_VERSION
from backend.llm.models import LLMRequest

if TYPE_CHECKING:
    # GraphExecutor is used only as a constructor parameter annotation
    # (InternalKnowledgeDeriver.__init__ below) -- never constructed,
    # isinstance-checked, or otherwise touched at runtime in this module
    # (`grep -n "GraphExecutor" backend/knowledge/extraction/internal_derivation.py`
    # -> the import and the one annotation, nothing else). `from __future__
    # import annotations` above makes that annotation a string at runtime,
    # so nothing evaluates it; the real import is needed only by type
    # checkers, and moving it here keeps `import backend.knowledge.storage`
    # (neo4j, and pandas/pyarrow behind it) off this module's runtime path.
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


def render_derivation_messages(
    utterance: str,
    assistant_response: str,
    signal_types: list[str] | frozenset[str],
    matched_patterns: list[str] | tuple[str, ...],
    existing_internal_entities: str,
) -> list[dict]:
    """Render the Stage 9 chat messages for internal knowledge derivation.

    Pure function -- no I/O, no LLM call. Shared by the in-process
    `InternalKnowledgeDeriver` and the extraction service, which receives
    `signal_types`/`matched_patterns`/`existing_internal_entities` from the
    backend's `DerivationInput` (the service cannot read the graph itself).

    Args:
        utterance: The user's message.
        assistant_response: MIST's response to the user.
        signal_types: Detected signal type names (e.g. "feedback").
        matched_patterns: Human-readable matched-pattern strings.
        existing_internal_entities: Pre-rendered summary of existing
            self-model entities, as produced by
            `InternalKnowledgeDeriver.fetch_existing_internal_entities`.

    Returns:
        A two-message list: system prompt, then the rendered user template.
    """
    user_message = INTERNAL_DERIVATION_USER_TEMPLATE.format(
        utterance=utterance,
        assistant_response=assistant_response,
        signal_types=", ".join(signal_types),
        matched_patterns=", ".join(matched_patterns),
        existing_internal_entities=existing_internal_entities,
    )
    return [
        {"role": "system", "content": INTERNAL_DERIVATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def parse_derivation_output(raw: str | None, *, strict: bool) -> list[dict]:
    """Parse Stage 9 LLM output into a list of self-model operations.

    Non-strict (`strict=False`, the in-process pipeline's historical
    behavior): returns `[]` on missing/unparseable JSON rather than
    raising -- `derive()` already logs and degrades to "no operations".

    Strict (`strict=True`, the extraction service): raises so the caller
    can distinguish "valid output with zero operations" from
    "unparseable output" instead of silently treating both as "nothing to
    derive".

    Args:
        raw: Raw string output from the LLM (may be None).
        strict: When True, raise on unparseable/malformed input instead of
            returning an empty list.

    Returns:
        The `operations` list from the parsed JSON object (possibly empty).

    Raises:
        ExtractionValidationError: `strict=True` and `raw` is empty/None,
            not parseable as a JSON object, or its `operations` key (when
            present) is not a list.
    """
    if not raw or not raw.strip():
        if strict:
            raise ExtractionValidationError("Empty internal-derivation output from LLM")
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        if strict:
            raise ExtractionValidationError(
                f"Could not parse internal-derivation output as JSON: {raw[:200]!r}"
            ) from exc
        return []

    if not isinstance(parsed, dict):
        if strict:
            raise ExtractionValidationError(
                f"Internal-derivation output is not a JSON object: {raw[:200]!r}"
            )
        return []

    operations = parsed.get("operations", [])
    if strict and not isinstance(operations, list):
        raise ExtractionValidationError(
            f"Internal-derivation 'operations' is not a list: {raw[:200]!r}"
        )
    return operations if isinstance(operations, list) else []


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

    def __init__(self, llm, executor: GraphExecutor, temperature: float = 0.0) -> None:
        """Initialize the deriver.

        Args:
            llm: LLM provider (satisfies LLMProvider protocol).
            executor: Async graph executor for writing internal entities.
            temperature: Sampling temperature for extraction calls. Defaults
                to 0.0 for backward compatibility; production deployments
                should pass LLMConfig.temperature from the KnowledgeConfig.
        """
        self._llm = llm
        self._executor = executor
        self._temperature = temperature
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
        existing = await self.fetch_existing_internal_entities()

        messages = render_derivation_messages(
            utterance=utterance,
            assistant_response=assistant_response,
            signal_types=signals.signal_types,
            matched_patterns=signals.matched_patterns,
            existing_internal_entities=existing,
        )

        # LLM call
        try:
            request = LLMRequest(
                messages=messages,
                json_mode=True,
                temperature=self._temperature,
            )
            from backend.llm.instrumented_provider import llm_call_context

            with llm_call_context(call_site="extraction.internal_derivation"):
                response = await self._llm.invoke(request)
            raw = response.content
        except Exception as e:
            logger.error("Internal derivation LLM call failed: %s", e)
            elapsed = (time.perf_counter() - start) * 1000
            return InternalDerivationResult(derivation_time_ms=elapsed, llm_called=True)

        # Parse response. Non-strict: falls back to [] on unparseable output
        # rather than raising -- this method's historical behavior.
        operations = parse_derivation_output(raw, strict=False)

        valid_ops = await self.apply_operations(
            operations, session_id=session_id, event_id=event_id
        )

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("Internal derivation: %d operations in %.1fms", len(valid_ops), elapsed)

        return InternalDerivationResult(
            operations=valid_ops,
            derivation_time_ms=elapsed,
            llm_called=True,
        )

    async def apply_operations(
        self, operations: list[dict], *, session_id: str, event_id: str
    ) -> tuple[dict, ...]:
        """Validate and apply a list of self-model operations to the graph.

        Each operation's `op` type is checked against `VALID_OPS`; an
        invalid type is skipped and logged. A valid op that fails to apply
        (graph write error) is also skipped and logged -- one bad operation
        never aborts the batch. This is the single apply path: `derive()`
        calls this method rather than looping over `_apply_operation` itself,
        and the extraction service's caller (the backend dispatcher) calls
        it directly with operations parsed from `DerivationOut.operations`.

        Args:
            operations: Candidate operations, e.g. from
                `parse_derivation_output`.
            session_id: Conversation session ID.
            event_id: Event store turn ID.

        Returns:
            The operations that were validated and successfully applied, in
            input order.
        """
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

        return tuple(valid_ops)

    async def fetch_existing_internal_entities(self) -> str:
        """Fetch existing internal entities for LLM context."""
        try:
            results = await self._executor.execute_query(
                f"MATCH (m:MistIdentity)-[r]->(e:{SELF_MODEL_LABEL}) "
                "WHERE e.knowledge_domain = 'internal' AND e.status = 'active' "
                "AND coalesce(r.is_latest_belief, true) "
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

    # Kept as an alias: the leading-underscore name predates the public
    # `fetch_existing_internal_entities` method the extraction service
    # (T1a) and any other external caller now use.
    _fetch_existing_internal_entities = fetch_existing_internal_entities

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
                "ontology_version": ONTOLOGY_VERSION,
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
                "e.ontology_version = $ontology_version"
            )
            if prop_sets:
                create_set += ", " + prop_sets

            # MERGE entity into the :__SelfModel__ partition + apply the typed
            # label (entity_type is one of SELF_MODEL_TYPES from the op mapping,
            # so it is safe to interpolate as a label).
            typed_label = params["entity_type"]
            if typed_label not in SELF_MODEL_TYPES:
                raise ExtractionError(f"Unknown self-model type: {typed_label}")
            await self._executor.execute_write(
                f"MERGE (e:{SELF_MODEL_LABEL} {{id: $entity_id}}) "
                f"ON CREATE SET {create_set} "
                "ON MATCH SET e.updated_at = $now, e.confidence = $confidence "
                f"SET e:{typed_label} "
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
                f"MATCH (e:{SELF_MODEL_LABEL} {{id: $entity_id}}) " f"SET {', '.join(set_clauses)}",
                params,
            )

        elif op_type == "DEPRECATE":
            entity_id = op.get("entity_id", "")
            reason = op.get("reason", "")
            if not entity_id:
                return

            await self._executor.execute_write(
                f"MATCH (e:{SELF_MODEL_LABEL} {{id: $entity_id}}) "
                "SET e.status = 'deprecated', e.deprecated_reason = $reason, "
                "e.updated_at = $now",
                {"entity_id": entity_id, "reason": reason, "now": now},
            )
