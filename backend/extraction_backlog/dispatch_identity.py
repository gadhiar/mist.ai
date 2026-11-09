"""The dispatch identity: one hash over the backend settings that decide what the cache holds.

An epoch's full identity is its ledger triple, the serving identity the
extraction service computes (`backend.knowledge.extraction.serving_identity`),
and this one, which the backend computes. Its components are the dispatcher's
history window and the gate settings: a Gate 2 or Gate 3 skip is cached and
replayed by every rebuild, so a tuned threshold that is not keyed splits cached
history silently (KNOWN_ISSUES.md, the significance-threshold entry).

The rule is the serving identity's: a setting is keyed unless it provably
cannot change the bytes the model returns or the decision the backend caches.
The settings that decide only when to retry or give up are listed in
`DISPATCHER_SETTINGS_EXCLUDED` with the reason; a completeness test fails on a
`DispatcherSettings` field that is in neither place.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .settings import DispatcherSettings

DISPATCH_IDENTITY_PREFIX = "di1-"


@dataclass(frozen=True, slots=True)
class DispatchIdentityComponents:
    """Everything the dispatch identity hashes.

    Attributes:
        history_messages: Conversation-history messages sent with each job
            (`DispatcherSettings.history_messages`).
        significance_threshold: The Gate 2 threshold as resolved for the
            "conversation" source, the one the dispatcher gates with.
        dedup_similarity_threshold: Gate 3's cosine-similarity cutoff.
        dedup_cache_size: How many recent turns Gate 3 compares against.
        dedup_cache_ttl_seconds: How long a turn stays in Gate 3's window.
        embedding_gates: Whether an embedding provider is wired. Without one,
            Gate 2's novelty is 1.0 and Gate 3 never fires.
    """

    history_messages: int
    significance_threshold: float
    dedup_similarity_threshold: float
    dedup_cache_size: int
    dedup_cache_ttl_seconds: int
    embedding_gates: bool


@dataclass(frozen=True, slots=True)
class DispatchIdentity:
    """A computed dispatch identity and the flat components it hashes."""

    identity: str
    components: Mapping[str, int | float | bool]


# The components read from the pipeline's gates, in the order of the dataclass.
GATE_FIELDS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(DispatchIdentityComponents) if f.name != "history_messages"
)

# Every `DispatcherSettings` field that is not a component, with the reason it
# cannot change what the cache holds for a turn.
DISPATCHER_SETTINGS_EXCLUDED: Mapping[str, str] = MappingProxyType(
    {
        "mode": "whether the dispatcher runs at all; it decides nothing about a turn it extracts",
        "service_url": "an address: where the extraction service is reached",
        "max_attempts": (
            "when to give up on a turn; a dead letter is a recorded, retryable skip "
            "(retry-dead-letters), not an extraction"
        ),
        "backoff_base_s": "when to retry, not what a retry returns",
        "backoff_cap_s": "when to retry, not what a retry returns",
        "request_timeout_s": (
            "a deadline; a timed-out job is retried or dead-lettered, never cached as a result"
        ),
        "idle_poll_s": "how often an idle dispatcher rescans the log",
        "stall_recheck_s": "how often a stalled dispatcher re-reads /v1/info",
    }
)


def components_dict(c: DispatchIdentityComponents) -> dict[str, int | float | bool]:
    """The components as a flat mapping of JSON scalars, the form stored and compared.

    Each value is coerced to its declared type, so an int threshold and the
    same float hash alike and a JSON round trip reads back what was hashed.
    """
    return {
        "history_messages": int(c.history_messages),
        "significance_threshold": float(c.significance_threshold),
        "dedup_similarity_threshold": float(c.dedup_similarity_threshold),
        "dedup_cache_size": int(c.dedup_cache_size),
        "dedup_cache_ttl_seconds": int(c.dedup_cache_ttl_seconds),
        "embedding_gates": bool(c.embedding_gates),
    }


def compute_dispatch_identity(c: DispatchIdentityComponents) -> str:
    """Return `"di1-"` and the sha256 of `components_dict(c)` as canonical JSON."""
    canonical = json.dumps(components_dict(c), sort_keys=True, separators=(",", ":"))
    return DISPATCH_IDENTITY_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dispatch_identity_components(
    gates: Mapping[str, float | int | bool], settings: DispatcherSettings
) -> DispatchIdentityComponents:
    """The components of a dispatcher with these gate settings and dispatcher settings.

    Args:
        gates: `ExtractionPipeline.dispatch_gate_settings()`, one value per
            name in `GATE_FIELDS`.
        settings: The dispatcher's settings; only `history_messages` is read.

    Raises:
        ValueError: `gates` names a field that is not in `GATE_FIELDS`, or
            lacks one. Either would leave a gate setting unkeyed.
    """
    if set(gates) != set(GATE_FIELDS):
        raise ValueError(
            f"dispatch gate settings {sorted(gates)} are not the keyed gate fields "
            f"{sorted(GATE_FIELDS)}"
        )
    return DispatchIdentityComponents(
        history_messages=settings.history_messages,
        significance_threshold=float(gates["significance_threshold"]),
        dedup_similarity_threshold=float(gates["dedup_similarity_threshold"]),
        dedup_cache_size=int(gates["dedup_cache_size"]),
        dedup_cache_ttl_seconds=int(gates["dedup_cache_ttl_seconds"]),
        embedding_gates=bool(gates["embedding_gates"]),
    )


def dispatch_identity_for(
    gates: Mapping[str, float | int | bool], settings: DispatcherSettings
) -> DispatchIdentity:
    """The dispatch identity of a dispatcher, with its components."""
    components = dispatch_identity_components(gates, settings)
    return DispatchIdentity(
        identity=compute_dispatch_identity(components),
        components=MappingProxyType(components_dict(components)),
    )
