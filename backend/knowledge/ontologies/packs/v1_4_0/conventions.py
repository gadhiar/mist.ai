"""The v1.4.0 pack's convention rules, in the order their phases run them.

The pack loader reads `RULES`. Each rule the rails run carries a pinned case: a row and
the payload the pack must turn it into, which the rail runner checks the first time it is
handed the pack. A pinned case feeds neither of the pack's hashes.

Three of the cases are payloads the extraction service wrote for probes of dev-lists, the
tuning set, in the run of prompt candidate c3r2 (`CASES_FROM_THE_TUNING_RUN`). Each holds
the payload's entities as written and, of each relationship, its two ends and its type:
a relationship's properties quote the utterance, and no rule of this pack reads them.
The other cases are written for their rule.

The vocabularies and the reserved names are stated here. The modules that held them
before still hold a copy, and the pack's test keeps the two equal.
"""

from __future__ import annotations

from collections.abc import Mapping

from backend.knowledge.ontologies.pack import (
    ID,
    PAYLOAD,
    PRE_LLM,
    AliasVocabulary,
    ConventionRule,
    DerivedEdge,
    IdExempt,
    PinnedCase,
    PlanVocabulary,
    ReservedIdentity,
    TypedId,
)

from .plugins import EVENT_DATE, METRIC_ID, TECHNOLOGY_VERSION
from .types import (
    DATE,
    EVENT,
    EXTRACTABLE_EDGE_TYPES,
    EXTRACTABLE_NODE_TYPES,
    METRIC,
    MIST_IDENTITY,
    OCCURRED_ON,
    TECHNOLOGY,
    USER,
)

# ---------------------------------------------------------------------------
# The data the rules state
# ---------------------------------------------------------------------------

# The names the system itself goes by. An entity with one of them is the one identity
# node, whatever type the model gave it, so extraction never adds a second node for the
# system: the edges that describe it take that node as their source.
RESERVED_ID = "mist-identity"
RESERVED_NAMES: tuple[str, ...] = (
    "mist",
    "mist.ai",
    "mist ai",
    "mist-ai",
    "the ai",
    "the-ai",
    "the assistant",
    "the-assistant",
)

# Types whose ids follow their own convention, so the id rail keeps their leading words.
# In the order the types are defined, since the pack's hash reads them in order.
ID_EXEMPT_TYPES: tuple[str, ...] = (
    MIST_IDENTITY.type_name,
    USER.type_name,
    DATE.type_name,
    METRIC.type_name,
)

# Looked up by the lowered display name, before the id is a slug: the special characters
# of these names do not survive one.
PRE_CANON_ALIASES: tuple[tuple[str, str], ...] = (
    ("c++", "c-plus-plus"),
    ("c#", "c-sharp"),
)

# A short name or another spelling, and the id it stands for. A key must be unambiguous:
# a short form that could name two technologies is left out, and the model's full name
# decides.
STATIC_ALIASES: tuple[tuple[str, str], ...] = (
    ("js", "javascript"),
    ("ts", "typescript"),
    ("py", "python"),
    ("rb", "ruby"),
    ("node.js", "nodejs"),
    ("node", "nodejs"),
    ("react.js", "react"),
    ("reactjs", "react"),
    ("react-js", "react"),
    ("vue.js", "vue"),
    ("vuejs", "vue"),
    ("vue-js", "vue"),
    ("angular.js", "angular"),
    ("angularjs", "angular"),
    ("next.js", "nextjs"),
    ("nuxt.js", "nuxtjs"),
    ("mongo", "mongodb"),
    ("postgres", "postgresql"),
    ("pg", "postgresql"),
    ("k8s", "kubernetes"),
    ("kube", "kubernetes"),
    ("vscode", "visual-studio-code"),
    ("vs-code", "visual-studio-code"),
    ("gcp", "google-cloud-platform"),
    ("aws", "amazon-web-services"),
    ("gh", "github"),
    ("cpp", "c-plus-plus"),
    ("csharp", "c-sharp"),
    ("golang", "go"),
    ("objc", "objective-c"),
    ("sklearn", "scikit-learn"),
    ("torch", "pytorch"),
)

# ---------------------------------------------------------------------------
# The pinned cases
# ---------------------------------------------------------------------------

# The rules whose case is a payload of the tuning run, and the tag of the probe.
CASES_FROM_THE_TUNING_RUN: Mapping[str, str] = {
    "static-aliases": "dvl-21-list-dash-laptop-cleanup",
    "metric-id": "dvl-39-list-long-load-test-numbers",
    "event-date": "dvl-15-list-paren-upcoming-dates",
}


def _case(
    entities: tuple[Mapping, ...],
    expected: tuple[Mapping, ...],
    relationships: tuple[Mapping, ...] = (),
    expected_relationships: tuple[Mapping, ...] = (),
) -> PinnedCase:
    return PinnedCase(
        entities=entities,
        relationships=relationships,
        expected_entities=expected,
        expected_relationships=expected_relationships,
    )


def _edge(source: str, target: str, edge_type: str) -> dict[str, str]:
    return {"source": source, "target": target, "type": edge_type}


def _entity(entity_id: str, name: str, entity_type: str) -> dict[str, object]:
    return {"id": entity_id, "name": name, "type": entity_type}


def _metric(entity_id: str, name: str, quantity: str, unit: str, value: float) -> dict[str, object]:
    """A Metric the model wrote with its properties, keyed as the payload keys them."""
    return {
        "id": entity_id,
        "name": name,
        "properties": {"quantity": quantity, "unit": unit, "value": value},
        "type": "Metric",
    }


_USER = _entity("user", "User", "User")

# Written for the rule. The system under two of its names, one of them typed as a person:
# both are the identity node, the first one written is kept, and the edge follows it.
_RESERVED_CASE = _case(
    (
        _entity("mist", "MIST", "Technology"),
        _entity("the-assistant", "the assistant", "Person"),
        _entity("voice-replies", "voice replies", "Skill"),
    ),
    (
        _entity("mist-identity", "MIST", "MistIdentity"),
        _entity("voice-replies", "voice replies", "Skill"),
    ),
    relationships=(_edge("the-assistant", "voice-replies", "MIST_HAS_CAPABILITY"),),
    expected_relationships=(_edge("mist-identity", "voice-replies", "MIST_HAS_CAPABILITY"),),
)

# Written for the rule. A Metric keeps the leading word every other type loses. The case
# also carries what the manifest states without a rule, since a case needs a rule to hold
# it: the user anchor's id written with a capital, both retired types, and a type the
# version does not have.
_ID_EXEMPT_CASE = _case(
    (
        _entity("User", "User", "User"),
        _entity("the-outbox-pattern", "the outbox pattern", "Concept"),
        _entity("a-dozen-retries", "a dozen retries", "Metric"),
        _entity("the-first-release", "the first release", "Milestone"),
        _entity("our-release-cadence", "our release cadence", "Topic"),
        _entity("a-hunch", "a hunch", "Feeling"),
    ),
    (
        _USER,
        _entity("outbox-pattern", "the outbox pattern", "Concept"),
        _entity("a-dozen-retries", "a dozen retries", "Metric"),
        {
            **_entity("first-release", "the first release", "Event"),
            "properties": {"event_type": "milestone"},
        },
        _entity("release-cadence", "our release cadence", "Concept"),
        _entity("hunch", "a hunch", "Abstraction"),
    ),
    relationships=(
        _edge("User", "the-outbox-pattern", "KNOWS"),
        _edge("User", "the-first-release", "EXPERIENCED"),
        _edge("User", "our-release-cadence", "INTERESTED_IN"),
    ),
    expected_relationships=(
        _edge("user", "outbox-pattern", "KNOWS"),
        _edge("user", "first-release", "EXPERIENCED"),
        _edge("user", "release-cadence", "INTERESTED_IN"),
    ),
)

# Written for the rule. The slug of either name is `c`, so the name decides the id.
_PRE_CANON_CASE = _case(
    (
        _USER,
        _entity("c++", "C++", "Technology"),
        _entity("c#", "C#", "Technology"),
    ),
    (
        _USER,
        _entity("c-plus-plus", "C++", "Technology"),
        _entity("c-sharp", "C#", "Technology"),
    ),
    relationships=(_edge("user", "c++", "USES"), _edge("user", "c#", "LEARNING")),
    expected_relationships=(
        _edge("user", "c-plus-plus", "USES"),
        _edge("user", "c-sharp", "LEARNING"),
    ),
)

# Written for the rule. The version in the name leaves the id.
_TECHNOLOGY_VERSION_CASE = _case(
    (_entity("quillscript-3-11", "Quillscript 3.11", "Technology"),),
    (_entity("quillscript", "Quillscript 3.11", "Technology"),),
)

# The tuning run's payload for dvl-21. One id is a short form of the vocabulary.
_DOCKER_DESKTOP = _entity("docker-desktop", "Docker Desktop", "Technology")
_PODMAN = _entity("podman", "Podman", "Technology")
_SLACK = _entity("slack", "Slack", "Technology")
_STATIC_ALIASES_CASE = _case(
    (_USER, _DOCKER_DESKTOP, _PODMAN, _SLACK, _entity("vs-code", "VS Code", "Technology")),
    (
        _USER,
        _DOCKER_DESKTOP,
        _PODMAN,
        _SLACK,
        _entity("visual-studio-code", "VS Code", "Technology"),
    ),
    relationships=(
        _edge("user", "docker-desktop", "USES"),
        _edge("user", "podman", "USES"),
        _edge("user", "slack", "USES"),
        _edge("user", "vs-code", "USES"),
    ),
    expected_relationships=(
        _edge("user", "docker-desktop", "USES"),
        _edge("user", "podman", "USES"),
        _edge("user", "slack", "USES"),
        _edge("user", "visual-studio-code", "USES"),
    ),
)

# The tuning run's payload for dvl-39, which writes several of its numbers more than once.
# The service entity is written twice, under two spellings of its id, and stays two
# entities: no rule joins them. A Metric that states its value takes the id the metric
# rail builds from it, and where that is the id of a Metric written earlier, the earlier
# one is kept.
_SERVICE = "saltmere-matchmaking-service"
_SERVICE_AGAIN = "salmere-matchmaking-service"
_STATED_FIRST = (
    _entity(_SERVICE, "Salmere matchmaking service", "Technology"),
    _metric("38-ms-p50-latency", "38 ms p50 latency", "p50-latency", "ms", 38),
    _metric("410-ms-p99-latency", "410 ms p99 latency", "p99-latency", "ms", 410),
    _metric(
        "9500-requests-per-second-peak-throughput",
        "9500 requests per second peak throughput",
        "peak-throughput",
        "requests per second",
        9500,
    ),
    _metric("0-2-percent-error-rate", "0.2 percent error rate", "error-rate", "percent", 0.2),
    _entity(_SERVICE_AGAIN, "Salmere matchmaking service", "Technology"),
    _entity("38-ms-latency", "38-ms-latency", "Metric"),
    _entity("410-ms-latency", "410-ms-latency", "Metric"),
    _entity("9500-requests-per-second-throughput", "9500-requests-per-second-throughput", "Metric"),
    _entity("71-percent-cpu", "71-percent-cpu", "Metric"),
    _entity("3-1-gb-memory", "3-1-gb-memory", "Metric"),
    _entity("2400-matches-per-minute", "2400-matches-per-minute", "Metric"),
    _entity("180-queue-depth", "180-queue-depth", "Metric"),
)
_CPU_AT_PEAK = _metric("cpu-at-peak", "CPU at peak", "cpu_usage", "percent", 71)
_MEMORY_AT_PEAK = _metric("memory-at-peak", "memory at peak", "memory_usage", "GB", 3.1)
_QUEUE_DEPTH_MAX = _metric("queue-depth-max", "queue depth max", "queue_depth", "depth", 180)
_COLD_START = _metric("cold-start", "cold start", "cold_start", "seconds", 2.4)
_COST_PER_HOUR = _metric("cost-per-hour", "cost per hour", "cost_per_hour", "euros", 3.8)
_STATED_FIRST_EDGES = (
    _edge(_SERVICE, "38-ms-p50-latency", "HAS_METRIC"),
    _edge(_SERVICE, "410-ms-p99-latency", "HAS_METRIC"),
    _edge(_SERVICE, "9500-requests-per-second-peak-throughput", "HAS_METRIC"),
    _edge(_SERVICE, "0-2-percent-error-rate", "HAS_METRIC"),
    _edge(_SERVICE_AGAIN, "38-ms-latency", "HAS_METRIC"),
    _edge(_SERVICE_AGAIN, "410-ms-latency", "HAS_METRIC"),
    _edge(_SERVICE_AGAIN, "9500-requests-per-second-throughput", "HAS_METRIC"),
    _edge(_SERVICE_AGAIN, "0-2-percent-error-rate", "HAS_METRIC"),
    _edge(_SERVICE_AGAIN, "71-percent-cpu", "HAS_METRIC"),
    _edge(_SERVICE_AGAIN, "3-1-gb-memory", "HAS_METRIC"),
    _edge(_SERVICE_AGAIN, "2400-matches-per-minute", "HAS_METRIC"),
    _edge(_SERVICE_AGAIN, "180-queue-depth", "HAS_METRIC"),
)
_METRIC_ID_CASE = _case(
    (
        _USER,
        *_STATED_FIRST,
        _metric("p50-latency", "p50 latency", "latency", "ms", 38),
        _metric("p99-latency", "p99 latency", "latency", "ms", 410),
        _metric("peak-throughput", "peak throughput", "throughput", "requests per second", 9500),
        _metric("error-rate", "error rate", "error_rate", "percent", 0.2),
        _CPU_AT_PEAK,
        _MEMORY_AT_PEAK,
        _metric(
            "matches-per-minute",
            "matches per minute",
            "matches_per_minute",
            "matches per minute",
            2400,
        ),
        _QUEUE_DEPTH_MAX,
        _COLD_START,
        _COST_PER_HOUR,
    ),
    (
        _USER,
        *_STATED_FIRST,
        {**_CPU_AT_PEAK, "id": "71-percent-cpu-usage"},
        {**_MEMORY_AT_PEAK, "id": "3-1-gb-memory-usage"},
        {**_QUEUE_DEPTH_MAX, "id": "180-depth-queue-depth"},
        {**_COLD_START, "id": "2-4-seconds-cold-start"},
        {**_COST_PER_HOUR, "id": "3-8-euros-cost-per-hour"},
    ),
    relationships=(
        *_STATED_FIRST_EDGES,
        _edge("user", "p50-latency", "HAS_METRIC"),
        _edge("user", "p99-latency", "HAS_METRIC"),
        _edge("user", "peak-throughput", "HAS_METRIC"),
        _edge("user", "error-rate", "HAS_METRIC"),
        _edge("user", "cpu-at-peak", "HAS_METRIC"),
        _edge("user", "memory-at-peak", "HAS_METRIC"),
        _edge("user", "matches-per-minute", "HAS_METRIC"),
        _edge("user", "queue-depth-max", "HAS_METRIC"),
    ),
    expected_relationships=(
        *_STATED_FIRST_EDGES,
        _edge("user", "38-ms-latency", "HAS_METRIC"),
        _edge("user", "410-ms-latency", "HAS_METRIC"),
        _edge("user", "9500-requests-per-second-throughput", "HAS_METRIC"),
        _edge("user", "0-2-percent-error-rate", "HAS_METRIC"),
        _edge("user", "71-percent-cpu-usage", "HAS_METRIC"),
        _edge("user", "3-1-gb-memory-usage", "HAS_METRIC"),
        _edge("user", "2400-matches-per-minute", "HAS_METRIC"),
        _edge("user", "180-depth-queue-depth", "HAS_METRIC"),
    ),
)

# The tuning run's payload for dvl-15. Each Event is dated to one day, so its id and its
# name gain that day, and every edge that names it follows.
_DATES = (
    _entity("2026-11-18", "2026-11-18", "Date"),
    _entity("2026-12-03", "2026-12-03", "Date"),
    _entity("2026-11-21", "2026-11-21", "Date"),
)
_EVENT_DATE_CASE = _case(
    (
        _USER,
        _entity("dentist", "Dentist", "Event"),
        _entity("ilse-thesis-defense", "Ilse thesis defense", "Event"),
        _entity("team-retro", "Team retro", "Event"),
        *_DATES,
    ),
    (
        _USER,
        _entity("dentist-2026-11-18", "Dentist 2026-11-18", "Event"),
        _entity("ilse-thesis-defense-2026-12-03", "Ilse thesis defense 2026-12-03", "Event"),
        _entity("team-retro-2026-11-21", "Team retro 2026-11-21", "Event"),
        *_DATES,
    ),
    relationships=(
        _edge("user", "dentist", "EXPERIENCED"),
        _edge("dentist", "2026-11-18", "OCCURRED_ON"),
        _edge("user", "ilse-thesis-defense", "EXPERIENCED"),
        _edge("ilse-thesis-defense", "2026-12-03", "OCCURRED_ON"),
        _edge("user", "team-retro", "EXPERIENCED"),
        _edge("team-retro", "2026-11-21", "OCCURRED_ON"),
    ),
    expected_relationships=(
        _edge("user", "dentist-2026-11-18", "EXPERIENCED"),
        _edge("dentist-2026-11-18", "2026-11-18", "OCCURRED_ON"),
        _edge("user", "ilse-thesis-defense-2026-12-03", "EXPERIENCED"),
        _edge("ilse-thesis-defense-2026-12-03", "2026-12-03", "OCCURRED_ON"),
        _edge("user", "team-retro-2026-11-21", "EXPERIENCED"),
        _edge("team-retro-2026-11-21", "2026-11-21", "OCCURRED_ON"),
    ),
)

# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

RULES: tuple[ConventionRule, ...] = (
    ConventionRule(
        id=f"reserved-{RESERVED_ID}",
        kind="reserved_identity",
        phase=PAYLOAD,
        params=ReservedIdentity(names=RESERVED_NAMES, id=RESERVED_ID, type=MIST_IDENTITY.type_name),
        pinned_case=_RESERVED_CASE,
    ),
    ConventionRule(
        id="n1-exempt",
        kind="id_exempt",
        phase=ID,
        params=IdExempt(types=ID_EXEMPT_TYPES),
        pinned_case=_ID_EXEMPT_CASE,
    ),
    ConventionRule(
        id="pre-canon-aliases",
        kind="alias_vocabulary",
        phase=ID,
        params=AliasVocabulary(aliases=PRE_CANON_ALIASES, match="name"),
        pinned_case=_PRE_CANON_CASE,
    ),
    ConventionRule(
        id=TECHNOLOGY_VERSION,
        kind="typed_id",
        phase=ID,
        params=TypedId(type=TECHNOLOGY.type_name, plugin=TECHNOLOGY_VERSION),
        pinned_case=_TECHNOLOGY_VERSION_CASE,
    ),
    ConventionRule(
        id="static-aliases",
        kind="alias_vocabulary",
        phase=ID,
        params=AliasVocabulary(aliases=STATIC_ALIASES),
        pinned_case=_STATIC_ALIASES_CASE,
    ),
    ConventionRule(
        id=METRIC_ID,
        kind="typed_id",
        phase=ID,
        params=TypedId(type=METRIC.type_name, plugin=METRIC_ID),
        pinned_case=_METRIC_ID_CASE,
    ),
    ConventionRule(
        id=EVENT_DATE,
        kind="derived_edge",
        phase=ID,
        params=DerivedEdge(
            source_type=EVENT.type_name,
            edge=OCCURRED_ON.type_name,
            target_type=DATE.type_name,
            plugin=EVENT_DATE,
        ),
        pinned_case=_EVENT_DATE_CASE,
    ),
    # What the turn plan may name: every type and predicate the model may produce.
    ConventionRule(
        id="plan-vocabulary",
        kind="plan_vocabulary",
        phase=PRE_LLM,
        params=PlanVocabulary(
            entity_types=tuple(EXTRACTABLE_NODE_TYPES),
            predicates=EXTRACTABLE_EDGE_TYPES,
            fan_out_predicates=EXTRACTABLE_EDGE_TYPES,
        ),
    ),
)
