"""The v1.4.0 pack's convention rules, in the order their phases run them.

The pack loader reads `RULES`. Each rule the rails run carries a pinned case: a row and
the payload the pack must turn it into, which the rail runner checks the first time it is
handed the pack. A pinned case feeds neither of the pack's hashes.

Three of the cases are payloads the extraction service wrote for probes of dev-lists, the
tuning set, in the run of prompt candidate c3r2 (`CASES_FROM_THE_TUNING_RUN`). Each holds
the payload's entities as written and, of each relationship, its two ends and its type:
a relationship's properties quote the utterance, and no rule of this pack reads them.
The other cases are written for their rule.

The vocabularies, the reserved names and the prompt's ontology half are stated here and
nowhere else: the engine reads each through the pack.
"""

from __future__ import annotations

from collections.abc import Mapping

from backend.knowledge.extraction.prompts import (
    CONSTRAINTS_SLOT,
    MID_RULES_SLOT,
    OPENING_RULES_SLOT,
    RULES_SLOT,
)
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
    PromptFragment,
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
# The service entity is written twice, under two spellings of its id and one name. No
# rule of this pack joins them: the engine's near-duplicate merge joins the two spellings
# under the first, so the second entity is gone and its edges are the first's. A Metric
# that states its value takes the id the metric rail builds from it, and where that is
# the id of a Metric written earlier, the earlier one is kept.
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
# What is stored for the two: the first service entity alone, holding its own edges and
# then the second's. The second's edge to `0-2-percent-error-rate` repeats one of the
# first's and is stored once.
_STORED_FIRST = tuple(entity for entity in _STATED_FIRST if entity["id"] != _SERVICE_AGAIN)
_STORED_FIRST_EDGES = (
    _edge(_SERVICE, "38-ms-p50-latency", "HAS_METRIC"),
    _edge(_SERVICE, "410-ms-p99-latency", "HAS_METRIC"),
    _edge(_SERVICE, "9500-requests-per-second-peak-throughput", "HAS_METRIC"),
    _edge(_SERVICE, "0-2-percent-error-rate", "HAS_METRIC"),
    _edge(_SERVICE, "38-ms-latency", "HAS_METRIC"),
    _edge(_SERVICE, "410-ms-latency", "HAS_METRIC"),
    _edge(_SERVICE, "9500-requests-per-second-throughput", "HAS_METRIC"),
    _edge(_SERVICE, "71-percent-cpu", "HAS_METRIC"),
    _edge(_SERVICE, "3-1-gb-memory", "HAS_METRIC"),
    _edge(_SERVICE, "2400-matches-per-minute", "HAS_METRIC"),
    _edge(_SERVICE, "180-queue-depth", "HAS_METRIC"),
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
        *_STORED_FIRST,
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
        *_STORED_FIRST_EDGES,
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
# The prompt's ontology half
# ---------------------------------------------------------------------------

# What the system prompt says about this ontology, written as the model reads it. The
# engine core (`backend.knowledge.extraction.prompts`) holds the rest of the prompt
# and a slot at each place one of these texts stands; the loader places each by its
# rule's position. The rule numbers run around the core's own rules, 4, 5 and 7 to 10,
# and the texts cite rules by number, so a rule added here takes the next free number.

# The types, the relationships, and how to choose between them.
PROMPT_CONSTRAINTS = """### Allowed Entity Types (21 -- use EXACTLY these strings):
User, Person, Organization, Technology, Skill, Project, Concept, Event,
Goal, Preference, Location, Date, Metric, Document, Pattern, Convention,
Mechanism, Strategy, DataStructure, Abstraction, MistIdentity

### Abstract-Type Tests (pick the child whose test clearly fits; if none clearly
### fits, emit Abstraction -- never guess between sibling abstract types):
###   Concept       an idea/method/principle you understand but cannot "run"
###   Skill         a competency you can be better or worse at
###   Pattern       a named reusable solution shape
###   Strategy      a goal-directed plan, may compose patterns
###   Mechanism     a concrete component that performs a function
###   Convention    a normative naming/format rule
###   DataStructure an addressable runtime data shape

### Allowed Relationship Types (use EXACTLY these strings):
USES, KNOWS, WORKS_ON, WORKS_AT, INTERESTED_IN, HAS_GOAL, PREFERS, DISLIKES, EXPERT_IN, LEARNING, STRUGGLES_WITH, DECIDED, EXPERIENCED, IS_A, PART_OF, RELATED_TO, DEPENDS_ON, USED_FOR, WORKS_WITH, KNOWS_PERSON, MEMBER_OF, IMPLEMENTED_WITH, MIST_HAS_CAPABILITY, MIST_HAS_TRAIT, MIST_HAS_PREFERENCE, OCCURRED_ON, HAS_METRIC, REFERENCES_DOCUMENT, PRECEDED_BY, MECHANISM_OF, OPERATES_ON, INPUT_TO, IMPROVES, COMPRISES, APPLICABLE_TO, STRATEGY_FOR, NAMING_CONVENTION_OF, RECOMMENDS, HAS_HABIT

### Glosses (a "topic" is a Concept; there is no Topic type):
- Organization: "A company, team, institution, community, or open-source project."
- KNOWS: "Indicates familiarity with a skill, concept, technology, or topic."
- INTERESTED_IN: "Indicates interest in a technology, concept, topic, or skill."
- EXPERT_IN: "Indicates expert-level proficiency in a technology, skill, or concept."
- LEARNING: "Indicates actively learning a technology, skill, or concept."
- WORKS_AT: employment at an organization, the employer, and nothing else; membership is MEMBER_OF.
- MEMBER_OF: "Indicates membership in an organization."

### Subject Scope (passed in as SUBJECT SCOPE below)
- `user-scope` utterances: the user is the subject. Use User-centric predicates (USES, LEARNING, WORKS_ON, etc.). source="user".
- `system-scope` utterances: MIST is the subject. Use MistIdentity-centric predicates (USES, IMPLEMENTED_WITH, MIST_HAS_CAPABILITY, MIST_HAS_TRAIT, MIST_HAS_PREFERENCE). source="mist-identity" with type="MistIdentity".
- `third-party` utterances: someone else is the subject. Use Person or Organization as source, or drop the relationship if the subject is ambiguous. Do NOT attribute third-party claims to the user. When the user names that someone only by their relation ("my coworker"), still emit that Person and user KNOWS_PERSON <relation noun> (Rule 24, Example 7); only the claim stays off the user.
- `unknown` scope: use utterance content to infer; prefer no relationship over wrong attribution.

### Relationship Direction Rules:
- Structural relationships flow from specific to general: "Combwright" IS_A "hive-logging app" (Rule 25).
- USES accepts User, Person, Organization, or MistIdentity as source.
- DEPENDS_ON sources are Technology, Project, Organization, or MistIdentity (never User).
- WORKS_WITH is undirected between Technology/Organization/MistIdentity and Technology; either orientation is valid.
- IMPLEMENTED_WITH / MIST_HAS_* predicates require a MistIdentity source (id="mist-identity").

"""

# Rules 1 to 3: the subject by scope, and how an entity's id is formed.
PROMPT_RULES_1_TO_3 = """1. Subject entity depends on scope. For `user-scope` utterances, create entity {"id": "user", "name": "User", "type": "User"}. For `system-scope` utterances, create entity {"id": "mist-identity", "name": "MIST", "type": "MistIdentity"}. For `third-party` utterances, use Person/Organization names directly as sources. For `unknown` scope, infer from content and prefer no relationship over wrong attribution.
2. Entity IDs are the utterance's own phrase for the thing: lowercased and hyphenated, with leading articles and possessive pronouns (a, an, the, my, our, your, his, her, their, its) dropped, and a plural kept as written ("React Native" -> "react-native"). Never invent an id. When a contiguous run of the utterance's words names the entity, that run is the id, neither lengthened nor shortened: "handles transient network failures" -> "transient-network-failures", not "transient-failure". An id differs from the utterance's phrase only where a rule builds it: the "user" and "mist-identity" anchors, ISO Date ids, version collapse (Rule 3), and Rules 21 to 29.
3. Collapse version specifics into canonical names: "Python 3.11" -> "python".
"""

# Rule 6: the assertion kind, with the predicates a preference's negation takes.
PROMPT_RULE_6 = """6. Assertion kind -- every relationship carries "assertion_kind": "assert" (the fact is true / began), "cease" (the fact stopped being true: "stopped", "quit", "left", "gave up on", "don't ... anymore"; set end_date if a stop date is stated), or "retract" (the fact was NEVER true: "I never actually", "I misspoke", "scratch that", "I was wrong about"). Self-corrections about the user's own facts ("scratch that, I don't work at Initech") are retractions of facts, NOT directives -- Rule 10 does not apply to them. Preference negation is NOT cessation: "don't like X" -> DISLIKES with assertion_kind=assert; "don't like X anymore" -> PREFERS X cease and DISLIKES X assert; neither edge gets a date. When unsure, use "assert".
"""

# Rules 11 to 37: the conventions for this ontology's types and predicates.
PROMPT_RULES_11_TO_37 = """11. Event vs Milestone -- the `Milestone` type is RETIRED. Represent milestone-class facts (shipped / launched / completed / achieved / promoted -- explicit accomplishments worth flagging) as an `Event` with `properties.event_type="milestone"`, anchored to a `Date` via `OCCURRED_ON` when a date is given. Use a plain `Event` (no `event_type`) for meetings, decisions (paired with `DECIDED`), deadlines, conferences, life events, and generic notable occurrences (paired with `EXPERIENCED`). Do NOT emit a `Milestone` entity -- the type no longer exists; `Event` with `event_type="milestone"` is the canonical representation.
12. Document engagement -- when the user is reading, finishing, working through, halfway through, reviewing, or otherwise engaging with a named artifact (book, paper, ADR, RFC, spec, article, blog post), the artifact is a `Document` and the relationship is `REFERENCES_DOCUMENT`. Do NOT use `LEARNING` for this -- `LEARNING` is reserved for `Technology`, `Skill`, or `Concept` targets. Do NOT use `WORKS_ON` -- `WORKS_ON` is reserved for `Project` targets. The verb (read / finished / studying / reviewing / halfway through) does not change the edge type; the target type does.
13. Temporal precedence -- when the user says "X happened after Y" / "X came after Y" / "X after Y" / "X following Y", emit a `PRECEDED_BY` edge from X to Y (X PRECEDED_BY Y means X happened after Y in time). X and Y are `Event` entities (milestone-class facts are `Event` with `event_type="milestone"`); Y can also be a `Date`. Do NOT use `RELATED_TO` for explicit temporal ordering -- `RELATED_TO` drops the directional time signal. Do NOT use only `OCCURRED_ON` for "X after <date>" -- the precedence relationship is the load-bearing fact, not the date anchor.
14. Mechanism / Pattern / Strategy entity selection -- when a noun phrase names a concrete operational component (a thing that performs a specific function inside a larger system: garbage collector, validator, retry-policy, exclusion-filter, rate-limiter, scheduler), classify it as `Mechanism`, NOT `Concept` and NOT `Technology`. When it names a reusable algorithmic shape (LRU, two-pointer, observer, hexagonal-architecture), classify it as `Pattern`. When it names a high-level approach to a goal (hybrid-retrieval, semantic-search, write-through-caching), classify it as `Strategy`. When it names a first-class data shape used by code (memoryfragment, hashmap, trie, preferencerecord), classify it as `DataStructure`. When it names a naming/formatting rule (camelCase, snake_case, PEP 8), classify it as `Convention`. These types absorb the most common RELATED_TO cases and unlock canonical predicates below; defaulting to `Concept` is a quality regression.
15. Mechanism / Pattern / Strategy predicates -- prefer canonical predicates over `RELATED_TO` whenever a sentence expresses one of these shapes. (a) `MECHANISM_OF` -- "X is the mechanism by which Y works"; source is Mechanism or Pattern, target is Concept / Technology / Strategy. (b) `OPERATES_ON` -- "X acts on Y" / "X manages Y"; source is Mechanism / Technology / Strategy / Pattern, target is DataStructure / Concept. Direction is from the actor to the thing acted upon. (c) `INPUT_TO` -- "X feeds into Y" / "X is consumed by Y"; source is the data, target is the consumer. (d) `IMPROVES` -- "X optimises Y" / "X mitigates Y" / "X reduces Y" / "X speeds up Y"; merged optimization-or-mitigation predicate. (e) `COMPRISES` -- "X is made up of Y" / "X consists of Y"; source is the whole, target is the part. (f) `APPLICABLE_TO` -- "X applies to Y" / "X works for Y"; substrate-oriented. (g) `STRATEGY_FOR` -- "X is a strategy for Y"; goal-oriented. (h) `NAMING_CONVENTION_OF` -- "X is the naming convention used for Y"; source is Convention. Do NOT emit `RELATED_TO` when one of these canonical shapes fits.
16. Date entities are anchors, not facts. Create a `Date` entity ONLY when an edge targets it: an Event anchoring via OCCURRED_ON, or a PRECEDED_BY edge targeting the date (Rule 13). When a date merely scopes a stative fact ("since May 2026", "started using X in 2026"), put it in start_date/end_date and do NOT create a Date entity.
17. Recommendations and habits. "X recommended/suggested Y" -> X RECOMMENDS Y (source is the recommender, NOT the user unless the user recommended it); when X is a person the user knows, ALSO emit user KNOWS_PERSON X. "I <do activity> every <interval>" -> user HAS_HABIT <activity> with temporal_expression carrying the interval; habitual statements are current stative facts, not Events. HAS_HABIT requires an explicit recurrence cadence ("every morning", "each Friday", "nightly", "twice a week"). A continuous activity stated with only a start point ("mentoring since 2026", "I've been running for years") is NOT a habit -- use the matching stative predicate (EXPERT_IN, LEARNING, INTERESTED_IN) with valid_from instead.
18. Extract the asserted fact, not incidental scope. A trailing prepositional phrase that merely scopes another fact ("recommended GraphQL FOR the new service", "switched to Memcached FOR the project", "covers 87 percent OF the codebase") is context, not a separate relationship -- do NOT manufacture a structural edge (STRATEGY_FOR / APPLICABLE_TO / WORKS_ON / PART_OF / USED_FOR) for it. Extract a structural edge ONLY when that relationship is itself the asserted fact: "I use Rust FOR backend work" -> rust USED_FOR backend-work IS the fact, so extract it.
19. Third-party facts -- when the user states a fact about another person ("my sister Anjali works in biotech"), extract BOTH: user KNOWS_PERSON the person AND the third party's own fact with the third party as source (person WORKS_AT org, or equivalent predicate). Do not collapse the third-party fact onto the user.
20. Abstract-type disambiguation -- when a noun phrase names an abstract knowledge object, apply the ABSTRACT-TYPE TESTS above. Emit the winning child type (Concept / Skill / Pattern / Strategy / Mechanism / Convention / DataStructure). Emit `Abstraction` ONLY when no child type's test clearly wins. Never guess between siblings: if the noun could be Concept OR Pattern, choose Abstraction rather than picking at random.
21. An activity keeps its object. The id of an activity, a capability or a habit keeps the object the utterance gives it, possessive pronouns dropped: "learning to carve soundboards" -> "carve-soundboards", not "carving" (Example 27). A habit id is the utterance's own phrase: "I work out every morning" -> "work-out" (Example 24). This shapes the id only; it adds no fact.
22. Goal ids. A Goal's id is the goal's verb phrase as the utterance words it, articles and possessive pronouns dropped: "my goal is to rebuild my old harpsichord" -> "rebuild-old-harpsichord" (Example 27), not a rephrasing such as "harpsichord-restoration".
23. Name modifiers. A name that only modifies another name gets no entity of its own, unless a list's lead-in carries it: in "the Ferrowind tuning guide" the entity is "ferrowind-tuning-guide", and Ferrowind gets no entity.
24. A relative named by the relation. A single person the user names only by their relation to the user ("my coworker") is a `Person` whose id is the relation noun ("coworker"). That person gets user KNOWS_PERSON <relation noun> and their own durable facts, as a named person would (Example 7). No entity for: a group, since a `Person` is one person; someone the utterance relates to another person rather than to the user; someone named without a possessive ("a coworker").
25. Kind nouns. When the utterance says what kind of thing a named entity is, beyond its entity type, the kind is a `Concept` and the entity gets IS_A to it: "Combwright, a hive-logging app" -> combwright IS_A hive-logging-app (Example 28). No IS_A edge when: the kind only restates the entity type, as the words of the type's description do ("a framework" for a Technology); IS_A cannot join the pair (it joins only Abstraction, Concept, Location, Organization, Skill and Technology); the kind is part of the name; or the entity stands for something other than the thing the kind describes.
26. Event ids and edges. An Event's id is the utterance's own phrase for it, articles and possessive pronouns dropped. Keep a named owner, dropping its possessive 's ("Maelle's housewarming" -> "maelle-housewarming"), and keep a place as the utterance words it (Example 29). The id holds no date word. Date words are: a month or weekday name or its abbreviation, a day or year number, an ordinal day ("30th"), an ISO date, "today", "tonight", "tomorrow", "yesterday", and "this", "next" or "last" before "week", "weekend", "month", "year" or a weekday. So "the June 3 regatta" is "regatta", not "june-3-regatta". A number that names the event itself is not a date word ("cluster-8-phase-6", Example 9). The date is carried only by the Event's OCCURRED_ON edge to its `Date` (Examples 9, 12 and 29). A milestone-class Event (shipped, launched, completed, promoted) gets OCCURRED_ON to its `Date` and no user EXPERIENCED edge. An Event stated with no participant gets OCCURRED_ON only. An Event the user took part in also gets user EXPERIENCED <event>. A decision the user made is user DECIDED <event>; DECIDED takes only a User source.
27. Metric properties. When the utterance gives a measured number, the Metric carries "properties": {"value": <the number as written>, "unit": <its unit, or null when it has none>, "quantity": <what is measured, or null>}. Write its id value-first, "<value>-<unit>-<quantity>" with empty parts left out and a decimal point as a hyphen: "0.94 tool_selection" -> "0-94-tool-selection" (Example 10). Attach it with HAS_METRIC, whose source is a User, Project, Technology, Skill, Concept or Goal; a metric about an Organization or an Event gets no edge.
28. Structured turns. A list, heading or table turn arrives as lines, each `<context> > <lead-in>: <item>`, the context (headings and parent bullets above, joined by " > ") optional. Extract every line of the Utterance. Lines marked `[user]:` are earlier units, for reference only: never extract them again (Example 30). A list item's id is the item's own words for the thing itself (Rule 2), with articles, possessive pronouns and a possessive 's dropped anywhere in it. Add no word of its lead-in, heading or context, and no cadence, scope, owner, purpose or quantity phrase, even one on its line: a cadence goes in temporal_expression, a scope or purpose in context. A word that only joins a dropped phrase goes with it: "Chores: wax our looms on Sundays for Orrin" -> "wax-looms". But an Event item keeps its named owner and stated place (Rule 26); a Metric item's quantity is every word it gives for what is measured, none dropped or added (Rule 27).
29. List rules. Read each item as a sentence completing the lead-in or heading above it. A lead-in, heading or parent bullet sets the subject or predicate of the items under it; it is not an entity unless it names one. An item referring back takes the earlier item's entity and id. An option nobody chose gets no edge. A checked box `[x]` is a done fact; an unchecked box `[ ]` is a plan with no edge, except under a goals lead-in, where it is HAS_GOAL. A table row, `<header>: <value>; ...`, is one fact about its first column. A list with nothing durable (an errand list) is empty. Itinerary items are Events: user EXPERIENCED <event>, and the Event OCCURRED_ON the date its line or heading gives (Rule 26).
30. Decisions. A decision by the user or their team is user DECIDED <decision Event>, the Event RELATED_TO the thing chosen unless a canonical predicate fits. It does not make its outcome current: no USES for a tool not yet adopted, no cease for one not yet retired. Someone else's decision gets the Event and the link, no DECIDED; its owner, even named by role, is an entity with no edge to it.
31. Stated places and organisations. A place or organisation an Event is tied to by "in", "at" or "with" is an entity RELATED_TO from the Event, not Rule 18's incidental scope: "a lantern parade in Corvasse" -> lantern-parade-in-corvasse RELATED_TO corvasse (Location).
32. Third-party events. A named person's own non-milestone Event also gets <person> EXPERIENCED <event>.
33. Topic links. An Event is RELATED_TO what the utterance says it concerned: a subject discussed or decided on, decisions made, a decision's object; something chosen for a subject links to that subject instead. Add no implied Event for them.
34. Named teams. A team, club or community named by its work, a proper name, or a group noun in its heading, lead-in, bullet or the lead-in above is an `Organization` when it places a person (on, in, joined, runs, leads, the speaker's possessive, a title with its group noun, listed under it); each, lead included, gets MEMBER_OF it. Id by Rule 2, a written group word kept, a heading's word alone. No team for a nameless group noun, even with a describing word (new, old, small, whole, current, remote, other), a role with no group noun, or a relationship category (colleagues, friends, family, household). The employer's name alone, with a group noun or before a team name is the employer, WORKS_AT. A named employer is a separate `Organization` that each person placed in the team also WORKS_AT; the team's id holds no employer name and no "at <employer>", and the team gets no PART_OF to the employer (Example 31). Its tools and decisions follow Rules 35 and 30, a name modifier Rule 23.
35. The team's tools. Tools the user's team runs are user USES; a possessive that only names a thing is not.
36. Personal milestones. A personal legal or financial step completed on a day is a milestone-class Event, no user EXPERIENCED.
37. Goal deadlines. A goal's stated time frame is its deadline: HAS_GOAL gets end_date at the frame's end (last day of the day, month or year named), start_date null. A bare month is the first such month after the reference date. A lead-in's frame covers its goals; a frame whose year is unclear gets no date.
"""

# The worked examples, in the order the model reads them. Each opens with the blank
# line that parts it from what stands before.
PROMPT_EXAMPLES: tuple[str, ...] = (
    """
### Example 1: User-scope simple usage statement
Subject scope: user-scope
Utterance: "I've been using Python for about 5 years"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "python", "name": "Python", "type": "Technology"}], "relationships": [{"source": "user", "target": "python", "type": "USES", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": "about 5 years", "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 2: User-scope cessation
Subject scope: user-scope
Utterance: "I used to work with Java but stopped"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "java", "name": "Java", "type": "Technology"}], "relationships": [{"source": "user", "target": "java", "type": "USES", "properties": {"confidence": 0.9, "temporal_status": "past", "start_date": null, "end_date": null, "temporal_expression": "used to", "context": null, "negated": false, "assertion_kind": "cease"}}]}
""",
    """
### Example 3: System-scope MIST tooling
Subject scope: system-scope
Utterance: "MIST uses LanceDB for vector search and Neo4j for the knowledge graph"
Output:
{"entities": [{"id": "mist-identity", "name": "MIST", "type": "MistIdentity"}, {"id": "lancedb", "name": "LanceDB", "type": "Technology"}, {"id": "neo4j", "name": "Neo4j", "type": "Technology"}], "relationships": [{"source": "mist-identity", "target": "lancedb", "type": "USES", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "vector search", "negated": false, "assertion_kind": "assert"}}, {"source": "mist-identity", "target": "neo4j", "type": "USES", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "knowledge graph", "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 4: System-scope implementation stack
Subject scope: system-scope
Utterance: "MIST is implemented with Python and llama.cpp"
Output:
{"entities": [{"id": "mist-identity", "name": "MIST", "type": "MistIdentity"}, {"id": "python", "name": "Python", "type": "Technology"}, {"id": "llama-cpp", "name": "llama.cpp", "type": "Technology"}], "relationships": [{"source": "mist-identity", "target": "python", "type": "IMPLEMENTED_WITH", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}, {"source": "mist-identity", "target": "llama-cpp", "type": "IMPLEMENTED_WITH", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 5: User-scope multi-entity with third-party coreference
Subject scope: user-scope
Utterance: "I'm learning React at work. My team lead Sarah uses it for everything"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "react", "name": "React", "type": "Technology"}, {"id": "sarah", "name": "Sarah", "type": "Person"}], "relationships": [{"source": "user", "target": "react", "type": "LEARNING", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "at work", "negated": false, "assertion_kind": "assert"}}, {"source": "user", "target": "sarah", "type": "KNOWS_PERSON", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "team lead", "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 6: System-scope personality traits
Subject scope: system-scope
Utterance: "MIST is warm and playful by default"
Output:
{"entities": [{"id": "mist-identity", "name": "MIST", "type": "MistIdentity"}, {"id": "warm", "name": "Warm", "type": "Concept"}, {"id": "playful", "name": "Playful", "type": "Concept"}], "relationships": [{"source": "mist-identity", "target": "warm", "type": "MIST_HAS_TRAIT", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "by default", "negated": false, "assertion_kind": "assert"}}, {"source": "mist-identity", "target": "playful", "type": "MIST_HAS_TRAIT", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "by default", "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 7: Third-party opinion -- the relative named by the relation is a Person; the opinion is not the user's
Subject scope: third-party
Utterance: "My coworker says Rust is really fast"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "coworker", "name": "Coworker", "type": "Person"}, {"id": "rust", "name": "Rust", "type": "Technology"}], "relationships": [{"source": "user", "target": "coworker", "type": "KNOWS_PERSON", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 8: No extractable knowledge
Subject scope: user-scope
Utterance: "Hey, how's it going?"
Output:
{"entities": [], "relationships": []}
""",
    """
### Example 9: Temporal -- milestone-class Event anchored to a Date via OCCURRED_ON
Subject scope: user-scope
Utterance: "We shipped Cluster 8 Phase 6 on 2026-04-22"
Output:
{"entities": [{"id": "cluster-8-phase-6", "name": "Cluster 8 Phase 6", "type": "Event", "properties": {"event_type": "milestone"}}, {"id": "2026-04-22", "name": "2026-04-22", "type": "Date"}], "relationships": [{"source": "cluster-8-phase-6", "target": "2026-04-22", "type": "OCCURRED_ON", "properties": {"confidence": 0.95, "temporal_status": "past", "start_date": "2026-04-22", "end_date": null, "temporal_expression": "on 2026-04-22", "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 10: Quantified -- Technology with a numeric Metric via HAS_METRIC
Subject scope: unknown
Utterance: "Gemma 4 E4B gets 0.94 tool_selection on the eval harness"
Output:
{"entities": [{"id": "gemma-4-e4b", "name": "Gemma 4 E4B", "type": "Technology"}, {"id": "0-94-tool-selection", "name": "0.94 tool_selection", "type": "Metric", "properties": {"value": 0.94, "unit": null, "quantity": "tool_selection"}}], "relationships": [{"source": "gemma-4-e4b", "target": "0-94-tool-selection", "type": "HAS_METRIC", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "eval harness", "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 11: Document reference -- user references an ADR via REFERENCES_DOCUMENT
Subject scope: user-scope
Utterance: "I read ADR-010 yesterday and I like the vault-as-canon pattern"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "adr-010", "name": "ADR-010", "type": "Document"}], "relationships": [{"source": "user", "target": "adr-010", "type": "REFERENCES_DOCUMENT", "properties": {"confidence": 0.9, "temporal_status": "past", "start_date": null, "end_date": null, "temporal_expression": "yesterday", "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 12: Temporal -- user-experienced Event anchored to a Date; the id is the bare phrase, the date only on OCCURRED_ON (contrast Example 9 milestone-class Event)
Subject scope: user-scope
Utterance: "I attended a glassblowing symposium on 2026-04-15"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "glassblowing-symposium", "name": "Glassblowing symposium", "type": "Event"}, {"id": "2026-04-15", "name": "2026-04-15", "type": "Date"}], "relationships": [{"source": "user", "target": "glassblowing-symposium", "type": "EXPERIENCED", "properties": {"confidence": 0.9, "temporal_status": "past", "start_date": "2026-04-15", "end_date": null, "temporal_expression": "on 2026-04-15", "context": null, "negated": false, "assertion_kind": "assert"}}, {"source": "glassblowing-symposium", "target": "2026-04-15", "type": "OCCURRED_ON", "properties": {"confidence": 0.95, "temporal_status": "past", "start_date": "2026-04-15", "end_date": null, "temporal_expression": "on 2026-04-15", "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 13: Document engagement -- active reading verbs trigger REFERENCES_DOCUMENT
Subject scope: user-scope
Utterance: "I'm working through the Pragmatic Programmer"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "pragmatic-programmer", "name": "Pragmatic Programmer", "type": "Document"}], "relationships": [{"source": "user", "target": "pragmatic-programmer", "type": "REFERENCES_DOCUMENT", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": "working through", "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 14: Temporal precedence -- "X after Y" triggers PRECEDED_BY (X PRECEDED_BY Y)
Subject scope: unknown
Utterance: "The deploy happened after the code review"
Output:
{"entities": [{"id": "deploy", "name": "Deploy", "type": "Event"}, {"id": "code-review", "name": "Code review", "type": "Event"}], "relationships": [{"source": "deploy", "target": "code-review", "type": "PRECEDED_BY", "properties": {"confidence": 0.9, "temporal_status": "past", "start_date": null, "end_date": null, "temporal_expression": "after", "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 15: Mechanism / OPERATES_ON -- a concrete operational component acting on data
Subject scope: unknown
Utterance: "The garbage collector reclaims unused memory fragments"
Output:
{"entities": [{"id": "garbage-collector", "name": "Garbage collector", "type": "Mechanism"}, {"id": "memory-fragments", "name": "Memory fragments", "type": "DataStructure"}], "relationships": [{"source": "garbage-collector", "target": "memory-fragments", "type": "OPERATES_ON", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "reclaims unused", "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 16: COMPRISES + INPUT_TO -- composition and data-flow direction
Subject scope: unknown
Utterance: "The retrieval pipeline consists of a query encoder and a reranker; user context feeds into the encoder"
Output:
{"entities": [{"id": "retrieval-pipeline", "name": "Retrieval pipeline", "type": "Strategy"}, {"id": "query-encoder", "name": "Query encoder", "type": "Mechanism"}, {"id": "reranker", "name": "Reranker", "type": "Mechanism"}, {"id": "user-context", "name": "User context", "type": "DataStructure"}], "relationships": [{"source": "retrieval-pipeline", "target": "query-encoder", "type": "COMPRISES", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}, {"source": "retrieval-pipeline", "target": "reranker", "type": "COMPRISES", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}, {"source": "user-context", "target": "query-encoder", "type": "INPUT_TO", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "feeds into", "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 17: IMPROVES -- mechanism reduces a harm or boosts a metric (merged optimize/mitigate)
Subject scope: unknown
Utterance: "Speculative decoding speeds up token generation, and the LRU cache cuts down on lookup latency"
Output:
{"entities": [{"id": "speculative-decoding", "name": "Speculative decoding", "type": "Mechanism"}, {"id": "token-generation", "name": "Token generation", "type": "Concept"}, {"id": "lru-cache", "name": "LRU cache", "type": "Mechanism"}, {"id": "lookup-latency", "name": "Lookup latency", "type": "Metric"}], "relationships": [{"source": "speculative-decoding", "target": "token-generation", "type": "IMPROVES", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "speeds up", "negated": false, "assertion_kind": "assert"}}, {"source": "lru-cache", "target": "lookup-latency", "type": "IMPROVES", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "cuts down on", "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 18: STRATEGY_FOR + APPLICABLE_TO + NAMING_CONVENTION_OF -- goal vs substrate vs naming rule
Subject scope: unknown
Utterance: "Hybrid retrieval is our strategy for memory recall and applies to long-context conversations. The MemoryFragment data structure follows camelCase naming"
Output:
{"entities": [{"id": "hybrid-retrieval", "name": "Hybrid retrieval", "type": "Strategy"}, {"id": "memory-recall", "name": "Memory recall", "type": "Goal"}, {"id": "long-context-conversations", "name": "Long-context conversations", "type": "Concept"}, {"id": "memoryfragment", "name": "MemoryFragment", "type": "DataStructure"}, {"id": "camelcase", "name": "camelCase", "type": "Convention"}], "relationships": [{"source": "hybrid-retrieval", "target": "memory-recall", "type": "STRATEGY_FOR", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}, {"source": "hybrid-retrieval", "target": "long-context-conversations", "type": "APPLICABLE_TO", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}, {"source": "camelcase", "target": "memoryfragment", "type": "NAMING_CONVENTION_OF", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 19: Mechanism / Pattern type discrimination -- a Pattern (LRU) vs the Mechanism (cache) that uses it
Subject scope: unknown
Utterance: "The LRU pattern is applicable to cache eviction and the cache mechanism comprises a doubly-linked list"
Output:
{"entities": [{"id": "lru", "name": "LRU", "type": "Pattern"}, {"id": "cache-eviction", "name": "Cache eviction", "type": "Concept"}, {"id": "cache", "name": "Cache", "type": "Mechanism"}, {"id": "doubly-linked-list", "name": "Doubly-linked list", "type": "DataStructure"}], "relationships": [{"source": "lru", "target": "cache-eviction", "type": "APPLICABLE_TO", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}, {"source": "cache", "target": "doubly-linked-list", "type": "COMPRISES", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 20: MECHANISM_OF -- phrasal variations beyond "X is the mechanism for Y"
Subject scope: unknown
Utterance: "Speculative decoding is the mechanism behind faster token generation, and the retry-policy mechanism handles transient network failures"
Output:
{"entities": [{"id": "speculative-decoding", "name": "Speculative decoding", "type": "Mechanism"}, {"id": "token-generation", "name": "Token generation", "type": "Concept"}, {"id": "retry-policy", "name": "Retry policy", "type": "Mechanism"}, {"id": "transient-network-failures", "name": "Transient network failures", "type": "Concept"}], "relationships": [{"source": "speculative-decoding", "target": "token-generation", "type": "MECHANISM_OF", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "behind faster", "negated": false, "assertion_kind": "assert"}}, {"source": "retry-policy", "target": "transient-network-failures", "type": "MECHANISM_OF", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "handles", "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 21: Cessation with stated stop date and abandoned learning
Subject scope: user-scope
Utterance: "I quit my job at Acme in March 2026, and I've given up on learning Haskell"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "acme", "name": "Acme", "type": "Organization"}, {"id": "haskell", "name": "Haskell", "type": "Technology"}], "relationships": [{"source": "user", "target": "acme", "type": "WORKS_AT", "properties": {"confidence": 0.95, "temporal_status": "past", "start_date": null, "end_date": "2026-03-01", "temporal_expression": "in March 2026", "context": null, "negated": false, "assertion_kind": "cease"}}, {"source": "user", "target": "haskell", "type": "LEARNING", "properties": {"confidence": 0.9, "temporal_status": "past", "start_date": null, "end_date": null, "temporal_expression": "given up on", "context": null, "negated": false, "assertion_kind": "cease"}}]}
""",
    """
### Example 22: Retraction -- a self-correction is a fact retraction, not a directive
Subject scope: user-scope
Utterance: "Scratch that -- I never actually used MongoDB"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "mongodb", "name": "MongoDB", "type": "Technology"}], "relationships": [{"source": "user", "target": "mongodb", "type": "USES", "properties": {"confidence": 0.95, "temporal_status": "past", "start_date": null, "end_date": null, "temporal_expression": "never", "context": null, "negated": false, "assertion_kind": "retract"}}]}
""",
    """
### Example 23: Third-party recommendation -- RECOMMENDS plus KNOWS_PERSON, user stays anchored
Subject scope: user-scope
Utterance: "My coworker Sarah recommended Postgres"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "sarah", "name": "Sarah", "type": "Person"}, {"id": "postgres", "name": "Postgres", "type": "Technology"}], "relationships": [{"source": "user", "target": "sarah", "type": "KNOWS_PERSON", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "coworker", "negated": false, "assertion_kind": "assert"}}, {"source": "sarah", "target": "postgres", "type": "RECOMMENDS", "properties": {"confidence": 0.9, "temporal_status": "past", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 24: Habit -- recurring activity via HAS_HABIT, no Date entity for the recurrence
Subject scope: user-scope
Utterance: "I work out every morning before standup"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "work-out", "name": "Work out", "type": "Concept"}], "relationships": [{"source": "user", "target": "work-out", "type": "HAS_HABIT", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": "every morning", "context": "before standup", "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 25: Third-party facts -- KNOWS_PERSON plus the third party's own WORKS_AT fact
Subject scope: user-scope
Utterance: "My sister Anjali works at Google in biotech"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "anjali", "name": "Anjali", "type": "Person"}, {"id": "google", "name": "Google", "type": "Organization"}], "relationships": [{"source": "user", "target": "anjali", "type": "KNOWS_PERSON", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "sister", "negated": false, "assertion_kind": "assert"}}, {"source": "anjali", "target": "google", "type": "WORKS_AT", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "biotech", "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 26: Abstraction fallback -- ambiguous abstract noun that fits no single child type
Subject scope: user-scope
Utterance: "I find the notion of semantic closure really interesting"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "semantic-closure", "name": "Semantic closure", "type": "Abstraction"}], "relationships": [{"source": "user", "target": "semantic-closure", "type": "INTERESTED_IN", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 27: Goal and activity ids -- the goal's verb phrase, the activity with its object
Subject scope: user-scope
Utterance: "My goal is to rebuild my old harpsichord, so I'm learning to carve soundboards"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "rebuild-old-harpsichord", "name": "Rebuild old harpsichord", "type": "Goal"}, {"id": "carve-soundboards", "name": "Carve soundboards", "type": "Skill"}], "relationships": [{"source": "user", "target": "rebuild-old-harpsichord", "type": "HAS_GOAL", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}, {"source": "user", "target": "carve-soundboards", "type": "LEARNING", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 28: Kind noun -- the named entity IS_A the kind the utterance gives it
Subject scope: user-scope
Utterance: "I track my beehives with Combwright, a hive-logging app"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "combwright", "name": "Combwright", "type": "Technology"}, {"id": "hive-logging-app", "name": "Hive-logging app", "type": "Concept"}], "relationships": [{"source": "user", "target": "combwright", "type": "USES", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "track my beehives", "negated": false, "assertion_kind": "assert"}}, {"source": "combwright", "target": "hive-logging-app", "type": "IS_A", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 29: Event id -- named owner and place kept, possessive 's dropped, date only on OCCURRED_ON
Subject scope: user-scope
Utterance: "I went to Maelle's housewarming in Dunmarrow on 2026-03-07"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "maelle-housewarming-in-dunmarrow", "name": "Maelle housewarming in Dunmarrow", "type": "Event"}, {"id": "2026-03-07", "name": "2026-03-07", "type": "Date"}], "relationships": [{"source": "user", "target": "maelle-housewarming-in-dunmarrow", "type": "EXPERIENCED", "properties": {"confidence": 0.9, "temporal_status": "past", "start_date": "2026-03-07", "end_date": null, "temporal_expression": "on 2026-03-07", "context": null, "negated": false, "assertion_kind": "assert"}}, {"source": "maelle-housewarming-in-dunmarrow", "target": "2026-03-07", "type": "OCCURRED_ON", "properties": {"confidence": 0.95, "temporal_status": "past", "start_date": "2026-03-07", "end_date": null, "temporal_expression": "on 2026-03-07", "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 30: List unit. Context: [user]: Bench diary > Crafts I'm learning: chip carving
Subject scope: user-scope
Utterance: "Bench diary > Crafts I'm learning: lost-wax casting"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "lost-wax-casting", "name": "Lost-wax casting", "type": "Skill"}], "relationships": [{"source": "user", "target": "lost-wax-casting", "type": "LEARNING", "properties": {"confidence": 0.9, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
    """
### Example 31: Rule 34
Subject scope: user-scope
Utterance: "I lead the Brackwater billing team at Quillmere"
Output:
{"entities": [{"id": "user", "name": "User", "type": "User"}, {"id": "brackwater-billing-team", "name": "Brackwater billing team", "type": "Organization"}, {"id": "quillmere", "name": "Quillmere", "type": "Organization"}], "relationships": [{"source": "user", "target": "brackwater-billing-team", "type": "MEMBER_OF", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": "lead", "negated": false, "assertion_kind": "assert"}}, {"source": "user", "target": "quillmere", "type": "WORKS_AT", "properties": {"confidence": 0.95, "temporal_status": "current", "start_date": null, "end_date": null, "temporal_expression": null, "context": null, "negated": false, "assertion_kind": "assert"}}]}
""",
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
        params=TypedId(type=METRIC.type_name, plugin=METRIC_ID, alignment="metric"),
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
    # The prompt's ontology half, each text in the slot of the engine core it stands in.
    ConventionRule(
        id="prompt-constraints",
        kind="prompt_fragment",
        phase=PRE_LLM,
        params=PromptFragment(text=PROMPT_CONSTRAINTS, position=CONSTRAINTS_SLOT),
    ),
    ConventionRule(
        id="prompt-rules-1-to-3",
        kind="prompt_fragment",
        phase=PRE_LLM,
        params=PromptFragment(text=PROMPT_RULES_1_TO_3, position=OPENING_RULES_SLOT),
    ),
    ConventionRule(
        id="prompt-rule-6",
        kind="prompt_fragment",
        phase=PRE_LLM,
        params=PromptFragment(text=PROMPT_RULE_6, position=MID_RULES_SLOT),
    ),
    ConventionRule(
        id="prompt-rules-11-to-37",
        kind="prompt_fragment",
        phase=PRE_LLM,
        params=PromptFragment(text=PROMPT_RULES_11_TO_37, position=RULES_SLOT),
    ),
    # The examples go to the core's examples slot whatever position their fragment names.
    ConventionRule(
        id="prompt-examples",
        kind="prompt_fragment",
        phase=PRE_LLM,
        params=PromptFragment(text="", position=RULES_SLOT, examples=PROMPT_EXAMPLES),
    ),
)
