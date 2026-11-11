"""The base fixture pack's rules: one of every kind, as data."""

from backend.knowledge.ontologies.pack import (
    ID,
    PAYLOAD,
    PRE_LLM,
    RELATION,
    AliasVocabulary,
    ConventionRule,
    DerivedEdge,
    EdgeSuppress,
    EdgeTrigger,
    FanOut,
    GainedEdge,
    IdExempt,
    ItemFilter,
    PinnedCase,
    PlanVocabulary,
    PluginRule,
    PromptFragment,
    ReservedIdentity,
    TypedId,
)

_MOORING_NUMBER_CASE = PinnedCase(
    entities=({"id": "the-north-mooring", "name": "North Mooring No. 7", "type": "Mooring"},),
    relationships=(),
    expected_entities=({"id": "mooring-7", "name": "North Mooring No. 7", "type": "Mooring"},),
    expected_relationships=(),
)

_MOORED_ID_CASE = PinnedCase(
    entities=(
        {"id": "Tolliver", "name": "Tolliver", "type": "Deckhand"},
        {"id": "mooring 12", "name": "Mooring 12", "type": "Mooring"},
    ),
    relationships=({"source": "Tolliver", "target": "mooring 12", "type": "MOORED_AT"},),
    expected_entities=(
        {"id": "tolliver-at-mooring-12", "name": "Tolliver mooring-12", "type": "Deckhand"},
        {"id": "mooring-12", "name": "Mooring 12", "type": "Mooring"},
    ),
    expected_relationships=(
        {"source": "tolliver-at-mooring-12", "target": "mooring-12", "type": "MOORED_AT"},
    ),
)

_CREW_ENTITIES = (
    {"id": "user", "name": "User", "type": "Deckhand"},
    {"id": "oswin-pike", "name": "Oswin Pike", "type": "Deckhand"},
    {"id": "brannoch", "name": "Brannoch", "type": "Deckhand"},
    {"id": "wend", "name": "Wend", "type": "Deckhand"},
    {"id": "mooring-3", "name": "Mooring 3", "type": "Mooring"},
)
_CREW_EDGES = (
    {
        "source": "oswin-pike",
        "target": "brannoch",
        "type": "CREWS_WITH",
        "properties": {"confidence": 0.8, "temporal_status": "current", "start_date": "2026-01-05"},
    },
    {
        "source": "wend",
        "target": "brannoch",
        "type": "CREWS_WITH",
        "properties": {"confidence": 0.9, "temporal_status": "past"},
    },
    {
        "source": "user",
        "target": "brannoch",
        "type": "CREWS_WITH",
        "properties": {"confidence": 0.95},
    },
    {
        "source": "brannoch",
        "target": "mooring-3",
        "type": "LODGES_AT",
        "properties": {"confidence": 0.9},
    },
)
_OSWIN_COPY = {"temporal_status": "current", "confidence": 0.8}
_USER_COPY = {"confidence": 0.95}

_CREW_CASE = PinnedCase(
    scope="user-scope",
    entities=_CREW_ENTITIES,
    relationships=(
        *_CREW_EDGES,
        {"source": "brannoch", "target": "oswin-pike", "type": "OUTRANKS"},
    ),
    expected_entities=_CREW_ENTITIES,
    expected_relationships=(
        *_CREW_EDGES,
        {
            "source": "oswin-pike",
            "target": "brannoch",
            "type": "SERVES_UNDER",
            "properties": _OSWIN_COPY,
        },
        {"source": "user", "target": "oswin-pike", "type": "KNOWS_HAND", "properties": _OSWIN_COPY},
        {
            "source": "oswin-pike",
            "target": "mooring-3",
            "type": "LODGES_AT",
            "properties": _OSWIN_COPY,
        },
        {"source": "user", "target": "brannoch", "type": "SERVES_UNDER", "properties": _USER_COPY},
        {"source": "user", "target": "mooring-3", "type": "LODGES_AT", "properties": _USER_COPY},
    ),
)

_SOLO_CASE = PinnedCase(
    entities=({"id": "Fenwick", "name": "Fenwick", "type": "Deckhand"},),
    relationships=({"source": "Fenwick", "target": "Fenwick", "type": "CREWS_WITH"},),
    expected_entities=({"id": "fenwick", "name": "Fenwick", "type": "Deckhand"},),
    expected_relationships=(),
)

RULES = (
    ConventionRule(
        id="harbourmaster",
        kind="reserved_identity",
        phase=PAYLOAD,
        params=ReservedIdentity(
            names=("the harbourmaster", "harbour-master"), id="harbourmaster", type="Deckhand"
        ),
    ),
    ConventionRule(
        id="mooring-exempt", kind="id_exempt", phase=ID, params=IdExempt(types=("Mooring",))
    ),
    ConventionRule(
        id="spoken-ranks",
        kind="alias_vocabulary",
        phase=ID,
        params=AliasVocabulary(aliases=(("cap'n", "captain"),), match="name"),
    ),
    ConventionRule(
        id="mooring-number",
        kind="typed_id",
        phase=ID,
        params=TypedId(type="Mooring", plugin="mooring-number"),
        pinned_case=_MOORING_NUMBER_CASE,
    ),
    ConventionRule(
        id="rank-aliases",
        kind="alias_vocabulary",
        phase=ID,
        params=AliasVocabulary(aliases=(("bosun", "boatswain"),), types=("Deckhand",)),
    ),
    ConventionRule(
        id="moored-id",
        kind="derived_edge",
        phase=ID,
        params=DerivedEdge(
            source_type="Deckhand", edge="MOORED_AT", target_type="Mooring", plugin="moored-id"
        ),
        pinned_case=_MOORED_ID_CASE,
    ),
    ConventionRule(
        id="home-mooring",
        kind="derived_edge",
        phase=RELATION,
        params=DerivedEdge(
            source_type="Deckhand",
            edge="LODGES_AT",
            target_type="Mooring",
            property="home_mooring",
            edge_properties=(("temporal_status", "current"),),
        ),
        prompt_rule="Rule F3",
    ),
    ConventionRule(
        id="crew-fan-out",
        kind="fan_out",
        phase=RELATION,
        params=FanOut(
            trigger=EdgeTrigger(
                edge="CREWS_WITH", source_types=("Deckhand",), target_types=("Deckhand",)
            ),
            gains=(
                GainedEdge(edge="SERVES_UNDER", source="member", target="anchor"),
                GainedEdge(
                    edge="KNOWS_HAND", source="user", target="member", scopes=("user-scope",)
                ),
                GainedEdge(edge="LODGES_AT", source="member", target="plugin:skipper-lodging"),
            ),
        ),
        prompt_rule="Rule F1",
        pinned_case=_CREW_CASE,
    ),
    ConventionRule(
        id="skipper-outranks",
        kind="edge_suppress",
        phase=RELATION,
        params=EdgeSuppress(
            edge="OUTRANKS",
            source_types=("Deckhand",),
            target_types=("Deckhand",),
            source_is_target_of=("CREWS_WITH",),
        ),
        prompt_rule="Rule F2",
    ),
    ConventionRule(
        id="no-solo-crew",
        kind="plugin",
        phase=RELATION,
        params=PluginRule(plugin="no-solo-crew"),
        prompt_rule="Rule F4",
        pinned_case=_SOLO_CASE,
    ),
    ConventionRule(
        id="unchecked-items",
        kind="item_filter",
        phase=PRE_LLM,
        params=ItemFilter(state="unchecked", keep_context_words=("drill", "watch list")),
    ),
    ConventionRule(
        id="plan-words",
        kind="plan_vocabulary",
        phase=PRE_LLM,
        params=PlanVocabulary(
            entity_types=("Deckhand", "Mooring"),
            predicates=("CREWS_WITH", "MOORED_AT"),
            fan_out_predicates=("SERVES_UNDER",),
        ),
    ),
    ConventionRule(
        id="crew-rule-text",
        kind="prompt_fragment",
        phase=PRE_LLM,
        params=PromptFragment(
            text="F1. Each deckhand placed in a crew also SERVES_UNDER its skipper.",
            position=1,
        ),
    ),
)
