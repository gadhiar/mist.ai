"""A synthetic payload, with invented names, that fires every rail of today's ontology.

`EXPECTED_ENTITIES` and `EXPECTED_RELATIONSHIPS` are what `id_rails.normalize_payload`
returned for it at eeff2a4, the commit before the rail runner existed: they were printed
by that commit's code and pasted here, so the runner is held to the old output and not
to its own.
"""

ENTITIES: list[dict] = [
    {"id": "User", "name": "Ilse", "type": "User"},
    {"id": "helper-bot", "name": "MIST", "type": "Organization"},
    {"id": "The AI", "name": "the helper", "type": "Concept"},
    {"id": "cpp-lang", "name": "C++", "type": "Technology"},
    {"id": "Quillscript 3.11", "name": "Quillscript 3.11", "type": "Technology"},
    {"id": "node 18", "name": "Node 18", "type": "Technology"},
    {"id": "k8s", "name": "K8s", "type": "Concept"},
    {"id": "harbour-opening", "name": "Harbour opening", "type": "Milestone"},
    {"id": "old-topic", "name": "Tide tables", "type": "Topic"},
    {
        "id": "cold-start",
        "name": "cold start",
        "type": "Metric",
        "properties": {"value": 2.4, "unit": "s", "quantity": "cold start"},
    },
    {"id": "ts", "name": "ts", "type": "Metric"},
    {"id": "tool-selection-0.94", "type": "Metric"},
    {"id": "the-odd-thing", "name": "The odd thing", "type": "NotARealType"},
    {"id": "The Brindle Works", "name": "The Brindle Works", "type": "Organization"},
    {"id": "brindle-works", "name": "Brindle Works", "type": "Organization"},
    {"id": "2026-05-30", "name": "2026-05-30", "type": "Date"},
    {"id": "2026-06-01", "name": "2026-06-01", "type": "Date"},
    {"id": "my-first-date", "name": "my first date", "type": "Date"},
    {"id": "regatta", "name": "Regatta", "type": "Event"},
    {"id": "launch-2026-01-02", "name": "Launch", "type": "Event"},
    {"name": "Nameless Co", "type": "Organization"},
    {"id": "my-the", "type": "Person"},
    {"id": "the-boss", "name": "Boss", "type": "User"},
    {"id": "our-ts", "name": "Our TS", "type": "Technology"},
]

RELATIONSHIPS: list[dict] = [
    {"source": "User", "target": "Quillscript 3.11", "type": "USES"},
    {"source": "User", "target": "node 18", "type": "USES"},
    {"source": "User", "target": "The Brindle Works", "type": "WORKS_AT"},
    {"source": "User", "target": "brindle-works", "type": "WORKS_AT"},
    {"source": "helper-bot", "target": "cpp-lang", "type": "IMPLEMENTED_WITH"},
    {"source": "User", "target": "regatta", "type": "EXPERIENCED"},
    {"source": "regatta", "target": "2026-05-30", "type": "OCCURRED_ON"},
    {"source": "launch-2026-01-02", "target": "2026-05-30", "type": "OCCURRED_ON"},
    {"source": "launch-2026-01-02", "target": "2026-06-01", "type": "OCCURRED_ON"},
    {"source": "harbour-opening", "target": "2026-05-30", "type": "OCCURRED_ON"},
    {"source": "User", "target": "ghost", "type": "USES"},
    {"source": "The AI", "target": "k8s", "type": "USES", "properties": {"confidence": 0.7}},
    {"source": "User", "target": "our-ts", "type": "USES"},
]

EXPECTED_ENTITIES: list[dict] = [
    {"id": "user", "name": "Ilse", "type": "User"},
    {"id": "mist-identity", "name": "MIST", "type": "MistIdentity"},
    {"id": "c-plus-plus", "name": "C++", "type": "Technology"},
    {"id": "quillscript", "name": "Quillscript 3.11", "type": "Technology"},
    {"id": "nodejs", "name": "Node 18", "type": "Technology"},
    {"id": "kubernetes", "name": "K8s", "type": "Concept"},
    {
        "id": "harbour-opening-2026-05-30",
        "name": "Harbour opening 2026-05-30",
        "type": "Event",
        "properties": {"event_type": "milestone"},
    },
    {"id": "old-topic", "name": "Tide tables", "type": "Concept"},
    {
        "id": "2-4-seconds-cold-start",
        "name": "cold start",
        "type": "Metric",
        "properties": {"value": 2.4, "unit": "s", "quantity": "cold start"},
    },
    {"id": "ts", "name": "ts", "type": "Metric"},
    {"id": "0-94-tool-selection", "type": "Metric"},
    {"id": "odd-thing", "name": "The odd thing", "type": "Abstraction"},
    {"id": "brindle-works", "name": "The Brindle Works", "type": "Organization"},
    {"id": "2026-05-30", "name": "2026-05-30", "type": "Date"},
    {"id": "2026-06-01", "name": "2026-06-01", "type": "Date"},
    {"id": "my-first-date", "name": "my first date", "type": "Date"},
    {"id": "regatta-2026-05-30", "name": "Regatta 2026-05-30", "type": "Event"},
    {"id": "launch-2026-01-02", "name": "Launch", "type": "Event"},
    {"name": "Nameless Co", "type": "Organization", "id": "nameless-co"},
    {"id": "the", "type": "Person"},
    {"id": "the-boss", "name": "Boss", "type": "User"},
]

EXPECTED_RELATIONSHIPS: list[dict] = [
    {"source": "user", "target": "quillscript", "type": "USES"},
    {"source": "user", "target": "nodejs", "type": "USES"},
    {"source": "user", "target": "brindle-works", "type": "WORKS_AT"},
    {"source": "mist-identity", "target": "c-plus-plus", "type": "IMPLEMENTED_WITH"},
    {"source": "user", "target": "regatta-2026-05-30", "type": "EXPERIENCED"},
    {"source": "regatta-2026-05-30", "target": "2026-05-30", "type": "OCCURRED_ON"},
    {"source": "launch-2026-01-02", "target": "2026-05-30", "type": "OCCURRED_ON"},
    {"source": "launch-2026-01-02", "target": "2026-06-01", "type": "OCCURRED_ON"},
    {"source": "harbour-opening-2026-05-30", "target": "2026-05-30", "type": "OCCURRED_ON"},
    {"source": "user", "target": "ghost", "type": "USES"},
    {
        "source": "mist-identity",
        "target": "kubernetes",
        "type": "USES",
        "properties": {"confidence": 0.7},
    },
    {"source": "user", "target": "ts", "type": "USES"},
]
