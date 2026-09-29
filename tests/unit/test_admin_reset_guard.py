"""The graph-reset guard (`admin.count_non_seed_entities`, `admin.reset_graph`).

KG-125 gave every seed node `provenance='seed'`, so a guard that tested only
node provenance passed a graph of seed nodes joined by extraction edges, or one
holding a clamped copy of a seed edge, and `reset_graph` wiped it without
`include_derived`. The guard now counts every `:__Entity__` node and every
relationship touching one that is not purely seed-applier-owned
(`admin.RESET_GUARD_CYPHER`).

A fake keyed on query text proves nothing about a Cypher predicate: it returns
whatever count the test hands it. `CypherSubsetGraph` below instead EVALUATES
the statements the reset path issues against an in-memory graph -- MATCH /
OPTIONAL MATCH over node and directed relationship patterns, WHERE with
Cypher's three-valued logic (NULL propagation through `=` / `<>`, `IS [NOT]
NULL`, `coalesce`, label tests), WITH / RETURN with `count` and grouping, and
`DETACH DELETE`. Anything outside that subset raises, so a query the fake
cannot read fails the test instead of being guessed at. What it cannot show is
Neo4j's own behaviour; `tests/integration/knowledge/test_reset_guard_eval.py`
runs the same cases against the disposable eval Neo4j.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from backend.errors import Neo4jQueryError
from backend.knowledge import admin
from tests.mocks.neo4j import FakeNeo4jConnection

# ---------------------------------------------------------------------------
# A Cypher-subset evaluator over an in-memory graph
# ---------------------------------------------------------------------------

_TOKEN = re.compile(
    r"\s*(?:(?P<str>'(?:[^'\\]|\\.)*')|(?P<num>\d+(?:\.\d+)?)|(?P<op><>|->|[()\[\]{},:.=+\-])"
    r"|(?P<ident>[A-Za-z_][A-Za-z_0-9]*))"
)
_KEYWORDS = frozenset(
    {"MATCH", "OPTIONAL", "WHERE", "WITH", "RETURN", "AS", "OR", "AND", "NOT", "IS", "NULL"}
    | {"DETACH", "DELETE", "COUNT", "COALESCE"}
)
_WRITE_KEYWORDS = frozenset({"CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DETACH", "CALL"})


class UnsupportedCypherError(AssertionError):
    """The statement uses syntax outside the subset this fake evaluates."""


def _tokenize(query: str) -> list[tuple[str, Any]]:
    tokens: list[tuple[str, Any]] = []
    pos = 0
    text = query.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None or m.end() == pos:
            raise UnsupportedCypherError(f"cannot tokenize at {text[pos:pos + 20]!r} in {query!r}")
        pos = m.end()
        if m.group("str") is not None:
            raw = m.group("str")[1:-1]
            tokens.append(("str", raw.replace("\\'", "'").replace("\\\\", "\\")))
        elif m.group("num") is not None:
            tokens.append(("num", float(m.group("num"))))
        elif m.group("op") is not None:
            tokens.append(("op", m.group("op")))
        else:
            word = m.group("ident")
            if word.upper() in _KEYWORDS:
                tokens.append(("kw", word.upper()))
            else:
                tokens.append(("ident", word))
    return tokens


@dataclass
class _Node:
    labels: frozenset[str]
    props: dict[str, Any]


@dataclass
class _Rel:
    type: str
    start: int
    end: int
    props: dict[str, Any]


@dataclass
class _NodePattern:
    var: str | None
    labels: list[str]


@dataclass
class _Pattern:
    start: _NodePattern
    rel_var: str | None = None
    rel_type: str | None = None
    end: _NodePattern | None = None


def _or3(values: list[bool | None]) -> bool | None:
    if any(v is True for v in values):
        return True
    if any(v is None for v in values):
        return None
    return False


def _and3(values: list[bool | None]) -> bool | None:
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


class _Parser:
    def __init__(self, tokens: list[tuple[str, Any]], query: str) -> None:
        self.tokens = tokens
        self.pos = 0
        self.query = query

    # -- token helpers --------------------------------------------------------
    def peek(self, offset: int = 0) -> tuple[str, Any] | None:
        i = self.pos + offset
        return self.tokens[i] if i < len(self.tokens) else None

    def at(self, kind: str, value: Any = None, offset: int = 0) -> bool:
        tok = self.peek(offset)
        return tok is not None and tok[0] == kind and (value is None or tok[1] == value)

    def take(self, kind: str, value: Any = None) -> Any:
        if not self.at(kind, value):
            raise UnsupportedCypherError(
                f"expected {kind} {value!r} at token {self.pos} ({self.peek()!r}) in {self.query!r}"
            )
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok[1]

    def done(self) -> bool:
        return self.pos >= len(self.tokens)

    def alias(self) -> str:
        """A projection alias; `count` is a keyword but also a legal alias."""
        if self.at("kw", "COUNT"):
            self.pos += 1
            return "count"
        return self.take("ident")

    # -- patterns ---------------------------------------------------------------
    def node_pattern(self) -> _NodePattern:
        self.take("op", "(")
        var = self.take("ident") if self.at("ident") else None
        labels: list[str] = []
        while self.at("op", ":"):
            self.take("op", ":")
            labels.append(self.take("ident"))
        self.take("op", ")")
        return _NodePattern(var, labels)

    def pattern(self) -> _Pattern:
        start = self.node_pattern()
        if not self.at("op", "-"):
            return _Pattern(start)
        self.take("op", "-")
        self.take("op", "[")
        rel_var = self.take("ident") if self.at("ident") else None
        rel_type = None
        if self.at("op", ":"):
            self.take("op", ":")
            rel_type = self.take("ident")
        self.take("op", "]")
        self.take("op", "->")
        return _Pattern(start, rel_var, rel_type, self.node_pattern())

    # -- expressions ------------------------------------------------------------
    def expr(self) -> Callable[[dict], Any]:
        parts = [self.and_expr()]
        while self.at("kw", "OR"):
            self.take("kw", "OR")
            parts.append(self.and_expr())
        if len(parts) == 1:
            return parts[0]
        return lambda env: _or3([p(env) for p in parts])

    def and_expr(self) -> Callable[[dict], Any]:
        parts = [self.not_expr()]
        while self.at("kw", "AND"):
            self.take("kw", "AND")
            parts.append(self.not_expr())
        if len(parts) == 1:
            return parts[0]
        return lambda env: _and3([p(env) for p in parts])

    def not_expr(self) -> Callable[[dict], Any]:
        if self.at("kw", "NOT"):
            self.take("kw", "NOT")
            inner = self.not_expr()

            def negate(env: dict) -> bool | None:
                value = inner(env)
                return None if value is None else not value

            return negate
        return self.comparison()

    def comparison(self) -> Callable[[dict], Any]:
        left = self.primary()
        if self.at("kw", "IS"):
            self.take("kw", "IS")
            negated = self.at("kw", "NOT")
            if negated:
                self.take("kw", "NOT")
            self.take("kw", "NULL")
            if negated:
                return lambda env: left(env) is not None
            return lambda env: left(env) is None
        if self.at("op", "=") or self.at("op", "<>"):
            op = self.take("op")
            right = self.primary()

            def compare(env: dict) -> bool | None:
                a, b = left(env), right(env)
                if a is None or b is None:
                    return None
                return (a == b) if op == "=" else (a != b)

            return compare
        return left

    def primary(self) -> Callable[[dict], Any]:
        if self.at("op", "("):
            self.take("op", "(")
            inner = self.expr()
            self.take("op", ")")
            return inner
        if self.at("str") or self.at("num"):
            value = self.tokens[self.pos][1]
            self.pos += 1
            return lambda env: value
        if self.at("kw", "COALESCE"):
            self.take("kw", "COALESCE")
            self.take("op", "(")
            args = [self.expr()]
            while self.at("op", ","):
                self.take("op", ",")
                args.append(self.expr())
            self.take("op", ")")

            def coalesce(env: dict) -> Any:
                for arg in args:
                    value = arg(env)
                    if value is not None:
                        return value
                return None

            return coalesce
        var = self.take("ident")
        if self.at("op", "."):
            self.take("op", ".")
            prop = self.take("ident")

            def prop_of(env: dict) -> Any:
                element = env[var]
                return None if element is None else element.props.get(prop)

            return prop_of
        if self.at("op", ":"):
            self.take("op", ":")
            label = self.take("ident")

            def has_label(env: dict) -> bool | None:
                element = env[var]
                if element is None:
                    return None
                if not isinstance(element, _Node):
                    raise UnsupportedCypherError(f"label test on a non-node in {self.query!r}")
                return label in element.labels

            return has_label
        return lambda env: env[var]

    # -- projections --------------------------------------------------------------
    def items(self) -> list[tuple[str, str, str]]:
        """[(kind, source, alias)] with kind 'count' or 'ref'."""
        items = []
        while True:
            if self.at("kw", "COUNT"):
                self.take("kw", "COUNT")
                self.take("op", "(")
                source = self.take("ident")
                self.take("op", ")")
                self.take("kw", "AS")
                items.append(("count", source, self.alias()))
            else:
                source = self.take("ident")
                alias = source
                if self.at("kw", "AS"):
                    self.take("kw", "AS")
                    alias = self.alias()
                items.append(("ref", source, alias))
            if not self.at("op", ","):
                return items
            self.take("op", ",")


def _project(rows: list[dict], items: list[tuple[str, str, str]]) -> list[dict]:
    if not any(kind == "count" for kind, _, _ in items):
        return [{alias: row[source] for _, source, alias in items} for row in rows]
    keys = [(source, alias) for kind, source, alias in items if kind == "ref"]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[s] for s, _ in keys), []).append(row)
    if not groups and not keys:
        groups[()] = []  # an aggregate with no grouping key yields one row
    out = []
    for key, members in groups.items():
        projected = {alias: key[i] for i, (_, alias) in enumerate(keys)}
        for kind, source, alias in items:
            if kind == "count":
                projected[alias] = sum(1 for m in members if m[source] is not None)
        out.append(projected)
    return out


@dataclass
class CypherSubsetGraph:
    """An in-memory graph that evaluates the reset path's Cypher (module docstring)."""

    nodes: dict[int, _Node] = field(default_factory=dict)
    rels: list[_Rel] = field(default_factory=list)
    queries: list[tuple[str, dict | None]] = field(default_factory=list)
    writes: list[tuple[str, dict | None]] = field(default_factory=list)

    def node(self, *labels: str, **props: Any) -> int:
        node_id = len(self.nodes) + 1
        self.nodes[node_id] = _Node(frozenset(labels), dict(props))
        return node_id

    def rel(self, start: int, rel_type: str, end: int, **props: Any) -> None:
        self.rels.append(_Rel(rel_type, start, end, dict(props)))

    # GraphConnection surface
    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def is_connected(self) -> bool:
        return True

    def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        self.queries.append((query, params))
        return self._run(query, allow_writes=False)

    def execute_write(self, query: str, params: dict | None = None) -> list[dict]:
        self.writes.append((query, params))
        return self._run(query, allow_writes=True)

    # evaluation
    def _node_matches(self, pattern: _NodePattern, node: _Node) -> bool:
        return all(label in node.labels for label in pattern.labels)

    def _expand(self, pattern: _Pattern, env: dict) -> list[dict]:
        found = []
        if pattern.end is None:
            for node in self.nodes.values():
                if self._node_matches(pattern.start, node):
                    found.append(
                        {**env, **({pattern.start.var: node} if pattern.start.var else {})}
                    )
            return found
        for rel in self.rels:
            start, end = self.nodes[rel.start], self.nodes[rel.end]
            if pattern.rel_type is not None and rel.type != pattern.rel_type:
                continue
            if not (
                self._node_matches(pattern.start, start) and self._node_matches(pattern.end, end)
            ):
                continue
            bound = dict(env)
            if pattern.start.var:
                bound[pattern.start.var] = start
            if pattern.rel_var:
                bound[pattern.rel_var] = rel
            if pattern.end.var:
                bound[pattern.end.var] = end
            found.append(bound)
        return found

    @staticmethod
    def _vars(pattern: _Pattern) -> list[str]:
        names = [pattern.start.var, pattern.rel_var, pattern.end.var if pattern.end else None]
        return [n for n in names if n]

    def _run(self, query: str, *, allow_writes: bool) -> list[dict]:
        tokens = _tokenize(query)
        words = {value.upper() for kind, value in tokens if kind in ("kw", "ident")}
        if not allow_writes and words & _WRITE_KEYWORDS:
            raise UnsupportedCypherError(f"write clause in a read query: {query!r}")
        parser = _Parser(tokens, query)
        rows: list[dict] = [{}]
        result: list[dict] = []
        while not parser.done():
            if parser.at("kw", "OPTIONAL") or parser.at("kw", "MATCH"):
                optional = parser.at("kw", "OPTIONAL")
                if optional:
                    parser.take("kw", "OPTIONAL")
                parser.take("kw", "MATCH")
                pattern = parser.pattern()
                where = None
                if parser.at("kw", "WHERE"):
                    parser.take("kw", "WHERE")
                    where = parser.expr()
                next_rows = []
                for env in rows:
                    matched = [
                        b for b in self._expand(pattern, env) if where is None or where(b) is True
                    ]
                    if not matched and optional:
                        matched = [{**env, **{v: None for v in self._vars(pattern)}}]
                    next_rows.extend(matched)
                rows = next_rows
            elif parser.at("kw", "WITH"):
                parser.take("kw", "WITH")
                rows = _project(rows, parser.items())
            elif parser.at("kw", "RETURN"):
                parser.take("kw", "RETURN")
                result = _project(rows, parser.items())
                if not parser.done():
                    raise UnsupportedCypherError(f"clauses after RETURN in {query!r}")
            elif allow_writes and parser.at("kw", "DETACH"):
                parser.take("kw", "DETACH")
                parser.take("kw", "DELETE")
                var = parser.take("ident")
                doomed = {id(env[var]) for env in rows if env[var] is not None}
                keep = {k: n for k, n in self.nodes.items() if id(n) not in doomed}
                self.rels = [r for r in self.rels if r.start in keep and r.end in keep]
                self.nodes = keep
            else:
                raise UnsupportedCypherError(
                    f"unsupported clause at {parser.peek()!r} in {query!r}"
                )
        return result

    # assertions
    def entity_deletes(self) -> list[str]:
        return [q for q, _ in self.writes if "DETACH DELETE" in q]


# ---------------------------------------------------------------------------
# The evaluator itself: it reads Cypher the way Neo4j does, NULLs included
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_a_bare_inequality_drops_a_missing_property_as_neo4j_does(self):
        g = CypherSubsetGraph()
        g.node("__Entity__")
        bare = g.execute_query(
            "MATCH (n:__Entity__) WHERE n.provenance <> 'seed' RETURN count(n) AS count"
        )
        coalesced = g.execute_query(
            "MATCH (n:__Entity__) WHERE coalesce(n.provenance, '') <> 'seed' "
            "RETURN count(n) AS count"
        )
        assert bare == [{"count": 0}]
        assert coalesced == [{"count": 1}]

    def test_an_aggregate_over_no_rows_is_one_row_of_zero(self):
        g = CypherSubsetGraph()
        assert g.execute_query("MATCH (n:__Entity__) RETURN count(n) AS count") == [{"count": 0}]

    def test_a_directed_pattern_matches_each_relationship_once(self):
        g = CypherSubsetGraph()
        a, b = g.node("__Entity__"), g.node("__Entity__")
        g.rel(a, "R", b)
        g.rel(a, "R", a)
        assert g.execute_query("MATCH (s)-[r]->(t) RETURN count(r) AS c") == [{"c": 2}]

    def test_syntax_outside_the_subset_raises(self):
        g = CypherSubsetGraph()
        with pytest.raises(UnsupportedCypherError):
            g.execute_query("MATCH (n) WHERE n.x IN [1, 2] RETURN count(n) AS c")
        with pytest.raises(UnsupportedCypherError):
            g.execute_query("MATCH (n) SET n.x = 1 RETURN count(n) AS c")


# ---------------------------------------------------------------------------
# Graph builders, one per acceptance case
# ---------------------------------------------------------------------------

SEED = {"provenance": "seed", "seed_version": "profile-v1"}
STAMPS = {"ontology_version": "1.2.0", "extraction_version": "v-x", "model_hash": "m-x"}


def _seed_pair(g: CypherSubsetGraph) -> tuple[int, int]:
    user = g.node("__Entity__", "User", id="user", **SEED)
    rust = g.node("__Entity__", "Technology", id="rust", **SEED)
    return user, rust


def pure_seed_graph(g: CypherSubsetGraph) -> None:
    """Nodes and edges exactly as the current applier writes them."""
    user, rust = _seed_pair(g)
    mist = g.node("__SelfModel__", "Trait", id="curious", **SEED)
    g.rel(user, "USES", rust, source_type="stated", confidence=1.0, **SEED)
    g.rel(mist, "HAS_TRAIT", user, source_type="stated", confidence=1.0, **SEED)


def seed_pair_with_extraction_edge(g: CypherSubsetGraph) -> None:
    user, rust = _seed_pair(g)
    g.rel(user, "USES", rust, provenance="extraction", **STAMPS)


def clamped_seed_copy(g: CypherSubsetGraph) -> None:
    user, rust = _seed_pair(g)
    g.rel(user, "USES", rust, is_latest_belief=False, **SEED)
    g.rel(
        user,
        "USES",
        rust,
        provenance="seed",
        seed_origin_version="profile-v1",
        source_type="stated",
        confidence=1.0,
        is_latest_belief=True,
        **STAMPS,
    )


def pre_kg125_seed_node(g: CypherSubsetGraph) -> None:
    g.node("__Entity__", "User", id="user", seed_version="profile-v1")


def adopted_relationship(g: CypherSubsetGraph) -> None:
    user, rust = _seed_pair(g)
    g.rel(user, "USES", rust, **SEED, **STAMPS)


def adopted_node(g: CypherSubsetGraph) -> None:
    g.node("__Entity__", "User", id="user", **SEED, **STAMPS)


def entity_to_non_entity_edge(g: CypherSubsetGraph) -> None:
    user, _ = _seed_pair(g)
    ctx = g.node("__Provenance__", "ConversationContext", id="ctx-s1")
    g.rel(user, "EXTRACTED_FROM", ctx)


def extraction_derived_node(g: CypherSubsetGraph) -> None:
    """What the pre-KG-125 guard already refused."""
    g.node("__Entity__", "Technology", id="python", provenance="extraction", **STAMPS)


# (builder, nodes counted, relationships counted)
GUARDED = [
    pytest.param(seed_pair_with_extraction_edge, 0, 1, id="seed-pair-extraction-edge"),
    pytest.param(clamped_seed_copy, 0, 1, id="clamped-seed-copy"),
    pytest.param(pre_kg125_seed_node, 1, 0, id="pre-kg125-seed-node"),
    pytest.param(adopted_relationship, 0, 1, id="adopted-relationship"),
    pytest.param(adopted_node, 1, 0, id="adopted-node"),
    pytest.param(entity_to_non_entity_edge, 0, 1, id="entity-to-non-entity-edge"),
    pytest.param(extraction_derived_node, 1, 0, id="extraction-derived-node"),
]


def _graph(builder: Callable[[CypherSubsetGraph], None]) -> CypherSubsetGraph:
    g = CypherSubsetGraph()
    builder(g)
    return g


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


class TestGuardCounts:
    @pytest.mark.parametrize("builder, nodes, relationships", GUARDED)
    def test_every_guarded_element_is_counted_once(self, builder, nodes, relationships):
        g = _graph(builder)
        assert admin.count_non_seed_entities(g) == nodes + relationships
        assert admin.count_reset_guard_elements(g) == (nodes, relationships)

    def test_a_pure_seed_graph_counts_nothing(self):
        assert admin.count_non_seed_entities(_graph(pure_seed_graph)) == 0

    def test_an_empty_graph_counts_nothing(self):
        assert admin.count_non_seed_entities(CypherSubsetGraph()) == 0

    def test_an_edge_between_two_non_entity_nodes_is_not_counted(self):
        """DETACH DELETE of `:__Entity__` nodes does not remove it, so it is not at stake."""
        g = CypherSubsetGraph()
        a = g.node("__SelfModel__", id="mist-identity")
        b = g.node("__Provenance__", "ConversationContext", id="ctx-s1")
        g.rel(a, "REFLECTED_IN", b)
        assert admin.count_non_seed_entities(g) == 0

    def test_counts_add_across_elements(self):
        g = _graph(clamped_seed_copy)
        entity_to_non_entity_edge(g)
        pre_kg125_seed_node(g)
        # Nodes: only the pre-KG-125 one (both seed pairs are pure seed).
        # Relationships: the clamped copy and the EXTRACTED_FROM edge; the
        # applier's own USES edge is not counted.
        assert admin.count_reset_guard_elements(g) == (1, 2)

    def test_the_guard_query_is_read_only(self):
        words = {value.upper() for kind, value in _tokenize(admin.RESET_GUARD_CYPHER)}
        assert not words & _WRITE_KEYWORDS

    def test_a_missing_row_fails_closed(self):
        connection = FakeNeo4jConnection(query_results=[])
        with pytest.raises(Neo4jQueryError):
            admin.count_non_seed_entities(connection)
        with pytest.raises(Neo4jQueryError):
            admin.reset_graph(connection, include_derived=False)
        connection.assert_no_writes()

    def test_a_row_without_the_columns_fails_closed(self):
        connection = FakeNeo4jConnection(query_results=[{"count": 0}])
        with pytest.raises(Neo4jQueryError):
            admin.reset_graph(connection, include_derived=False)
        connection.assert_no_writes()


class TestReset:
    @pytest.mark.parametrize("builder, nodes, relationships", GUARDED)
    def test_refuses_without_include_derived_and_deletes_nothing(
        self, builder, nodes, relationships
    ):
        g = _graph(builder)
        before_nodes, before_rels = dict(g.nodes), list(g.rels)
        with pytest.raises(Neo4jQueryError) as excinfo:
            admin.reset_graph(g, include_derived=False)
        assert g.writes == []
        assert g.entity_deletes() == []
        assert (g.nodes, g.rels) == (before_nodes, before_rels)
        message = str(excinfo.value)
        assert f"{nodes} non-seed nodes and {relationships} non-seed relationships" in message
        assert "include_derived=True" in message

    @pytest.mark.parametrize(
        "builder",
        [p.values[0] for p in GUARDED] + [pure_seed_graph],
        ids=[p.id for p in GUARDED] + ["pure-seed-graph"],
    )
    def test_include_derived_resets_every_case(self, builder):
        g = _graph(builder)
        result = admin.reset_graph(g, include_derived=True)
        assert not any("__Entity__" in n.labels for n in g.nodes.values())
        assert not any("__Provenance__" in n.labels for n in g.nodes.values())
        assert "MATCH (n:__Entity__) DETACH DELETE n" in g.entity_deletes()
        assert result["nodes_removed"] >= 1

    def test_a_pure_seed_graph_resets_without_include_derived(self):
        g = _graph(pure_seed_graph)
        result = admin.reset_graph(g, include_derived=False)
        assert g.entity_deletes() == ["MATCH (n:__Entity__) DETACH DELETE n"]
        # The self-model node survives; the edge joining it to `user` goes
        # with `user`.
        assert [n.props["id"] for n in g.nodes.values()] == ["curious"]
        assert g.rels == []
        assert result == {
            "nodes_removed": 2,
            "relationships_removed": 1,
            "provenance_nodes_removed": 0,
        }
