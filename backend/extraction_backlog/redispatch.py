r"""`redispatch-check`: prove that re-applying an applied turn leaves the graph unchanged.

FE-025 (Linear MIS-175, decision D6). For ONE turn the dispatcher has already
applied under the active epoch, re-run `ExtractionPipeline.apply_cached_turn`
from the turn's cached extraction, fingerprint the whole graph before and
after, and report whether anything changed. The turn is rebuilt exactly as
`ExtractionDispatcher._apply` builds it (`TurnToApply` from the logged row,
the cache row from `BacklogStore.get_cached` under the active epoch). The
progress object is `DiscardingProgress`: `curated` is False, so Stages 3-8 and
the Stage 9 operations all re-run, and `mark_curated` / `mark_applied` write
nothing. The check never writes the event store or the extraction cache.

OPERATOR COMMAND
----------------
Take a backup first (`CUTOVER.md` 5.1, `python -m scripts.backup.dump`): the
re-apply WRITES the live graph, and if it is not idempotent the graph is
changed. Stop the backend, then run the CLI in a one-off container, from Git
Bash on the host:

    docker compose stop mist-backend
    MSYS_NO_PATHCONV=1 docker compose run --rm mist-backend \
      python -m backend.extraction_backlog.admin redispatch-check --event-id <EVENT_ID>

The `docker compose exec` form cannot pass: it runs inside the running
backend, and the backend-up probe then finds it listening on
`localhost:8001` and refuses.

EXIT CODES
----------
- 0: the fingerprints are identical.
- 1: the check could not run: Neo4j unreachable, fingerprinting failed, the
  apply raised, or the apply reported curation stage errors while the
  fingerprints were identical (a partial apply proves nothing). After an
  apply that raised, the graph may be partially re-written: compare it with
  the backup.
- 2: refused, nothing applied and the graph not opened: no active epoch; the
  turn has no `applied` marker in `extraction_applied` for the active epoch
  (a `curated`-only marker is not applied); no cache row for the turn under
  the active epoch's `extraction_version` / `model_hash`; the cache row is a
  skip (its apply is a no-op, so the check would prove nothing); the turn is
  not in the conversation log; this environment's writer stamps differ from
  the active epoch (the dispatcher's own guard,
  `ExtractionDispatcher._writer_stamp_mismatch`, which would also refuse to
  apply); or the backend is not positively down.
- 3: the fingerprints differ. A bounded summary is printed: element counts
  before and after, the added / removed / changed counts, each differing
  property name with how many elements it changed on, and the first
  `DEFAULT_DIFF_LIMIT` differing elements by key.

THE BACKEND-UP PROBE
--------------------
`probe_backend` sends `GET /health` (plain `http.client`, no proxy) to every
endpoint in `eval_isolation.LIVE_WS_ENDPOINTS`, the repository's list of the
live backend's addresses: `mist-backend:8001` (the compose service, as seen
from a container on the compose network), `localhost:8001` and
`127.0.0.1:8001` (the published port, as seen from the host, or the backend
itself from inside its own container). The backend counts as down only when
EVERY endpoint gives a positive "not there": the connection was refused, or
the host name does not resolve (`EAI_NONAME`, or `EAI_NODATA` where the
platform defines it). Any HTTP response at all, a timeout, a temporary DNS
failure (`EAI_AGAIN`), an unreachable network or a protocol error is not
proof, and refuses. Probing all three is what keeps the one-off container
from passing on its own empty `localhost` while the backend is up.

THE FINGERPRINT
---------------
`fetch_graph_fingerprint` reads every node (labels, all properties) and every
relationship (type, endpoints, all properties) through `GraphExecutor`. Each
element becomes canonical JSON (sorted keys, list order kept) with only
`updated_at` removed (`EXCLUDED_PROPERTIES`), and is filed under a stable key:
a node by its `id` property; a relationship by its endpoints' ids, its type,
and its `version_key` / `valid_from` / `valid_to` where present. An element
with no `id` is keyed by its labels. Elements that share a key are kept as a
sorted multiset, so a key collision is compared, not lost. The digest is the
SHA-256 of the sorted key/body lines; the order the driver returns rows in
cannot change it. Neo4j's `elementId` is never used.

WHAT A RE-APPLY IS KNOWN TO CHANGE (found by reading, not excluded here)
------------------------------------------------------------------------
- `derived_at` on every EXTRACTED_FROM edge the turn's entities write:
  `CurationGraphWriter._extracted_from_clause` sets `r.derived_at = $now` on
  ON MATCH as well as ON CREATE, and `write()` takes `now` from
  `datetime.now(UTC)`. A wall-clock audit field (`canonical_serialize.
  AUDIT_FIELDS` excludes it), but this check does not, so any turn with at
  least one entity exits 3 on it until that is decided.
- `source_utterance_id` on those EXTRACTED_FROM edges is last-writer-wins
  (set on ON MATCH). Re-applying a turn that is not the latest turn of its
  session to extract an entity rewrites it back to this turn's event id, and
  the `_upsert_entity` replay guard, which keys on it, then no longer
  suppresses the confidence reinforce. Choose the most recently applied turn.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TextIO

from neo4j.exceptions import DriverError, Neo4jError

from backend.errors import MistError
from backend.knowledge.eval_isolation import LIVE_WS_ENDPOINTS, EvalIsolationError
from backend.knowledge.extraction.pipeline import ApplyReport, TurnToApply
from backend.knowledge.extraction_cache import OUTCOME_SKIPPED

from .store import STAGE_APPLIED, BacklogStore, Epoch

if TYPE_CHECKING:
    from backend.knowledge.config import KnowledgeConfig
    from backend.knowledge.curation.graph_writer import RebuildStamps
    from backend.knowledge.extraction.pipeline import ApplyProgress
    from backend.knowledge.storage.graph_executor import GraphExecutor

EXIT_IDENTICAL = 0
EXIT_CANNOT_RUN = 1
EXIT_REFUSED = 2
EXIT_DIFFERS = 3

# The only property the fingerprint ignores. Anything else a re-apply changes
# is a finding (see the module docstring), not something to filter out here.
EXCLUDED_PROPERTIES: frozenset[str] = frozenset({"updated_at"})

DEFAULT_DIFF_LIMIT = 20
BACKEND_PROBE_TIMEOUT_S = 3.0
HEALTH_PATH = "/health"

LIVE_GRAPH_WARNING = (
    "[WARNING] redispatch-check re-applies this turn to the LIVE graph. If the apply is "
    "not idempotent, the graph is changed and stays changed: a backup must exist "
    "(CUTOVER.md 5.1, python -m scripts.backup.dump) before this runs."
)

# Failures of the graph side (open, fingerprint, apply) that mean "could not
# run" (exit 1). The MIST error tree covers Neo4jConnectionError and
# Neo4jQueryError; the driver errors are the ones `Neo4jConnection` does not
# wrap; EvalIsolationError is a refused connect; OSError covers sockets and
# the embedding model's files.
_GRAPH_ERRORS = (MistError, EvalIsolationError, DriverError, Neo4jError, OSError)

NODES_CYPHER = "MATCH (n) RETURN labels(n) AS labels, properties(n) AS properties"
RELATIONSHIPS_CYPHER = (
    "MATCH (s)-[r]->(t) "
    "RETURN type(r) AS type, s.id AS start_id, labels(s) AS start_labels, "
    "t.id AS end_id, labels(t) AS end_labels, properties(r) AS properties"
)


# ---------------------------------------------------------------------------
# Backend-up probe
# ---------------------------------------------------------------------------

VERDICT_DOWN = "down"
VERDICT_UP = "up"
VERDICT_UNKNOWN = "unknown"

# getaddrinfo codes that mean "this name does not exist". EAI_NODATA is not
# defined on every platform.
_NAME_NOT_FOUND_CODES = frozenset(
    code
    for code in (getattr(socket, "EAI_NONAME", None), getattr(socket, "EAI_NODATA", None))
    if code is not None
)

HealthGet = Callable[[str, int, float], int]


@dataclass(frozen=True, slots=True)
class EndpointProbe:
    """One endpoint's answer to `GET /health`."""

    host: str
    port: int
    verdict: str  # VERDICT_DOWN | VERDICT_UP | VERDICT_UNKNOWN
    detail: str


@dataclass(frozen=True, slots=True)
class BackendProbe:
    """Every endpoint's answer. `down` only when each one positively says so."""

    endpoints: tuple[EndpointProbe, ...]

    @property
    def down(self) -> bool:
        """True when there is at least one endpoint and every one is `down`."""
        return bool(self.endpoints) and all(e.verdict == VERDICT_DOWN for e in self.endpoints)


def http_get_health(host: str, port: int, timeout: float) -> int:
    """`GET /health` on `host:port`; the HTTP status. Raises what the socket raises."""
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", HEALTH_PATH)
        return connection.getresponse().status
    finally:
        connection.close()


def classify_endpoint(host: str, port: int, get: HealthGet, timeout: float) -> EndpointProbe:
    """Probe one endpoint and classify the outcome, failing closed.

    `down` for a refused connection or a name that does not resolve; `up` for
    any HTTP response; `unknown` for everything else.
    """
    try:
        status = get(host, port, timeout)
    except ConnectionRefusedError as exc:
        return EndpointProbe(host, port, VERDICT_DOWN, f"connection refused ({exc})")
    except socket.gaierror as exc:
        if exc.errno in _NAME_NOT_FOUND_CODES:
            return EndpointProbe(host, port, VERDICT_DOWN, f"name does not resolve ({exc})")
        return EndpointProbe(host, port, VERDICT_UNKNOWN, f"name lookup failed ({exc})")
    except TimeoutError as exc:
        return EndpointProbe(host, port, VERDICT_UNKNOWN, f"timed out ({exc})")
    except http.client.HTTPException as exc:
        return EndpointProbe(
            host, port, VERDICT_UNKNOWN, f"protocol error ({type(exc).__name__}: {exc})"
        )
    except OSError as exc:
        return EndpointProbe(host, port, VERDICT_UNKNOWN, f"{type(exc).__name__}: {exc}")
    return EndpointProbe(host, port, VERDICT_UP, f"answered HTTP {status}")


def probe_backend(
    endpoints: Iterable[tuple[str, int]] = LIVE_WS_ENDPOINTS,
    *,
    get: HealthGet = http_get_health,
    timeout: float = BACKEND_PROBE_TIMEOUT_S,
) -> BackendProbe:
    """Probe every live backend endpoint (sorted, so the output order is stable)."""
    return BackendProbe(
        tuple(classify_endpoint(host, port, get, timeout) for host, port in sorted(endpoints))
    )


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """Encode the non-JSON values Neo4j returns (temporal, spatial, bytes)."""
    if isinstance(value, bytes | bytearray):
        return {"$type": "bytes", "$value": bytes(value).hex()}
    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        return {"$type": type(value).__name__, "$value": iso_format()}
    return {"$type": type(value).__name__, "$value": str(value)}


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, list order kept."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default
    )


def _without_excluded(properties: Mapping[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (properties or {}).items() if k not in EXCLUDED_PROPERTIES}


def _endpoint_key(node_id: Any, labels: Iterable[str] | None) -> str:
    if node_id is not None:
        return canonical_json(node_id)
    return "<no id>:" + ":".join(sorted(labels or ()))


def node_key(row: Mapping[str, Any]) -> str:
    """A node's stable key: its `id` property, else its labels."""
    properties = row.get("properties") or {}
    return "node " + _endpoint_key(properties.get("id"), row.get("labels"))


def relationship_key(row: Mapping[str, Any]) -> str:
    """Endpoints' ids, type, and version_key / valid_from / valid_to where present."""
    properties = row.get("properties") or {}
    key = (
        f"rel {_endpoint_key(row.get('start_id'), row.get('start_labels'))}"
        f"-[{row.get('type')}]->"
        f"{_endpoint_key(row.get('end_id'), row.get('end_labels'))}"
    )
    for name in ("version_key", "valid_from", "valid_to"):
        if properties.get(name) is not None:
            key += f" {name}={canonical_json(properties[name])}"
    return key


def _node_body(row: Mapping[str, Any]) -> str:
    return canonical_json(
        {
            "labels": sorted(row.get("labels") or ()),
            "properties": _without_excluded(row.get("properties")),
        }
    )


def _relationship_body(row: Mapping[str, Any]) -> str:
    return canonical_json(
        {
            "type": row.get("type"),
            "start": _endpoint_key(row.get("start_id"), row.get("start_labels")),
            "end": _endpoint_key(row.get("end_id"), row.get("end_labels")),
            "properties": _without_excluded(row.get("properties")),
        }
    )


@dataclass(frozen=True, slots=True)
class GraphFingerprint:
    """A whole-graph fingerprint: digest, counts, and each key's canonical bodies."""

    digest: str
    node_count: int
    relationship_count: int
    elements: Mapping[str, tuple[str, ...]] = field(repr=False)


def fingerprint_graph(
    nodes: Iterable[Mapping[str, Any]], relationships: Iterable[Mapping[str, Any]]
) -> GraphFingerprint:
    """Fingerprint node rows (`labels`, `properties`) and relationship rows.

    A relationship row carries `type`, `start_id`, `start_labels`, `end_id`,
    `end_labels` and `properties`, the columns `RELATIONSHIPS_CYPHER` returns.
    """
    grouped: dict[str, list[str]] = {}
    node_count = 0
    for row in nodes:
        grouped.setdefault(node_key(row), []).append(_node_body(row))
        node_count += 1
    relationship_count = 0
    for row in relationships:
        grouped.setdefault(relationship_key(row), []).append(_relationship_body(row))
        relationship_count += 1
    elements = {key: tuple(sorted(bodies)) for key, bodies in grouped.items()}
    digest = hashlib.sha256()
    for key in sorted(elements):
        for body in elements[key]:
            digest.update(f"{key}\t{body}\n".encode())
    return GraphFingerprint(
        digest=digest.hexdigest(),
        node_count=node_count,
        relationship_count=relationship_count,
        elements=elements,
    )


async def fetch_graph_fingerprint(executor: GraphExecutor) -> GraphFingerprint:
    """Read the whole graph through `GraphExecutor` and fingerprint it."""
    nodes = await executor.execute_query(NODES_CYPHER)
    relationships = await executor.execute_query(RELATIONSHIPS_CYPHER)
    return fingerprint_graph(nodes, relationships)


@dataclass(frozen=True, slots=True)
class ElementChange:
    """One key whose elements differ between two fingerprints."""

    key: str
    kind: str  # 'added' | 'removed' | 'changed'
    detail: str


@dataclass(frozen=True, slots=True)
class FingerprintDiff:
    """What differs between two fingerprints, sorted by key."""

    added: int
    removed: int
    changed: int
    changes: tuple[ElementChange, ...]
    property_counts: Mapping[str, int]


def _differing_fields(before: str, after: str) -> list[str]:
    old, new = json.loads(before), json.loads(after)
    names = [
        k for k in sorted(set(old) | set(new)) if k != "properties" and old.get(k) != new.get(k)
    ]
    old_props, new_props = old.get("properties") or {}, new.get("properties") or {}
    names += [
        k
        for k in sorted(set(old_props) | set(new_props))
        if (k in old_props) != (k in new_props) or old_props.get(k) != new_props.get(k)
    ]
    return names


def diff_fingerprints(before: GraphFingerprint, after: GraphFingerprint) -> FingerprintDiff:
    """Every key whose elements differ, and which properties changed on them."""
    changes: list[ElementChange] = []
    property_counts: Counter[str] = Counter()
    added = removed = changed = 0
    for key in sorted(set(before.elements) | set(after.elements)):
        old = before.elements.get(key)
        new = after.elements.get(key)
        if old == new:
            continue
        if old is None:
            added += 1
            changes.append(ElementChange(key, "added", f"{len(new or ())} element(s)"))
        elif new is None:
            removed += 1
            changes.append(ElementChange(key, "removed", f"{len(old)} element(s)"))
        else:
            changed += 1
            if len(old) == 1 and len(new) == 1:
                fields = _differing_fields(old[0], new[0])
                property_counts.update(fields)
                detail = ", ".join(fields)
            else:
                detail = f"{len(old)} element(s) before, {len(new)} after, under one key"
            changes.append(ElementChange(key, "changed", detail))
    return FingerprintDiff(
        added=added,
        removed=removed,
        changed=changed,
        changes=tuple(changes),
        property_counts=dict(property_counts),
    )


def format_diff(
    before: GraphFingerprint,
    after: GraphFingerprint,
    diff: FingerprintDiff,
    *,
    limit: int = DEFAULT_DIFF_LIMIT,
) -> list[str]:
    """The bounded diff summary printed on exit 3."""
    lines = [
        f"[redispatch] nodes {before.node_count} -> {after.node_count}, "
        f"relationships {before.relationship_count} -> {after.relationship_count}",
        f"[redispatch] differing keys: added={diff.added} removed={diff.removed} "
        f"changed={diff.changed}",
    ]
    if diff.property_counts:
        counted = ", ".join(
            f"{name} x{count}"
            for name, count in sorted(diff.property_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        lines.append(f"[redispatch] differing properties on changed elements: {counted}")
    for change in diff.changes[:limit]:
        lines.append(f"  {change.kind} {change.key}: {change.detail}")
    hidden = len(diff.changes) - limit
    if hidden > 0:
        lines.append(f"  ... and {hidden} more differing key(s)")
    return lines


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


class DiscardingProgress:
    """An `ApplyProgress` that says "not curated" and writes no marker.

    `curated` False makes `apply_cached_turn` re-run Stages 3-8 before the
    Stage 9 operations. The mark calls are counted, for the report and the
    tests, and otherwise dropped.
    """

    def __init__(self) -> None:
        self.curated_marks = 0
        self.applied_marks = 0

    @property
    def curated(self) -> bool:
        """Always False: the full apply re-runs."""
        return False

    def mark_curated(self) -> None:
        """Discarded."""
        self.curated_marks += 1

    def mark_applied(self) -> None:
        """Discarded."""
        self.applied_marks += 1


ApplyTurn = Callable[[TurnToApply, Mapping[str, Any], "ApplyProgress"], Awaitable[ApplyReport]]


@dataclass(frozen=True, slots=True)
class RedispatchGraph:
    """The graph side of the check, opened only after every refusal has passed.

    `apply_turn` is `ExtractionPipeline.apply_cached_turn` in production;
    `fingerprint` reads the graph `apply_turn` writes; `close` releases both.
    """

    apply_turn: ApplyTurn
    fingerprint: Callable[[], Awaitable[GraphFingerprint]]
    close: Callable[[], None]


def _writer_stamp_mismatch(writer_stamps: RebuildStamps, epoch: Epoch) -> str | None:
    """The dispatcher's own guard, called on a stand-in holding only the stamps."""
    from .dispatcher import ExtractionDispatcher

    return ExtractionDispatcher._writer_stamp_mismatch(
        SimpleNamespace(_writer_stamps=writer_stamps), epoch  # type: ignore[arg-type]
    )


def _refuse(out: TextIO, reason: str) -> int:
    print(f"[redispatch] REFUSED (exit 2, nothing applied): {reason}", file=out)
    return EXIT_REFUSED


async def run_redispatch_check(
    event_id: str,
    *,
    store: BacklogStore,
    writer_stamps: RebuildStamps,
    backend_probe: Callable[[], BackendProbe],
    open_graph: Callable[[], RedispatchGraph],
    out: TextIO,
    diff_limit: int = DEFAULT_DIFF_LIMIT,
) -> int:
    """Re-apply one applied turn and compare whole-graph fingerprints.

    Args:
        event_id: The applied turn's event id.
        store: The backlog over the event store and the extraction cache; only
            read.
        writer_stamps: The stamps a backend started from this environment
            writes (`factories.writer_stamps_from_config`).
        backend_probe: Called once, after the store checks pass.
        open_graph: Called once, only after every refusal check has passed.
        out: Where the report goes.
        diff_limit: How many differing keys the exit-3 summary lists.

    Returns:
        0 identical, 1 could not run, 2 refused, 3 differ (module docstring).
    """
    epoch = store.active_epoch()
    if epoch is None:
        return _refuse(out, "no epoch in the ledger")
    stage = store.event_store.get_extraction_applied(epoch.epoch_id).get(event_id)
    if stage != STAGE_APPLIED:
        found = "no marker" if stage is None else f"marker stage {stage!r}"
        return _refuse(
            out,
            f"event {event_id} has {found} in extraction_applied for {epoch.label}; "
            "only an applied turn can be re-dispatched",
        )
    cached = store.get_cached(event_id, epoch)
    if cached is None:
        return _refuse(
            out,
            f"no extraction cache row for event {event_id} under {epoch.label} "
            f"(extraction_version={epoch.extraction_version}, model_hash={epoch.model_hash})",
        )
    if cached.get("outcome") == OUTCOME_SKIPPED:
        return _refuse(
            out,
            f"the cache row for event {event_id} is a skip "
            f"(skip_reason={cached.get('skip_reason')}); its apply writes nothing to the "
            "graph, so the check would prove nothing. Choose an extracted turn",
        )
    turn_row = store.get_turn(event_id)
    if turn_row is None:
        return _refuse(out, f"event {event_id} is not in the conversation log")
    mismatch = _writer_stamp_mismatch(writer_stamps, epoch)
    if mismatch is not None:
        return _refuse(out, mismatch)

    probe = backend_probe()
    for endpoint in probe.endpoints:
        print(
            f"[redispatch] backend {endpoint.host}:{endpoint.port}: {endpoint.verdict} "
            f"({endpoint.detail})",
            file=out,
        )
    if not probe.down:
        return _refuse(
            out,
            "the backend is not positively down (every endpoint must refuse the connection "
            "or fail to resolve); stop it with `docker compose stop mist-backend` and run "
            "this with `docker compose run --rm`",
        )

    turn = TurnToApply(
        event_id=event_id,
        session_id=str(turn_row["session_id"]),
        user_utterance=str(turn_row["user_utterance"]),
        recorded_at=str(turn_row["timestamp"]),
    )
    print(LIVE_GRAPH_WARNING, file=out)
    print(
        f"[redispatch] event {event_id} (session {turn.session_id}, logged {turn.recorded_at}) "
        f"under {epoch.label}",
        file=out,
    )

    try:
        graph = open_graph()
    except _GRAPH_ERRORS as exc:
        print(
            f"[redispatch] could not open the graph (exit 1, nothing applied): "
            f"{type(exc).__name__}: {exc}",
            file=out,
        )
        return EXIT_CANNOT_RUN
    try:
        try:
            before = await graph.fingerprint()
        except _GRAPH_ERRORS as exc:
            print(
                f"[redispatch] could not fingerprint the graph (exit 1, nothing applied): "
                f"{type(exc).__name__}: {exc}",
                file=out,
            )
            return EXIT_CANNOT_RUN
        print(
            f"[redispatch] before: {before.node_count} node(s), "
            f"{before.relationship_count} relationship(s), sha256={before.digest}",
            file=out,
        )
        progress = DiscardingProgress()
        try:
            report = await graph.apply_turn(turn, cached, progress)
        except _GRAPH_ERRORS as exc:
            print(
                f"[redispatch] the apply raised (exit 1): {type(exc).__name__}: {exc}. "
                "The graph may be partially re-written; compare it with the backup.",
                file=out,
            )
            return EXIT_CANNOT_RUN
        try:
            after = await graph.fingerprint()
        except _GRAPH_ERRORS as exc:
            print(
                f"[redispatch] could not fingerprint the graph after the apply (exit 1): "
                f"{type(exc).__name__}: {exc}",
                file=out,
            )
            return EXIT_CANNOT_RUN
    finally:
        graph.close()

    print(
        f"[redispatch] after: {after.node_count} node(s), "
        f"{after.relationship_count} relationship(s), sha256={after.digest}",
        file=out,
    )
    print(
        f"[redispatch] apply: curation_resumed={report.curation_resumed} "
        f"derivation_operations_applied={report.derivation_operations_applied} "
        f"(markers discarded: curated x{progress.curated_marks}, "
        f"applied x{progress.applied_marks})",
        file=out,
    )
    for error in report.stage_errors:
        print(f"  curation stage error: {error}", file=out)

    if after.digest != before.digest:
        for line in format_diff(before, after, diff_fingerprints(before, after), limit=diff_limit):
            print(line, file=out)
        print("[redispatch] fingerprints DIFFER (exit 3): the re-apply changed the graph", file=out)
        return EXIT_DIFFERS
    if report.stage_errors:
        print(
            "[redispatch] fingerprints identical, but curation reported stage errors, so part "
            "of the apply did not run; this does not show idempotency (exit 1)",
            file=out,
        )
        return EXIT_CANNOT_RUN
    print("[redispatch] fingerprints IDENTICAL (exit 0)", file=out)
    return EXIT_IDENTICAL


# ---------------------------------------------------------------------------
# Production wiring
# ---------------------------------------------------------------------------


def open_live_graph_from_env(config: KnowledgeConfig) -> RedispatchGraph:
    """Connect to the graph `config` names and build the backend's extraction pipeline.

    The pipeline comes from `factories.build_extraction_pipeline`, the builder
    the backend's conversation handler uses, over one Neo4j connection that
    the fingerprint reads through as well. Building it runs
    `GraphStore.ensure_mist_identity` (an ON CREATE-only MERGE), before the
    first fingerprint. Needs Neo4j and the embedding model; not exercised by
    the unit tier.

    Raises:
        Neo4jConnectionError: Neo4j is unreachable (and the other
            `_GRAPH_ERRORS` a connect or build can raise).
    """
    from backend.factories import (
        build_extraction_pipeline,
        build_graph_executor,
        build_graph_store,
        build_neo4j_connection,
    )

    connection = build_neo4j_connection(config)
    built = False
    try:
        graph_store = build_graph_store(config, connection=connection)
        pipeline = build_extraction_pipeline(config, graph_store=graph_store)
        executor = build_graph_executor(config, connection)
        built = True
    finally:
        if not built:
            connection.disconnect()

    def close() -> None:
        cache = pipeline.extraction_cache
        if cache is not None:
            cache.close()
        connection.disconnect()

    async def fingerprint() -> GraphFingerprint:
        return await fetch_graph_fingerprint(executor)

    return RedispatchGraph(
        apply_turn=pipeline.apply_cached_turn, fingerprint=fingerprint, close=close
    )
