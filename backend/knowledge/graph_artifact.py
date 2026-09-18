"""Lossless codec for the MIST graph backup artifact.

WHY THIS EXISTS
    `cmd_graph_backup` serialised the full graph with
    `json.dumps(payload, indent=2, default=str)`
    (`git log -S "indent=2, default=str" -- scripts/mist_admin.py`). `default=str`
    is reached for every value `json` cannot encode, so each Neo4j temporal,
    spatial Point and byte string in the graph was written as its `str()` form.
    Nothing in the file records that it happened. A loader written against such a
    file restores `created_at` as `str` where the captured value was a `DateTime`
    -- the property's TYPE changes and no gate in this repo compares types.
    `canonical_serialize` excludes the stamp triple and `embedding`
    (`grep -n "EXCLUDED" backend/knowledge/canonical_serialize.py`), so a
    type-degraded restore is invisible to the determinism gates too.

    This module is therefore the only writer and the only reader of that
    artifact: every value is either tagged so it can be reconstructed exactly, or
    refused by name. There is no third branch, and in particular no coercion.

WHAT IS TAGGED, AND WHY str() IS ENOUGH FOR TEMPORALS
    All four `neo4j.time` temporals round-trip exactly through
    `str()` -> `<cls>.from_iso_format()`, nanosecond precision included, on the
    pinned driver (neo4j 5.24.0, `grep -n "^neo4j==" requirements.txt`).
    Measured in the backend container, Python 3.11:
        DateTime(2026, 9, 17, 12, 0, 0, 123456789) -> "2026-09-17T12:00:00.123456789"
            -> DateTime.from_iso_format(...) == original -> True
        Duration(months=1, days=2, seconds=3, nanoseconds=4) -> "P1M2DT3.000000004S"
            -> Duration.from_iso_format(...) == original -> True
        Time(12, 30, 45, 123456789) and Date(2026, 9, 17) likewise -> True

WHY A POINT IS RECONSTRUCTED FROM ITS SRID AND NOT AS A BARE `Point`
    `neo4j.spatial.Point` equality is type-sensitive: a bare `Point((1.5, 2.5))`
    with `.srid` assigned compares `False` against `CartesianPoint((1.5, 2.5))`,
    measured on 5.24.0. Decoding therefore constructs the concrete class the srid
    names; `CartesianPoint((1.5, 2.5)) == CartesianPoint((1.5, 2.5))` and the
    srid survives, measured `True` for all four srids in `_POINT_CLASSES`.

WHY THE ENVELOPE CARRIES A VERSION
    A backup format with no version field can be read exactly once -- by the
    build that wrote it. `load_artifact` refuses anything it does not recognise
    rather than guessing, and names what it is holding, because the artifact an
    operator reaches for in a recovery is the one written by an older build.

RELATIONSHIP TO `scripts/hydration/snapshot.py`
    That module carries its own older codec (`_encode_value` / `_decode_value`,
    `grep -n "_encode_value" scripts/hydration/snapshot.py`). It tags the four
    temporals and refuses everything else, spatial Points included, and it is
    NOT imported here: this codec is a superset, converging them would change the
    hydration artifact format, and the two are versioned separately.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from neo4j.spatial import CartesianPoint, Point, WGS84Point
from neo4j.time import Date, DateTime, Duration, Time

logger = logging.getLogger(__name__)

GRAPH_ARTIFACT_FORMAT = "mist.graph-artifact"
GRAPH_ARTIFACT_VERSION = 1

# Every version this build can READ. A frozenset rather than a max: a future
# build that drops support for a version must say so by removing it here, so the
# refusal names the gap instead of silently attempting a best-effort load.
SUPPORTED_ARTIFACT_VERSIONS = frozenset({1})

# Marks a reconstructable value. Cannot collide with a genuine property: Neo4j
# has no map property type, so a dict in a property position is always ours.
NEO4J_TYPE_TAG = "__neo4j_type__"

_TEMPORAL_TYPES: tuple[type, ...] = (Date, Time, DateTime, Duration)
_TEMPORAL_DECODERS = {cls.__name__: cls.from_iso_format for cls in _TEMPORAL_TYPES}

# srid -> (concrete class, coordinate count). The pairs are the four srids the
# Neo4j 5 spatial type system defines; an artifact naming any other srid is
# refused rather than decoded into a class that would not compare equal.
_POINT_CLASSES: dict[int, tuple[type[Point], int]] = {
    7203: (CartesianPoint, 2),
    9157: (CartesianPoint, 3),
    4326: (WGS84Point, 2),
    4979: (WGS84Point, 3),
}

_POINT_TAG = "Point"
_BYTES_TAG = "Bytes"


class GraphArtifactError(RuntimeError):
    """Raised when a graph artifact cannot be written or read without data loss.

    Local to this module, matching the `HydrationError` / `IsolatedRootError`
    precedent: the failure is about an artifact's contents, not about a Neo4j
    query or connection, so it does not belong in the `MistError` I/O hierarchy.
    """


def _zone_name(tzinfo: Any) -> str | None:
    """Return the IANA name a tzinfo exposes, or None when it has no name.

    Two spellings are checked because both reach a property value: `pytz`
    timezones carry `.zone` and `zoneinfo.ZoneInfo` carries `.key`. A fixed
    offset (`pytz.FixedOffset(60)`, `datetime.timezone`) has neither, which is
    the case that returns None.
    """
    for attribute in ("zone", "key"):
        name = getattr(tzinfo, attribute, None)
        if isinstance(name, str) and name:
            return name
    return None


def _resolve_zone(name: str) -> Any | None:
    """Resolve an IANA zone name to a tzinfo, or None when this host cannot.

    Both spellings are tried because neither is guaranteed: the backend image
    has `pytz` 2026.1.post1 but no `tzdata`, so `ZoneInfo("Europe/London")`
    raises `ZoneInfoNotFoundError` there while `pytz.timezone("Europe/London")`
    succeeds (measured in the backend container).

    Returning None rather than raising is deliberate, and it is not a
    best-effort load: the zone NAME is extra fidelity, not part of the value's
    identity. A zoned DateTime stringifies to a fixed offset
    (`...T12:00:00.000000000+01:00`), and the value decoded from that offset
    compares EQUAL to the original zoned value -- measured `True` on 5.24.0. So
    a host whose tz database lacks the name still restores an equal value, and
    refusing the whole artifact over a cosmetic tzinfo difference would turn a
    tzdata mismatch into a failed disaster recovery.
    """
    # Both lookups raise a KeyError subclass for an unknown name --
    # `zoneinfo.ZoneInfoNotFoundError` and `pytz.exceptions.UnknownTimeZoneError`
    # -- and ImportError when the library itself is absent. Neither is bare
    # `Exception`: a different failure here is a bug and must not be swallowed.
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except (ImportError, KeyError, ValueError):
        pass
    try:
        import pytz

        return pytz.timezone(name)
    except (ImportError, KeyError, ValueError):
        return None


def encode_value(value: Any, *, key: str, where: str) -> Any:
    """Encode one property value into JSON, losslessly or not at all.

    Args:
        value: A property value as the Neo4j driver returned it.
        key: The property name, named in the refusal so an operator can find it.
        where: The node or relationship the property sits on, same purpose.

    Returns:
        A JSON-encodable value: a scalar verbatim, a list of encoded values, or
        a tagged object carrying everything `decode_value` needs.

    Raises:
        GraphArtifactError: When the value has no verified round trip. Refusing
            is the point -- a coerced value restores a property whose type
            differs from the captured one, and nothing downstream detects that.
    """
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, _TEMPORAL_TYPES):
        encoded: dict[str, Any] = {NEO4J_TYPE_TAG: type(value).__name__, "iso": str(value)}
        if isinstance(value, DateTime):
            zone = _zone_name(getattr(value, "tzinfo", None))
            if zone is not None:
                encoded["zone"] = zone
        return encoded
    # Before the list arm: `Point` subclasses tuple, and a tuple is not a list,
    # so the ordering is documentation rather than load-bearing -- but a Point
    # reaching the refusal below would be a silent capability regression.
    if isinstance(value, Point):
        srid = getattr(value, "srid", None)
        if srid is None:
            raise GraphArtifactError(
                f"{where}: property {key!r} is a Point with no srid, so the class "
                "needed to reconstruct it cannot be determined. A Point read from "
                "Neo4j always carries one; this value did not come from the driver."
            )
        return {
            NEO4J_TYPE_TAG: _POINT_TAG,
            "srid": int(srid),
            "coordinates": [float(coordinate) for coordinate in value],
        }
    if isinstance(value, bytes | bytearray):
        return {NEO4J_TYPE_TAG: _BYTES_TAG, "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, list):
        return [encode_value(item, key=key, where=where) for item in value]
    raise GraphArtifactError(
        f"{where}: property {key!r} has type {type(value).__name__}, which this "
        f"artifact format (version {GRAPH_ARTIFACT_VERSION}) cannot round-trip. "
        "Coercing it would restore a graph that differs from the one captured. "
        "Extend the format and its version rather than losing the value."
    )


def decode_value(value: Any) -> Any:
    """Invert `encode_value` so a restored property keeps its captured type.

    Args:
        value: A value read back out of an artifact.

    Returns:
        The driver-native value: a temporal, a Point of the srid's concrete
        class, `bytes`, a list of decoded values, or the scalar unchanged.

    Raises:
        GraphArtifactError: When a tagged object names a type or srid this build
            cannot reconstruct, or when a bare map appears in a property
            position. Neo4j has no map property type, so an untagged dict here
            means the artifact was written by something other than this codec.
    """
    if isinstance(value, dict):
        if NEO4J_TYPE_TAG not in value:
            raise GraphArtifactError(
                f"artifact carries an untagged map {sorted(value)} in a property "
                f"position. Neo4j has no map property type, so every dict here must "
                f"carry {NEO4J_TYPE_TAG!r}; this one does not and cannot be restored."
            )
        return _decode_tagged(value)
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    return value


def _decode_tagged(value: Mapping[str, Any]) -> Any:
    """Reconstruct one tagged value. Split out to keep `decode_value` flat."""
    tag = value[NEO4J_TYPE_TAG]
    if tag in _TEMPORAL_DECODERS:
        return _decode_temporal(tag, value)
    if tag == _POINT_TAG:
        return _decode_point(value)
    if tag == _BYTES_TAG:
        return _decode_bytes(value)
    raise GraphArtifactError(
        f"artifact carries an unknown tagged value type {tag!r}. This build knows "
        f"{sorted([*_TEMPORAL_DECODERS, _POINT_TAG, _BYTES_TAG])}. Restoring it would "
        "change the property's type, so it is refused."
    )


def _decode_temporal(tag: str, value: Mapping[str, Any]) -> Any:
    """Rebuild a temporal from its ISO form, re-attaching a named zone if present."""
    if "iso" not in value:
        raise GraphArtifactError(f"tagged {tag!r} value has no 'iso' field: {sorted(value)}")
    decoded = _TEMPORAL_DECODERS[tag](value["iso"])
    zone = value.get("zone")
    if tag != DateTime.__name__ or not zone or decoded.tzinfo is None:
        return decoded
    resolved = _resolve_zone(zone)
    if resolved is None:
        # Not an error: see `_resolve_zone`. The offset-carrying value already
        # compares equal to the captured one, so the loss is the zone's NAME.
        logger.warning(
            "artifact DateTime names zone %r, which this host's tz database does "
            "not resolve; restoring the fixed offset from the ISO form instead",
            zone,
        )
        return decoded
    return decoded.astimezone(resolved)


def _decode_point(value: Mapping[str, Any]) -> Point:
    """Rebuild a Point as the concrete class its srid names. See the module docstring."""
    if "srid" not in value or "coordinates" not in value:
        raise GraphArtifactError(
            f"tagged Point value needs 'srid' and 'coordinates'; got {sorted(value)}"
        )
    srid = value["srid"]
    entry = _POINT_CLASSES.get(srid) if isinstance(srid, int) else None
    if entry is None:
        raise GraphArtifactError(
            f"artifact carries a Point with srid {srid!r}, which this build cannot "
            f"map to a concrete spatial class. Known srids: {sorted(_POINT_CLASSES)}."
        )
    point_class, dimension = entry
    coordinates = value["coordinates"]
    if not isinstance(coordinates, list | tuple) or len(coordinates) != dimension:
        raise GraphArtifactError(
            f"artifact carries a Point with srid {srid} whose 'coordinates' is "
            f"{coordinates!r}; that srid is {dimension}-dimensional and needs a list "
            "of that many numbers."
        )
    return point_class(tuple(float(coordinate) for coordinate in coordinates))


def _decode_bytes(value: Mapping[str, Any]) -> bytes:
    """Rebuild a byte string.

    Always `bytes`, never `bytearray`: the driver returns `bytes` for a Neo4j
    byte-array property, so `bytes` is the faithful restore type. A `bytearray`
    is accepted on the capture side because a Python caller may hold one, and it
    is written under the same tag.
    """
    if "base64" not in value:
        raise GraphArtifactError(f"tagged Bytes value has no 'base64' field: {sorted(value)}")
    try:
        return base64.b64decode(value["base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise GraphArtifactError(f"tagged Bytes value is not valid base64: {exc}") from exc


def encode_properties(props: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """Encode a whole property map.

    Runs BEFORE any `json.dumps`. Ordering matters: `json` raises a bare
    `TypeError` naming only the type, while this raises naming the property, the
    node or relationship it sits on, and what to do about it.
    """
    return {key: encode_value(value, key=key, where=where) for key, value in props.items()}


def decode_properties(props: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a whole property map read back out of an artifact."""
    return {key: decode_value(value) for key, value in props.items()}


def build_artifact(
    *,
    nodes: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Sequence[str]],
    counts: Mapping[str, int],
    stamps: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the versioned envelope around a captured graph.

    Takes DRIVER-NATIVE property values and encodes them here, so an artifact
    with raw temporals in it cannot be constructed by mistake. Passing already
    encoded properties raises instead of double-encoding: `encode_value` refuses
    a dict, which is what an encoded value is.

    Args:
        nodes: Rows of `{"id", "labels", "properties"}` as read from the graph.
        relationships: Rows of `{"source", "type", "target", "properties"}`.
        schema: `{"constraints": [...], "indexes": [...]}` of createStatements.
        counts: `{"nodes": int, "relationships": int}`, recorded so a truncated
            artifact is detectable without a graph to compare against.
        stamps: The version-stamp triple. RECORDED, never enforced on read --
            see `load_artifact`.
        source: `{"uri": ..., "database": ...}`, so a restore operator can see
            which instance the artifact came from.

    Returns:
        The artifact dict, ready for `dumps_artifact`.

    Raises:
        GraphArtifactError: When any property value has no verified round trip.
    """
    encoded_nodes = []
    for node in nodes:
        labels = sorted(node.get("labels") or [])
        where = f"node {node.get('id')!r} {labels}"
        encoded_nodes.append(
            {
                "id": node.get("id"),
                "labels": labels,
                "properties": encode_properties(node.get("properties") or {}, where=where),
            }
        )

    encoded_relationships = []
    for relationship in relationships:
        where = (
            f"relationship {relationship.get('type')!r} "
            f"{relationship.get('source')!r} -> {relationship.get('target')!r}"
        )
        encoded_relationships.append(
            {
                "source": relationship.get("source"),
                "type": relationship.get("type"),
                "target": relationship.get("target"),
                "properties": encode_properties(relationship.get("properties") or {}, where=where),
            }
        )

    return {
        "format": GRAPH_ARTIFACT_FORMAT,
        "format_version": GRAPH_ARTIFACT_VERSION,
        "captured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": {"uri": source.get("uri"), "database": source.get("database")},
        "stamps": dict(stamps),
        "counts": {
            "nodes": int(counts.get("nodes", len(encoded_nodes))),
            "relationships": int(counts.get("relationships", len(encoded_relationships))),
        },
        "schema": {
            "constraints": list(schema.get("constraints", [])),
            "indexes": list(schema.get("indexes", [])),
        },
        "nodes": encoded_nodes,
        "relationships": encoded_relationships,
    }


def load_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the envelope and decode every property back to driver types.

    FAIL-CLOSED, with no auto-upgrade and no partial load. The version check
    covers the FORMAT and nothing else: `stamps` is recorded for audit and
    deliberately not enforced, because a disaster-recovery artifact that
    self-invalidates on an `EXTRACTION_VERSION` bump is worse than no artifact.
    That is the opposite of `SnapshotManifest.assert_fresh`
    (`grep -n "def assert_fresh" scripts/hydration/manifest.py`), which is
    correct for a test fixture that must match the code under test and wrong
    here.

    Args:
        payload: The parsed JSON of an artifact file.

    Returns:
        The artifact with `nodes` and `relationships` properties decoded.

    Raises:
        GraphArtifactError: When the envelope is missing, names another format,
            names a version this build cannot read, or carries a value this
            build cannot reconstruct.
    """
    if "format" not in payload or "format_version" not in payload:
        raise GraphArtifactError(
            "this file has no format/format_version envelope, so it predates the "
            f"versioned {GRAPH_ARTIFACT_FORMAT} artifact -- it is most likely the "
            "output of an older `mist_admin.py graph-backup`, which wrote the graph "
            "through `json.dumps(..., default=str)`. Such a file cannot be restored "
            "safely: every temporal, Point and byte string in it is already a "
            "string and its original type is unrecoverable."
        )
    if payload["format"] != GRAPH_ARTIFACT_FORMAT:
        raise GraphArtifactError(
            f"artifact declares format {payload['format']!r}; this loader reads only "
            f"{GRAPH_ARTIFACT_FORMAT!r}."
        )
    version = payload["format_version"]
    # `isinstance(True, int)` is True and `True in frozenset({1})` is True, so a
    # bool version would pass the membership test unchecked.
    if isinstance(version, bool) or not isinstance(version, int):
        raise GraphArtifactError(
            f"artifact declares format_version {version!r}, which is not an integer. "
            f"This build reads versions {sorted(SUPPORTED_ARTIFACT_VERSIONS)}."
        )
    if version not in SUPPORTED_ARTIFACT_VERSIONS:
        raise GraphArtifactError(
            f"artifact is format_version {version}; this build reads "
            f"{sorted(SUPPORTED_ARTIFACT_VERSIONS)}. It is not upgraded in place: "
            "guessing at an unknown version's semantics is how a restore silently "
            "loses a field. Use a build that lists this version."
        )

    nodes = _require_sequence(payload, "nodes")
    relationships = _require_sequence(payload, "relationships")
    schema = payload.get("schema") or {}

    return {
        "format": payload["format"],
        "format_version": version,
        "captured_at": payload.get("captured_at"),
        "source": dict(payload.get("source") or {}),
        "stamps": dict(payload.get("stamps") or {}),
        "counts": dict(payload.get("counts") or {}),
        "schema": {
            "constraints": list(schema.get("constraints", [])),
            "indexes": list(schema.get("indexes", [])),
        },
        "nodes": [
            {
                "id": node.get("id"),
                "labels": list(node.get("labels") or []),
                "properties": decode_properties(node.get("properties") or {}),
            }
            for node in nodes
        ],
        "relationships": [
            {
                "source": relationship.get("source"),
                "type": relationship.get("type"),
                "target": relationship.get("target"),
                "properties": decode_properties(relationship.get("properties") or {}),
            }
            for relationship in relationships
        ],
    }


def _require_sequence(payload: Mapping[str, Any], key: str) -> Sequence[Mapping[str, Any]]:
    """Refuse an envelope whose node or relationship list is missing or not a list."""
    value = payload.get(key)
    if not isinstance(value, list):
        raise GraphArtifactError(
            f"artifact has no {key!r} list (found {type(value).__name__}). An artifact "
            "missing a whole leg is truncated, not empty."
        )
    return value


def dumps_artifact(artifact: Mapping[str, Any]) -> str:
    """Serialise an artifact to JSON text, refusing non-finite floats.

    `allow_nan=False` is the whole point of this function existing. Python's
    `json` writes `NaN`, `Infinity` and `-Infinity` by default, which no strict
    JSON parser accepts -- so a single non-finite float in one 384-float
    embedding would produce a backup file that cannot be read back at all. This
    turns that into a loud failure at CAPTURE time, when the graph is still
    there to re-read.

    Raises:
        GraphArtifactError: When the artifact holds a non-finite float or any
            value `json` cannot encode.
    """
    try:
        return json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        raise GraphArtifactError(
            f"artifact cannot be serialised as strict JSON: {exc}. A non-finite float "
            "(NaN or Infinity) in a property would be written as a token no strict "
            "JSON parser reads back, so the capture fails here instead."
        ) from exc
