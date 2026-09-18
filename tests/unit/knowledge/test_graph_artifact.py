"""The graph backup codec: what survives a round trip, and what is refused.

Subject: `backend/knowledge/graph_artifact.py`.

WHY THIS EXISTS. `mist_admin graph-backup` wrote the full graph through
`json.dumps(payload, indent=2, default=str)`. `default=str` is reached for every
value `json` cannot encode, so a `DateTime` property became a string, a Point
became a string, and the file recorded nothing about it having happened. There
was also no loader in the repo (`grep -rn "graph_snapshots" --include=*.py`
finds only writers), so the backup was write-only and its fidelity was never
tested by anything.

Equality here is DRIVER equality on the reconstructed object, not a string
comparison of its ISO form, because that is the property the restore depends on:
a value that prints the same but compares unequal restores a different graph.
The embedding case asserts EXACT list equality for the same reason -- a
float-tolerant comparison would pass on a codec that silently rounded.
"""

from __future__ import annotations

import json
import math
from datetime import datetime as stdlib_datetime

import pytest
import pytz
from neo4j.spatial import CartesianPoint, WGS84Point
from neo4j.time import Date, DateTime, Duration, Time

from backend.knowledge.graph_artifact import (
    GRAPH_ARTIFACT_FORMAT,
    GRAPH_ARTIFACT_VERSION,
    NEO4J_TYPE_TAG,
    SUPPORTED_ARTIFACT_VERSIONS,
    GraphArtifactError,
    build_artifact,
    decode_properties,
    decode_value,
    dumps_artifact,
    encode_properties,
    encode_value,
    load_artifact,
)

_EMBEDDING = [round(i * 0.0013, 12) for i in range(384)]

_ZONED = pytz.timezone("Europe/London").localize(DateTime(2026, 9, 17, 12, 0, 0))


def _round_trip(value):
    """Encode one value and decode it back, through JSON text.

    Through the TEXT, not just through the dicts: a codec can produce a
    structure that is correct in memory and unwritable as strict JSON, and this
    module exists because of what serialisation did to these values.
    """
    encoded = encode_properties({"probe": value}, where="node ['__Entity__']")
    reparsed = json.loads(dumps_artifact({"properties": encoded}))
    return decode_properties(reparsed["properties"])["probe"]


class TestExactRoundTrip:
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(Date(2026, 9, 17), id="date"),
            pytest.param(Time(12, 30, 45, 123456789), id="time-nanoseconds"),
            pytest.param(DateTime(2026, 9, 17, 12, 0, 0, 123456789), id="datetime-nanoseconds"),
            pytest.param(_ZONED, id="datetime-zoned"),
            pytest.param(Duration(months=1, days=2, seconds=3, nanoseconds=4), id="duration"),
        ],
    )
    def test_temporals_survive_exactly(self, value):
        decoded = _round_trip(value)

        assert decoded == value
        assert type(decoded) is type(value)

    def test_zoned_datetime_records_the_zone_name(self):
        """The offset carries the value; the name is extra fidelity on top of it.

        `str(DateTime)` emits `+01:00`, never `Europe/London`, so without this
        field the restored value would be equal but would have lost which zone
        it was expressed in.
        """
        encoded = encode_value(_ZONED, key="created_at", where="node ['MistIdentity']")

        assert encoded[NEO4J_TYPE_TAG] == "DateTime"
        assert encoded["zone"] == "Europe/London"

    def test_zoned_datetime_restores_the_named_zone_when_the_host_resolves_it(self):
        """Equality is the acceptance; the name is restored where tz data allows.

        The backend image has `pytz` but no `tzdata`, so `ZoneInfo` cannot
        resolve any name there and `pytz` can -- `_resolve_zone` tries both. A
        host that resolves neither still restores an EQUAL value from the fixed
        offset, which is why this asserts equality unconditionally and the name
        only when it came back.
        """
        decoded = _round_trip(_ZONED)

        assert decoded == _ZONED
        zone = getattr(decoded.tzinfo, "zone", None) or getattr(decoded.tzinfo, "key", None)
        assert zone in {"Europe/London", None}

    def test_naive_datetime_has_no_zone_field(self):
        encoded = encode_value(
            DateTime(2026, 9, 17, 12, 0, 0), key="created_at", where="node ['X']"
        )

        assert "zone" not in encoded

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(CartesianPoint((1.5, 2.5)), id="cartesian-2d-7203"),
            pytest.param(CartesianPoint((1.5, 2.5, 3.5)), id="cartesian-3d-9157"),
            pytest.param(WGS84Point((1.5, 2.5)), id="wgs84-2d-4326"),
            pytest.param(WGS84Point((1.5, 2.5, 3.5)), id="wgs84-3d-4979"),
        ],
    )
    def test_points_survive_exactly_including_their_srid(self, value):
        """Point equality is TYPE-sensitive, so the srid decides the class.

        Measured on neo4j 5.24.0: a bare `Point((1.5, 2.5))` with `.srid`
        assigned compares `False` against `CartesianPoint((1.5, 2.5))`. Decoding
        into a bare Point would therefore produce a value that prints right and
        compares wrong.
        """
        decoded = _round_trip(value)

        assert decoded == value
        assert type(decoded) is type(value)
        assert decoded.srid == value.srid

    def test_bytes_survive_exactly(self):
        value = bytes(range(256))

        decoded = _round_trip(value)

        assert decoded == value
        assert isinstance(decoded, bytes)

    def test_bytearray_is_captured_and_restores_as_bytes(self):
        """The driver returns `bytes` for a Neo4j byte array, so `bytes` is faithful."""
        decoded = _round_trip(bytearray(b"mist"))

        assert decoded == b"mist"
        assert isinstance(decoded, bytes)

    def test_a_384_float_embedding_survives_exactly(self):
        """EXACT list equality. Not cosine similarity, not `pytest.approx`.

        A codec that rounded or truncated an embedding would restore a graph
        whose retrieval is quietly worse, and `canonical_serialize` excludes
        `embedding` (`grep -n "embedding" backend/knowledge/canonical_serialize.py`),
        so no determinism gate in this repo would notice.
        """
        decoded = _round_trip(_EMBEDDING)

        assert decoded == _EMBEDDING
        assert len(decoded) == 384

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(None, id="null"),
            pytest.param(True, id="bool"),
            pytest.param(7, id="int"),
            pytest.param(0.5, id="float"),
            pytest.param("rust", id="str"),
            pytest.param(["a", "b"], id="list-of-str"),
        ],
    )
    def test_json_native_values_pass_through_untagged(self, value):
        encoded = encode_value(value, key="probe", where="node ['X']")

        assert encoded == value
        assert _round_trip(value) == value

    def test_a_list_of_temporals_recurses(self):
        value = [Date(2026, 9, 17), Date(2026, 9, 18)]

        assert _round_trip(value) == value


class TestUnencodableValuesAreRefused:
    def test_an_unsupported_type_names_the_property_and_refuses(self):
        """A stdlib datetime is exactly what `default=str` used to swallow."""
        with pytest.raises(GraphArtifactError) as excinfo:
            encode_properties(
                {"created_at": stdlib_datetime(2026, 9, 17)},
                where="node ['__Entity__', 'Technology']",
            )

        message = str(excinfo.value)
        assert "created_at" in message
        assert "datetime" in message
        assert "node ['__Entity__', 'Technology']" in message

    def test_a_set_is_refused_rather_than_coerced_to_a_list(self):
        with pytest.raises(GraphArtifactError, match="aliases"):
            encode_properties({"aliases": {"a", "b"}}, where="node ['X']")

    def test_an_unencodable_value_inside_a_list_is_refused(self):
        with pytest.raises(GraphArtifactError, match="history"):
            encode_properties({"history": [1, stdlib_datetime(2026, 9, 17)]}, where="node ['X']")


class TestDecodeRefusals:
    def test_an_unknown_tag_is_refused(self):
        with pytest.raises(GraphArtifactError, match="LocalDateTime"):
            decode_value({NEO4J_TYPE_TAG: "LocalDateTime", "iso": "2026-09-17T12:00:00"})

    def test_an_unknown_point_srid_is_refused(self):
        """Refused rather than guessed: no other srid maps to a known class."""
        with pytest.raises(GraphArtifactError, match="1234"):
            decode_value({NEO4J_TYPE_TAG: "Point", "srid": 1234, "coordinates": [1.0, 2.0]})

    def test_a_point_whose_dimension_contradicts_its_srid_is_refused(self):
        with pytest.raises(GraphArtifactError, match="2-dimensional"):
            decode_value({NEO4J_TYPE_TAG: "Point", "srid": 7203, "coordinates": [1.0, 2.0, 3.0]})

    def test_an_untagged_map_in_a_property_position_is_refused(self):
        """Neo4j has no map property type, so this cannot be a genuine value."""
        with pytest.raises(GraphArtifactError, match="untagged map"):
            decode_value({"iso": "2026-09-17"})

    def test_a_tagged_temporal_with_no_iso_field_is_refused(self):
        with pytest.raises(GraphArtifactError, match="'iso'"):
            decode_value({NEO4J_TYPE_TAG: "Date"})


class TestLoadArtifactVersionRule:
    def test_an_artifact_with_no_envelope_is_refused_and_says_what_it_is(self):
        """This is today's `graph-backup` output, so an operator WILL hit it."""
        legacy = {"nodes": [], "relationships": [], "node_count": 0, "rel_count": 0}

        with pytest.raises(GraphArtifactError) as excinfo:
            load_artifact(legacy)

        message = str(excinfo.value)
        assert "predates" in message
        assert "graph-backup" in message

    def test_a_foreign_format_is_refused(self):
        with pytest.raises(GraphArtifactError, match="mist.hydration-snapshot"):
            load_artifact(
                {
                    "format": "mist.hydration-snapshot",
                    "format_version": 1,
                    "nodes": [],
                    "relationships": [],
                }
            )

    def test_an_unknown_version_is_refused_naming_both_versions(self):
        """Fail-closed: no auto-upgrade, no best-effort read of a future shape."""
        with pytest.raises(GraphArtifactError) as excinfo:
            load_artifact(
                {
                    "format": GRAPH_ARTIFACT_FORMAT,
                    "format_version": 99,
                    "nodes": [],
                    "relationships": [],
                }
            )

        message = str(excinfo.value)
        assert "99" in message
        assert str(sorted(SUPPORTED_ARTIFACT_VERSIONS)) in message

    def test_a_boolean_version_does_not_slip_through_the_membership_test(self):
        """`True == 1`, so `True in frozenset({1})` is True without this check."""
        with pytest.raises(GraphArtifactError, match="not an integer"):
            load_artifact(
                {
                    "format": GRAPH_ARTIFACT_FORMAT,
                    "format_version": True,
                    "nodes": [],
                    "relationships": [],
                }
            )

    def test_a_missing_node_list_is_refused_as_truncation(self):
        with pytest.raises(GraphArtifactError, match="truncated"):
            load_artifact(
                {
                    "format": GRAPH_ARTIFACT_FORMAT,
                    "format_version": GRAPH_ARTIFACT_VERSION,
                    "relationships": [],
                }
            )


class TestDumpsArtifact:
    def test_a_non_finite_float_fails_at_capture_time(self):
        """`allow_nan=False`. Python's json would write a bare `NaN` token.

        No strict JSON parser reads that back, so the default would produce a
        backup file that cannot be loaded at all -- discovered during a
        recovery. This fails while the graph is still there to re-read.
        """
        artifact = _artifact_with_property("embedding", [0.1, math.nan, 0.3])

        with pytest.raises(GraphArtifactError, match="non-finite"):
            dumps_artifact(artifact)

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan], ids=["inf", "-inf", "nan"])
    def test_every_non_finite_float_is_caught(self, value):
        with pytest.raises(GraphArtifactError):
            dumps_artifact(_artifact_with_property("score", value))

    def test_a_finite_artifact_serialises(self):
        text = dumps_artifact(_artifact_with_property("score", 0.9))

        assert json.loads(text)["nodes"][0]["properties"]["score"] == 0.9


def _artifact_with_property(key: str, value):
    return build_artifact(
        nodes=[{"id": "rust", "labels": ["__Entity__"], "properties": {key: value}}],
        relationships=[],
        schema={"constraints": [], "indexes": []},
        counts={"nodes": 1, "relationships": 0},
        stamps={},
        source={"uri": "bolt://mist-neo4j-dev:7687", "database": "neo4j"},
    )


class TestArtifactEnvelope:
    def test_build_artifact_stamps_the_format_and_version(self):
        artifact = _artifact_with_property("score", 0.9)

        assert artifact["format"] == GRAPH_ARTIFACT_FORMAT
        assert artifact["format_version"] == GRAPH_ARTIFACT_VERSION
        assert artifact["captured_at"].endswith("Z")
        assert artifact["source"] == {"uri": "bolt://mist-neo4j-dev:7687", "database": "neo4j"}

    def test_build_artifact_records_stamps_without_load_artifact_enforcing_them(self):
        """A DR artifact that self-invalidates on a version bump is worse than none."""
        artifact = build_artifact(
            nodes=[],
            relationships=[],
            schema={"constraints": [], "indexes": []},
            counts={"nodes": 0, "relationships": 0},
            stamps={"extraction_version": "1999-01-01-r0", "ontology_version": "0.0.1"},
            source={"uri": "bolt://x:7687", "database": None},
        )

        loaded = load_artifact(json.loads(dumps_artifact(artifact)))

        assert loaded["stamps"]["extraction_version"] == "1999-01-01-r0"

    def test_a_whole_graph_round_trips_through_text(self):
        artifact = build_artifact(
            nodes=[
                {
                    "id": "mist-identity",
                    "labels": ["MistIdentity", "__SelfModel__"],
                    "properties": {"created_at": _ZONED, "embedding": _EMBEDDING},
                }
            ],
            relationships=[
                {
                    "source": "mist-identity",
                    "type": "HAS_TRAIT",
                    "target": "curious",
                    "properties": {"first_seen_at": DateTime(2026, 9, 17, 1, 2, 3, 4)},
                }
            ],
            schema={"constraints": ["CREATE CONSTRAINT `c` FOR (n:X) REQUIRE n.id IS UNIQUE"]},
            counts={"nodes": 1, "relationships": 1},
            stamps={"ontology_version": "1.3.0"},
            source={"uri": "bolt://mist-neo4j-dev:7687", "database": "neo4j"},
        )

        loaded = load_artifact(json.loads(dumps_artifact(artifact)))

        node = loaded["nodes"][0]
        assert node["properties"]["created_at"] == _ZONED
        assert node["properties"]["embedding"] == _EMBEDDING
        assert loaded["relationships"][0]["properties"]["first_seen_at"] == DateTime(
            2026, 9, 17, 1, 2, 3, 4
        )
        assert loaded["schema"]["constraints"][0].startswith("CREATE CONSTRAINT")
        assert loaded["counts"] == {"nodes": 1, "relationships": 1}

    def test_build_artifact_refuses_already_encoded_properties(self):
        """Double encoding is a caller error, and it is loud rather than silent."""
        encoded = {NEO4J_TYPE_TAG: "Date", "iso": "2026-09-17"}

        with pytest.raises(GraphArtifactError, match="dict"):
            build_artifact(
                nodes=[{"id": "a", "labels": [], "properties": {"day": encoded}}],
                relationships=[],
                schema={},
                counts={"nodes": 1, "relationships": 0},
                stamps={},
                source={"uri": "bolt://x:7687", "database": None},
            )
