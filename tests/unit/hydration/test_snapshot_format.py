"""R1.4.6 T4: the artifact-format guards, and the output-path guard.

These cover the parts of `snapshot.py` that hold no database connection. The
graph load/dump round trip needs a running dev Neo4j and belongs to the
integration layer; what is pinned here is every place the format can silently
lose or corrupt something.
"""

import json
from pathlib import Path

import pytest
import pytz
from neo4j.time import Date, DateTime, Duration, Time

from scripts.hydration.manifest import HydrationError
from scripts.hydration.snapshot import (
    _assert_artifact_dir_safe,
    _canonical,
    _ddl_name,
    _decode_props,
    _encode_props,
    _label_clause,
    _node_key,
    _quote_ident,
    _strip_sqlite_sidecars,
    dump,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestPropertyEncoding:
    @pytest.mark.parametrize(
        "props",
        [
            pytest.param({"id": "person-raj"}, id="string"),
            pytest.param({"confidence": 0.87}, id="float"),
            pytest.param({"vram_gb": 12}, id="int"),
            pytest.param({"is_latest_belief": True}, id="bool"),
            pytest.param({"valid_to": None}, id="null"),
            pytest.param({"embedding": [0.1, 0.2, 0.3]}, id="list-of-float"),
            pytest.param({"evidence": ["a", "b"]}, id="list-of-string"),
        ],
    )
    def test_json_native_values_pass_through_unchanged(self, props):
        assert _encode_props(props, "node ['__Entity__']") == props

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(DateTime(2026, 8, 4, 4, 31, 18, 5000000, tzinfo=pytz.UTC), id="zoned"),
            pytest.param(DateTime(2026, 8, 3, 10, 30, 0, 123456789), id="nanosecond-precision"),
            pytest.param(Date(2026, 8, 3), id="date"),
            pytest.param(Time(10, 30, 0), id="time"),
            pytest.param(Duration(days=3, seconds=42), id="duration"),
        ],
    )
    def test_temporals_survive_the_round_trip_exactly(self, value):
        # Not hypothetical: the self-model seeding path writes `created_at` as
        # a ZONED DATETIME, so a real hydrated graph contains one on first boot.
        # Stringifying it would restore text where the original was a temporal.
        encoded = _encode_props({"created_at": value}, "node ['MistIdentity']")

        assert json.loads(json.dumps(encoded)) == encoded  # actually JSON-safe
        assert _decode_props(encoded)["created_at"] == value

    def test_the_tag_records_the_concrete_type(self):
        encoded = _encode_props({"created_at": Date(2026, 8, 3)}, "node ['X']")

        assert encoded["created_at"]["__neo4j_type__"] == "Date"
        assert encoded["created_at"]["iso"] == "2026-08-03"

    def test_temporals_inside_a_list_are_encoded_too(self):
        value = [Date(2026, 8, 3), Date(2026, 8, 4)]
        encoded = _encode_props({"history": value}, "node ['X']")

        assert _decode_props(encoded)["history"] == value

    def test_refuses_a_value_whose_round_trip_is_unverified(self):
        # An encoding nobody has proven lossless is worse than a refusal.
        with pytest.raises(HydrationError, match="cannot round-trip"):
            _encode_props({"location": object()}, "relationship USES")

    def test_refusal_names_the_property_and_the_owner(self):
        with pytest.raises(HydrationError) as excinfo:
            _encode_props({"where": object()}, "relationship LEARNING")

        assert "where" in str(excinfo.value)
        assert "relationship LEARNING" in str(excinfo.value)

    def test_refuses_an_unrepresentable_element_inside_a_list(self):
        with pytest.raises(HydrationError, match="cannot round-trip"):
            _encode_props({"history": [object()]}, "node ['LearningEvent']")

    def test_decoding_an_unknown_tag_refuses_rather_than_changing_the_type(self):
        # A newer artifact tagging a type this tree cannot rebuild must not
        # restore silently as a dict.
        with pytest.raises(HydrationError, match="unknown tagged value type"):
            _decode_props({"p": {"__neo4j_type__": "Point", "iso": "POINT(1 2)"}})


class TestNodeKey:
    def test_same_content_yields_the_same_key(self):
        first = _node_key(["__Entity__", "Person"], {"id": "raj", "name": "Raj"})
        second = _node_key(["Person", "__Entity__"], {"name": "Raj", "id": "raj"})

        # Label order and property order are not content, so neither may change
        # the key -- otherwise a re-dump of an unchanged graph would diff.
        assert first == second

    def test_different_content_yields_a_different_key(self):
        first = _node_key(["__Entity__"], {"id": "raj"})
        second = _node_key(["__Entity__"], {"id": "raj2"})

        assert first != second

    def test_label_set_participates_in_the_key(self):
        first = _node_key(["__Entity__"], {"id": "raj"})
        second = _node_key(["__Entity__", "__SelfModel__"], {"id": "raj"})

        assert first != second


class TestCanonicalJson:
    def test_key_order_does_not_affect_output(self):
        assert _canonical({"b": 1, "a": 2}) == _canonical({"a": 2, "b": 1})

    def test_non_ascii_is_preserved_rather_than_escaped(self):
        assert "é" in _canonical({"name": "café"})


class TestIdentifierQuoting:
    def test_quotes_a_label(self):
        assert _quote_ident("__Entity__") == "`__Entity__`"

    def test_refuses_a_backtick_which_would_break_out_of_the_quoting(self):
        with pytest.raises(HydrationError, match="unquotable"):
            _quote_ident("Bad`Label")

    def test_refuses_an_empty_identifier(self):
        with pytest.raises(HydrationError, match="unquotable"):
            _quote_ident("")


class TestLabelClause:
    """Regression: the separator is load-bearing and silent when wrong.

    Joining quoted labels with "" produced `` `__Entity__``Person` ``, which
    Cypher reads as ONE label named ``__Entity__`Person`` -- a doubled backtick
    is the escape for a literal one. Restore then created nodes carrying a
    single nonsense label instead of three, and the only symptom, several
    hundred lines later, was "created 0 of 1 USES relationships".
    """

    def test_multiple_labels_are_colon_separated(self):
        assert _label_clause(["__Entity__", "Person"]) == "`__Entity__`:`Person`"

    def test_a_single_label_needs_no_separator(self):
        assert _label_clause(["__RestoreKey__"]) == "`__RestoreKey__`"

    def test_no_doubled_backticks_which_cypher_reads_as_one_escaped_label(self):
        assert "``" not in _label_clause(["__Entity__", "Person", "__RestoreKey__"])

    def test_refuses_a_label_that_would_break_out_of_the_quoting(self):
        with pytest.raises(HydrationError, match="unquotable"):
            _label_clause(["__Entity__", "Bad`Label"])


class TestDdlName:
    """The name's POSITION varies by DDL kind, which is why this scans.

    Every statement below is verbatim `createStatement` output captured from
    the live Neo4j 5 instance on 2026-08-03 -- not a plausible reconstruction.
    An earlier version read token 2 unconditionally, which is correct for
    CONSTRAINT and returns the literal string "INDEX" for every index. Restore
    then matched no existing index, re-created all of them, and Neo4j rejected
    it with EquivalentSchemaRuleAlreadyExists.
    """

    @pytest.mark.parametrize(
        "statement, expected",
        [
            pytest.param(
                "CREATE CONSTRAINT `entity_id_unique` FOR (n:`__Entity__`) "
                "REQUIRE (n.`id`) IS UNIQUE",
                "entity_id_unique",
                id="constraint-name-at-token-2",
            ),
            pytest.param(
                "CREATE RANGE INDEX `entity_type_idx` FOR (n:`__Entity__`) ON (n.`entity_type`)",
                "entity_type_idx",
                id="range-index-name-at-token-3",
            ),
            pytest.param(
                "CREATE VECTOR INDEX `entity_embeddings` FOR (n:`__Entity__`) "
                "ON (n.`embedding`) OPTIONS {indexConfig: {`vector.dimensions`: 384}}",
                "entity_embeddings",
                id="vector-index-name-at-token-3",
            ),
            pytest.param(
                "CREATE RANGE INDEX `rel_applicable_to_src_utt_idx` "
                "FOR ()-[r:`APPLICABLE_TO`]-() ON (r.`source_utterance_id`)",
                "rel_applicable_to_src_utt_idx",
                id="relationship-property-index",
            ),
            pytest.param("DROP INDEX foo", None, id="not-a-create"),
            pytest.param("CREATE", None, id="truncated"),
        ],
    )
    def test_extracts_the_object_name(self, statement, expected):
        assert _ddl_name(statement) == expected

    def test_a_quoted_object_named_index_is_not_mistaken_for_the_keyword(self):
        statement = "CREATE RANGE INDEX `index` FOR (n:`X`) ON (n.`y`)"
        assert _ddl_name(statement) == "index"


class TestSqliteSidecars:
    """A WAL database leaves -wal/-shm behind whenever it is opened."""

    def test_strips_both_sidecars(self, tmp_path):
        db = tmp_path / "event_store.db"
        db.write_bytes(b"")
        for suffix in ("-wal", "-shm"):
            Path(f"{db}{suffix}").write_bytes(b"x")

        _strip_sqlite_sidecars(db)

        assert not Path(f"{db}-wal").exists()
        assert not Path(f"{db}-shm").exists()
        assert db.exists()

    def test_is_a_no_op_when_there_are_none(self, tmp_path):
        db = tmp_path / "event_store.db"
        db.write_bytes(b"")

        _strip_sqlite_sidecars(db)

        assert db.exists()


class TestDumpFailsBeforeWriting:
    """Producer identity is computed FIRST, so a refusal writes nothing.

    Regression: identity used to be computed last, so a dump that could not
    read its corpus still left behind a directory holding graph.json, the
    databases and the vault -- an artifact with no manifest, which is exactly
    the "cannot state what produced it" state `SnapshotManifest.read` refuses.
    The tool was manufacturing the corruption it warns about.
    """

    def test_a_missing_corpus_leaves_no_artifact_directory(self, tmp_path):
        dev_root = tmp_path / "dev-state"
        dev_root.mkdir()
        artifact_dir = tmp_path / "artifacts" / "label"

        with pytest.raises(HydrationError, match="corpus"):
            dump(
                dev_root=dev_root,
                neo4j_uri="bolt://mist-neo4j-dev:7687",
                artifact_dir=artifact_dir,
                corpus_path=tmp_path / "absent-corpus.jsonl",
            )

        # Neither the artifact nor its staging sibling survives -- and the fact
        # that this test needs no database proves the refusal happens before
        # the graph is even read.
        assert not artifact_dir.exists()
        assert not artifact_dir.with_name(artifact_dir.name + ".partial").exists()


class TestArtifactDirSafety:
    def test_refuses_an_output_dir_that_is_a_live_directory(self):
        # dump() replaces its output directory, so `--artifact data` would
        # delete the live event store.
        with pytest.raises(HydrationError, match="is, or contains, the live state"):
            _assert_artifact_dir_safe(REPO_ROOT / "data")

    def test_refuses_an_output_dir_that_contains_a_live_directory(self):
        with pytest.raises(HydrationError, match="is, or contains, the live state"):
            _assert_artifact_dir_safe(REPO_ROOT)

    def test_refuses_a_populated_directory_that_is_not_an_artifact(self, tmp_path):
        target = tmp_path / "someones-work"
        target.mkdir()
        (target / "notes.md").write_text("important\n", encoding="utf-8")

        with pytest.raises(HydrationError, match="not a hydration artifact"):
            _assert_artifact_dir_safe(target)

    def test_accepts_overwriting_an_existing_artifact(self, tmp_path):
        target = tmp_path / "r1.4.6-initial"
        target.mkdir()
        (target / "manifest.json").write_text("{}", encoding="utf-8")

        _assert_artifact_dir_safe(target)

    def test_accepts_a_new_directory_under_the_snapshot_root(self, tmp_path):
        _assert_artifact_dir_safe(tmp_path / "new-label")

    def test_accepts_the_default_snapshot_root_which_sits_under_data(self):
        # Artifacts live under data/ beside data/golden-log/, following the
        # convention this repo already uses for large generated fixtures. The
        # "sits under a live root" arm of assert_isolated_root must therefore
        # NOT apply to the artifact path -- only "is" and "contains" do.
        _assert_artifact_dir_safe(REPO_ROOT / "data" / "hydration-snapshots" / "label")
