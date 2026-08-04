"""R1.4.6 T4: staleness detection, and the artifact-format guards around it.

The property that matters is the one nobody has to remember: producer identity
is READ from `backend.knowledge.version_stamps`, never restated, so bumping
`EXTRACTION_VERSION` invalidates every artifact on the next restore without the
bumper knowing this module exists. The first test pins exactly that -- it fails
if anyone ever hardcodes a stamp here.
"""

import json
from pathlib import Path

import pytest

from backend.knowledge.version_stamps import EXTRACTION_VERSION, ONTOLOGY_VERSION
from scripts.hydration.manifest import (
    ARTIFACT_SCHEMA_VERSION,
    HydrationError,
    SnapshotIdentity,
    SnapshotManifest,
    assert_fresh,
    digest_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def build_identity(**overrides) -> SnapshotIdentity:
    """A valid identity with every field overridable."""
    fields = {
        "ontology_version": "v1.4.0",
        "extraction_version": "2026-06-14-r5",
        "model_hash": "gemma-x|emb:all-MiniLM-L6-v2",
        "corpus_path": "data/golden-log/golden-log.jsonl",
        "corpus_sha256": "a" * 64,
        "corpus_file_count": 1,
    }
    fields.update(overrides)
    return SnapshotIdentity(**fields)


def build_manifest(identity: SnapshotIdentity | None = None, **overrides) -> SnapshotManifest:
    fields = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at": "2026-08-03T12:00:00+00:00",
        "identity": identity or build_identity(),
        "contents": {"graph_nodes": 412, "graph_relationships": 903},
        "source": {"neo4j_uri": "bolt://mist-neo4j-dev:7687", "dev_root": "/app/dev-state"},
    }
    fields.update(overrides)
    return SnapshotManifest(**fields)


class TestIdentityDerivesFromTheSingleAuthority:
    def test_current_reads_the_stamps_rather_than_restating_them(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("{}\n", encoding="utf-8")

        identity = SnapshotIdentity.current(corpus)

        assert identity.ontology_version == ONTOLOGY_VERSION
        assert identity.extraction_version == EXTRACTION_VERSION

    def test_model_hash_carries_the_embedding_model(self, tmp_path):
        # `compose_model_hash` folds the embedding model in, because a swapped
        # embedding model can flip a near-threshold merge and change the graph.
        # An artifact that ignored it would restore as "fresh" across that swap.
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("{}\n", encoding="utf-8")

        identity = SnapshotIdentity.current(corpus)

        assert "|emb:" in identity.model_hash


class TestCorpusDigest:
    def test_digest_changes_when_the_corpus_content_changes(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("turn one\n", encoding="utf-8")
        before, _ = digest_corpus(corpus)
        corpus.write_text("turn one changed\n", encoding="utf-8")
        after, _ = digest_corpus(corpus)

        assert before != after

    def test_directory_digest_changes_when_a_file_is_renamed(self, tmp_path):
        # T2 may re-cut the 87 turns into a multi-session corpus directory where
        # filenames carry the ordering, so a rename is a corpus change.
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "session-01.jsonl").write_text("x\n", encoding="utf-8")
        before, count = digest_corpus(corpus)
        (corpus / "session-01.jsonl").rename(corpus / "session-02.jsonl")
        after, _ = digest_corpus(corpus)

        assert count == 1
        assert before != after

    def test_missing_corpus_refuses_rather_than_recording_unknown(self, tmp_path):
        with pytest.raises(HydrationError, match="does not exist"):
            digest_corpus(tmp_path / "absent.jsonl")

    def test_empty_corpus_directory_refuses(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        with pytest.raises(HydrationError, match="holds no files"):
            digest_corpus(corpus)


class TestDriftDetection:
    @pytest.mark.parametrize(
        "field, value",
        [
            pytest.param("ontology_version", "v1.5.0", id="ontology-bump"),
            pytest.param("extraction_version", "2026-09-01-r6", id="prompt-bump"),
            pytest.param("model_hash", "other|emb:x", id="model-swap"),
            pytest.param("corpus_sha256", "b" * 64, id="corpus-rewrite"),
            pytest.param("corpus_path", "data/corpus-v2/", id="corpus-moved"),
            pytest.param("corpus_file_count", 12, id="corpus-recut"),
        ],
    )
    def test_every_producer_input_is_a_staleness_trigger(self, field, value):
        drift = build_identity().drift_against(build_identity(**{field: value}))
        assert drift == [field]

    def test_identical_identities_report_no_drift(self):
        assert build_identity().drift_against(build_identity()) == []

    def test_multiple_changes_are_all_reported(self):
        other = build_identity(extraction_version="new", model_hash="new")
        assert set(build_identity().drift_against(other)) == {"extraction_version", "model_hash"}


class TestAssertFresh:
    def test_refuses_a_stale_artifact_and_names_the_drifted_field(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("{}\n", encoding="utf-8")
        current = SnapshotIdentity.current(corpus)
        stale = build_manifest(
            SnapshotIdentity(
                ontology_version=current.ontology_version,
                extraction_version="2020-01-01-r0",
                model_hash=current.model_hash,
                corpus_path=current.corpus_path,
                corpus_sha256=current.corpus_sha256,
                corpus_file_count=current.corpus_file_count,
            )
        )

        with pytest.raises(HydrationError, match="STALE") as excinfo:
            assert_fresh(stale, corpus)

        assert "extraction_version" in str(excinfo.value)
        assert "2020-01-01-r0" in str(excinfo.value)

    def test_accepts_an_artifact_produced_by_this_tree(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("{}\n", encoding="utf-8")

        assert_fresh(build_manifest(SnapshotIdentity.current(corpus)), corpus)

    def test_a_prompt_bump_invalidates_without_touching_the_manifest(self, tmp_path, monkeypatch):
        # The mechanism the design rests on: the bumper edits version_stamps and
        # every artifact goes stale, with no step in this package to remember.
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("{}\n", encoding="utf-8")
        manifest = build_manifest(SnapshotIdentity.current(corpus))
        assert_fresh(manifest, corpus)

        monkeypatch.setattr("scripts.hydration.manifest.EXTRACTION_VERSION", "2026-12-31-r9")

        with pytest.raises(HydrationError, match="extraction_version"):
            assert_fresh(manifest, corpus)


class TestManifestRoundTrip:
    def test_write_then_read_preserves_the_identity(self, tmp_path):
        original = build_manifest()
        original.write(tmp_path)

        assert SnapshotManifest.read(tmp_path).identity == original.identity

    def test_manifest_json_is_stable_across_writes(self, tmp_path):
        # A regenerated artifact should diff on what changed, not on key order.
        first = build_manifest().to_json()
        second = build_manifest().to_json()

        assert first == second

    def test_missing_manifest_refuses(self, tmp_path):
        with pytest.raises(HydrationError, match="not found"):
            SnapshotManifest.read(tmp_path)

    def test_future_artifact_schema_version_refuses(self, tmp_path):
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION + 1,
                    "created_at": "2026-08-03T12:00:00+00:00",
                    "identity": build_identity().to_dict(),
                    "contents": {},
                    "source": {},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(HydrationError, match="artifact_schema_version"):
            SnapshotManifest.read(tmp_path)

    def test_identity_missing_a_field_refuses_rather_than_defaulting(self, tmp_path):
        incomplete = build_identity().to_dict()
        del incomplete["corpus_sha256"]
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at": "2026-08-03T12:00:00+00:00",
                    "identity": incomplete,
                    "contents": {},
                    "source": {},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(HydrationError, match="missing"):
            SnapshotManifest.read(tmp_path)
