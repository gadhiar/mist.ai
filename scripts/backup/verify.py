"""Prove a restore: compare a restored graph, stores and vault with the artifact they came from.

Plan v1, task T7; the contract is Interfaces and contracts item 7. A backup counts once it has
been restored and shown equal to what was captured, and this is the "shown equal" half. The
drill project runs it in the release image after `restore`, against the drill's own Neo4j and
standby directories (`deploy/daily/RESTORE.md`).

    python -m scripts.backup.verify --artifact DIR --graph-uri URI --state-root DIR
        [--vault-root DIR]

WHAT IS COMPARED, AND AGAINST WHAT.
    - Graph: the node and relationship counts of the artifact's `graph.json` against the
      restored graph, and every node's `embedding` for exact equality, element by element.
      The restored graph is read with `dump_full_graph_json`, the reader the dump itself uses,
      so both sides are read by one rule. Embeddings are matched by node `id`, the key restore
      re-anchors relationships by. Exact `==` on floats, never cosine similarity: a vector that
      is merely close has been rewritten, and every similarity threshold in the codebase would
      pass it.
    - Stores: for every store the manifest records with a non-null `file`, the row counts of
      the artifact's copy against `<state-root>/<store>`, table by table. The artifact's copy is
      read from a temporary copy, because the drill mounts the backup root read-only and opening
      a WAL database creates a `-shm` file beside it.
    - Vault: the number of files in the artifact's `vault/` tree against the vault root,
      `.git` included, because restore copies the whole tree and the vault's commits exist
      nowhere else (`count_vault_files`, the dump's own counter).

    The output reports one total per store; the match flag uses the per-table counts, so two
    stores whose totals agree while a table differs still read `mismatch`.

THE OUTPUT CARRIES NO CONTENT. One JSON object of counts and flags, keyed by the fixed store
filenames. No node property, no text, no path from the data and no table name appears in it, so
the drill record written from it into `ops/` holds nothing from Raj's memory.

REFUSALS (exit 2), each before the graph is read:
    - a graph URI that fails `assert_neo4j_dev_isolated`, checked before any connection;
    - a state root or vault root that fails `assert_isolated_root`;
    - an artifact with no manifest this build reads, or a graph leg it cannot decode.

Exit codes: 0 every count and every embedding match; 1 anything differs, or the comparison could
not complete (the graph did not answer); 2 refused, nothing read from the target.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from backend.errors import MistError
from backend.interfaces import GraphConnection
from backend.knowledge.admin import dump_full_graph_json
from backend.knowledge.eval_isolation import (
    EvalIsolationError,
    IsolatedRootError,
    assert_isolated_root,
    assert_neo4j_dev_isolated,
)

from .dump import VAULT_DIRNAME, count_vault_files
from .errors import BackupError, BackupManifestError, RestorePreflightError
from .manifest import read_manifest
from .restore import DEFAULT_TARGET_VAULT_DIRNAME, load_graph_leg
from .stores import STORES_DIRNAME, read_back_store

logger = logging.getLogger(__name__)

EXIT_MATCH = 0
EXIT_MISMATCH = 1
EXIT_REFUSED = 2

EMBEDDING_PROPERTY = "embedding"
PURPOSE = "restore verification"


@dataclass(frozen=True, slots=True)
class CountPair:
    """One count read from the artifact and from the restored target."""

    artifact: int | None
    restored: int | None

    @property
    def equal(self) -> bool:
        """Both sides were read and agree; an unread side is never a match."""
        return self.artifact is not None and self.artifact == self.restored

    def to_dict(self) -> dict[str, int | None]:
        """The `{"artifact": n, "restored": n}` pair of item 7."""
        return {"artifact": self.artifact, "restored": self.restored}


@dataclass(frozen=True, slots=True)
class StoreComparison:
    """One store's per-table row counts on each side, None for a side that could not be read."""

    artifact: Mapping[str, int] | None
    restored: Mapping[str, int] | None

    @property
    def equal(self) -> bool:
        """Every table's row count agrees; a side that could not be read is never a match."""
        return (
            self.artifact is not None
            and self.restored is not None
            and dict(self.artifact) == dict(self.restored)
        )

    def to_dict(self) -> dict[str, int | None]:
        """Each side's total rows: item 7 reports one number per store, not per table."""
        return {
            "artifact": None if self.artifact is None else sum(self.artifact.values()),
            "restored": None if self.restored is None else sum(self.restored.values()),
        }


@dataclass(frozen=True, slots=True)
class VerifyReport:
    """The item 7 comparison, before it is rendered."""

    nodes: CountPair
    relationships: CountPair
    embeddings_equal: bool
    stores: Mapping[str, StoreComparison]
    vault_files: CountPair

    @property
    def matched(self) -> bool:
        """Whether every comparison agrees, which is what exit 0 means."""
        return (
            self.nodes.equal
            and self.relationships.equal
            and self.embeddings_equal
            and all(store.equal for store in self.stores.values())
            and self.vault_files.equal
        )

    def to_dict(self) -> dict[str, Any]:
        """The item 7 JSON object, in its documented key order."""
        return {
            "result": "match" if self.matched else "mismatch",
            "nodes": self.nodes.to_dict(),
            "relationships": self.relationships.to_dict(),
            "embeddings_equal": self.embeddings_equal,
            "stores": {name: store.to_dict() for name, store in self.stores.items()},
            "vault_files": self.vault_files.to_dict(),
        }


def check_targets(
    graph_uri: str, state_root: Path | str, vault_root: Path | str | None
) -> tuple[Path, Path]:
    """Refuse a live graph or a live directory; return the resolved state and vault roots.

    The guards are the shared ones, called unchanged. Verify only reads, but a verify pointed
    at the live instance would print counts of live data under a drill's name, and a drill that
    "matched" against the live graph proves nothing about the artifact.

    Raises:
        EvalIsolationError: The URI is not a dev endpoint.
        IsolatedRootError: The state root or the vault root is, sits under, or contains live
            state, or sits under a marked or registered live instance.
    """
    assert_neo4j_dev_isolated(graph_uri)
    state = Path(state_root).expanduser().resolve()
    assert_isolated_root(state, purpose=PURPOSE)
    vault = (
        state / DEFAULT_TARGET_VAULT_DIRNAME
        if vault_root is None
        else Path(vault_root).expanduser().resolve()
    )
    assert_isolated_root(vault, purpose=PURPOSE)
    return state, vault


def _embedding_key(value: Any) -> Any:
    """A hashable, exact form of an embedding value; a list keeps every float as it is."""
    if isinstance(value, list | tuple):
        return tuple(value)
    return value


def _embeddings(nodes: Iterable[Mapping[str, Any]]) -> Counter:
    """Every node's embedding as a multiset of `(id, exact vector)`.

    A multiset rather than a dict, so a duplicated id or a node with no id is still compared
    rather than silently collapsed into one entry.
    """
    return Counter(
        (node.get("id"), _embedding_key(node["properties"][EMBEDDING_PROPERTY]))
        for node in nodes
        if node.get("properties", {}).get(EMBEDDING_PROPERTY) is not None
    )


def _store_counts(path: Path) -> dict[str, int] | None:
    """Per-table row counts of one store, or None when it is absent or unsound.

    An unsound file reads as None rather than raising: a store that cannot be read back is a
    mismatch, which is exactly what the drill must report.
    """
    if not path.is_file():
        return None
    try:
        return dict(read_back_store(path).row_counts)
    except BackupError as exc:
        logger.warning("[verify] %s could not be read back: %s", path.name, exc)
        return None


def _artifact_store_counts(artifact_dir: Path, filename: str) -> dict[str, int] | None:
    """Count an artifact's store copy through a scratch copy, leaving the artifact unopened."""
    source = artifact_dir / STORES_DIRNAME / filename
    if not source.is_file():
        return None
    with tempfile.TemporaryDirectory(prefix="mist-verify-") as scratch:
        copy = Path(scratch) / filename
        shutil.copyfile(source, copy)
        return _store_counts(copy)


def compare_stores(
    artifact_dir: Path, stores: Mapping[str, Mapping[str, Any]], state_root: Path
) -> dict[str, StoreComparison]:
    """Compare every store the artifact holds with the target's file of the same name.

    A store the manifest records with `file` null was absent at capture, so restore left the
    target's own copy in place; there is nothing of the artifact's to compare it with.
    """
    return {
        filename: StoreComparison(
            artifact=_artifact_store_counts(artifact_dir, filename),
            restored=_store_counts(state_root / filename),
        )
        for filename, entry in sorted(stores.items())
        if entry.get("file") is not None
    }


def verify_restore(
    *,
    artifact_dir: Path | str,
    graph_uri: str,
    state_root: Path | str,
    vault_root: Path | str | None,
    connection: GraphConnection,
) -> VerifyReport:
    """Compare one restored target with the artifact it was restored from.

    Every guard and every artifact check runs before the first read of the graph.

    Raises:
        EvalIsolationError: The URI is not a dev endpoint.
        IsolatedRootError: The state root or the vault root is live.
        BackupManifestError: The artifact has no manifest this build reads.
        RestorePreflightError: The artifact's graph leg cannot be decoded.
        MistError: The restored graph could not be read.
    """
    state, vault = check_targets(graph_uri, state_root, vault_root)
    source = Path(artifact_dir)
    manifest = read_manifest(source)
    artifact_graph = load_graph_leg(source)

    restored_graph = dump_full_graph_json(connection)

    artifact_vault = source / VAULT_DIRNAME
    return VerifyReport(
        nodes=CountPair(len(artifact_graph["nodes"]), len(restored_graph["nodes"])),
        relationships=CountPair(
            len(artifact_graph["relationships"]), len(restored_graph["relationships"])
        ),
        embeddings_equal=_embeddings(artifact_graph["nodes"])
        == _embeddings(restored_graph["nodes"]),
        stores=compare_stores(source, manifest.stores, state),
        vault_files=CountPair(count_vault_files(artifact_vault)[1], count_vault_files(vault)[1]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.backup.verify",
        description=(
            "Compare a restored graph, state root and vault with the backup artifact they "
            "were restored from. Prints one JSON object of counts and match flags."
        ),
        epilog="Exit codes: 0 match; 1 mismatch or the comparison could not complete; 2 refused.",
    )
    parser.add_argument("--artifact", required=True, help="The artifact restored FROM.")
    parser.add_argument(
        "--graph-uri",
        required=True,
        help="Bolt URI of the restored graph; must pass assert_neo4j_dev_isolated.",
    )
    parser.add_argument("--state-root", required=True, help="The restored state root.")
    parser.add_argument(
        "--vault-root",
        default=None,
        help=f"The restored vault tree (default: <state-root>/{DEFAULT_TARGET_VAULT_DIRNAME}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one verification; prints the item 7 JSON on stdout and everything else on stderr."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # The guards run before the backend's connection class is even imported, as in restore:
    # a mistyped target costs nothing and receives no connection attempt.
    try:
        check_targets(args.graph_uri, args.state_root, args.vault_root)
    except (EvalIsolationError, IsolatedRootError) as exc:
        print(f"[verify] REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    from backend.knowledge.config import get_config
    from backend.knowledge.storage.neo4j_connection import Neo4jConnection

    connection: Any = None
    try:
        connection = Neo4jConnection(replace(get_config().neo4j, uri=args.graph_uri))
        connection.connect()
        report = verify_restore(
            artifact_dir=Path(args.artifact),
            graph_uri=args.graph_uri,
            state_root=Path(args.state_root),
            vault_root=None if args.vault_root is None else Path(args.vault_root),
            connection=connection,
        )
    except (
        EvalIsolationError,
        IsolatedRootError,
        BackupManifestError,
        RestorePreflightError,
    ) as exc:
        print(f"[verify] REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except MistError as exc:
        print(f"[verify] FAILED: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return EXIT_MISMATCH
    finally:
        if connection is not None:
            connection.disconnect()

    print(json.dumps(report.to_dict()))
    return EXIT_MATCH if report.matched else EXIT_MISMATCH


if __name__ == "__main__":
    sys.exit(main())
