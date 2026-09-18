"""Capture live MIST.AI state to an artifact directory outside the machine's working copy.

MIS-140 T1. Losing `./data` and losing the graph are SEPARATE failures: Neo4j
lives on the named volumes `mist-neo4j-data` / `mist-neo4j-logs`
(`grep -n "mist-neo4j-data" docker-compose.yml` -> :105-107,179-181), not on the
`./data` bind mount. `mist-memory/` is covered by nothing at all today: it is
gitignored (`grep -n "mist-memory" .gitignore` -> :39) and `git ls-files
mist-memory` returns zero files, so the corpus exists only on the operator's
disk. This dump covers all three in one pass.

WHAT IS CAPTURED
    stores/event_store.db       conversation sessions, turn events, epoch ledger
    stores/extraction_cache.db  cached extraction results
    stores/vault_sidecar.db     vault chunk index with embeddings
    graph.json                  every node and relationship, all partitions,
                                embeddings as exact `list[float]`, temporals and
                                Points tagged rather than stringified
    vault/                      a copy of the `mist-memory/` tree
    manifest.json               layout version, created_at, per-file sha256,
                                per-store row counts, graph counts, git HEAD and
                                the producer stamps

WHAT IS NOT CAPTURED, AND WHY
    `vector_store/` (LanceDB) is excluded as derived. It holds document chunks
    written by the ingestion pipeline
    (`grep -n "_vector_store.store_chunks" backend/knowledge/ingestion/pipeline.py`
    -> :224), and it measures 5.0K on the live deployment
    (`du -sh data/vector_store` -> 5.0K), so its size is not the argument; being
    reconstructible by re-ingesting is. UNVERIFIED CAVEAT, stated rather than
    papered over: this task did not establish that every ingested source
    document still exists outside `data/`. If one does not, its chunks are not
    recoverable from this artifact.

    `vault_sidecar.db` IS captured even though it too is derived, from
    `mist-memory/`. The reason is cost, not principle: the sidecar holds
    per-chunk embeddings, so rebuilding it re-runs the embedding model over the
    whole corpus, and it is small (3923968 bytes measured on live).

WHY A LOGICAL GRAPH EXPORT AND NOT `neo4j-admin database dump`
    `neo4j-admin database dump` requires the database STOPPED and its output is
    coupled to the server version that wrote it. Both defeat the point of a
    live-safe, read-only capture that can run on a schedule against a serving
    stack. The graph leg is instead the versioned JSON artifact from MIS-140 T3
    (`grep -n "def dump_full_graph_artifact" backend/knowledge/admin.py` ->
    :926), which is self-describing, checked on read by FORMAT version only, and
    carries embeddings as exact floats.

SCHEDULABLE BY CONSTRUCTION, NOT SCHEDULED
    Nothing here installs a timer. The dump is read-only against live state
    (SQLite opened `mode=ro`, Cypher reads only), takes no input, and returns
    deterministic exit codes:

        0  the artifact was written and its manifest is in place
        2  the destination was refused -- nothing was written
        1  the dump failed part way; the directory keeps its `.partial` suffix

    Arming a schedule is deliberately held until a restore has been rehearsed on
    the host. A backup nobody has restored is a rumour.

Usage:
    MIST_BACKUP_ROOT=/mnt/backup/mist python -m scripts.backup.dump
    python -m scripts.backup.dump --output /mnt/backup/mist --label pre-upgrade
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.errors import MistError
from backend.interfaces import GraphConnection
from backend.knowledge.admin import dump_full_graph_artifact, graph_version_stamps
from backend.knowledge.eval_isolation import REPO_ROOT
from backend.knowledge.graph_artifact import GraphArtifactError, dumps_artifact

from .destination import assert_backup_destination, resolve_backup_root
from .errors import BackupDestinationError, BackupError
from .manifest import (
    BACKUP_LAYOUT,
    BACKUP_LAYOUT_VERSION,
    PARTIAL_SUFFIX,
    BackupManifest,
    digest_artifact_files,
    utc_now_iso,
)
from .stores import LIVE_STORE_FILENAMES, capture_stores

logger = logging.getLogger(__name__)

GRAPH_FILENAME = "graph.json"
VAULT_DIRNAME = "vault"

# Both spellings of each live directory are expressed as one path so the same
# code is correct on the host and inside mist-backend, where the repo is bind
# mounted at `/app` (`grep -n "/app" docker-compose.yml`).
DEFAULT_STATE_ROOT = REPO_ROOT / "data"
DEFAULT_VAULT_ROOT = REPO_ROOT / "mist-memory"

# Recorded in the manifest so a restore operator can see what was left out
# without reading this file. Relative to the state root.
EXCLUDED_FROM_STATE_ROOT = ["vector_store"]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_DESTINATION_REFUSED = 2


@dataclass(frozen=True, slots=True)
class DumpReport:
    """What one dump actually captured."""

    artifact_dir: Path
    stores_present: int
    stores_absent: tuple[str, ...]
    graph_nodes: int
    graph_relationships: int
    vault_files: int


def read_git_head(repo_root: Path = REPO_ROOT) -> str | None:
    """Return the current commit SHA, or None when it cannot be established.

    Recorded so an operator can see which build produced an artifact. It is
    NEVER a gate: a dump must not fail because `git` is missing from the image
    or because the deployment is not a checkout, so every failure here returns
    None and the manifest carries `"git_head": null`.
    """
    try:
        # Fixed argv and no shell, so the repo path cannot be interpreted.
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    head = completed.stdout.strip()
    return head or None


def copy_vault(vault_root: Path, artifact_dir: Path) -> int:
    """Copy the vault corpus into the artifact; return the file count.

    An absent vault root returns 0 rather than raising. It is absent in every
    fresh clone -- `mist-memory/` is gitignored with zero tracked files -- and a
    dump that refused to run there would be a dump nobody could test.

    Raises:
        BackupError: When the tree exists but cannot be copied.
    """
    destination = artifact_dir / VAULT_DIRNAME
    if not vault_root.is_dir():
        return 0
    try:
        shutil.copytree(vault_root, destination)
    except OSError as exc:
        raise BackupError(
            f"could not copy the vault corpus from {vault_root}: {exc}. This tree is "
            "backed up by nothing else -- it is gitignored and untracked -- so the "
            "dump fails rather than completing without it."
        ) from exc
    return sum(1 for path in destination.rglob("*") if path.is_file())


def capture_graph(
    connection: GraphConnection,
    artifact_dir: Path,
    *,
    source_uri: str,
    database: str | None,
    stamps: dict[str, str],
) -> dict[str, Any]:
    """Write the graph leg and return its manifest entry.

    Args:
        connection: Read-only use only; the artifact is built from two MATCH
            queries and the schema `SHOW` statements.
        source_uri: The bolt URI being captured, recorded in the artifact.
        database: Database name, or None when the deployment has one.
        stamps: From `graph_version_stamps`. Recorded, never enforced.

    Returns:
        `{"file", "format", "format_version", "nodes", "relationships"}`.

    Raises:
        GraphArtifactError: When a property value cannot be round-tripped, or
            the artifact holds a non-finite float. Both fail here, at capture
            time, while the graph is still in front of the operator.
    """
    artifact = dump_full_graph_artifact(
        connection,
        source_uri=source_uri,
        database=database,
        stamps=stamps,
    )
    (artifact_dir / GRAPH_FILENAME).write_text(dumps_artifact(artifact), encoding="utf-8")
    counts = artifact["counts"]
    return {
        "file": GRAPH_FILENAME,
        "format": artifact["format"],
        "format_version": artifact["format_version"],
        "nodes": int(counts["nodes"]),
        "relationships": int(counts["relationships"]),
    }


def run_dump(
    *,
    backup_root: Path | str,
    connection: GraphConnection,
    source_uri: str,
    database: str | None,
    stamps: dict[str, str],
    state_root: Path | None = None,
    vault_root: Path | None = None,
    label: str | None = None,
) -> DumpReport:
    """Write one complete artifact directory under `backup_root`.

    The destination is re-checked here rather than trusted from the caller, so
    no code path reaches a write without passing the guard -- a library caller,
    a future command and the CLI all go through the same arms.

    The artifact is built under `<label>.partial` and RENAMED once the manifest
    is written. That rename is what makes completeness observable: a directory
    without the suffix has a manifest, and a directory with it is a dump that
    died. The retention leg can therefore tell a finished artifact from a
    half-written one without inspecting its contents.

    Args:
        backup_root: Destination root, already resolved or not.
        connection: Graph connection to read from.
        source_uri: Bolt URI recorded in the graph artifact.
        database: Database name recorded in the graph artifact.
        stamps: Producer stamps, recorded and never enforced.
        state_root: Directory holding the SQLite stores. Defaults to
            `<repo>/data`.
        vault_root: The `mist-memory/` corpus. Defaults to `<repo>/mist-memory`.
        label: Artifact directory name. Defaults to the UTC timestamp.

    Returns:
        A `DumpReport` naming the finished directory and what it holds.

    Raises:
        BackupDestinationError: When the destination is refused.
        BackupError: When a leg fails, or the target directory already exists.
    """
    root = assert_backup_destination(backup_root)
    state = state_root if state_root is not None else DEFAULT_STATE_ROOT
    vault = vault_root if vault_root is not None else DEFAULT_VAULT_ROOT
    created_at = utc_now_iso()
    name = label or created_at.replace(":", "").replace("-", "")
    final_dir = root / name
    working_dir = root / f"{name}{PARTIAL_SUFFIX}"

    for existing in (final_dir, working_dir):
        if existing.exists():
            raise BackupError(
                f"refusing to write {final_dir}: {existing} already exists. An "
                "artifact is never overwritten in place, because a failed dump "
                "would then have destroyed the last good one. Pass a different "
                "--label, or move the existing directory aside."
            )

    working_dir.mkdir(parents=True)
    captures = capture_stores(state, working_dir)
    vault_files = copy_vault(vault, working_dir)
    graph_entry = capture_graph(
        connection,
        working_dir,
        source_uri=source_uri,
        database=database,
        stamps=stamps,
    )

    manifest = BackupManifest(
        layout=BACKUP_LAYOUT,
        layout_version=BACKUP_LAYOUT_VERSION,
        created_at=created_at,
        label=name,
        git_head=read_git_head(),
        stamps=dict(stamps),
        source={
            "repo_root": str(REPO_ROOT),
            "state_root": str(state),
            "vault_root": str(vault),
            "graph_uri": source_uri,
            "graph_database": database,
        },
        files=digest_artifact_files(working_dir),
        stores={c.filename: c.to_manifest_entry() for c in captures},
        graph=graph_entry,
        vault={
            "directory": VAULT_DIRNAME,
            "source_present": vault.is_dir(),
            "file_count": vault_files,
        },
        excluded=list(EXCLUDED_FROM_STATE_ROOT),
    )
    manifest.write(working_dir)
    working_dir.rename(final_dir)

    absent = tuple(c.filename for c in captures if not c.present)
    return DumpReport(
        artifact_dir=final_dir,
        stores_present=sum(1 for c in captures if c.present),
        stores_absent=absent,
        graph_nodes=graph_entry["nodes"],
        graph_relationships=graph_entry["relationships"],
        vault_files=vault_files,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the dump CLI parser. No prompts and no positional arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.backup.dump",
        description=(
            "Capture the SQLite stores, the Neo4j graph and the vault corpus into "
            "one artifact directory outside this working copy."
        ),
        epilog=(
            "Exit codes: 0 artifact written; 2 destination refused, nothing "
            "written; 1 dump failed, the partial directory is left in place."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Backup root directory. Overrides MIST_BACKUP_ROOT, which is otherwise "
            "required -- this tool has no default destination."
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Artifact directory name (default: the UTC timestamp of the run).",
    )
    parser.add_argument(
        "--state-root",
        default=None,
        help=f"Directory holding the SQLite stores (default: {DEFAULT_STATE_ROOT}).",
    )
    parser.add_argument(
        "--vault-root",
        default=None,
        help=f"The vault corpus to copy (default: {DEFAULT_VAULT_ROOT}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log at DEBUG level.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one dump against the configured live stack.

    The destination is resolved BEFORE the graph connection is opened, so a
    misconfigured `MIST_BACKUP_ROOT` costs nothing and refuses immediately.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        backup_root = resolve_backup_root(args.output)
    except BackupDestinationError as exc:
        print(f"[backup] REFUSED: {exc}", file=sys.stderr)
        return EXIT_DESTINATION_REFUSED

    # Imported here, not at module scope, so `--help` and every destination
    # refusal above work without the neo4j driver installed.
    from backend.knowledge.config import get_config
    from backend.knowledge.storage.neo4j_connection import Neo4jConnection

    config = get_config()
    connection = Neo4jConnection(config.neo4j)
    connection.connect()
    try:
        report = run_dump(
            backup_root=backup_root,
            connection=connection,
            source_uri=config.neo4j.uri,
            database=config.neo4j.database,
            stamps=graph_version_stamps(config),
            state_root=None if args.state_root is None else Path(args.state_root),
            vault_root=None if args.vault_root is None else Path(args.vault_root),
            label=args.label,
        )
    except BackupDestinationError as exc:
        print(f"[backup] REFUSED: {exc}", file=sys.stderr)
        return EXIT_DESTINATION_REFUSED
    # `GraphArtifactError` is a `RuntimeError` and not a `MistError`
    # (`grep -n "class GraphArtifactError" backend/knowledge/graph_artifact.py`
    # -> :95), so it needs naming here. It is exactly what the graph leg raises
    # on a value that cannot be round-tripped or a non-finite float -- the
    # commonest way a capture fails -- and without this arm that exits with a
    # traceback instead of the documented code, leaving the `.partial` directory
    # unexplained.
    except (MistError, GraphArtifactError) as exc:
        print(f"[backup] FAILED: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED
    finally:
        connection.disconnect()

    print(
        f"[backup] Wrote {report.artifact_dir} -- "
        f"{report.stores_present} of {len(LIVE_STORE_FILENAMES)} stores, "
        f"{report.graph_nodes} nodes, {report.graph_relationships} relationships, "
        f"{report.vault_files} vault files."
    )
    for filename in report.stores_absent:
        print(f"[backup] WARNING: named store {filename} was not found under the state root.")
    if report.vault_files == 0:
        print("[backup] WARNING: the vault leg captured no files.")
    if report.graph_nodes == 0:
        print("[backup] WARNING: the graph is empty; this artifact restores no nodes.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
