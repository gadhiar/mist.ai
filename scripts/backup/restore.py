"""Put a backup artifact back, into a target that has proved it is not live.

MIS-140 T2, and the leg that makes the dump leg mean something: a backup nobody
has restored is a rumour. It is also the leg where irreversible mistakes live,
so it is deliberately IMPOSSIBLE TO RUN UNATTENDED. Four gates, every run, and
no flag disables any of them:

    1. `--target-root` is REQUIRED. There is no default and none may be added.
       The default in the tool this package replaces is what put graph backups
       inside the live state root
       (`git show ebe1b0d:scripts/mist_admin.py | grep -n 'data/graph_snapshots'`
       -> :427).
    2. A TYPED CONFIRMATION TOKEN equal to the RESOLVED target path
       (`assert_target_confirmed` in `scripts/backup/target.py`).
    3. A POSITIVE HANDSHAKE: the target carries the marker file
       `MIST_RESTORE_TARGET`, placed by the operator
       (`assert_restore_target_root`), on top of -- not instead of --
       `assert_isolated_root` for the root and `assert_neo4j_dev_isolated` for
       the graph URI.
    4. A PRE-RESTORE BACKUP of the target, taken FIRST, through T1's `run_dump`.
       If it fails, nothing is overwritten. The state about to be destroyed is
       captured before it is destroyed, which is the only reason a restore into
       the wrong target is survivable at all.

WHAT IS DESTRUCTIVE, AND IN WHAT ORDER
    THE WHOLE ARTIFACT IS READ AND DECODED BEFORE ANYTHING IS TOUCHED. Two
    separate checks, and both are preflight:

    - every file the manifest names is re-digested (`verify_artifact_files`);
    - the graph leg is parsed, version-checked and decoded, and its relationship
      endpoints resolved (`load_graph_leg`).

    The second check is preflight because of a defect this file used to have:
    `load_artifact` ran LAST, so an artifact whose digests were all valid but
    whose `format_version` this build cannot read replaced the stores and the
    vault and only then refused, leaving a half-restored target. Digest validity
    and decodability are different properties, and only checking the first one
    early was a claim this docstring made that the code did not keep.

    Then, in order: the stores are replaced file by file, the vault tree is
    replaced wholesale, and the already-decoded graph is written last through
    `backend.knowledge.admin.restore_graph_from_artifact`, which detach-deletes
    the target graph before loading. The graph write is last because it is the
    only leg that clears its target as part of the load; putting it first would
    mean a failure in the store leg left a target with neither its old graph nor
    its old stores.

WHAT IT DOES NOT DO
    It does not reimplement the graph codec or the graph loader. `load_artifact`
    and `restore_graph_from_artifact` are MIS-140 T3 and are called, not copied
    (`grep -n "def restore_graph_from_artifact" backend/knowledge/admin.py` ->
    :1125). It does not write a second destination guard: `resolve_backup_root`
    from T1 decides where the pre-restore backup lands.

Exit codes:
    0  the target was restored
    2  refused -- target, confirmation, destination, graph URI, an artifact that
       failed preflight, or a pre-restore backup that failed. In every one of
       these cases NOTHING was written to the target.
    1  a leg failed after the pre-restore backup succeeded. The pre-restore
       artifact named in the output is the way back.

Usage (see `scripts/backup/README.md` for the rehearsal this belongs to, and run
it once without `--confirm-target` to be shown the exact token to type):
    MIST_BACKUP_ROOT=/mnt/backup/mist python -m scripts.backup.restore
        --artifact /mnt/backup/mist/20260917T030000Z
        --target-root ./dev-state
        --target-graph-uri bolt://localhost:7690
        --confirm-target /abs/resolved/path/to/dev-state
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from backend.errors import MistError
from backend.interfaces import GraphConnection
from backend.knowledge.admin import (
    assert_artifact_is_relinkable,
    graph_version_stamps,
    restore_graph_from_artifact,
)
from backend.knowledge.eval_isolation import EvalIsolationError, assert_neo4j_dev_isolated
from backend.knowledge.graph_artifact import GraphArtifactError, load_artifact

from .destination import resolve_backup_root
from .dump import GRAPH_FILENAME, VAULT_DIRNAME, DumpReport, run_dump
from .errors import (
    BackupDestinationError,
    BackupError,
    BackupManifestError,
    RestoreAbortedError,
    RestoreConfirmationError,
    RestorePreflightError,
    RestoreTargetError,
)
from .manifest import BackupManifest, read_manifest, sha256_file, utc_now_iso
from .stores import STORES_DIRNAME
from .target import assert_restore_target_root, assert_target_confirmed

logger = logging.getLogger(__name__)

# Where the vault tree lands under the target root, unless the operator says
# otherwise. Matches the dev-hydration stack, whose backend is configured with
# MIST_VAULT_ROOT=/app/dev-state/vault (`docker-compose.dev-hydration.yml:129`).
DEFAULT_TARGET_VAULT_DIRNAME = "vault"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2


@dataclass(frozen=True, slots=True)
class RestoreReport:
    """What one restore actually put back, and where the way back is."""

    artifact_dir: Path
    target_root: Path
    pre_restore_artifact: Path
    stores_restored: tuple[str, ...]
    stores_absent_from_artifact: tuple[str, ...]
    vault_files: int
    graph_deleted: int
    graph_nodes: int
    graph_relationships: int
    schema_statements: int


def verify_artifact_files(artifact_dir: Path, manifest: BackupManifest) -> None:
    """Re-digest every file the manifest names, before anything is overwritten.

    Ordered before the destructive legs on purpose. A truncated artifact
    discovered half way through a restore leaves the target holding neither its
    old state nor a complete copy of the new; discovered here, it leaves the
    target untouched and the operator free to reach for an older artifact.

    Digest validity is NOT decodability: a file can match its recorded sha256
    exactly and still be an artifact this build cannot read, because the version
    it declares is one this build does not know. `load_graph_leg` is the second
    preflight check for that reason.

    Raises:
        RestorePreflightError: When a named file is missing, or its bytes no
            longer match the digest recorded at capture time.
    """
    missing: list[str] = []
    corrupt: list[str] = []
    for relative, entry in sorted(manifest.files.items()):
        path = artifact_dir / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if sha256_file(path) != entry.get("sha256"):
            corrupt.append(relative)

    if missing or corrupt:
        raise RestorePreflightError(
            f"refusing to restore from {artifact_dir}: "
            f"{len(missing)} file(s) named by the manifest are absent "
            f"({sorted(missing)[:3]}) and {len(corrupt)} no longer match their "
            f"recorded sha256 ({sorted(corrupt)[:3]}). Nothing has been written to "
            "the target. Use a different artifact; this one cannot be trusted to "
            "replace anything."
        )


def _clear_sqlite_sidecars(db_path: Path) -> None:
    """Delete the `-wal`/`-shm` files beside a store that is about to be replaced.

    This is not tidiness. A restored `.db` paired with the TARGET's old `-wal`
    is a database SQLite may open and then apply stale committed frames into --
    a silently wrong store rather than a loud failure. The artifact's copies
    carry no sidecars at all, because `capture_stores` checkpoints and strips
    them (`grep -n "_strip_sqlite_sidecars" scripts/backup/stores.py` -> :67,187),
    so the correct post-restore state is a `.db` with no sidecars beside it.
    """
    for suffix in ("-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)


def restore_stores(artifact_dir: Path, target_root: Path, manifest: BackupManifest) -> list[str]:
    """Replace the target's store files with the artifact's copies.

    Driven by the MANIFEST's `stores` map rather than by a glob over the
    artifact, for the same reason the capture leg enumerates names
    (`grep -n "ENUMERATED ON PURPOSE" scripts/backup/stores.py` -> :3): a
    directory listing is whatever happens to be there, and a restore that loads
    whatever happens to be there is how an old `event_store.pre-reset-backup`
    becomes the live event store.

    Each store is written to a temporary file in the target and then moved onto
    the final name with `os.replace`, so a failure part way through leaves the
    previous file intact rather than a half-written one.

    Args:
        artifact_dir: The verified artifact.
        target_root: The resolved, marked target.
        manifest: The artifact's manifest.

    Returns:
        The filenames actually written, in manifest order.

    Raises:
        BackupError: When a store the manifest records as present is missing
            from the artifact, or cannot be written into the target.
    """
    written: list[str] = []
    for filename, entry in manifest.stores.items():
        if not entry.get("present"):
            continue
        source = artifact_dir / STORES_DIRNAME / filename
        if not source.is_file():
            raise BackupError(
                f"artifact {artifact_dir} records {filename} as present but "
                f"{source} does not exist. Refusing to continue: the target now "
                f"holds {written} from this artifact and its own copy of the rest, "
                "which is a mixture of two points in time. Restore an intact "
                "artifact over it."
            )
        destination = target_root / filename
        staging = target_root / f"{filename}.restore-tmp"
        try:
            shutil.copyfile(source, staging)
            os.replace(staging, destination)
        except OSError as exc:
            staging.unlink(missing_ok=True)
            raise BackupError(
                f"could not write {destination}: {exc}. Is a backend still running "
                "against this target? Stop it and re-run."
            ) from exc
        _clear_sqlite_sidecars(destination)
        written.append(filename)
    return written


def restore_vault(artifact_dir: Path, target_vault_root: Path) -> int:
    """Replace the target's vault tree with the artifact's copy; return the file count.

    REPLACES rather than merges. A merge leaves behind every file the target had
    and the artifact did not, so the restored vault would be a union of two
    corpora and no `mist-memory/` that ever existed. The target's tree is
    already inside the pre-restore artifact by the time this runs.

    An artifact with NO vault directory leaves the target's vault ALONE and
    returns 0. Deleting a tree to replace it with nothing is not a restore, and
    `mist-memory/` is absent from every fresh clone -- it is gitignored with
    zero tracked files (`git ls-files mist-memory` -> empty) -- so artifacts
    without a vault leg are ordinary, not exceptional.

    Raises:
        BackupError: When the tree cannot be replaced.
    """
    source = artifact_dir / VAULT_DIRNAME
    if not source.is_dir():
        return 0
    try:
        if target_vault_root.exists():
            shutil.rmtree(target_vault_root)
        shutil.copytree(source, target_vault_root)
    except OSError as exc:
        raise BackupError(
            f"could not replace the vault tree at {target_vault_root}: {exc}. The "
            "target is now partially restored; the pre-restore artifact taken at the "
            "start of this run holds its previous contents."
        ) from exc
    return sum(1 for path in target_vault_root.rglob("*") if path.is_file())


def load_graph_leg(artifact_dir: Path) -> dict[str, Any]:
    """Parse, version-check and decode the graph leg. PREFLIGHT: writes nothing.

    Runs before the first store is replaced, which is the fix for a defect this
    module shipped with: the decode used to happen last, so an artifact with
    valid digests and an unreadable `format_version` took the stores and the
    vault with it before refusing.

    `assert_artifact_is_relinkable` runs here as well, for the same reason. It
    is also the first statement of `restore_graph_from_artifact`
    (`grep -n "assert_artifact_is_relinkable(artifact)"
    backend/knowledge/admin.py` -> :1160), so calling it here does not replace
    that check -- it moves the same refusal to before the target is touched
    rather than after.

    Every failure is translated to `RestorePreflightError` because
    `GraphArtifactError` is a `RuntimeError`, not a `MistError`, and would
    otherwise escape this package's `except` arms as a raw traceback.

    Returns:
        The decoded artifact, ready for `write_graph`.

    Raises:
        RestorePreflightError: When the graph file is absent, is not JSON,
            declares a format or version this build does not read, carries a
            value it cannot reconstruct, or has an endpoint no node provides.
    """
    path = artifact_dir / GRAPH_FILENAME
    if not path.is_file():
        raise RestorePreflightError(
            f"artifact {artifact_dir} has no {GRAPH_FILENAME}. The graph is the leg "
            "that cannot be rebuilt from anything else on disk, so a restore does "
            "not proceed without it. Nothing has been written to the target."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RestorePreflightError(
            f"{path} could not be read as JSON: {exc}. Nothing has been written to " "the target."
        ) from exc
    try:
        artifact = load_artifact(payload)
        assert_artifact_is_relinkable(artifact)
    except GraphArtifactError as exc:
        raise RestorePreflightError(
            f"refusing to restore from {artifact_dir}: its graph leg cannot be "
            f"loaded by this build ({exc}). Nothing has been written to the target, "
            "and its stores, vault and graph are untouched. A digest check alone "
            "would not have caught this: the file is intact and this build cannot "
            "read it."
        ) from exc
    return artifact


def write_graph(connection: GraphConnection, artifact: dict[str, Any]) -> dict[str, int]:
    """Write an ALREADY-DECODED artifact into `connection`, replacing what is there.

    DESTRUCTIVE, and the last leg of a restore for that reason. Calls MIS-140 T3
    and adds nothing: `restore_graph_from_artifact` detach-deletes the target and
    loads. The isolation guard is the CALLER's job by that function's own
    docstring (`grep -n "The caller is responsible for the isolation guard"
    backend/knowledge/admin.py` -> :1131), and `run_restore` is where it runs.

    Returns:
        `{"deleted", "schema_statements", "nodes", "relationships"}`.

    Raises:
        GraphArtifactError: When the server creates a number of relationships
            not equal to the batch. That is a genuine part-way failure, so it is
            NOT translated into a preflight error: the target really is
            partially loaded at that point, and `main` exits 1.
    """
    return restore_graph_from_artifact(connection, artifact)


def take_pre_restore_backup(
    *,
    target_root: Path,
    target_vault_root: Path,
    connection: GraphConnection,
    target_graph_uri: str,
    database: str | None,
    stamps: dict[str, str],
    backup_root: Path,
    label: str | None = None,
    dump: Callable[..., DumpReport] = run_dump,
) -> Path:
    """Capture the target's current state through T1's dump leg, or refuse the restore.

    Args:
        target_root: The resolved target; also the state root of the capture.
        target_vault_root: The target's vault tree.
        connection: The TARGET's graph connection, read from here.
        target_graph_uri: Recorded in the captured graph artifact.
        database: Recorded in the captured graph artifact.
        stamps: Recorded, never enforced.
        backup_root: Where the pre-restore artifact lands. Already guarded by
            `resolve_backup_root`, and re-checked inside `run_dump`.
        label: Artifact name. Defaults to `pre-restore-<UTC timestamp>`.
        dump: The capture function. Injected so the failure path is testable
            without breaking a real dump, per this repository's
            no-hidden-construction rule.

    Returns:
        The finished pre-restore artifact directory.

    Raises:
        RestoreAbortedError: When the capture fails for ANY reason the dump leg
            can raise. The restore then does not proceed -- a restore whose
            safety net failed is a one-way door, and the operator has not agreed
            to one.

            Both arms are needed for that "any" to be true. `MistError` covers
            this package's own failures; `GraphArtifactError` is a `RuntimeError`
            (`grep -n "class GraphArtifactError"
            backend/knowledge/graph_artifact.py` -> :95) and is exactly what the
            graph leg of a capture raises, so without the second arm the
            commonest capture failure escaped this translation, printed a
            traceback, and exited with the code documented as "the pre-restore
            artifact is the way back" -- when no such artifact existed.
    """
    name = label or f"pre-restore-{utc_now_iso().replace(':', '').replace('-', '')}"
    try:
        report = dump(
            backup_root=backup_root,
            connection=connection,
            source_uri=target_graph_uri,
            database=database,
            stamps=dict(stamps),
            state_root=target_root,
            vault_root=target_vault_root,
            label=name,
        )
    except (MistError, GraphArtifactError) as exc:
        raise RestoreAbortedError(
            f"REFUSING TO RESTORE: the pre-restore backup of {target_root} failed "
            f"({exc.__class__.__name__}: {exc}). Nothing has been overwritten. This "
            "restore replaces the target's stores, its vault and its whole graph, "
            "and without that capture there is no way back from a mistake. Fix the "
            f"capture first -- check the destination {backup_root} is writable and "
            "that no other run is using this label -- then re-run."
        ) from exc
    return report.artifact_dir


def run_restore(
    *,
    artifact_dir: Path | str,
    target_root: Path | str,
    target_graph_uri: str,
    connection: GraphConnection,
    confirm_token: str | None,
    stamps: dict[str, str],
    backup_root: Path | str | None = None,
    database: str | None = None,
    target_vault_root: Path | str | None = None,
    dump: Callable[..., DumpReport] = run_dump,
    pre_restore_label: str | None = None,
) -> RestoreReport:
    """Restore one artifact into one target, after all four gates pass.

    The gate ORDER is chosen so that the cheapest and least reversible checks
    run before anything is read or written, so that each refusal is the accurate
    one, and so that EVERY check that can refuse an artifact happens before the
    first byte of the target changes:

        target root resolved and not live -> handshake marker -> graph URI ->
        typed token -> backup destination -> artifact manifest and digests ->
        graph leg decoded and relinkable -> pre-restore backup -> stores ->
        vault -> graph write

    Everything left of `pre-restore backup` leaves the target bit-for-bit
    unchanged.

    Args:
        artifact_dir: The backup artifact to restore FROM.
        target_root: The state root to restore INTO. Never defaulted.
        target_graph_uri: The target's bolt URI. Passed through
            `assert_neo4j_dev_isolated` unchanged, so only dev endpoints pass.
        connection: A connection to the TARGET graph. Written to.
        confirm_token: What the operator typed. Must equal the resolved target.
        stamps: Recorded in the pre-restore artifact.
        backup_root: Where the pre-restore artifact lands. Defaults to
            `MIST_BACKUP_ROOT` through T1's guard.
        database: The target's database name, recorded in the pre-restore
            artifact.
        target_vault_root: The target's vault tree. Defaults to
            `<target-root>/vault`.
        dump: The capture function used for the pre-restore backup.
        pre_restore_label: Name for the pre-restore artifact.

    Returns:
        A `RestoreReport` naming the pre-restore artifact, which is the way back.

    Raises:
        RestoreTargetError: The target is live, absent, or unmarked.
        EvalIsolationError: The graph URI is not a dev endpoint. Deliberately
            NOT wrapped: it is the shared guard's refusal and reads better in
            its own words.
        RestoreConfirmationError: The token is absent or not the resolved path.
        BackupDestinationError: The pre-restore destination is unset or refused.
        BackupManifestError: The artifact has no manifest, or one this build
            cannot read.
        RestorePreflightError: The artifact failed a digest, version, decode or
            re-anchoring check. The target is untouched.
        RestoreAbortedError: The pre-restore backup failed. The target is
            untouched.
        BackupError: A leg failed after the pre-restore backup succeeded.
        GraphArtifactError: The graph WRITE failed part way, leaving a partially
            loaded graph. Not translated, because unlike every entry above it,
            this one does not mean the target is untouched.
    """
    resolved_target = assert_restore_target_root(target_root)
    assert_neo4j_dev_isolated(target_graph_uri)
    assert_target_confirmed(resolved_target, confirm_token)

    resolved_root = (
        resolve_backup_root(purpose="pre-restore backup")
        if backup_root is None
        else resolve_backup_root(backup_root, purpose="pre-restore backup")
    )
    vault_root = (
        resolved_target / DEFAULT_TARGET_VAULT_DIRNAME
        if target_vault_root is None
        else Path(target_vault_root)
    )

    source = Path(artifact_dir)
    manifest = read_manifest(source)
    verify_artifact_files(source, manifest)
    # Decoded here, written at the very end. Both halves of the graph leg used
    # to happen after the stores were replaced, which made every version and
    # decode refusal a half-restore.
    graph_artifact = load_graph_leg(source)

    pre_restore = take_pre_restore_backup(
        target_root=resolved_target,
        target_vault_root=vault_root,
        connection=connection,
        target_graph_uri=target_graph_uri,
        database=database,
        stamps=stamps,
        backup_root=resolved_root,
        label=pre_restore_label,
        dump=dump,
    )
    logger.info("[restore] Pre-restore backup of the target is at %s", pre_restore)

    stores_written = restore_stores(source, resolved_target, manifest)
    vault_files = restore_vault(source, vault_root)
    graph = write_graph(connection, graph_artifact)

    absent = tuple(
        filename for filename, entry in manifest.stores.items() if not entry.get("present")
    )
    return RestoreReport(
        artifact_dir=source,
        target_root=resolved_target,
        pre_restore_artifact=pre_restore,
        stores_restored=tuple(stores_written),
        stores_absent_from_artifact=absent,
        vault_files=vault_files,
        graph_deleted=int(graph["deleted"]),
        graph_nodes=int(graph["nodes"]),
        graph_relationships=int(graph["relationships"]),
        schema_statements=int(graph["schema_statements"]),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the restore CLI parser.

    `--target-root` and `--confirm-target` have no defaults and never will. The
    epilog states the four gates, because the operator reading `--help` at 3am
    is deciding whether to run this at all.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.backup.restore",
        description=(
            "DESTRUCTIVE: replace a target's SQLite stores, vault tree and entire "
            "graph with the contents of a backup artifact."
        ),
        epilog=(
            "Four gates run every time and no flag disables any of them: "
            "--target-root is required; the target must carry the marker file "
            "MIST_RESTORE_TARGET; the resolved target path must be typed back via "
            "--confirm-target; and a pre-restore backup of the target is taken "
            "first, which must succeed. Exit codes: 0 restored; 2 refused, target "
            "untouched; 1 a leg failed after the pre-restore backup was taken."
        ),
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="Backup artifact directory to restore FROM (the one holding manifest.json).",
    )
    parser.add_argument(
        "--target-root",
        required=True,
        help=(
            "State root to restore INTO. REQUIRED, no default. Must carry the "
            "MIST_RESTORE_TARGET marker file and must not be live state."
        ),
    )
    parser.add_argument(
        "--target-graph-uri",
        required=True,
        help=(
            "Bolt URI of the target graph, e.g. bolt://localhost:7690 for the "
            "dev-hydration stack. Checked by assert_neo4j_dev_isolated."
        ),
    )
    parser.add_argument(
        "--confirm-target",
        default=None,
        help=(
            "The RESOLVED absolute target path, typed back. Run without it once to "
            "be shown the exact string to pass."
        ),
    )
    parser.add_argument(
        "--target-vault-root",
        default=None,
        help="Target vault tree (default: <target-root>/vault).",
    )
    parser.add_argument(
        "--backup-root",
        default=None,
        help=(
            "Where the pre-restore backup lands. Overrides MIST_BACKUP_ROOT, which "
            "is otherwise required -- there is no default destination."
        ),
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Target database name, recorded in the pre-restore artifact.",
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level.")
    return parser


def _print_report(report: RestoreReport) -> None:
    """Print what was restored, leading with the way back."""
    print(f"[restore] Pre-restore backup of the target: {report.pre_restore_artifact}")
    print(
        f"[restore] Into {report.target_root}: "
        f"{len(report.stores_restored)} store(s) {list(report.stores_restored)}, "
        f"{report.vault_files} vault file(s), "
        f"{report.graph_nodes} nodes and {report.graph_relationships} relationships "
        f"(deleted {report.graph_deleted} pre-existing nodes, applied "
        f"{report.schema_statements} schema statements)."
    )
    for filename in report.stores_absent_from_artifact:
        print(
            f"[restore] WARNING: {filename} was absent when this artifact was taken, "
            "so the target keeps its own copy of it."
        )


def main(argv: list[str] | None = None) -> int:
    """Run one restore. Every refusal prints and returns 2 with the target untouched."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # The URI guard runs FIRST, before the backend is even imported: a mistyped
    # target then costs nothing, receives no connection attempt, and its refusal
    # does not depend on the neo4j driver being installed.
    try:
        assert_neo4j_dev_isolated(args.target_graph_uri)
    except EvalIsolationError as exc:
        print(f"[restore] REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    from backend.knowledge.config import get_config
    from backend.knowledge.storage.neo4j_connection import Neo4jConnection

    config = get_config()
    connection: Any = None
    try:
        connection = Neo4jConnection(replace(config.neo4j, uri=args.target_graph_uri))
        connection.connect()
        report = run_restore(
            artifact_dir=Path(args.artifact),
            target_root=Path(args.target_root),
            target_graph_uri=args.target_graph_uri,
            connection=connection,
            confirm_token=args.confirm_target,
            stamps=graph_version_stamps(config),
            backup_root=args.backup_root,
            database=args.database,
            target_vault_root=(
                None if args.target_vault_root is None else Path(args.target_vault_root)
            ),
        )
    # Exit 2 is a PROMISE that the target is untouched, so this tuple holds
    # exactly the failures raised before the first store is replaced -- which
    # now includes every artifact check, because the graph leg is decoded during
    # preflight rather than written first and checked later.
    except (
        RestoreTargetError,
        RestoreConfirmationError,
        BackupDestinationError,
        BackupManifestError,
        RestorePreflightError,
        RestoreAbortedError,
    ) as exc:
        print(f"[restore] REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except EvalIsolationError as exc:
        print(f"[restore] REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    # `GraphArtifactError` is a `RuntimeError`, not a `MistError`, so it needs
    # its own name here or it escapes as a traceback. Reaching this arm means
    # the graph WRITE failed part way; the target is partially restored and the
    # pre-restore artifact is the way back, which is what exit 1 documents.
    except (MistError, GraphArtifactError) as exc:
        print(f"[restore] FAILED: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED
    finally:
        if connection is not None:
            connection.disconnect()

    _print_report(report)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
