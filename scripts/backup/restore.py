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

STAGE THEN SWAP: THE SIX PHASES
    THE WHOLE ARTIFACT IS READ AND DECODED BEFORE ANYTHING IS TOUCHED, and so is
    the target. A restore runs in six named phases, and every intermediate state
    between them is recorded in `restore.in-progress.json` in the target root:

    1. PREFLIGHT. Read-only, both halves. The artifact's files are re-digested
       (`verify_artifact_files`) and its graph leg is parsed, version-checked and
       decoded (`load_graph_leg`); then the TARGET is checked
       (`assert_target_is_restorable`). Exit 2, target bit-for-bit untouched.
    2. PRE-RESTORE BACKUP. The first consequential step, and the fourth gate.
    3. STAGE. Every store is copied to `<name>.db.incoming` and the vault tree to
       `vault.incoming/`, each a SIBLING of the file or directory it will
       replace. Nothing live is touched, so a failure here costs nothing and
       needs no recovery.
    4. GRAPH. Clear and batched load, through
       `backend.knowledge.admin.restore_graph_from_artifact`. Still the one leg
       that is not atomic: it clears the target as part of the load, so a failure
       leaves a partially loaded graph. It runs BEFORE the swap so that the leg
       which cannot be made atomic fails while the stores and the vault are still
       the target's own.
    5. COMMIT. Per store: `os.replace` onto the live name, then that store's
       `-wal`/`-shm` are unlinked, then the next store. Then the vault:
       `vault` -> `vault.previous`, `vault.incoming` -> `vault`.
    6. CLEANUP. `vault.previous` is removed.

    Phase 1's second artifact check is preflight because of a defect this file
    used to have: `load_artifact` ran LAST, so an artifact whose digests were all
    valid but whose `format_version` this build cannot read replaced the stores
    and the vault and only then refused. Digest validity and decodability are
    different properties.

WHAT "ATOMIC" MEANS HERE, EXACTLY
    `os.replace` is atomic PER STORE AND PER RENAME. Three stores are three
    atomic operations, not one: a crash between them leaves some stores new and
    some old, which is why `restore.in-progress.json` records `stores_committed`
    as it goes.

    The vault swap is TWO renames, because Windows cannot rename a directory onto
    an existing one. There is therefore a window in which `vault/` does not
    exist -- it is `vault.previous/` and `vault.incoming/`. That window is
    irreducible without transactional NTFS, which is deprecated. This is "atomic
    per store and per rename". It is NOT atomic per tree, and nothing in this
    package should ever say that it is.

    Staging is a sibling of its destination on purpose, and that is an INVARIANT
    rather than a runtime check: same-volume is then guaranteed by construction,
    which is the precondition `os.replace` needs to be atomic at all. No flag
    makes the staging location configurable, because such a flag would silently
    take that guarantee away.

    The dump leg already works this way -- it builds under `<label>.partial` and
    renames once the manifest is written
    (`grep -n "RENAMED once the manifest" scripts/backup/dump.py` -> :245) -- and
    the retention leg treats a leftover `.partial` as the detectable signature of
    a dump that died and protects it from pruning
    (`grep -n "ARE PROTECTED BY ARM 3" scripts/backup/prune.py` -> :32).

WHAT IT DOES NOT DO
    It does not reimplement the graph codec or the graph loader. `load_artifact`
    and `restore_graph_from_artifact` are MIS-140 T3 and are called, not copied
    (`grep -n "def restore_graph_from_artifact" backend/knowledge/admin.py` ->
    :1125). It does not write a second destination guard: `resolve_backup_root`
    from T1 decides where the pre-restore backup lands.

Exit codes:
    0  the target was restored
    2  refused -- target, confirmation, destination, graph URI, an artifact that
       failed preflight, a TARGET that failed preflight, or a pre-restore backup
       that failed. In every one of these cases nothing in the target was
       OVERWRITTEN. Not quite the same as "nothing was written": the pre-restore
       capture opens the target's stores, and opening a WAL database creates
       `-shm`/`-wal` sidecars beside it (`tests/unit/backup/test_dump.py:104-118`
       asserts exactly that). No store contents, vault file or graph node changes
       on any exit-2 path.
    1  a phase failed after the pre-restore backup succeeded. The pre-restore
       artifact named in the output is the way back, and
       `<target>/restore.in-progress.json` says which phases had completed. A
       phase-3 failure is the benign case: nothing live was touched.

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
import stat
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from backend.errors import MistError
from backend.interfaces import GraphConnection
from backend.knowledge.admin import (
    assert_artifact_is_relinkable,
    graph_version_stamps,
    read_schema_ddl,
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
    RestoreTargetStateError,
)
from .manifest import BackupManifest, read_manifest, sha256_file, utc_now_iso
from .stores import STORES_DIRNAME
from .target import assert_restore_target_root, assert_target_confirmed

logger = logging.getLogger(__name__)

# Where the vault tree lands under the target root, unless the operator says
# otherwise. Matches the dev-hydration stack, whose backend is configured with
# MIST_VAULT_ROOT=/app/dev-state/vault (`docker-compose.dev-hydration.yml:129`).
DEFAULT_TARGET_VAULT_DIRNAME = "vault"

# Appended to a store filename or the vault directory name while phase 3 writes
# it. Always a SIBLING of the destination, never a path the caller chooses: see
# the invariant in the module docstring.
STAGING_SUFFIX = ".incoming"

# Where the live vault tree is renamed aside during phase 5, and what phase 6
# removes. Its presence in a target is the signature of a restore that died
# between the two vault renames.
VAULT_PREVIOUS_SUFFIX = ".previous"

# Written into the target root for the duration of the consequential sequence.
# Without it, a restore that dies in the graph leg leaves NOTHING on disk saying
# so, and the operator has to remember. The question at 3am is not "can this be
# in a bad state", it is "can I tell what state this is in".
RESTORE_PROGRESS_FILENAME = "restore.in-progress.json"

# Bump when the marker's key set or a key's meaning changes. A reader that does
# not know the version prints the file verbatim rather than interpreting it.
RESTORE_MARKER_VERSION = 1

# Phase 3 writes a second copy of the stores and the vault beside the originals,
# and phase 5's `vault.previous` coexists with the new `vault` until phase 6, so
# the peak occupancy of a restore's data inside the target is about twice the
# artifact's stores plus vault. Requiring that much FREE space is therefore a
# deliberate over-estimate -- only the staged copies are genuinely new bytes,
# the renames that follow cost none -- and the surplus is headroom rather than a
# measured requirement.
_STAGED_PEAK_MULTIPLIER = 2

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2


@dataclass(frozen=True, slots=True)
class RestoreReport:
    """What one restore actually put back, and where the way back is.

    Attributes:
        artifact_dir: The artifact restored FROM.
        target_root: The resolved target restored INTO.
        pre_restore_artifact: The target's state as captured before phase 3.
        stores_restored: Store filenames committed in phase 5, in commit order.
        stores_absent_from_artifact: Stores the manifest records as absent at
            capture time, so the target keeps its own copies of them.
        vault_files: EVERY file in the restored vault tree, `.git` plumbing
            included. On the live corpus most of this number is git objects.
        vault_corpus_files: Files in the restored vault tree with no `.git` path
            segment -- the notes, and the only one of these two numbers that
            answers "did my notes come back".
        graph_deleted: Nodes the phase-4 clear removed from the target.
        graph_nodes: Nodes loaded in phase 4.
        graph_relationships: Relationships loaded in phase 4.
        schema_statements: Artifact DDL statements executed in phase 4.
    """

    artifact_dir: Path
    target_root: Path
    pre_restore_artifact: Path
    stores_restored: tuple[str, ...]
    stores_absent_from_artifact: tuple[str, ...]
    vault_files: int
    vault_corpus_files: int
    graph_deleted: int
    graph_nodes: int
    graph_relationships: int
    schema_statements: int


@dataclass(frozen=True, slots=True)
class RestoreProgressMarker:
    """What `restore.in-progress.json` says, between one phase and the next.

    Frozen, and advanced with `dataclasses.replace`, so a phase transition is one
    expression that produces a new value and one write. A mutable record would
    let a field be updated without the file being rewritten, and a marker that
    disagrees with the disk is worse than no marker.

    The file is rewritten through its own `.tmp` and `os.replace`, so a crash
    during a rewrite leaves the PREVIOUS marker intact rather than a truncated
    one. It is deleted only on success.

    Attributes:
        started_utc: When the consequential sequence began.
        artifact_dir: The artifact being restored FROM.
        artifact_label: That artifact's manifest label.
        target_root: The resolved target.
        pre_restore_artifact: The way back, and the first thing an operator
            reading a stale marker needs.
        staged: Phase 3 completed; staged copies exist and nothing live has
            been touched.
        graph_committed: Phase 4 completed.
        graph_nodes: Nodes loaded by phase 4. 0 until it completes.
        graph_relationships: Relationships loaded by phase 4.
        stores_committed: The store filenames already `os.replace`d onto their
            live names, in commit order. Per store rather than a single flag,
            because `os.replace` is atomic PER STORE and three stores are three
            atomic operations.
        vault_committed: Both vault renames completed.
        vault_previous: Where the target's previous vault tree was renamed
            aside, while it still exists. `None` before phase 5 and after phase
            6 has removed it; a non-`None` value in a stale marker is a tree
            holding the target's own notes.
    """

    started_utc: str
    artifact_dir: str
    artifact_label: str
    target_root: str
    pre_restore_artifact: str
    staged: bool = False
    graph_committed: bool = False
    graph_nodes: int = 0
    graph_relationships: int = 0
    stores_committed: tuple[str, ...] = field(default_factory=tuple)
    vault_committed: bool = False
    vault_previous: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the marker in its on-disk shape."""
        return {
            "marker_version": RESTORE_MARKER_VERSION,
            "started_utc": self.started_utc,
            "artifact_dir": self.artifact_dir,
            "artifact_label": self.artifact_label,
            "target_root": self.target_root,
            "pre_restore_artifact": self.pre_restore_artifact,
            "phases": {
                "staged": self.staged,
                "graph": {
                    "committed": self.graph_committed,
                    "nodes": self.graph_nodes,
                    "relationships": self.graph_relationships,
                },
                "stores_committed": list(self.stores_committed),
                "vault_committed": self.vault_committed,
                "vault_previous": self.vault_previous,
            },
        }

    def write(self, target_root: Path) -> Path:
        """Rewrite the marker in `target_root` through a `.tmp` and `os.replace`.

        Returns:
            The marker path.

        Raises:
            BackupError: When the marker cannot be written. A restore whose
                progress record cannot be kept is one whose intermediate states
                are not nameable, which is the property this whole design buys.
        """
        path = restore_progress_marker_path(target_root)
        staging = path.with_name(f"{path.name}.tmp")
        try:
            staging.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(staging, path)
        except OSError as exc:
            staging.unlink(missing_ok=True)
            raise BackupError(
                f"could not write the restore progress marker {path}: {exc}. The "
                "restore stops here rather than proceeding unrecorded: without this "
                "file a failure in a later phase leaves nothing on disk saying which "
                "phases had completed."
            ) from exc
        return path


def restore_progress_marker_path(target_root: Path | str) -> Path:
    """Return where a restore's progress marker sits for `target_root`.

    Exposed so the refusal, the writer and the tests all name one path rather
    than each joining the filename themselves.
    """
    return Path(target_root) / RESTORE_PROGRESS_FILENAME


def _clear_readonly_and_retry(function: Callable[[str], None], path: str, excinfo: Any) -> None:
    """`shutil.rmtree` error handler: clear the read-only bit, then retry once.

    THE DEFECT THIS EXISTS FOR IS MIS-157, AND IT IS WINDOWS-SPECIFIC. The live
    `mist-memory/` is a git repository with no remote and no upstream, so its
    `.git` is captured deliberately -- those commits exist nowhere else. Git
    marks loose objects read-only (`-r--r--r--`), and on Windows `shutil.rmtree`
    raises `WinError 5` on a read-only file unless the handler clears the
    attribute first. Under stage-then-swap this is off the critical path -- phase
    5 renames the live vault aside rather than removing it -- but phase 6 still
    removes `vault.previous`, so it still bites there.

    `onerror=` and NOT `onexc=`: `shutil.rmtree(onexc=...)` landed in Python
    3.12 and this codebase targets 3.11
    (`grep -n "target-version" pyproject.toml` -> :59,78). The three-argument
    `(function, path, excinfo)` signature is the `onerror=` contract.

    The parent directory is chmod-ed as well as the entry itself, because a
    directory that denies write permission is what blocks an unlink on POSIX,
    while a read-only FILE is what blocks it on Windows. Both routes end in the
    same retry.

    WHAT THE LINUX TEST FOR THIS CAN AND CANNOT PROVE: on Linux a read-only file
    is removable, so the Windows case cannot be reproduced here at all. The
    genuine non-mocked Linux reproduction is a read-only containing DIRECTORY
    (`chmod 0o555`), which raises a real `PermissionError` through this same
    handler. That proves the handler is wired in and that the retry succeeds
    after the attribute is cleared. It does NOT prove the Windows read-only-file
    behaviour; only the host rehearsal does.
    """
    del excinfo
    parent = os.path.dirname(path)
    if os.path.isdir(parent):
        os.chmod(parent, os.stat(parent).st_mode | stat.S_IWUSR | stat.S_IXUSR)
    if os.path.lexists(path):
        os.chmod(path, os.stat(path).st_mode | stat.S_IWUSR)
    function(path)


def remove_tree(path: Path) -> None:
    """Remove a directory tree, clearing read-only entries rather than failing on them.

    The one place in this package that deletes a tree the vault may have written,
    so the MIS-157 handler is wired in here rather than at each call site.
    """
    if not path.exists():
        return
    shutil.rmtree(path, onerror=_clear_readonly_and_retry)


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


@dataclass(frozen=True, slots=True)
class StagedStore:
    """One store copied into the target and not yet committed onto its live name."""

    filename: str
    staged: Path
    destination: Path


def stage_stores(
    artifact_dir: Path, target_root: Path, manifest: BackupManifest
) -> list[StagedStore]:
    """PHASE 3 for the stores: copy each one to `<name>.incoming`. Touches nothing live.

    Driven by the MANIFEST's `stores` map rather than by a glob over the
    artifact, for the same reason the capture leg enumerates names
    (`grep -n "ENUMERATED ON PURPOSE" scripts/backup/stores.py` -> :3): a
    directory listing is whatever happens to be there, and a restore that loads
    whatever happens to be there is how an old `event_store.pre-reset-backup`
    becomes the live event store.

    Every staged file is a SIBLING of the store it will replace, so the
    `os.replace` in phase 5 is a same-volume rename and therefore atomic. That is
    an invariant of the layout, not something checked at runtime.

    Staging is consequence-free by design, which is also why this package has no
    writability probe before the exit-2 boundary: staging IS the writability
    test, and a probe that wrote before that boundary would break the promise
    that exit 2 means nothing was written.

    A failure removes every file staged so far. Leaving megabytes of orphaned
    `.incoming` in the target is the one way staging could cost something, and
    the progress marker -- not the leftovers -- is this leg's durable signature.

    Args:
        artifact_dir: The verified artifact.
        target_root: The resolved, marked target.
        manifest: The artifact's manifest.

    Returns:
        One `StagedStore` per store, in manifest order, ready for `commit_stores`.

    Raises:
        BackupError: When a store the manifest records as present is missing
            from the artifact, or cannot be copied into the target.
    """
    staged: list[StagedStore] = []
    try:
        for filename, entry in manifest.stores.items():
            if not entry.get("present"):
                continue
            source = artifact_dir / STORES_DIRNAME / filename
            if not source.is_file():
                raise BackupError(
                    f"artifact {artifact_dir} records {filename} as present but "
                    f"{source} does not exist. Refusing to continue. Nothing in the "
                    "target has been replaced: this failure happened while staging "
                    "copies beside the live files, so the target still holds its own "
                    "stores, its own vault and its own graph. Restore an intact "
                    "artifact."
                )
            entry_staged = target_root / f"{filename}{STAGING_SUFFIX}"
            try:
                shutil.copyfile(source, entry_staged)
            except OSError as exc:
                raise BackupError(
                    f"could not stage {filename} at {entry_staged}: {exc}. Nothing in "
                    "the target has been replaced. Is the target read-only, or out of "
                    "space?"
                ) from exc
            staged.append(
                StagedStore(
                    filename=filename, staged=entry_staged, destination=target_root / filename
                )
            )
    except BackupError:
        discard_staged_stores(staged)
        raise
    return staged


def discard_staged_stores(staged: Iterable[StagedStore]) -> None:
    """Remove staged store copies that were never committed. Touches nothing live."""
    for entry in staged:
        entry.staged.unlink(missing_ok=True)


def commit_stores(
    staged: Iterable[StagedStore], *, on_commit: Callable[[str], None] | None = None
) -> list[str]:
    """PHASE 5 for the stores: `os.replace` each one, then clear ITS sidecars, then the next.

    ATOMIC PER STORE. Each `os.replace` is atomic on its own; three stores are
    three atomic operations, not one, so a failure at store 2 leaves store 1
    committed and store 3 untouched. `on_commit` fires after each rename so the
    progress marker records exactly which ones landed.

    THE SIDECAR CLEAR BELONGS HERE AND NOT IN PHASE 3, and the reason is
    correctness rather than ordering taste. Deleting the live `-wal` during
    staging would discard committed-but-uncheckpointed frames from the OLD
    database -- damage to the target before this run has committed to replacing
    it, which would make "staging failed, the target is untouched" a lie. See
    `_clear_sqlite_sidecars`.

    Each store's sidecars are unlinked immediately after THAT store's rename
    rather than renaming all three and then unlinking all three, which keeps the
    window in which a restored `.db` sits beside a stale `-wal` to microseconds.
    The staged files themselves carry no sidecars: the artifact's copies have
    none, because `capture_stores` checkpoints and strips them
    (`grep -n "_strip_sqlite_sidecars" scripts/backup/stores.py` -> :67,187).

    Args:
        staged: The output of `stage_stores`.
        on_commit: Called with each filename just after its rename and sidecar
            clear. Injected rather than writing the marker from here, so this
            function stays about renames.

    Returns:
        The filenames committed, in order.

    Raises:
        BackupError: When a rename fails. The stores named in the message are
            already live; the rest are not.
    """
    committed: list[str] = []
    for entry in staged:
        try:
            os.replace(entry.staged, entry.destination)
        except OSError as exc:
            raise BackupError(
                f"could not commit {entry.destination}: {exc}. {len(committed)} "
                f"store(s) {committed} are already the artifact's and the rest are "
                "still the target's own, so this target is a mixture of two points in "
                "time. Is a backend still running against it? Stop it, then use the "
                "pre-restore artifact named above."
            ) from exc
        _clear_sqlite_sidecars(entry.destination)
        committed.append(entry.filename)
        if on_commit is not None:
            on_commit(entry.filename)
    return committed


def count_vault_files(vault_root: Path) -> tuple[int, int]:
    """Count a vault tree twice: corpus notes, and every file including git plumbing.

    TWO NUMBERS BECAUSE ONE OF THEM WAS MISLEADING. The rehearsal reported "117
    vault files" as though that measured the corpus. Measured on the host it was
    13 notes and 104 git objects -- 89% plumbing -- and an operator reading that
    line after a recovery could not tell whether their notes had come back.

    `.git` is captured deliberately and must stay captured: the live
    `mist-memory/` is a git repository with no remote and no upstream, so its
    commits exist nowhere else and excluding them would introduce a new
    data-loss mode. It just must not be counted as notes.

    Args:
        vault_root: The tree to count. A path that does not exist counts as
            `(0, 0)` rather than raising.

    Returns:
        `(corpus, total)`. `corpus` counts files with no `.git` path segment --
        the notes. `total` counts every file in the tree, plumbing included.
    """
    if not vault_root.is_dir():
        return (0, 0)
    corpus = 0
    total = 0
    for path in vault_root.rglob("*"):
        if not path.is_file():
            continue
        total += 1
        if ".git" not in path.relative_to(vault_root).parts:
            corpus += 1
    return (corpus, total)


def stage_vault(artifact_dir: Path, target_vault_root: Path) -> Path | None:
    """PHASE 3 for the vault: copy the artifact's tree to `vault.incoming/`. Nothing live.

    An artifact with NO vault directory returns `None` and leaves the target's
    vault ALONE. Deleting a tree to replace it with nothing is not a restore, and
    `mist-memory/` is absent from every fresh clone -- it is gitignored with
    zero tracked files (`git ls-files mist-memory` -> empty) -- so artifacts
    without a vault leg are ordinary, not exceptional.

    The staged tree is a SIBLING of `target_vault_root`, so phase 5's renames are
    same-volume. That holds even when `--target-vault-root` points outside the
    target root, because the staging name is derived from the destination rather
    than from the target root.

    Returns:
        The staged tree, or `None` when the artifact has no vault leg.

    Raises:
        BackupError: When the tree cannot be staged. Nothing live has changed.
    """
    source = artifact_dir / VAULT_DIRNAME
    if not source.is_dir():
        return None
    staged = target_vault_root.with_name(f"{target_vault_root.name}{STAGING_SUFFIX}")
    try:
        remove_tree(staged)
        shutil.copytree(source, staged)
    except OSError as exc:
        raise BackupError(
            f"could not stage the vault tree at {staged}: {exc}. Nothing in the target "
            "has been replaced: the target still holds its own vault, stores and "
            "graph."
        ) from exc
    return staged


def commit_vault(staged_vault: Path, target_vault_root: Path) -> Path | None:
    """PHASE 5 for the vault: `vault` -> `vault.previous`, then `vault.incoming` -> `vault`.

    TWO RENAMES, NOT ONE, and the window between them is real: `vault/` does not
    exist while it is `vault.previous/` and `vault.incoming/`. Windows cannot
    rename a directory onto an existing one, so a single atomic directory swap is
    unavailable, and the window is irreducible without transactional NTFS --
    which is deprecated. Each rename is atomic; the PAIR is not. Nothing here
    should ever be described as atomic per tree.

    REPLACES rather than merges, which is why the old tree is renamed aside
    rather than merged into: a merge leaves behind every file the target had and
    the artifact did not, producing a corpus that is the union of two vaults and
    equal to neither. The target's tree is already inside the pre-restore
    artifact by the time this runs, and `vault.previous` is a second copy of it
    until phase 6 removes it.

    Returns:
        The path the previous tree was renamed to, or `None` when the target had
        no vault directory to rename aside -- in which case only one rename
        happened and there is nothing for phase 6 to remove.

    Raises:
        BackupError: When either rename fails. The message says which of the two
            windows the target is in.
    """
    previous: Path | None = None
    candidate = target_vault_root.with_name(f"{target_vault_root.name}{VAULT_PREVIOUS_SUFFIX}")
    try:
        if target_vault_root.exists():
            remove_tree(candidate)
            os.replace(target_vault_root, candidate)
            previous = candidate
    except OSError as exc:
        raise BackupError(
            f"could not rename the target's vault {target_vault_root} aside to "
            f"{candidate}: {exc}. The target's own vault is still in place and the "
            f"artifact's is staged at {staged_vault}; no vault file has been lost."
        ) from exc
    try:
        os.replace(staged_vault, target_vault_root)
    except OSError as exc:
        raise BackupError(
            f"could not move the staged vault {staged_vault} onto {target_vault_root}: "
            f"{exc}. The target has NO vault directory right now: its previous tree is "
            f"at {candidate} and the artifact's is at {staged_vault}. Rename one of "
            "them back by hand -- both trees are intact, and this window between the "
            "two renames is the one a directory swap cannot avoid on Windows."
        ) from exc
    return previous


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


def parse_ddl_object_name(statement: str) -> str | None:
    """Pull the object name out of a captured `createStatement`, or return None.

    DELIBERATELY THIS PACKAGE'S OWN PARSER rather than an import from
    `backend/knowledge/admin.py`. The preflight's job is to decide whether that
    module's schema replacement will be able to name every statement it is about
    to execute, and a check that reuses the implementation it is checking cannot
    fail when the implementation is wrong.

    The name follows the CONSTRAINT or INDEX keyword and its POSITION varies:
    Neo4j 5 emits ``CREATE CONSTRAINT `n` FOR ...`` (token 2) but ``CREATE RANGE
    INDEX `n` FOR ...`` and ``CREATE VECTOR INDEX `n` ...`` (token 3). A bare
    keyword only is matched, because an object actually NAMED "index" arrives
    backtick-quoted and must not be read as the keyword.

    Returns:
        The unquoted object name, or `None` when the statement is not a CREATE or
        names nothing this can find.
    """
    parts = statement.split()
    if not parts or parts[0].upper() != "CREATE":
        return None
    for position, token in enumerate(parts):
        if token.upper() in {"CONSTRAINT", "INDEX"} and "`" not in token:
            if position + 1 < len(parts):
                return parts[position + 1].strip("`")
            return None
    return None


def assert_target_graph_answers_schema_reads(connection: GraphConnection) -> dict[str, list[str]]:
    """PREFLIGHT: the target graph answers `SHOW CONSTRAINTS` / `SHOW INDEXES`. Read-only.

    Runs before the pre-restore backup, so a target whose graph is down costs no
    dump. `read_schema_ddl` is called rather than reimplemented -- it issues both
    statements and nothing else (`grep -n "SHOW CONSTRAINTS"
    backend/knowledge/admin.py` -> :889) -- and its result is read here rather
    than acted on.

    Returns:
        The target's own schema DDL, for the caller to log.

    Raises:
        RestoreTargetStateError: When either read fails. Every driver failure
            arrives as a `MistError` subclass, because `Neo4jConnection` wraps
            `Neo4jError` into `Neo4jQueryError` and `Neo4jConnectionError`
            (`grep -n "raise Neo4jQueryError"
            backend/knowledge/storage/neo4j_connection.py` -> :96,119), so this
            arm needs no bare `except Exception`.
    """
    try:
        return read_schema_ddl(connection)
    except MistError as exc:
        raise RestoreTargetStateError(
            f"refusing to restore: the target graph did not answer its schema reads "
            f"({exc.__class__.__name__}: {exc}). The restore stops here, where the "
            "target is still bit-for-bit untouched and no pre-restore backup has been "
            "spent. Start the target graph, check --target-graph-uri, and re-run."
        ) from exc


def assert_artifact_ddl_names_parse(schema: dict[str, list[str]]) -> None:
    """PREFLIGHT: every DDL statement in the artifact has a parseable object name.

    THIS IS THE CHECK FOR THE MIS-140 REHEARSAL'S DIRECT CAUSE. A statement whose
    name cannot be parsed is invisible to every name-keyed decision the schema
    legs make -- it can neither be skipped nor compared -- so it silently becomes
    "always execute", and the server rejects it. Discovering that here costs
    nothing; discovering it during the graph leg cost a half-restored target.

    Read-only: it inspects the already-decoded artifact and touches neither the
    target nor the artifact on disk.

    Args:
        schema: The artifact's `schema` block, `{"constraints": [...],
            "indexes": [...]}`.

    Raises:
        RestoreTargetStateError: When any statement's object name does not parse.
    """
    unparseable = [
        statement
        for statement in list(schema.get("constraints") or []) + list(schema.get("indexes") or [])
        if parse_ddl_object_name(statement) is None
    ]
    if unparseable:
        raise RestoreTargetStateError(
            f"refusing to restore: {len(unparseable)} schema statement(s) in this "
            f"artifact name no object this build can parse: {unparseable[:3]}. A "
            "statement that cannot be named cannot be compared against anything on "
            "the target, so it would be executed unconditionally and rejected part "
            "way through the graph leg. The target is still bit-for-bit untouched. "
            "Use an artifact captured by a build whose schema capture this one reads."
        )


def staged_peak_bytes(manifest: BackupManifest) -> int:
    """How much free space phase 3 needs in the target, sized from the manifest.

    Summed from the manifest's per-file `bytes` for the `stores/` and `vault/`
    legs rather than from a constant, because the figure a constant would encode
    is whatever the vault happened to weigh when it was written. The graph leg is
    excluded: it is written into Neo4j, not into the target root.

    Returns:
        The required free bytes -- see `_STAGED_PEAK_MULTIPLIER` for why it is a
        multiple of the artifact's own size rather than equal to it.
    """
    prefixes = (f"{STORES_DIRNAME}/", f"{VAULT_DIRNAME}/")
    payload = sum(
        int(entry.get("bytes") or 0)
        for relative, entry in manifest.files.items()
        if relative.startswith(prefixes)
    )
    return payload * _STAGED_PEAK_MULTIPLIER


def assert_free_space_for_staging(target_root: Path, manifest: BackupManifest) -> None:
    """PREFLIGHT: the target's volume can hold the staged copies. Read-only.

    `shutil.disk_usage` reads the filesystem's own accounting and writes nothing,
    which is what lets this sit before the exit-2 boundary. There is deliberately
    no write probe here: staging IS this package's writability test, and it is
    consequence-free, so a probe that wrote before the boundary would break the
    promise that exit 2 means nothing was written.

    Raises:
        RestoreTargetStateError: When the free space is below the staged peak, or
            when the volume cannot be measured at all.
    """
    required = staged_peak_bytes(manifest)
    try:
        free = shutil.disk_usage(target_root).free
    except OSError as exc:
        raise RestoreTargetStateError(
            f"refusing to restore: the free space on {target_root} could not be read "
            f"({exc}). The target is still bit-for-bit untouched."
        ) from exc
    if free < required:
        raise RestoreTargetStateError(
            f"refusing to restore into {target_root}: staging needs about "
            f"{required / 1_000_000:.1f} MB free and the volume has "
            f"{free / 1_000_000:.1f} MB. A restore copies the artifact's stores and "
            "vault in beside the live ones before replacing anything, so it needs room "
            "for both at once. Nothing has been written to the target. Free space and "
            "re-run."
        )


def read_restore_progress_marker_text(target_root: Path) -> str | None:
    """Return the raw text of `restore.in-progress.json`, or None when there is none.

    RAW TEXT, not a parsed marker. The refusal prints this verbatim, so a marker
    written by a version this build does not know is still shown in full rather
    than reduced to the fields this build happens to understand.
    """
    path = restore_progress_marker_path(target_root)
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"<{path} exists but could not be read: {exc}>"


def assert_no_restore_in_progress(target_root: Path) -> None:
    """PREFLIGHT: no earlier restore left a progress marker behind. Read-only.

    THE REFUSAL HAS NO OVERRIDE FLAG AND NONE MAY BE ADDED. The operator deletes
    the file by hand. A `--force`-shaped flag would be a new bypass in a tool
    whose whole design is that it cannot be run unattended; a manual delete is
    self-documenting, cannot be scripted into a cron job by accident, and the act
    of deleting IS the acknowledgement.

    The marker's contents are printed in full, including `pre_restore_artifact` --
    the way back from whatever the previous run did, and the next thing the
    operator needs.

    Raises:
        RestoreTargetStateError: When the marker exists. The target is untouched.
    """
    contents = read_restore_progress_marker_text(target_root)
    if contents is None:
        return
    path = restore_progress_marker_path(target_root)
    raise RestoreTargetStateError(
        f"refusing to restore into {target_root}: an earlier restore left "
        f"{path} behind, so it did not finish. Its contents:\n"
        f"{contents}\n"
        "Read `phases` above to see how far that run got, and "
        "`pre_restore_artifact` for the state it captured before starting. This "
        "restore does NOT proceed over a half-applied one, and there is no flag that "
        "makes it: once you have decided what to do about the target, delete "
        f"{path} by hand and re-run. Deleting it is the acknowledgement."
    )


def assert_target_is_restorable(
    *,
    target_root: Path,
    connection: GraphConnection,
    manifest: BackupManifest,
    graph_artifact: dict[str, Any],
) -> None:
    """PHASE 1, target half: four read-only checks, all before the pre-restore backup.

    Ordered AFTER the artifact checks and BEFORE `take_pre_restore_backup`, which
    is the one point where the target is byte-for-byte untouched -- not even a
    `-wal` sidecar exists yet, because the capture has not opened its stores --
    and where a target fault therefore costs no dump.

    Every check reads and none writes, so the exit-2 promise survives all four:

    1. the target graph answers its schema reads;
    2. every DDL statement in the artifact names an object this build can parse;
    3. the volume has room for the staged copies;
    4. no earlier restore left `restore.in-progress.json` behind.

    Raises:
        RestoreTargetStateError: From any of the four. A subclass of
            `RestorePreflightError`, which `main` already treats as exit 2, so
            adding these refusals widened no guard and edited no `except` tuple.
    """
    target_schema = assert_target_graph_answers_schema_reads(connection)
    logger.info(
        "[restore] Target graph answered: %d constraint(s), %d index(es).",
        len(target_schema.get("constraints") or []),
        len(target_schema.get("indexes") or []),
    )
    assert_artifact_ddl_names_parse(graph_artifact.get("schema") or {})
    assert_free_space_for_staging(target_root, manifest)
    assert_no_restore_in_progress(target_root)


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
    graph_writer: Callable[[GraphConnection, dict[str, Any]], dict[str, int]] = write_graph,
    pre_restore_label: str | None = None,
) -> RestoreReport:
    """Restore one artifact into one target, after all four gates pass.

    The ORDER is chosen so that the cheapest and least reversible checks run
    before anything is read or written, so that each refusal is the accurate one,
    and so that EVERY check that can refuse -- of the artifact OR of the target --
    happens before the first byte of the target changes:

        target root resolved and not live -> handshake marker -> graph URI ->
        typed token -> backup destination -> artifact manifest and digests ->
        graph leg decoded and relinkable -> TARGET preflight -> pre-restore
        backup -> [3] stage -> [4] graph -> [5] commit -> [6] cleanup

    Everything left of `pre-restore backup` leaves the target bit-for-bit
    unchanged. Everything from the pre-restore backup onward is recorded in
    `restore.in-progress.json` in the target root, which is deleted only on
    success.

    THE HEADLINE PROPERTY: the graph leg runs BEFORE the swap. It is the one leg
    that clears its target as part of the load and so cannot be made atomic, and
    a failure in it now costs no store and no vault file -- the target keeps its
    own, still staged beside the artifact's copies. That is the MIS-140 rehearsal
    failure, and it is the reason for the phase order.

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
        graph_writer: The phase-4 graph write. Injected for the same reason
            `dump` is -- it is the one leg that cannot be made atomic, so its
            failure path is the one that most needs a test, and there is no other
            way to fail it with the earlier phases already run.
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
        RestoreTargetStateError: The TARGET failed one of the four read-only
            checks in `assert_target_is_restorable`. A subclass of the above, so
            it is already exit 2 and the target is equally untouched.
        RestoreAbortedError: The pre-restore backup failed. Nothing in the
            target was overwritten, though the capture will have opened its
            stores -- see the exit-code note in the module docstring.
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

    assert_target_is_restorable(
        target_root=resolved_target,
        connection=connection,
        manifest=manifest,
        graph_artifact=graph_artifact,
    )

    # PHASE 2. The exit-2 boundary is crossed here and nowhere else.
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

    marker = RestoreProgressMarker(
        started_utc=utc_now_iso(),
        artifact_dir=str(source),
        artifact_label=manifest.label,
        target_root=str(resolved_target),
        pre_restore_artifact=str(pre_restore),
    )
    marker.write(resolved_target)

    # PHASE 3. Copies beside the live files. A failure here costs nothing, so it
    # is the only phase whose cleanup is a plain unlink of its own output.
    staged_stores = stage_stores(source, resolved_target, manifest)
    try:
        staged_vault = stage_vault(source, vault_root)
    except BackupError:
        # `stage_stores` discards its own partial output; the vault leg failing
        # after it succeeded is the case that would otherwise orphan it.
        discard_staged_stores(staged_stores)
        raise
    marker = replace(marker, staged=True)
    marker.write(resolved_target)

    # PHASE 4. The one non-atomic leg, run while the stores and the vault on disk
    # are still the target's own.
    graph = graph_writer(connection, graph_artifact)
    marker = replace(
        marker,
        graph_committed=True,
        graph_nodes=int(graph["nodes"]),
        graph_relationships=int(graph["relationships"]),
    )
    marker.write(resolved_target)

    # PHASE 5. Atomic per store and per rename -- never per tree.
    def record_committed_store(filename: str) -> None:
        nonlocal marker
        marker = replace(marker, stores_committed=marker.stores_committed + (filename,))
        marker.write(resolved_target)

    stores_written = commit_stores(staged_stores, on_commit=record_committed_store)

    vault_previous: Path | None = None
    if staged_vault is not None:
        vault_previous = commit_vault(staged_vault, vault_root)
        marker = replace(
            marker,
            vault_committed=True,
            vault_previous=None if vault_previous is None else str(vault_previous),
        )
        marker.write(resolved_target)

    vault_corpus_files, vault_files = (
        count_vault_files(vault_root) if staged_vault is not None else (0, 0)
    )

    # PHASE 6. The previous tree is redundant once the swap has landed: the
    # pre-restore artifact already holds it.
    if vault_previous is not None:
        remove_tree(vault_previous)
        marker = replace(marker, vault_previous=None)
        marker.write(resolved_target)

    restore_progress_marker_path(resolved_target).unlink(missing_ok=True)

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
        vault_corpus_files=vault_corpus_files,
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
    """Print what was restored, leading with the way back.

    The two vault numbers are printed side by side and each is labelled with what
    it counts. A single "N vault file(s)" line read as a corpus measure is how
    the rehearsal came to report 117 notes when 13 were notes and 104 were git
    objects; an ambiguous label sitting beside a precise one would re-lay that
    trap, so neither number is printed bare.
    """
    print(f"[restore] Pre-restore backup of the target: {report.pre_restore_artifact}")
    print(
        f"[restore] Into {report.target_root}: "
        f"{len(report.stores_restored)} store(s) {list(report.stores_restored)}, "
        f"{report.vault_corpus_files} vault corpus file(s) "
        f"({report.vault_files} including .git plumbing), "
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
