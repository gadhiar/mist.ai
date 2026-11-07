"""Run backups on a schedule inside the compose project, and refuse to let a bad dump prune good ones.

Plan v1, task T6; the contract is Interfaces and contracts item 6. The
`mist-backup` service runs `python -m scripts.backup.scheduler`. It checks every
`MIST_BACKUP_CHECK_MINUTES` and takes a backup when there is no recorded
success or the last one is older than `MIST_BACKUP_INTERVAL_HOURS`. A schedule
of "last success plus the interval", rather than a wall-clock time, is what
catches up after the host slept or the stack was down.

ONE ATTEMPT.
    1. `scripts.backup.dump.main` runs in-process with `--label <UTC timestamp>
       --state-root /app/data --vault-root /app/mist-memory`. The backup root
       comes from `MIST_BACKUP_ROOT`, which the dump reads and guards itself.
    2. Exit 2 is `refused`, any other non-zero code `failed`. Either waits
       `MIST_BACKUP_RETRY_MINUTES` before the next attempt.
    3. Exit 0 is judged by the new artifact's manifest (`read_manifest`). The
       success rule: `graph.nodes` > 0, `vault.corpus_file_count` > 0, every
       store the last success held is present, and `graph.nodes` is at least
       `MIST_BACKUP_MIN_NODE_RATIO` times the last success's.
    4. A success runs `scripts.backup.prune.main` with `--retain
       $MIST_BACKUP_RETAIN --confirm`.
    5. The outcome is written to `MIST_BACKUP_STATUS_PATH` (`status.py`), which
       `BackupProbe` reads for `/health`.

A STORE IS PRESENT WHEN ITS `file` IS NOT NULL. The manifest's `stores` map
names every live store whether or not it was captured, with `file` null for an
absent one (`grep -n '"file": None if' scripts/backup/stores.py` -> :86), so a
key-membership check would pass a dump that lost a store.

WHY A DUMP THAT FAILS THE RULE IS MOVED ASIDE. Prune keeps the newest
`--retain` artifacts by manifest date, so seven empty dumps in a row, each
followed by prune, would delete seven good ones. A `suspect` artifact is moved
with `os.replace` to `<backup root>/suspect/<label>` and prune does not run.
Prune never looks inside `suspect/`: it considers only the root's immediate
children (`grep -n "root.iterdir" scripts/backup/prune.py` -> :122), and
`suspect/` itself has no manifest. Only the newest suspect is kept, because the
older ones were never successes. The next attempt waits the full interval, not
the retry window, so a lasting shrink costs one dump a day rather than one an
hour. `--accept-suspect LABEL` makes a reviewed suspect the new baseline.

A dump that exits 0 with a manifest `read_manifest` refuses is also `suspect`:
item 6 makes a success require a readable manifest that meets the rule. A
failed rule is recorded as `suspect` even when the move into `suspect/` fails.
In that case the artifact is deleted from the backup root instead, because it
was never a success and a later prune would count it as a retained backup. If
that deletion fails too, prune is skipped for as long as the directory remains
in the root.

WHAT AN ATTEMPT CATCHES. `MistError` (which covers `BackupError`, and the
`Neo4jConnectionError` that the dump's `connection.connect()` raises outside its
own handling, `grep -n "connection.connect()" scripts/backup/dump.py` -> :529),
`GraphArtifactError` (a `RuntimeError`, not a `MistError`:
`grep -n "class GraphArtifactError" backend/knowledge/graph_artifact.py` -> :95),
`OSError` and `sqlite3.Error`. Each is recorded as `failed` and the loop keeps
running. Anything else is a bug, and it propagates so the container restarts
with a traceback rather than looping over it.

SIGTERM sets a flag, and the loop exits 0 once the current attempt finishes.
That covers only what the process is given time to do. Docker kills a container
that has not exited within its stop grace period (10 s unless the compose file
sets `stop_grace_period`), so a dump still running at that point dies part
way. What it leaves behind is a `<label>.partial` directory with no manifest:
the dump builds the artifact there (`grep -n "working_dir.mkdir"
scripts/backup/dump.py` -> :362) and renames it only after the manifest is
written (`grep -n "working_dir.rename(final_dir)" scripts/backup/dump.py` ->
:407). Prune never counts or deletes it, because `is_backup_artifact_dir`
answers False without a `manifest.json` (`grep -n "manifest_path.is_file()"
scripts/backup/manifest.py` -> :259) and `scan_artifacts` skips any directory
that fails that predicate (`grep -n "if not is_backup_artifact_dir(child)"
scripts/backup/prune.py` -> :123). The status file still shows the attempt
before the killed one, so the next start makes a new attempt under a new
label. The partial stays until an operator deletes it, as the backup README's
retention section describes.

Exit codes:
    loop             0 after SIGTERM; 2 when the configuration is refused
    --once           0 success, the dump's code for a failure (1 when the
                     attempt raised), 3 suspect
    --accept-suspect 0 accepted, 2 refused with nothing changed, 1 failed
                     part way
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import signal
import sqlite3
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType

from backend.errors import MistError
from backend.knowledge.graph_artifact import GraphArtifactError

from . import dump as backup_dump
from . import prune as backup_prune
from .destination import BACKUP_ROOT_ENV, assert_backup_destination
from .errors import BackupError, BackupManifestError
from .manifest import BackupManifest, is_backup_artifact_dir, read_manifest
from .status import (
    RESULT_FAILED,
    RESULT_OK,
    RESULT_REFUSED,
    RESULT_SUSPECT,
    BackupStatus,
    ResultName,
    SuccessCounts,
    read_status,
    write_status,
)

logger = logging.getLogger(__name__)

DumpEntry = Callable[[list[str]], int]
PruneEntry = Callable[[list[str]], int]
Clock = Callable[[], datetime]
Sleeper = Callable[[float], None]

STATUS_PATH_ENV = "MIST_BACKUP_STATUS_PATH"
INTERVAL_HOURS_ENV = "MIST_BACKUP_INTERVAL_HOURS"
RETAIN_ENV = "MIST_BACKUP_RETAIN"
CHECK_MINUTES_ENV = "MIST_BACKUP_CHECK_MINUTES"
RETRY_MINUTES_ENV = "MIST_BACKUP_RETRY_MINUTES"
MIN_NODE_RATIO_ENV = "MIST_BACKUP_MIN_NODE_RATIO"

# The defaults equal the compose file's (plan v1, Interfaces and contracts
# item 1), so a variable dropped from the environment changes nothing.
DEFAULT_INTERVAL_HOURS = 24.0
DEFAULT_RETAIN = 7
DEFAULT_CHECK_MINUTES = 10.0
DEFAULT_RETRY_MINUTES = 60.0
DEFAULT_MIN_NODE_RATIO = 0.5

# The mist-backup container's mounts of the instance's state and vault.
DEFAULT_STATE_ROOT = Path("/app/data")
DEFAULT_VAULT_ROOT = Path("/app/mist-memory")

SUSPECT_DIRNAME = "suspect"

# The dump's own default label shape (`scripts/backup/dump.py:349`), from the
# scheduler's clock so tests can pin it.
LABEL_FORMAT = "%Y%m%dT%H%M%SZ"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_SUSPECT = 3

ATTEMPT_ERRORS: tuple[type[BaseException], ...] = (
    MistError,
    GraphArtifactError,
    OSError,
    sqlite3.Error,
)

# How often the default sleeper wakes to check for SIGTERM. Python resumes an
# interrupted `time.sleep` after the handler returns (PEP 475), so without
# slicing a stop during a ten-minute wait would wait out the ten minutes and be
# killed by Docker's stop timeout first.
_STOP_POLL_SECONDS = 1.0


class SchedulerConfigError(BackupError):
    """Raised when the scheduler's environment is missing a value or holds an invalid one."""


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """The scheduler's settings, normally read from the mist-backup environment.

    Attributes:
        backup_root: `MIST_BACKUP_ROOT`, or None when unset. None is not refused
            here: the dump refuses it with exit 2, and recording that as
            `refused` tells `/health` more than a scheduler that never starts.
        status_path: Where the status file is written; required.
        interval: How old the last success may get before the next dump.
        check_interval: How often the loop checks whether a dump is due.
        retry: The least wait after a failed or refused attempt.
        retain: Passed to prune as `--retain`.
        min_node_ratio: The share of the last success's graph nodes a new dump
            must hold to count as a success.
        state_root: Passed to the dump as `--state-root`.
        vault_root: Passed to the dump as `--vault-root`.
    """

    backup_root: Path | None
    status_path: Path
    interval: timedelta
    check_interval: timedelta
    retry: timedelta
    retain: int
    min_node_ratio: float
    state_root: Path = DEFAULT_STATE_ROOT
    vault_root: Path = DEFAULT_VAULT_ROOT

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> SchedulerConfig:
        """Build from the environment of plan v1, Interfaces and contracts item 1.

        Raises:
            SchedulerConfigError: When `MIST_BACKUP_STATUS_PATH` is unset, or a
                number is unparsable or out of range. Without a status path no
                outcome could be recorded, so the scheduler refuses to start.
        """
        status_raw = environ.get(STATUS_PATH_ENV, "").strip()
        if not status_raw:
            raise SchedulerConfigError(
                f"{STATUS_PATH_ENV} is not set. The scheduler records every attempt "
                "there for /health, and runs no backup it cannot record."
            )
        root_raw = environ.get(BACKUP_ROOT_ENV, "").strip()
        ratio = _env_float(environ, MIN_NODE_RATIO_ENV, DEFAULT_MIN_NODE_RATIO, allow_zero=True)
        if ratio > 1:
            raise SchedulerConfigError(
                f"{MIN_NODE_RATIO_ENV}={ratio} is above 1, which would make every dump "
                "that does not grow the graph suspect."
            )
        return cls(
            backup_root=Path(root_raw) if root_raw else None,
            status_path=Path(status_raw),
            interval=timedelta(
                hours=_env_float(environ, INTERVAL_HOURS_ENV, DEFAULT_INTERVAL_HOURS)
            ),
            check_interval=timedelta(
                minutes=_env_float(environ, CHECK_MINUTES_ENV, DEFAULT_CHECK_MINUTES)
            ),
            retry=timedelta(minutes=_env_float(environ, RETRY_MINUTES_ENV, DEFAULT_RETRY_MINUTES)),
            retain=_env_int(environ, RETAIN_ENV, DEFAULT_RETAIN),
            min_node_ratio=ratio,
        )


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    """What one attempt did.

    Attributes:
        label: The artifact label the attempt passed to the dump.
        result: `ok`, `failed`, `refused` or `suspect`.
        exit_code: The dump's exit code, recorded as `last_exit_code`; 1 when
            the attempt raised before the dump returned one.
    """

    label: str
    result: ResultName
    exit_code: int

    @property
    def cli_exit_code(self) -> int:
        """The `--once` exit code: 0 success, 3 suspect, the dump's code otherwise."""
        if self.result == RESULT_OK:
            return EXIT_OK
        if self.result == RESULT_SUSPECT:
            return EXIT_SUSPECT
        return self.exit_code if self.exit_code != EXIT_OK else EXIT_FAILED


class StopFlag:
    """Set by the SIGTERM handler; checked between attempts and during the sleep.

    A plain attribute rather than `threading.Event`: `Event.set` takes a
    non-reentrant lock that `Event.wait` holds briefly, so calling it from a
    signal handler that interrupts the main thread inside `wait` can deadlock.
    Assigning an attribute takes no lock.
    """

    __slots__ = ("_requested",)

    def __init__(self) -> None:
        self._requested = False

    def request(self) -> None:
        """Ask the loop to exit after the current attempt."""
        self._requested = True

    def is_set(self) -> bool:
        """True once a stop has been requested."""
        return self._requested

    def wait(self, seconds: float, *, sleep: Sleeper = time.sleep) -> None:
        """Sleep up to `seconds`, returning within one poll interval of a stop request."""
        remaining = seconds
        while remaining > 0 and not self._requested:
            step = min(_STOP_POLL_SECONDS, remaining)
            sleep(step)
            remaining -= step


def install_sigterm_handler(
    stop: StopFlag,
) -> Callable[[int, FrameType | None], object] | int | None:
    """Make SIGTERM request a stop; return the previous handler so a caller can restore it."""

    def _request_stop(_signum: int, _frame: FrameType | None) -> None:
        stop.request()

    return signal.signal(signal.SIGTERM, _request_stop)


def utc_now() -> datetime:
    """The production clock: an aware UTC datetime."""
    return datetime.now(UTC)


def present_stores(manifest: BackupManifest) -> tuple[str, ...]:
    """The stores whose manifest entry has a non-null `file`, sorted. Never by key membership."""
    return tuple(
        sorted(
            name
            for name, entry in manifest.stores.items()
            if isinstance(entry, dict) and entry.get("file") is not None
        )
    )


def success_counts(manifest: BackupManifest) -> SuccessCounts:
    """The counts the status file keeps from a success, for the next attempt's comparison.

    A count that is missing or not a non-negative integer reads as 0, which the
    success rule then refuses: an artifact that cannot state what it holds is
    not evidence of a good backup.
    """
    return SuccessCounts(
        graph_nodes=_count(manifest.graph.get("nodes")),
        vault_corpus_files=_count(manifest.vault.get("corpus_file_count")),
        stores=present_stores(manifest),
    )


def suspect_reasons(
    manifest: BackupManifest,
    baseline: SuccessCounts | None,
    min_node_ratio: float,
) -> tuple[str, ...]:
    """Apply item 6's success rule; return why it failed, or an empty tuple for a success.

    Args:
        manifest: The new artifact's manifest.
        baseline: The last success's counts, or None before the first success,
            when only the two emptiness checks apply.
        min_node_ratio: `MIST_BACKUP_MIN_NODE_RATIO`.
    """
    counts = success_counts(manifest)
    reasons: list[str] = []
    if counts.graph_nodes <= 0:
        reasons.append("the graph leg holds no nodes")
    if counts.vault_corpus_files <= 0:
        reasons.append("the vault leg holds no corpus files")
    if baseline is not None:
        lost = [name for name in baseline.stores if name not in counts.stores]
        if lost:
            reasons.append(f"stores the last success held have no file: {lost}")
        floor = min_node_ratio * baseline.graph_nodes
        if counts.graph_nodes < floor:
            reasons.append(
                f"{counts.graph_nodes} graph nodes is below {min_node_ratio} x the last "
                f"success's {baseline.graph_nodes}"
            )
    return tuple(reasons)


def next_attempt_due(status: BackupStatus | None, config: SchedulerConfig) -> datetime | None:
    """When the next attempt is due; None means now.

    The due time is the latest of: the last success plus the interval; after a
    `suspect`, that attempt plus the interval; after a `failed` or `refused`,
    that attempt plus the retry window. A missing or corrupt status file has no
    success, so it is due now.
    """
    if status is None:
        return None
    candidates: list[datetime] = []
    if status.last_success_at is not None:
        candidates.append(status.last_success_at + config.interval)
    if status.last_attempt_at is not None:
        if status.last_result == RESULT_SUSPECT:
            candidates.append(status.last_attempt_at + config.interval)
        elif status.last_result in (RESULT_FAILED, RESULT_REFUSED):
            candidates.append(status.last_attempt_at + config.retry)
    return max(candidates) if candidates else None


def quarantine(artifact_dir: Path, backup_root: Path) -> Path:
    """Move a suspect artifact to `<backup root>/suspect/<label>` with `os.replace`.

    Returns:
        The artifact's new path.

    Raises:
        OSError: When `suspect/` cannot be created or the move fails. The
            artifact is then still in the backup root.
    """
    suspect_root = backup_root / SUSPECT_DIRNAME
    suspect_root.mkdir(exist_ok=True)
    destination = suspect_root / artifact_dir.name
    os.replace(artifact_dir, destination)
    return destination


def delete_older_suspects(suspect_root: Path, *, keep: str) -> None:
    """Delete every directory in `suspect/` except `keep`.

    Called only after the newest suspect has been moved in, so a failure part
    way leaves an extra suspect rather than none. Only directories are deleted,
    and never through a symlink; any other entry is left alone.

    Raises:
        OSError: When a deletion fails.
    """
    for child in suspect_root.iterdir():
        if child.name == keep:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)


def count_artifacts(backup_root: Path) -> int:
    """How many backup artifacts sit directly under the root, by prune's own predicate."""
    return sum(1 for child in backup_root.iterdir() if is_backup_artifact_dir(child))


class BackupScheduler:
    """The loop, one attempt, and the decision between them.

    Every dependency on time or on another module's side effects is injected:
    the clock, the sleeper, and the dump and prune entry points. Production
    passes `utc_now`, `StopFlag.wait`, `scripts.backup.dump.main` and
    `scripts.backup.prune.main` (see `main`).
    """

    def __init__(
        self,
        *,
        config: SchedulerConfig,
        dump: DumpEntry,
        prune: PruneEntry,
        clock: Clock,
        sleeper: Sleeper,
        stop: StopFlag,
    ) -> None:
        self._config = config
        self._dump = dump
        self._prune = prune
        self._clock = clock
        self._sleeper = sleeper
        self._stop = stop
        # Labels of rule-failing dumps that could be neither moved into
        # `suspect/` nor deleted from the root. Prune is skipped while any of
        # them is still there, because prune would count it as a retained
        # backup. In-process only: a restart forgets it, so the logged error
        # naming the label is what an operator acts on after a restart.
        self._suspects_in_root: set[str] = set()

    def run_forever(self) -> int:
        """Check, attempt when due, sleep; exit 0 once a stop is requested."""
        logger.info(
            "[backup-scheduler] started: interval %s, check every %s, retry after %s, "
            "retain %d, status %s",
            self._config.interval,
            self._config.check_interval,
            self._config.retry,
            self._config.retain,
            self._config.status_path,
        )
        while not self._stop.is_set():
            self.tick()
            if self._stop.is_set():
                break
            self._sleeper(self._config.check_interval.total_seconds())
        logger.info("[backup-scheduler] stop requested; exiting after the current attempt.")
        return EXIT_OK

    def run_once(self) -> int:
        """Make one attempt whether or not one is due; return its `--once` exit code."""
        return self.attempt().cli_exit_code

    def tick(self) -> AttemptOutcome | None:
        """Make an attempt if one is due; return it, or None when nothing was due."""
        status = read_status(self._config.status_path)
        due = next_attempt_due(status, self._config)
        if due is not None and self._clock() < due:
            logger.debug("[backup-scheduler] next attempt due at %s", due.isoformat())
            return None
        return self._attempt(status)

    def attempt(self) -> AttemptOutcome:
        """Make one attempt and record it, whatever the status file says is due."""
        return self._attempt(read_status(self._config.status_path))

    def _attempt(self, previous: BackupStatus | None) -> AttemptOutcome:
        before = previous if previous is not None else BackupStatus()
        started = self._clock()
        label = started.astimezone(UTC).strftime(LABEL_FORMAT)
        try:
            outcome, counts = self._dump_and_judge(label, before.last_success_counts)
        except ATTEMPT_ERRORS as exc:
            logger.error(
                "[backup-scheduler] attempt %s failed: %s: %s", label, type(exc).__name__, exc
            )
            outcome, counts = AttemptOutcome(label, RESULT_FAILED, EXIT_FAILED), None

        succeeded = outcome.result == RESULT_OK
        retained = before.retained
        if succeeded:
            pruned = self._prune_after_success()
            if pruned is not None:
                retained = pruned

        finished = self._clock()
        if succeeded:
            success_at, success_counts_now = started, counts
        else:
            # Re-read just before the write, so an `--accept-suspect` that ran
            # while this attempt was in flight keeps the baseline it recorded.
            # This narrows the race; it does not close it, and there is no lock.
            latest = read_status(self._config.status_path)
            source = latest if latest is not None else before
            success_at, success_counts_now = source.last_success_at, source.last_success_counts
        self._record(
            BackupStatus(
                last_attempt_at=started,
                last_result=outcome.result,
                last_exit_code=outcome.exit_code,
                last_success_at=success_at,
                last_label=label,
                last_duration_s=round((finished - started).total_seconds(), 3),
                retained=retained,
                last_success_counts=success_counts_now,
            )
        )
        return outcome

    def _dump_and_judge(
        self, label: str, baseline: SuccessCounts | None
    ) -> tuple[AttemptOutcome, SuccessCounts | None]:
        code = self._dump(
            [
                "--label",
                label,
                "--state-root",
                str(self._config.state_root),
                "--vault-root",
                str(self._config.vault_root),
            ]
        )
        if code != backup_dump.EXIT_OK:
            result = (
                RESULT_REFUSED if code == backup_dump.EXIT_DESTINATION_REFUSED else RESULT_FAILED
            )
            logger.error("[backup-scheduler] dump %s exited %d (%s)", label, code, result)
            return AttemptOutcome(label, result, code), None

        root = self._backup_root()
        artifact = root / label
        try:
            manifest = read_manifest(artifact)
        except BackupManifestError as exc:
            self._set_aside(artifact, root, (f"its manifest cannot be read: {exc}",))
            return AttemptOutcome(label, RESULT_SUSPECT, code), None
        reasons = suspect_reasons(manifest, baseline, self._config.min_node_ratio)
        if reasons:
            self._set_aside(artifact, root, reasons)
            return AttemptOutcome(label, RESULT_SUSPECT, code), None

        counts = success_counts(manifest)
        logger.info(
            "[backup-scheduler] dump %s succeeded: %d graph nodes, %d corpus files, stores %s",
            label,
            counts.graph_nodes,
            counts.vault_corpus_files,
            list(counts.stores),
        )
        return AttemptOutcome(label, RESULT_OK, code), counts

    def _set_aside(self, artifact: Path, root: Path, reasons: tuple[str, ...]) -> None:
        """Take a rule-failing artifact out of the backup root, so prune never counts it.

        Never raises: the attempt is `suspect` whatever happens here. The
        preferred outcome is a move into `suspect/`; the fallback is deletion;
        the last resort is skipping prune while the directory remains.
        """
        label = artifact.name
        try:
            moved = quarantine(artifact, root)
        except OSError as exc:
            logger.error(
                "[backup-scheduler] dump %s is SUSPECT (%s) and could not be moved into "
                "%s: %s. Deleting it from the backup root: it was never a success, and a "
                "later prune would count it as a retained backup.",
                label,
                "; ".join(reasons),
                root / SUSPECT_DIRNAME,
                exc,
            )
            self._delete_from_root(artifact)
            return
        logger.warning(
            "[backup-scheduler] dump %s is SUSPECT and was moved to %s; prune did not "
            "run. Reasons: %s. If the shrink is legitimate, accept it as the new "
            "baseline with --accept-suspect %s.",
            label,
            moved,
            "; ".join(reasons),
            label,
        )
        try:
            delete_older_suspects(moved.parent, keep=label)
        except OSError as exc:
            logger.error(
                "[backup-scheduler] could not delete older suspects in %s: %s",
                moved.parent,
                exc,
            )

    def _delete_from_root(self, artifact: Path) -> None:
        if not os.path.lexists(artifact):
            return
        try:
            shutil.rmtree(artifact)
        except OSError as exc:
            self._suspects_in_root.add(artifact.name)
            logger.error(
                "[backup-scheduler] could not delete suspect dump %s from the backup root: "
                "%s. Prune is skipped while %s exists. Delete it by hand.",
                artifact.name,
                exc,
                artifact,
            )

    def _prune_after_success(self) -> int | None:
        """Run prune; return how many artifacts remain, or None when that is unknown.

        A failed prune does not undo the success: the new artifact is good and
        recorded as such. The failure is logged, and the count reflects what
        is actually on disk. Prune is skipped while a suspect dump that could
        not be set aside is still in the root.
        """
        try:
            root = self._backup_root()
            self._suspects_in_root = {
                label for label in self._suspects_in_root if os.path.lexists(root / label)
            }
            if self._suspects_in_root:
                logger.error(
                    "[backup-scheduler] prune skipped: suspect dump(s) %s are still in %s, "
                    "and prune would count them as retained backups.",
                    sorted(self._suspects_in_root),
                    root,
                )
                return None
            code = self._prune(["--retain", str(self._config.retain), "--confirm"])
            if code != backup_prune.EXIT_OK:
                logger.error("[backup-scheduler] prune exited %d after a success", code)
            return count_artifacts(root)
        except ATTEMPT_ERRORS as exc:
            logger.error(
                "[backup-scheduler] prune after a success failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return None

    def _backup_root(self) -> Path:
        if self._config.backup_root is None:
            raise BackupError(
                f"{BACKUP_ROOT_ENV} is not set, so the artifact the dump reported "
                "cannot be located."
            )
        return assert_backup_destination(self._config.backup_root, purpose="backup scheduler")

    def _record(self, status: BackupStatus) -> None:
        try:
            write_status(self._config.status_path, status)
        except OSError as exc:
            logger.error(
                "[backup-scheduler] could not write %s: %s. /health keeps reporting the "
                "previous status.",
                self._config.status_path,
                exc,
            )


def accept_suspect(config: SchedulerConfig, label: str) -> int:
    """Make the held suspect `label` the new baseline; never dumps and never prunes.

    Refuses (exit 2, nothing changed) unless `suspect/<label>` holds a manifest
    this build reads with a parsable `created_at`, and `<backup root>/<label>`
    does not already exist. Otherwise it moves the artifact back with
    `os.replace` and records it as the last success: `last_result: "ok"`,
    `last_success_at` from the manifest's `created_at`, `last_label`, and
    `last_success_counts` from the manifest. The other fields keep their values.

    Returns:
        0 accepted; 2 refused with nothing changed; 1 the move or the status
        write failed.
    """
    if config.backup_root is None:
        logger.error("[backup-scheduler] REFUSED: %s is not set.", BACKUP_ROOT_ENV)
        return EXIT_REFUSED
    if label in {"", ".", "..", SUSPECT_DIRNAME} or Path(label).name != label or "\\" in label:
        logger.error("[backup-scheduler] REFUSED: %r is not an artifact label.", label)
        return EXIT_REFUSED
    try:
        root = assert_backup_destination(config.backup_root, purpose="backup scheduler")
    except BackupError as exc:
        logger.error("[backup-scheduler] REFUSED: %s", exc)
        return EXIT_REFUSED

    held = root / SUSPECT_DIRNAME / label
    if not held.is_dir():
        logger.error("[backup-scheduler] REFUSED: no suspect artifact is held at %s.", held)
        return EXIT_REFUSED
    try:
        manifest = read_manifest(held)
        created_at = manifest.created_at_datetime()
    except BackupError as exc:
        logger.error("[backup-scheduler] REFUSED: %s", exc)
        return EXIT_REFUSED
    destination = root / label
    if destination.exists():
        logger.error(
            "[backup-scheduler] REFUSED: %s already exists, and an artifact is never "
            "overwritten in place.",
            destination,
        )
        return EXIT_REFUSED

    previous = read_status(config.status_path)
    before = previous if previous is not None else BackupStatus()
    try:
        os.replace(held, destination)
    except OSError as exc:
        logger.error("[backup-scheduler] FAILED: could not move %s back: %s", held, exc)
        return EXIT_FAILED
    try:
        write_status(
            config.status_path,
            replace(
                before,
                last_result=RESULT_OK,
                last_success_at=created_at,
                last_label=label,
                last_success_counts=success_counts(manifest),
            ),
        )
    except OSError as exc:
        logger.error(
            "[backup-scheduler] FAILED: %s was moved back to %s, but %s could not be "
            "written: %s",
            label,
            destination,
            config.status_path,
            exc,
        )
        return EXIT_FAILED
    logger.info("[backup-scheduler] accepted %s as the new baseline.", label)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Build the scheduler CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.backup.scheduler",
        description=(
            "Take a backup whenever the last success is older than "
            f"{INTERVAL_HOURS_ENV}, checking every {CHECK_MINUTES_ENV}, and record "
            f"each attempt in {STATUS_PATH_ENV}."
        ),
        epilog=(
            "Exit codes: the loop exits 0 after SIGTERM; --once exits 0 for a success, "
            "the dump's code for a failure and 3 for a suspect dump; --accept-suspect "
            "exits 0 accepted, 2 refused, 1 failed. A refused configuration exits 2."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Make one attempt now and exit.")
    mode.add_argument(
        "--accept-suspect",
        metavar="LABEL",
        default=None,
        help="Move suspect/LABEL back and record it as the new baseline. Makes no dump.",
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level.")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    dump: DumpEntry = backup_dump.main,
    prune: PruneEntry = backup_prune.main,
    clock: Clock = utc_now,
    sleeper: Sleeper | None = None,
) -> int:
    """Run the loop, one attempt (`--once`), or `--accept-suspect LABEL`."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = SchedulerConfig.from_env(os.environ)
    except SchedulerConfigError as exc:
        logger.error("[backup-scheduler] REFUSED: %s", exc)
        return EXIT_REFUSED

    if args.accept_suspect is not None:
        return accept_suspect(config, args.accept_suspect)

    stop = StopFlag()
    scheduler = BackupScheduler(
        config=config,
        dump=dump,
        prune=prune,
        clock=clock,
        sleeper=sleeper if sleeper is not None else stop.wait,
        stop=stop,
    )
    if args.once:
        return scheduler.run_once()
    install_sigterm_handler(stop)
    return scheduler.run_forever()


def _env_float(
    environ: Mapping[str, str], name: str, default: float, *, allow_zero: bool = False
) -> float:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SchedulerConfigError(f"{name}={raw!r} is not a number.") from exc
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        bound = "of at least 0" if allow_zero else "above 0"
        raise SchedulerConfigError(f"{name}={raw!r} must be a finite number {bound}.")
    return value


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SchedulerConfigError(f"{name}={raw!r} is not an integer.") from exc
    if value < 1:
        raise SchedulerConfigError(f"{name}={value} must be at least 1.")
    return value


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


if __name__ == "__main__":
    sys.exit(main())
