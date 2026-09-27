"""Snapshot LIVE and dev state before and after the smoke run, and adjudicate the delta.

This is the isolation proof. The smoke stack is designed so it CANNOT reach
live state -- `docker-compose.live-path-smoke.yml` bind-mounts only
`./smoke-state` (its LAYER 1) and sets MIST_EVAL_ISOLATION with a smoke-only
MIST_EVAL_NEO4J_HOSTS (its LAYER 3). This script does not trust that design; it
measures it, before and after, and prints what moved.

WHY EVERY SQLITE READ HERE IS A READ-ONLY URI
---------------------------------------------
`open_readonly` opens with `mode=ro` through a `file:` URI. That is a
containment property of THIS script, not a style preference. The `--phase pre`
snapshot reads the LIVE `data/event_store.db` while the live backend has it
open in WAL mode. A read-only URI connection cannot write that database, cannot
create `-wal` or `-shm` sidecars next to it, and cannot take a write lock -- so
even a bug in this script cannot touch live state. A default
`sqlite3.connect(path)` has none of those properties: it creates the file if
absent and can acquire write locks.

The same rule covers `dev-state/event_store.db`. That one is additionally NEVER
hashed: it holds the hydration fixture, its WAL is actively written when the
dev stack is up, and a hash computed over a file being written concurrently
reports a difference that means nothing. Row counts are the comparable measure.

WHY NEO4J IS REACHED BY `docker exec`, NOT THE DRIVER
-----------------------------------------------------
This script runs on the HOST during the experiment. Importing the `neo4j`
driver would make the isolation proof depend on a host pip install, which is a
new way for the measuring instrument to fail on the day it is needed. Shelling
out to `docker exec <container> cypher-shell` needs nothing on the host but
Docker, which the experiment already requires.

USAGE
-----
Run from the repository root:

    python -m scripts.smoke.baseline --phase pre  --out <dir>/baseline-pre.json
    python -m scripts.smoke.baseline --phase post --out <dir>/baseline-post.json
    python -m scripts.smoke.baseline --compare <dir>/baseline-pre.json <dir>/baseline-post.json

Put `<dir>` OUTSIDE the repository. Teardown runs `rm -rf smoke-state`, and
`smoke-state/` is not in `.gitignore` -- see RUNBOOK.md.

COMPARE EXIT CODES
------------------
    0  every delta is UNCHANGED or EXPECTED
    1  at least one CONTAMINATION
    2  no contamination found, but at least one check was UNAVAILABLE in one
       phase, so isolation was not PROVEN. Distinct from 0 on purpose: a check
       that could not run must not read as a check that passed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# scripts/smoke/baseline.py -> scripts/smoke -> scripts -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Tables counted in every event store this script reads.
COUNTED_TABLES = ("conversation_sessions", "conversation_turn_events", "curation_job_runs")

#: Verdicts `--compare` assigns to each observed delta.
UNCHANGED = "UNCHANGED"
EXPECTED = "EXPECTED"
REVIEW = "REVIEW"
CONTAMINATION = "CONTAMINATION"
UNAVAILABLE = "UNAVAILABLE"

EXIT_CLEAN = 0
EXIT_CONTAMINATED = 1
EXIT_UNDECIDED = 2


@dataclass(frozen=True, slots=True)
class CommandResult:
    """One external command's outcome."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """Whether the command exited zero."""
        return self.returncode == 0


@dataclass(frozen=True, slots=True)
class Delta:
    """One adjudicated pre/post difference."""

    check: str
    field: str
    pre: Any
    post: Any
    verdict: str
    reasoning: str


# ---------------------------------------------------------------------------
# Shared primitives. `assert_artifacts.py` imports these three rather than
# redefining them, so the read-only-URI guarantee and the cypher-shell output
# parser have ONE definition and cannot drift apart between the two scripts.
# ---------------------------------------------------------------------------


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open a SQLite database read-only, through a `file:` URI.

    `Path.as_uri()` rather than string concatenation because this runs on
    Windows: it produces `file:///D:/...` with the drive letter and any spaces
    percent-encoded, which SQLite's URI parser accepts. A hand-built
    `"file:" + str(path)` leaves a bare `D:\\...` that SQLite reads as a
    relative path plus an unknown authority.

    Args:
        path: The database file. It must already exist -- `mode=ro` will not
            create it, which is exactly the property wanted.

    Returns:
        A connection on which any write raises `sqlite3.OperationalError`.

    Raises:
        sqlite3.OperationalError: If the file does not exist or cannot be read.
    """
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def run_command(argv: list[str], *, timeout: float = 120.0) -> CommandResult:
    """Run one external command with no shell.

    No shell, so nothing in `argv` is word-split or glob-expanded, and a value
    containing a quote or a space cannot change the command that runs.

    Args:
        argv: Program and arguments.
        timeout: Seconds before the child is killed.

    Returns:
        A `CommandResult`. A missing executable or a timeout is reported as a
        non-zero result with the reason in `stderr`, never raised: a snapshot
        must record "docker was not reachable" rather than abort the run.
    """
    try:
        completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(tuple(argv), 127, "", f"executable not found: {exc}")
    except subprocess.TimeoutExpired:
        return CommandResult(tuple(argv), 124, "", f"timed out after {timeout:.0f}s")
    return CommandResult(tuple(argv), completed.returncode, completed.stdout, completed.stderr)


def parse_cypher_plain(text: str) -> list[dict[str, str]]:
    """Parse `cypher-shell --format plain` output into row dicts.

    Plain format emits one comma-separated header line followed by one line per
    row, with string values quoted. `csv.reader` handles the quoting, so a value
    containing a comma does not shift the columns -- but only with
    `skipinitialspace=True`: without it a `, "Corvid Analytics, Ltd"` field
    begins with a space rather than a quote, csv treats it as unquoted, and the
    value silently truncates at its internal comma.

    Args:
        text: The command's stdout.

    Returns:
        One dict per data row, keyed by header. Empty list when the output has
        no data rows (including when it is empty or whitespace only).
    """
    rows = [row for row in csv.reader(io.StringIO(text.strip()), skipinitialspace=True) if row]
    if len(rows) < 2:
        return []
    header = [column.strip() for column in rows[0]]
    return [dict(zip(header, [cell.strip() for cell in row])) for row in rows[1:]]


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


def table_counts(db_path: Path) -> dict[str, Any]:
    """Count the three tables of interest in one event store.

    Args:
        db_path: Path to the SQLite event store.

    Returns:
        `{"status": "ok", "counts": {...}}`, or `{"status": "unavailable",
        "error": ...}` when the file is absent or unreadable. A missing
        database is reported, never counted as zero -- `--compare` treats
        "0 to 0" as proof of isolation and must not be handed that by accident.
    """
    if not db_path.exists():
        return {"status": "unavailable", "error": f"no such file: {db_path}"}
    try:
        conn = open_readonly(db_path)
    except sqlite3.Error as exc:
        return {"status": "unavailable", "error": f"could not open read-only: {exc}"}
    try:
        counts: dict[str, Any] = {}
        for table in COUNTED_TABLES:
            try:
                counts[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            except sqlite3.Error as exc:
                counts[table] = f"unavailable: {exc}"
        return {"status": "ok", "counts": counts}
    finally:
        conn.close()


def newest_curation_rows(db_path: Path, limit: int = 3) -> dict[str, Any]:
    """Read the newest curation ledger rows.

    The ledger is `curation_job_runs` (`backend/event_store/schema.sql:100-112`).
    `examined` is carried because it is the column the whole experiment is
    about: `backend/event_store/schema.sql:95-96` records that `examined = 0`
    means the job looked at nothing, which is distinguishable from a job that
    looked and found nothing only because this column exists.

    Args:
        db_path: Path to the SQLite event store.
        limit: How many rows.

    Returns:
        `{"status": "ok", "rows": [...]}` or `{"status": "unavailable", ...}`.
    """
    if not db_path.exists():
        return {"status": "unavailable", "error": f"no such file: {db_path}"}
    try:
        conn = open_readonly(db_path)
    except sqlite3.Error as exc:
        return {"status": "unavailable", "error": f"could not open read-only: {exc}"}
    try:
        cursor = conn.execute(
            "SELECT run_id, job_name, trigger_source, started_at, outcome, examined, produced "
            "FROM curation_job_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        columns = [description[0] for description in cursor.description]
        return {"status": "ok", "rows": [dict(zip(columns, row)) for row in cursor.fetchall()]}
    except sqlite3.Error as exc:
        return {"status": "unavailable", "error": str(exc)}
    finally:
        conn.close()


def git_status(repo_path: Path) -> dict[str, Any]:
    """Capture `git status --porcelain` for a directory.

    Args:
        repo_path: Directory that may be a git repository.

    Returns:
        `{"status": "ok", "porcelain": [...]}` with one entry per line, or
        `{"status": "unavailable", "error": ...}` when the directory is absent
        or is not a git repository.
    """
    if not repo_path.exists():
        return {"status": "unavailable", "error": f"no such directory: {repo_path}"}
    result = run_command(["git", "-C", str(repo_path), "status", "--porcelain"])
    if not result.ok:
        return {"status": "unavailable", "error": result.stderr.strip() or result.stdout.strip()}
    return {"status": "ok", "porcelain": [line for line in result.stdout.splitlines() if line]}


def file_listing(root: Path, *, skip_dirs: tuple[str, ...] = (".git",)) -> dict[str, Any]:
    """List files under `root` with size and mtime.

    Args:
        root: Directory to walk.
        skip_dirs: Directory names pruned from the walk. `.git` is skipped by
            default: `git_status` already reports what changed in the working
            tree, and git's own object churn would otherwise swamp the diff.

    Returns:
        `{"status": "ok", "files": {relpath: {"size": int, "mtime": float}}}`
        or `{"status": "unavailable", "error": ...}`.
    """
    if not root.exists():
        return {"status": "unavailable", "error": f"no such directory: {root}"}
    files: dict[str, dict[str, float]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in skip_dirs]
        for name in filenames:
            full = Path(dirpath) / name
            try:
                stat = full.stat()
            except OSError:
                continue
            files[full.relative_to(root).as_posix()] = {
                "size": stat.st_size,
                "mtime": round(stat.st_mtime, 3),
            }
    return {"status": "ok", "files": dict(sorted(files.items()))}


def neo4j_counts(container: str, user: str, password: str, database: str) -> dict[str, Any]:
    """Read three node/edge counts out of a Neo4j container.

    Args:
        container: Container name, e.g. `mist-neo4j` (`docker-compose.yml:98`).
        user: Neo4j username.
        password: Neo4j password.
        database: Neo4j database name.

    Returns:
        `{"status": "ok", "counts": {...}}` or `{"status": "unavailable",
        "error": ..., "counts": {...}}` when any query failed. A failed query
        leaves its key out of `counts` rather than writing zero.
    """
    queries = {
        "nodes": "MATCH (n) RETURN count(n) AS value",
        "entities": "MATCH (n:__Entity__) RETURN count(n) AS value",
        "extracted_from_edges": "MATCH ()-[r:EXTRACTED_FROM]->() RETURN count(r) AS value",
    }
    counts: dict[str, int] = {}
    errors: list[str] = []
    for name, query in queries.items():
        result = run_command(
            [
                "docker", "exec", container, "cypher-shell",
                "-u", user, "-p", password, "-d", database,
                "--format", "plain", query,
            ]
        )
        if not result.ok:
            errors.append(f"{name}: {result.stderr.strip() or result.stdout.strip()}")
            continue
        rows = parse_cypher_plain(result.stdout)
        if not rows or "value" not in rows[0]:
            errors.append(f"{name}: unparseable output {result.stdout!r}")
            continue
        try:
            counts[name] = int(rows[0]["value"])
        except ValueError:
            errors.append(f"{name}: non-integer count {rows[0]['value']!r}")
    if errors:
        return {"status": "unavailable", "error": "; ".join(errors), "counts": counts}
    return {"status": "ok", "counts": counts}


def docker_ps() -> dict[str, Any]:
    """List running containers as `<id> <name>` lines.

    Returns:
        `{"status": "ok", "containers": {name: id}}` or
        `{"status": "unavailable", "error": ...}`.
    """
    result = run_command(["docker", "ps", "--format", "{{.ID}} {{.Names}}"])
    if not result.ok:
        return {"status": "unavailable", "error": result.stderr.strip() or result.stdout.strip()}
    containers: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            containers[parts[1].strip()] = parts[0].strip()
    return {"status": "ok", "containers": dict(sorted(containers.items()))}


def collect(
    phase: str,
    repo_root: Path,
    *,
    neo4j_container: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
) -> dict[str, Any]:
    """Take one full snapshot.

    Args:
        phase: `pre` or `post`, recorded in the output.
        repo_root: Repository root; every path below is relative to it.
        neo4j_container: LIVE Neo4j container name.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.

    Returns:
        The snapshot document, JSON-serialisable.
    """
    live_db = repo_root / "data" / "event_store.db"
    live_vault = repo_root / "mist-memory"
    dev_vault = repo_root / "dev-state" / "vault"
    dev_db = repo_root / "dev-state" / "event_store.db"

    return {
        "phase": phase,
        "captured_at": datetime.now(UTC).isoformat(),
        "repo_root": str(repo_root),
        "checks": {
            "B1_live_event_store": {
                "path": str(live_db),
                "tables": table_counts(live_db),
                "newest_curation_rows": newest_curation_rows(live_db),
            },
            "B2_live_vault": {
                "path": str(live_vault),
                "git": git_status(live_vault),
                "listing": file_listing(live_vault),
            },
            "B3_dev_state": {
                "vault_path": str(dev_vault),
                "db_path": str(dev_db),
                "git": git_status(dev_vault),
                "listing": file_listing(dev_vault),
                "tables": table_counts(dev_db),
            },
            "B4_live_neo4j": {
                "container": neo4j_container,
                **neo4j_counts(neo4j_container, neo4j_user, neo4j_password, neo4j_database),
            },
            "B8_docker_ps": docker_ps(),
        },
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _get(snapshot: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Walk a nested dict, returning `default` at the first missing key."""
    node: Any = snapshot
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _compare_counts(
    check: str,
    pre: dict[str, Any],
    post: dict[str, Any],
    rules: dict[str, tuple[str, str]],
) -> list[Delta]:
    """Compare a `{status, counts}` block field by field under per-field rules.

    Args:
        check: Check name for the report.
        pre: The pre-phase block.
        post: The post-phase block.
        rules: field -> (verdict when it GREW, reasoning). A field that did not
            change is always UNCHANGED regardless of its rule.

    Returns:
        One `Delta` per field, plus one UNAVAILABLE delta when either phase
        could not read the source.
    """
    if pre.get("status") != "ok" or post.get("status") != "ok":
        return [
            Delta(
                check=check,
                field="(whole check)",
                pre=pre.get("status"),
                post=post.get("status"),
                verdict=UNAVAILABLE,
                reasoning=(
                    "one or both phases could not read this source, so no claim of "
                    f"isolation can be made about it. pre: {pre.get('error')!r}; "
                    f"post: {post.get('error')!r}"
                ),
            )
        ]

    deltas: list[Delta] = []
    pre_counts = pre.get("counts", {})
    post_counts = post.get("counts", {})
    for field in sorted(set(pre_counts) | set(post_counts)):
        before = pre_counts.get(field)
        after = post_counts.get(field)
        if before == after:
            deltas.append(
                Delta(check, field, before, after, UNCHANGED, "identical in both phases")
            )
            continue
        verdict, reasoning = rules.get(
            field, (CONTAMINATION, "no adjudication rule for this field; treated as contamination")
        )
        deltas.append(Delta(check, field, before, after, verdict, reasoning))
    return deltas


def _compare_listing(check: str, pre: dict[str, Any], post: dict[str, Any]) -> list[Delta]:
    """Compare two file listings, reporting added, removed and modified paths."""
    if pre.get("status") != "ok" or post.get("status") != "ok":
        return [
            Delta(
                check,
                "listing",
                pre.get("status"),
                post.get("status"),
                UNAVAILABLE,
                f"pre: {pre.get('error')!r}; post: {post.get('error')!r}",
            )
        ]
    before = pre.get("files", {})
    after = post.get("files", {})
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    if not (added or removed or modified):
        return [Delta(check, "listing", "identical", "identical", UNCHANGED, "no file changed")]
    return [
        Delta(
            check,
            "listing",
            {"files": len(before)},
            {"added": added, "removed": removed, "modified": modified},
            CONTAMINATION,
            "the smoke stack does not bind-mount this directory "
            "(docker-compose.live-path-smoke.yml LAYER 1), so nothing it runs can "
            "write here. A change means something else did -- identify it before "
            "trusting any result from this run.",
        )
    ]


def _compare_git(check: str, pre: dict[str, Any], post: dict[str, Any]) -> list[Delta]:
    """Compare two `git status --porcelain` captures."""
    if pre.get("status") != "ok" or post.get("status") != "ok":
        return [
            Delta(
                check,
                "git status --porcelain",
                pre.get("status"),
                post.get("status"),
                UNAVAILABLE,
                f"pre: {pre.get('error')!r}; post: {post.get('error')!r}",
            )
        ]
    before = pre.get("porcelain", [])
    after = post.get("porcelain", [])
    if before == after:
        return [
            Delta(
                check,
                "git status --porcelain",
                f"{len(before)} line(s)",
                f"{len(after)} line(s)",
                UNCHANGED,
                "identical working-tree status",
            )
        ]
    return [
        Delta(
            check,
            "git status --porcelain",
            before,
            after,
            CONTAMINATION,
            "the working tree of a directory the smoke stack cannot write changed "
            "during the window",
        )
    ]


def _compare_curation_rows(pre: dict[str, Any], post: dict[str, Any]) -> list[Delta]:
    """Adjudicate new rows in the LIVE curation ledger.

    This is the one genuinely ambiguous delta in the whole comparison, so it is
    adjudicated in the output rather than left to the reader.

    The live scheduler runs on its own timer for the life of the live backend
    (`backend/knowledge/curation/scheduler.py:293-327`, which sleeps 60s and
    re-checks). So new rows during the experiment window are live behaving
    normally, NOT evidence that the smoke stack reached live. Two things make
    that adjudication safe rather than convenient: the smoke backend writes its
    ledger to `/app/smoke-state/event_store.db`
    (`docker-compose.live-path-smoke.yml:182`), and the live store is not
    bind-mounted into it at all.

    A `trigger_source` other than `scheduled` is flagged REVIEW rather than
    accepted: `scheduler.py:158` writes `manual` only from `run_all_once`, which
    is an operator action, and nobody should be taking one during the window.

    Args:
        pre: The pre-phase `newest_curation_rows` block.
        post: The post-phase block.

    Returns:
        A list of deltas, one per new row plus a summary when nothing is new.
    """
    if pre.get("status") != "ok" or post.get("status") != "ok":
        return [
            Delta(
                "B1_live_event_store",
                "newest_curation_rows",
                pre.get("status"),
                post.get("status"),
                UNAVAILABLE,
                f"pre: {pre.get('error')!r}; post: {post.get('error')!r}",
            )
        ]
    pre_ids = {row.get("run_id") for row in pre.get("rows", [])}
    new_rows = [row for row in post.get("rows", []) if row.get("run_id") not in pre_ids]
    if not new_rows:
        return [
            Delta(
                "B1_live_event_store",
                "newest_curation_rows",
                f"{len(pre.get('rows', []))} row(s)",
                "no new rows",
                UNCHANGED,
                "the live scheduler happened not to fire during the window",
            )
        ]

    deltas: list[Delta] = []
    for row in new_rows:
        if row.get("trigger_source") == "scheduled":
            examined = row.get("examined")
            deltas.append(
                Delta(
                    "B1_live_event_store",
                    f"new curation row {row.get('job_name')}",
                    "absent",
                    row,
                    EXPECTED,
                    "trigger_source='scheduled': the LIVE scheduler's own timer "
                    "(backend/knowledge/curation/scheduler.py:293-327). This is live "
                    "behaving normally, not contamination -- the smoke backend's "
                    "ledger is /app/smoke-state/event_store.db "
                    "(docker-compose.live-path-smoke.yml:182) and the live store is "
                    "not mounted into it. "
                    + (
                        f"examined={examined} is the live symptom this experiment "
                        "exists to explain, not a new event."
                        if examined in (0, None)
                        else f"examined={examined}: unexpected on live, whose "
                        "conversation_turn_events table was measured empty during "
                        "planning. Worth a look, but not contamination by itself."
                    ),
                )
            )
        else:
            deltas.append(
                Delta(
                    "B1_live_event_store",
                    f"new curation row {row.get('job_name')}",
                    "absent",
                    row,
                    REVIEW,
                    f"trigger_source={row.get('trigger_source')!r} is not 'scheduled'. "
                    "'manual' comes from `run_all_once` "
                    "(backend/knowledge/curation/scheduler.py:158), an explicit "
                    "operator action. Find out who ran it during the window.",
                )
            )
    return deltas


def compare(pre: dict[str, Any], post: dict[str, Any]) -> list[Delta]:
    """Adjudicate every pre/post difference.

    Args:
        pre: The `--phase pre` snapshot.
        post: The `--phase post` snapshot.

    Returns:
        Every delta, in check order.
    """
    deltas: list[Delta] = []

    deltas += _compare_counts(
        "B1_live_event_store",
        _get(pre, "checks", "B1_live_event_store", "tables", default={}),
        _get(post, "checks", "B1_live_event_store", "tables", default={}),
        rules={
            "conversation_sessions": (
                CONTAMINATION,
                "LIVE conversation_sessions must be exactly unchanged. Nobody is "
                "talking to live during the window, and the smoke backend writes to "
                "/app/smoke-state/event_store.db "
                "(docker-compose.live-path-smoke.yml:182). There is no benign "
                "explanation for a delta here.",
            ),
            "conversation_turn_events": (
                CONTAMINATION,
                "LIVE conversation_turn_events must be exactly unchanged, for the "
                "same reason as conversation_sessions. A new turn row on live is the "
                "precise failure this whole isolation design exists to prevent.",
            ),
            "curation_job_runs": (
                EXPECTED,
                "LIVE curation_job_runs MAY grow: the live scheduler runs on its own "
                "timer (backend/knowledge/curation/scheduler.py:293-327). The rows "
                "themselves are adjudicated below, one by one.",
            ),
        },
    )
    deltas += _compare_curation_rows(
        _get(pre, "checks", "B1_live_event_store", "newest_curation_rows", default={}),
        _get(post, "checks", "B1_live_event_store", "newest_curation_rows", default={}),
    )

    deltas += _compare_git(
        "B2_live_vault",
        _get(pre, "checks", "B2_live_vault", "git", default={}),
        _get(post, "checks", "B2_live_vault", "git", default={}),
    )
    deltas += _compare_listing(
        "B2_live_vault",
        _get(pre, "checks", "B2_live_vault", "listing", default={}),
        _get(post, "checks", "B2_live_vault", "listing", default={}),
    )

    deltas += _compare_git(
        "B3_dev_state",
        _get(pre, "checks", "B3_dev_state", "git", default={}),
        _get(post, "checks", "B3_dev_state", "git", default={}),
    )
    deltas += _compare_listing(
        "B3_dev_state",
        _get(pre, "checks", "B3_dev_state", "listing", default={}),
        _get(post, "checks", "B3_dev_state", "listing", default={}),
    )
    deltas += _compare_counts(
        "B3_dev_state",
        _get(pre, "checks", "B3_dev_state", "tables", default={}),
        _get(post, "checks", "B3_dev_state", "tables", default={}),
        rules={
            table: (
                CONTAMINATION,
                "dev-state holds the hydration fixture that "
                "docker-compose.dev-hydration.yml:21-23 records as costing 87 LLM "
                "turns to reproduce. The smoke stack does not mount ./dev-state "
                "(docker-compose.live-path-smoke.yml LAYER 1), so it cannot have "
                "written here.",
            )
            for table in COUNTED_TABLES
        },
    )

    deltas += _compare_counts(
        "B4_live_neo4j",
        _get(pre, "checks", "B4_live_neo4j", default={}),
        _get(post, "checks", "B4_live_neo4j", default={}),
        rules={
            field: (
                CONTAMINATION,
                "the LIVE graph must be exactly unchanged. The smoke backend sets "
                "MIST_EVAL_ISOLATION=1 with a smoke-only MIST_EVAL_NEO4J_HOSTS "
                "(docker-compose.live-path-smoke.yml:206-207), which makes "
                "Neo4jConnection.connect() refuse a live driver at "
                "backend/knowledge/storage/neo4j_connection.py:40. A delta here "
                "means that guard did not hold.",
            )
            for field in ("nodes", "entities", "extracted_from_edges")
        },
    )

    deltas += _compare_docker_ps(
        _get(pre, "checks", "B8_docker_ps", default={}),
        _get(post, "checks", "B8_docker_ps", default={}),
    )
    return deltas


def _compare_docker_ps(pre: dict[str, Any], post: dict[str, Any]) -> list[Delta]:
    """Compare running containers, expecting only smoke containers to appear."""
    if pre.get("status") != "ok" or post.get("status") != "ok":
        return [
            Delta(
                "B8_docker_ps",
                "containers",
                pre.get("status"),
                post.get("status"),
                UNAVAILABLE,
                f"pre: {pre.get('error')!r}; post: {post.get('error')!r}",
            )
        ]
    before = pre.get("containers", {})
    after = post.get("containers", {})
    deltas: list[Delta] = []

    for name, container_id in sorted(before.items()):
        if name not in after:
            deltas.append(
                Delta(
                    "B8_docker_ps",
                    name,
                    container_id,
                    "gone",
                    CONTAMINATION,
                    "a container that was running before the experiment is gone. "
                    "Every teardown command in RUNBOOK.md names its services "
                    "explicitly for this reason; a bare `docker compose down` "
                    "reconciles the whole project including live.",
                )
            )
        elif after[name] != container_id:
            deltas.append(
                Delta(
                    "B8_docker_ps",
                    name,
                    container_id,
                    after[name],
                    CONTAMINATION,
                    "same name, different container id: this container was recreated "
                    "during the window. A live service must not be recreated by a "
                    "smoke command.",
                )
            )

    for name in sorted(set(after) - set(before)):
        verdict, reasoning = (
            (EXPECTED, "a smoke container, expected to appear between the phases")
            if "smoke" in name
            else (REVIEW, "a container with no 'smoke' in its name appeared during the window")
        )
        deltas.append(Delta("B8_docker_ps", name, "absent", after[name], verdict, reasoning))

    if not deltas:
        deltas.append(
            Delta(
                "B8_docker_ps",
                "containers",
                f"{len(before)} running",
                f"{len(after)} running",
                UNCHANGED,
                "same names, same ids",
            )
        )
    return deltas


def render_comparison(deltas: list[Delta]) -> tuple[str, int]:
    """Format the comparison report and pick the exit code.

    Args:
        deltas: What `compare` produced.

    Returns:
        (report text, exit code). See the module docstring for the codes.
    """
    lines = ["=" * 78, "BASELINE COMPARISON", "=" * 78]
    by_verdict: dict[str, int] = {}
    for delta in deltas:
        by_verdict[delta.verdict] = by_verdict.get(delta.verdict, 0) + 1

    for delta in deltas:
        if delta.verdict == UNCHANGED:
            lines.append(f"[{delta.verdict}] {delta.check}.{delta.field}: {delta.pre}")
            continue
        lines.append(f"[{delta.verdict}] {delta.check}.{delta.field}")
        lines.append(f"    pre : {json.dumps(delta.pre, default=str)}")
        lines.append(f"    post: {json.dumps(delta.post, default=str)}")
        lines.append(f"    why : {delta.reasoning}")

    lines.append("-" * 78)
    lines.append(
        "  ".join(f"{verdict}={count}" for verdict, count in sorted(by_verdict.items()))
    )

    if by_verdict.get(CONTAMINATION):
        lines.append(
            "VERDICT: CONTAMINATED. Live or dev state moved during the window. Do not "
            "report the artifact results as valid until each CONTAMINATION above has "
            "an explanation that is not the smoke stack."
        )
        code = EXIT_CONTAMINATED
    elif by_verdict.get(UNAVAILABLE):
        lines.append(
            "VERDICT: UNDECIDED. Nothing looks contaminated, but at least one check "
            "could not run in one of the phases, so isolation was not PROVEN. A check "
            "that could not run is not a check that passed."
        )
        code = EXIT_UNDECIDED
    else:
        lines.append(
            "VERDICT: CLEAN. Every delta is UNCHANGED or EXPECTED. "
            + (
                "Read the REVIEW items above before relying on this."
                if by_verdict.get(REVIEW)
                else "No REVIEW items."
            )
        )
        code = EXIT_CLEAN
    lines.append("=" * 78)
    return "\n".join(lines), code


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI."""
    parser = argparse.ArgumentParser(
        prog="baseline.py",
        description="Snapshot live and dev state, or compare two snapshots.",
    )
    parser.add_argument("--phase", choices=("pre", "post"), help="Take a snapshot.")
    parser.add_argument("--out", type=Path, help="Where to write the snapshot JSON.")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("PRE_JSON", "POST_JSON"),
        type=Path,
        help="Adjudicate the deltas between two snapshots.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root.")
    parser.add_argument(
        "--neo4j-container",
        default="mist-neo4j",
        help="LIVE Neo4j container name (docker-compose.yml:98). Default: mist-neo4j.",
    )
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "password"))
    parser.add_argument("--neo4j-database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, defaulting to `sys.argv[1:]`.

    Returns:
        A process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.compare:
        pre = json.loads(args.compare[0].read_text(encoding="utf-8"))
        post = json.loads(args.compare[1].read_text(encoding="utf-8"))
        report, code = render_comparison(compare(pre, post))
        print(report)
        return code

    if not args.phase or not args.out:
        parser.error("either --phase with --out, or --compare PRE_JSON POST_JSON")

    snapshot = collect(
        args.phase,
        args.repo_root.resolve(),
        neo4j_container=args.neo4j_container,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

    unavailable = [
        name
        for name, check in snapshot["checks"].items()
        if _has_unavailable(check)
    ]
    print(f"[baseline] {args.phase} snapshot written to {args.out}")
    for name, check in snapshot["checks"].items():
        print(f"[baseline]   {name}: {_summarise_check(check)}")
    if unavailable:
        print(
            "[baseline] WARNING: these checks could not be read and will compare as "
            f"UNAVAILABLE, not as clean: {', '.join(unavailable)}"
        )
    return EXIT_CLEAN


def _has_unavailable(check: Any) -> bool:
    """Whether any nested block in a check reported `status: unavailable`."""
    if isinstance(check, dict):
        if check.get("status") == "unavailable":
            return True
        return any(_has_unavailable(value) for value in check.values())
    return False


def _summarise_check(check: dict[str, Any]) -> str:
    """One-line summary of a check block for the snapshot's stdout."""
    parts: list[str] = []
    tables = check.get("tables")
    if isinstance(tables, dict):
        parts.append(
            f"tables={tables.get('counts')}"
            if tables.get("status") == "ok"
            else f"tables=unavailable({tables.get('error')})"
        )
    if check.get("status") == "ok" and "counts" in check:
        parts.append(f"counts={check['counts']}")
    elif check.get("status") == "unavailable":
        parts.append(f"unavailable({check.get('error')})")
    if isinstance(check.get("containers"), dict):
        parts.append(f"{len(check['containers'])} container(s)")
    listing = check.get("listing")
    if isinstance(listing, dict) and listing.get("status") == "ok":
        parts.append(f"{len(listing.get('files', {}))} file(s)")
    return "; ".join(parts) or "captured"


if __name__ == "__main__":
    sys.exit(main())
