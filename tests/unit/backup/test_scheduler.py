"""The in-compose backup scheduler (plan v1, task T6; Interfaces and contracts item 6).

Every test injects the clock, the sleeper and the dump and prune callables, so no
test waits, reaches a graph or depends on the wall clock. The fake dump writes a
real artifact directory with a real manifest under `tmp_path`, because the
scheduler's verdict is read from that manifest through `read_manifest`, and a
fake that skipped it would leave the success rule untested.
"""

from __future__ import annotations

import json
import shutil
import signal
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from backend.errors import Neo4jConnectionError
from backend.knowledge.graph_artifact import GraphArtifactError
from scripts.backup import prune as backup_prune
from scripts.backup import scheduler as scheduler_module
from scripts.backup.destination import BACKUP_ROOT_ENV
from scripts.backup.errors import BackupError
from scripts.backup.manifest import BACKUP_LAYOUT, BACKUP_LAYOUT_VERSION, BackupManifest
from scripts.backup.scheduler import (
    BackupScheduler,
    SchedulerConfig,
    StopFlag,
    accept_suspect,
    install_sigterm_handler,
    main,
)
from scripts.backup.status import BackupStatus, SuccessCounts, read_status, write_status
from scripts.backup.stores import LIVE_STORE_FILENAMES

from .conftest import STAMPS

ITEM_6_KEYS = [
    "v",
    "last_attempt_at",
    "last_result",
    "last_exit_code",
    "last_success_at",
    "last_label",
    "last_duration_s",
    "retained",
    "last_success_counts",
]

START = datetime(2026, 10, 23, 3, 0, 0, tzinfo=UTC)
ALL_STORES_PRESENT = {name: f"stores/{name}" for name in LIVE_STORE_FILENAMES}


def write_artifact(
    root: Path,
    label: str,
    *,
    created_at: str = "2026-10-23T03:00:00Z",
    nodes: int = 100,
    corpus_files: int = 10,
    store_files: dict[str, str | None] | None = None,
) -> Path:
    """Write an artifact directory whose manifest has the shape the dump writes.

    `stores` always carries every name in `LIVE_STORE_FILENAMES`, with `file`
    null for an absent store, exactly as `StoreCapture.to_manifest_entry` renders
    it (`scripts/backup/stores.py:84-89`).
    """
    files = dict(ALL_STORES_PRESENT if store_files is None else store_files)
    artifact = root / label
    artifact.mkdir(parents=True)
    (artifact / "graph.json").write_text("{}", encoding="utf-8")
    BackupManifest(
        layout=BACKUP_LAYOUT,
        layout_version=BACKUP_LAYOUT_VERSION,
        created_at=created_at,
        label=label,
        git_head=None,
        stamps=dict(STAMPS),
        source={},
        files={},
        stores={
            name: {
                "present": files.get(name) is not None,
                "file": files.get(name),
                "row_counts": {},
                "uncounted_tables": {},
            }
            for name in LIVE_STORE_FILENAMES
        },
        graph={"file": "graph.json", "nodes": nodes, "relationships": 0},
        vault={
            "directory": "vault",
            "source_present": True,
            "corpus_file_count": corpus_files,
            "file_count": corpus_files,
        },
        excluded=["vector_store"],
    ).write(artifact)
    return artifact


class FakeClock:
    """A settable UTC clock."""

    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


@dataclass
class DumpStep:
    """What one fake dump call does: write an artifact, return a code, or raise."""

    exit_code: int = 0
    nodes: int = 100
    corpus_files: int = 10
    store_files: dict[str, str | None] | None = None
    raises: BaseException | None = None
    write: bool = True
    side_effect: Callable[[], None] | None = None
    manifest_text: str | None = None


@dataclass
class FakeDump:
    """Stands in for `scripts.backup.dump.main`: same argv, same exit codes."""

    backup_root: Path
    clock: FakeClock
    steps: list[DumpStep] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, argv: list[str]) -> int:
        self.calls.append(list(argv))
        step = self.steps.pop(0) if self.steps else DumpStep()
        if step.side_effect is not None:
            step.side_effect()
        if step.raises is not None:
            raise step.raises
        if step.exit_code == 0 and step.write:
            label = argv[argv.index("--label") + 1]
            artifact = write_artifact(
                self.backup_root,
                label,
                created_at=self.clock().strftime("%Y-%m-%dT%H:%M:%SZ"),
                nodes=step.nodes,
                corpus_files=step.corpus_files,
                store_files=step.store_files,
            )
            if step.manifest_text is not None:
                (artifact / "manifest.json").write_text(step.manifest_text, encoding="utf-8")
        return step.exit_code


@dataclass
class FakePrune:
    """Stands in for `scripts.backup.prune.main`; records each argv."""

    exit_code: int = 0
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, argv: list[str]) -> int:
        self.calls.append(list(argv))
        return self.exit_code


@dataclass
class FakeSleeper:
    """Advances the fake clock by each requested sleep; requests a stop after N sleeps."""

    clock: FakeClock
    stop: StopFlag
    stop_after: int
    calls: list[float] = field(default_factory=list)

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(timedelta(seconds=seconds))
        if len(self.calls) >= self.stop_after:
            self.stop.request()


@pytest.fixture
def backup_root(tmp_path: Path) -> Path:
    root = tmp_path / "backup"
    root.mkdir()
    return root


@pytest.fixture
def config(tmp_path: Path, backup_root: Path) -> SchedulerConfig:
    ops = tmp_path / "ops"
    ops.mkdir()
    return SchedulerConfig(
        backup_root=backup_root,
        status_path=ops / "backup-status.json",
        interval=timedelta(hours=24),
        check_interval=timedelta(minutes=10),
        retry=timedelta(minutes=60),
        retain=7,
        min_node_ratio=0.5,
        state_root=tmp_path / "state",
        vault_root=tmp_path / "mist-memory",
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def dump(backup_root: Path, clock: FakeClock) -> FakeDump:
    return FakeDump(backup_root=backup_root, clock=clock)


@pytest.fixture
def prune() -> FakePrune:
    return FakePrune()


@pytest.fixture
def stop() -> StopFlag:
    return StopFlag()


def make_scheduler(
    config: SchedulerConfig,
    dump: Callable[[list[str]], int],
    prune: Callable[[list[str]], int],
    clock: FakeClock,
    *,
    stop: StopFlag | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> BackupScheduler:
    flag = stop if stop is not None else StopFlag()
    return BackupScheduler(
        config=config,
        dump=dump,
        prune=prune,
        clock=clock,
        sleeper=sleeper if sleeper is not None else (lambda _seconds: None),
        stop=flag,
    )


def seed_success(config: SchedulerConfig, *, at: datetime, nodes: int = 100) -> None:
    """Write a status file recording one success at `at`, holding every store."""
    write_status(
        config.status_path,
        BackupStatus(
            last_attempt_at=at,
            last_result="ok",
            last_exit_code=0,
            last_success_at=at,
            last_label="seeded",
            last_duration_s=1.0,
            retained=1,
            last_success_counts=SuccessCounts(
                graph_nodes=nodes,
                vault_corpus_files=10,
                stores=tuple(sorted(LIVE_STORE_FILENAMES)),
            ),
        ),
    )


def refuse_rmtree_of(monkeypatch: pytest.MonkeyPatch, name: str) -> dict[str, bool]:
    """Make `shutil.rmtree` raise for a directory called `name` while the switch is on.

    Every other path is deleted normally. Returns the switch, so a test can let
    the deletion succeed later without undoing the shared `monkeypatch`, which
    also holds the unit tier's isolation environment.
    """
    switch = {"refuse": True}
    real_rmtree = shutil.rmtree

    def refusing_rmtree(path, *args, **kwargs):
        if switch["refuse"] and Path(path).name == name:
            raise OSError(f"permission denied: {path}")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(scheduler_module.shutil, "rmtree", refusing_rmtree)
    return switch


def status_json(config: SchedulerConfig) -> dict:
    return json.loads(config.status_path.read_text(encoding="utf-8"))


class TestCatchUpInterval:
    def test_with_no_status_file_it_runs_at_once(self, config, dump, prune, clock) -> None:
        scheduler = make_scheduler(config, dump, prune, clock)

        outcome = scheduler.tick()

        assert outcome is not None
        assert len(dump.calls) == 1
        assert status_json(config)["last_result"] == "ok"

    @pytest.mark.parametrize(
        ("hours_ago", "expected_runs"),
        [
            pytest.param(23, 0, id="23h-ago-not-due"),
            pytest.param(25, 1, id="25h-ago-due"),
        ],
    )
    def test_a_success_runs_again_only_after_the_interval(
        self, config, dump, prune, clock, hours_ago: int, expected_runs: int
    ) -> None:
        seed_success(config, at=START - timedelta(hours=hours_ago))
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        assert len(dump.calls) == expected_runs

    def test_a_corrupt_status_file_counts_as_no_success_and_is_overwritten(
        self, config, dump, prune, clock
    ) -> None:
        config.status_path.write_text("{truncated", encoding="utf-8")
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        assert len(dump.calls) == 1
        assert status_json(config)["last_result"] == "ok"


class TestRetryWindow:
    @pytest.mark.parametrize(
        ("minutes_later", "expected_runs"),
        [
            pytest.param(59, 1, id="59-minutes-not-due"),
            pytest.param(60, 2, id="60-minutes-due"),
        ],
    )
    def test_a_failure_retries_after_the_retry_window_and_not_before(
        self, config, dump, prune, clock, minutes_later: int, expected_runs: int
    ) -> None:
        seed_success(config, at=START - timedelta(hours=30))
        dump.steps = [DumpStep(exit_code=1)]
        scheduler = make_scheduler(config, dump, prune, clock)
        scheduler.tick()
        assert status_json(config)["last_result"] == "failed"

        clock.advance(timedelta(minutes=minutes_later))
        scheduler.tick()

        assert len(dump.calls) == expected_runs

    def test_a_refusal_also_waits_the_retry_window(self, config, dump, prune, clock) -> None:
        dump.steps = [DumpStep(exit_code=2)]
        scheduler = make_scheduler(config, dump, prune, clock)
        scheduler.tick()

        clock.advance(timedelta(minutes=59))
        scheduler.tick()

        assert status_json(config)["last_result"] == "refused"
        assert len(dump.calls) == 1


class TestDumpInvocation:
    def test_dump_gets_a_utc_timestamp_label_and_the_two_roots(
        self, config, dump, prune, clock
    ) -> None:
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        assert dump.calls == [
            [
                "--label",
                "20261023T030000Z",
                "--state-root",
                str(config.state_root),
                "--vault-root",
                str(config.vault_root),
            ]
        ]

    def test_the_default_roots_are_the_container_paths(self) -> None:
        config = SchedulerConfig.from_env(
            {"MIST_BACKUP_ROOT": "/backup", "MIST_BACKUP_STATUS_PATH": "/app/ops/s.json"}
        )

        assert config.state_root == Path("/app/data")
        assert config.vault_root == Path("/app/mist-memory")
        assert config.interval == timedelta(hours=24)
        assert config.check_interval == timedelta(minutes=10)
        assert config.retry == timedelta(minutes=60)
        assert config.retain == 7
        assert config.min_node_ratio == 0.5


class TestSuccessRule:
    def test_a_success_runs_prune_with_retain_and_confirm(self, config, dump, prune, clock) -> None:
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        assert prune.calls == [["--retain", "7", "--confirm"]]

    def test_a_success_records_the_counts_for_the_next_comparison(
        self, config, dump, prune, clock
    ) -> None:
        dump.steps = [
            DumpStep(
                nodes=42,
                corpus_files=13,
                store_files={**ALL_STORES_PRESENT, "extraction_cache.db": None},
            )
        ]
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        raw = status_json(config)
        assert raw["last_success_counts"] == {
            "graph_nodes": 42,
            "vault_corpus_files": 13,
            "stores": ["event_store.db", "vault_sidecar.db"],
        }
        assert raw["last_success_at"] == "2026-10-23T03:00:00Z"
        assert raw["last_exit_code"] == 0
        assert raw["retained"] == 1

    @pytest.mark.parametrize(
        "step",
        [
            pytest.param(DumpStep(nodes=0), id="zero-graph-nodes"),
            pytest.param(DumpStep(corpus_files=0), id="zero-corpus-files"),
            pytest.param(
                DumpStep(store_files={**ALL_STORES_PRESENT, "vault_sidecar.db": None}),
                id="store-file-null-where-baseline-had-it",
            ),
            pytest.param(DumpStep(nodes=49), id="fewer-than-half-the-nodes"),
        ],
    )
    def test_a_dump_that_fails_the_rule_is_suspect_moved_and_not_pruned(
        self, config, dump, prune, clock, backup_root: Path, step: DumpStep
    ) -> None:
        seed_success(config, at=START - timedelta(hours=25), nodes=100)
        dump.steps = [step]
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        label = "20261023T030000Z"
        assert status_json(config)["last_result"] == "suspect"
        assert not (backup_root / label).exists()
        assert (backup_root / "suspect" / label / "manifest.json").is_file()
        assert prune.calls == []

    def test_every_store_key_with_the_same_non_null_files_is_not_suspect(
        self, config, dump, prune, clock
    ) -> None:
        seed_success(config, at=START - timedelta(hours=25), nodes=100)
        dump.steps = [DumpStep(nodes=100, store_files=dict(ALL_STORES_PRESENT))]
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        assert status_json(config)["last_result"] == "ok"
        assert len(prune.calls) == 1

    def test_a_store_absent_from_the_baseline_may_stay_absent(
        self, config, dump, prune, clock
    ) -> None:
        write_status(
            config.status_path,
            BackupStatus(
                last_attempt_at=START - timedelta(hours=25),
                last_result="ok",
                last_exit_code=0,
                last_success_at=START - timedelta(hours=25),
                last_success_counts=SuccessCounts(
                    graph_nodes=100,
                    vault_corpus_files=10,
                    stores=("event_store.db", "vault_sidecar.db"),
                ),
            ),
        )
        dump.steps = [DumpStep(store_files={**ALL_STORES_PRESENT, "extraction_cache.db": None})]
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        assert status_json(config)["last_result"] == "ok"

    def test_exactly_the_minimum_node_ratio_is_a_success(self, config, dump, prune, clock) -> None:
        seed_success(config, at=START - timedelta(hours=25), nodes=100)
        dump.steps = [DumpStep(nodes=50)]
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        assert status_json(config)["last_result"] == "ok"


class TestSuspectHandling:
    def test_the_suspect_artifact_is_moved_by_rename_not_copied(
        self, config, dump, prune, clock, backup_root: Path
    ) -> None:
        # A rename keeps the directory's inode; a copy-and-delete would not.
        seed_success(config, at=START - timedelta(hours=25))
        inodes: dict[str, int] = {}
        dump.steps = [DumpStep(nodes=0)]

        def dump_then_record(argv: list[str]) -> int:
            code = dump(argv)
            label = argv[argv.index("--label") + 1]
            inodes["before"] = (backup_root / label).stat().st_ino
            return code

        scheduler = make_scheduler(config, dump_then_record, prune, clock)

        scheduler.tick()

        moved = backup_root / "suspect" / "20261023T030000Z"
        assert moved.stat().st_ino == inodes["before"]

    def test_the_suspect_status_keeps_the_last_success(self, config, dump, prune, clock) -> None:
        seed_success(config, at=START - timedelta(hours=25), nodes=100)
        dump.steps = [DumpStep(nodes=0)]
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        raw = status_json(config)
        assert raw["last_success_at"] == "2026-10-22T02:00:00Z"
        assert raw["last_success_counts"]["graph_nodes"] == 100
        assert raw["last_exit_code"] == 0
        assert raw["last_label"] == "20261023T030000Z"

    def test_after_a_suspect_the_next_attempt_waits_the_full_interval(
        self, config, dump, prune, clock
    ) -> None:
        seed_success(config, at=START - timedelta(hours=30))
        dump.steps = [DumpStep(nodes=0)]
        scheduler = make_scheduler(config, dump, prune, clock)
        scheduler.tick()

        clock.advance(timedelta(minutes=60))
        scheduler.tick()
        clock.advance(timedelta(hours=22, minutes=59))
        scheduler.tick()
        runs_before_interval = len(dump.calls)
        clock.advance(timedelta(minutes=1))
        scheduler.tick()

        assert runs_before_interval == 1
        assert len(dump.calls) == 2

    def test_seven_suspects_keep_every_success_and_only_the_newest_suspect(
        self, config, dump, clock, backup_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The REAL prune, retaining 7: had any suspect attempt been followed by
        # prune with its artifact left in the root, the oldest success below
        # would have been deleted.
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(backup_root))
        successes = [
            write_artifact(backup_root, f"good-{day}", created_at=f"2026-10-{day:02d}T03:00:00Z")
            for day in range(15, 22)
        ]
        seed_success(config, at=START - timedelta(hours=25))
        dump.steps = [DumpStep(nodes=0) for _ in range(7)]
        scheduler = make_scheduler(config, dump, backup_prune.main, clock)

        labels = []
        for _ in range(7):
            labels.append(clock().strftime("%Y%m%dT%H%M%SZ"))
            scheduler.tick()
            clock.advance(timedelta(hours=24))

        assert len(dump.calls) == 7
        assert all(path.is_dir() for path in successes)
        assert sorted(p.name for p in (backup_root / "suspect").iterdir()) == [labels[-1]]
        assert status_json(config)["last_result"] == "suspect"

    def test_when_the_move_fails_the_suspect_is_deleted_and_every_good_artifact_survives(
        self, config, dump, clock, backup_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `suspect` as a regular file makes every move into `suspect/` fail. Six
        # good artifacts plus the final success is exactly `--retain 7`, so the
        # REAL prune keeps them all unless a suspect is left in the root to be
        # counted in their place.
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(backup_root))
        (backup_root / "suspect").write_text("not a directory", encoding="utf-8")
        successes = [
            write_artifact(backup_root, f"good-{day}", created_at=f"2026-10-{day:02d}T03:00:00Z")
            for day in range(16, 22)
        ]
        seed_success(config, at=START - timedelta(hours=25))
        dump.steps = [DumpStep(nodes=0) for _ in range(7)] + [DumpStep()]
        scheduler = make_scheduler(config, dump, backup_prune.main, clock)

        exit_codes = []
        for _ in range(8):
            outcome = scheduler.tick()
            assert outcome is not None
            exit_codes.append(outcome.cli_exit_code)
            clock.advance(timedelta(hours=24))

        assert exit_codes == [3] * 7 + [0]
        assert all(path.is_dir() for path in successes)
        assert {p.name for p in backup_root.iterdir() if p.is_dir()} == {
            *(path.name for path in successes),
            "20261030T030000Z",
        }

    def test_when_the_suspect_cannot_be_deleted_either_prune_is_skipped(
        self, config, dump, prune, clock, backup_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (backup_root / "suspect").write_text("not a directory", encoding="utf-8")
        refuse_rmtree_of(monkeypatch, "20261023T030000Z")
        dump.steps = [DumpStep(nodes=0), DumpStep()]
        scheduler = make_scheduler(config, dump, prune, clock)

        first = scheduler.tick()
        clock.advance(timedelta(hours=24))
        scheduler.tick()

        assert first is not None
        assert first.cli_exit_code == 3
        assert (backup_root / "20261023T030000Z").is_dir()
        assert status_json(config)["last_result"] == "ok"
        assert prune.calls == []

    def test_prune_resumes_once_the_undeletable_suspect_is_gone(
        self, config, dump, prune, clock, backup_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (backup_root / "suspect").write_text("not a directory", encoding="utf-8")
        switch = refuse_rmtree_of(monkeypatch, "20261023T030000Z")
        dump.steps = [DumpStep(nodes=0), DumpStep()]
        scheduler = make_scheduler(config, dump, prune, clock)
        scheduler.tick()

        switch["refuse"] = False
        shutil.rmtree(backup_root / "20261023T030000Z")
        clock.advance(timedelta(hours=24))
        scheduler.tick()

        assert prune.calls == [["--retain", "7", "--confirm"]]

    @pytest.mark.parametrize(
        ("step", "expected_result"),
        [
            pytest.param(DumpStep(exit_code=1), "failed", id="failed"),
            pytest.param(DumpStep(nodes=0), "suspect", id="suspect"),
        ],
    )
    def test_a_baseline_accepted_during_an_attempt_survives_its_result(
        self, config, dump, prune, clock, step: DumpStep, expected_result: str
    ) -> None:
        seed_success(config, at=START - timedelta(hours=72), nodes=100)

        def accept_meanwhile() -> None:
            write_status(
                config.status_path,
                BackupStatus(
                    last_result="ok",
                    last_success_at=START - timedelta(hours=1),
                    last_label="accepted",
                    last_success_counts=SuccessCounts(
                        graph_nodes=30, vault_corpus_files=4, stores=("event_store.db",)
                    ),
                ),
            )

        step.side_effect = accept_meanwhile
        dump.steps = [step]
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        raw = status_json(config)
        assert raw["last_result"] == expected_result
        assert raw["last_success_at"] == "2026-10-23T02:00:00Z"
        assert raw["last_success_counts"]["graph_nodes"] == 30


class TestCaughtErrors:
    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(Neo4jConnectionError("graph unreachable"), id="neo4j-connection"),
            pytest.param(BackupError("leg failed"), id="backup-error"),
            pytest.param(GraphArtifactError("non-finite float"), id="graph-artifact"),
            pytest.param(OSError("no space left"), id="os-error"),
            pytest.param(sqlite3.OperationalError("database is locked"), id="sqlite"),
        ],
    )
    def test_an_attempt_error_is_recorded_as_failed(
        self, config, dump, prune, clock, error: BaseException
    ) -> None:
        dump.steps = [DumpStep(raises=error)]
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        raw = status_json(config)
        assert raw["last_result"] == "failed"
        assert raw["last_exit_code"] == 1
        assert prune.calls == []

    def test_an_exception_outside_the_set_propagates(self, config, dump, prune, clock) -> None:
        dump.steps = [DumpStep(raises=ValueError("a bug"))]
        scheduler = make_scheduler(config, dump, prune, clock)

        with pytest.raises(ValueError, match="a bug"):
            scheduler.tick()

    def test_the_loop_takes_its_next_tick_after_a_neo4j_connection_error(
        self, config, dump, prune, clock, stop
    ) -> None:
        dump.steps = [DumpStep(raises=Neo4jConnectionError("graph unreachable")), DumpStep()]
        sleeper = FakeSleeper(clock=clock, stop=stop, stop_after=7)
        scheduler = make_scheduler(config, dump, prune, clock, stop=stop, sleeper=sleeper)

        code = scheduler.run_forever()

        assert code == 0
        assert sleeper.calls == [600.0] * 7
        assert len(dump.calls) == 2
        assert status_json(config)["last_result"] == "ok"

    @pytest.mark.parametrize(
        "manifest_text",
        [
            pytest.param("{truncated", id="unparsable"),
            pytest.param(
                json.dumps({"layout": BACKUP_LAYOUT, "layout_version": 99, "created_at": "x"}),
                id="unreadable-layout",
            ),
        ],
    )
    def test_an_unreadable_manifest_after_exit_0_is_suspect_and_moved(
        self, config, dump, prune, clock, backup_root: Path, manifest_text: str
    ) -> None:
        dump.steps = [DumpStep(manifest_text=manifest_text)]
        scheduler = make_scheduler(config, dump, prune, clock)

        outcome = scheduler.tick()

        label = "20261023T030000Z"
        assert outcome is not None
        assert outcome.cli_exit_code == 3
        assert status_json(config)["last_result"] == "suspect"
        assert not (backup_root / label).exists()
        assert (backup_root / "suspect" / label).is_dir()
        assert prune.calls == []

    def test_a_failed_prune_does_not_undo_the_success(self, config, dump, clock) -> None:
        prune = FakePrune(exit_code=1)
        scheduler = make_scheduler(config, dump, prune, clock)

        outcome = scheduler.tick()

        assert outcome is not None
        assert outcome.exit_code == 0
        assert status_json(config)["last_result"] == "ok"


class TestPruneOnlyAfterSuccess:
    @pytest.mark.parametrize(
        ("step", "expected_prunes"),
        [
            pytest.param(DumpStep(), 1, id="ok"),
            pytest.param(DumpStep(exit_code=1), 0, id="failed"),
            pytest.param(DumpStep(exit_code=2), 0, id="refused"),
            pytest.param(DumpStep(nodes=0), 0, id="suspect"),
            pytest.param(DumpStep(raises=OSError("gone")), 0, id="raised"),
        ],
    )
    def test_prune_runs_only_after_a_success(
        self, config, dump, prune, clock, step: DumpStep, expected_prunes: int
    ) -> None:
        dump.steps = [step]
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        assert len(prune.calls) == expected_prunes


class TestStatusFileContract:
    @pytest.mark.parametrize(
        "step",
        [
            pytest.param(DumpStep(), id="ok"),
            pytest.param(DumpStep(exit_code=1), id="failed"),
            pytest.param(DumpStep(nodes=0), id="suspect"),
        ],
    )
    def test_the_file_has_exactly_the_item_6_keys(
        self, config, dump, prune, clock, step: DumpStep
    ) -> None:
        dump.steps = [step]
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        assert list(status_json(config)) == ITEM_6_KEYS


class TestOnce:
    @pytest.fixture
    def env(self, config: SchedulerConfig, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(config.backup_root))
        monkeypatch.setenv("MIST_BACKUP_STATUS_PATH", str(config.status_path))

    @pytest.mark.parametrize(
        ("step", "expected_code"),
        [
            pytest.param(DumpStep(), 0, id="success-0"),
            pytest.param(DumpStep(exit_code=1), 1, id="dump-failed-1"),
            pytest.param(DumpStep(exit_code=2), 2, id="dump-refused-2"),
            pytest.param(DumpStep(nodes=0), 3, id="suspect-3"),
            pytest.param(DumpStep(raises=Neo4jConnectionError("down")), 1, id="raised-1"),
        ],
    )
    def test_once_makes_one_attempt_and_exits_with_its_code(
        self, env, config, dump, prune, clock, step: DumpStep, expected_code: int
    ) -> None:
        seed_success(config, at=START - timedelta(hours=1))
        dump.steps = [step]

        code = main(["--once"], dump=dump, prune=prune, clock=clock)

        assert code == expected_code
        assert len(dump.calls) == 1

    def test_a_missing_status_path_is_refused_before_any_dump(
        self, config, dump, prune, clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(config.backup_root))
        monkeypatch.delenv("MIST_BACKUP_STATUS_PATH", raising=False)

        code = main(["--once"], dump=dump, prune=prune, clock=clock)

        assert code == 2
        assert dump.calls == []


class TestStop:
    def test_sigterm_during_an_attempt_exits_0_after_the_attempt(
        self, config, dump, prune, clock, stop
    ) -> None:
        dump.steps = [DumpStep(side_effect=lambda: signal.raise_signal(signal.SIGTERM))]
        sleeper = FakeSleeper(clock=clock, stop=stop, stop_after=100)
        scheduler = make_scheduler(config, dump, prune, clock, stop=stop, sleeper=sleeper)
        previous = install_sigterm_handler(stop)
        try:
            code = scheduler.run_forever()
        finally:
            signal.signal(signal.SIGTERM, previous)

        assert code == 0
        assert status_json(config)["last_result"] == "ok"
        assert len(prune.calls) == 1
        assert sleeper.calls == []

    def test_a_stop_during_the_sleep_exits_without_another_attempt(
        self, config, dump, prune, clock, stop
    ) -> None:
        sleeper = FakeSleeper(clock=clock, stop=stop, stop_after=1)
        scheduler = make_scheduler(config, dump, prune, clock, stop=stop, sleeper=sleeper)

        code = scheduler.run_forever()

        assert code == 0
        assert len(dump.calls) == 1

    def test_the_stop_flag_wait_returns_at_once_once_requested(self) -> None:
        flag = StopFlag()
        flag.request()
        slept: list[float] = []

        flag.wait(3600, sleep=slept.append)

        assert slept == []

    def test_a_stop_requested_mid_wait_ends_the_wait_within_one_slice(self) -> None:
        flag = StopFlag()
        slept: list[float] = []

        def sleep(seconds: float) -> None:
            slept.append(seconds)
            if len(slept) == 3:
                flag.request()

        flag.wait(600, sleep=sleep)

        assert slept == [1.0, 1.0, 1.0]

    def test_the_cli_loop_installs_a_handler_that_stops_after_the_attempt(
        self, config, dump, prune, clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(config.backup_root))
        monkeypatch.setenv("MIST_BACKUP_STATUS_PATH", str(config.status_path))
        dump.steps = [DumpStep(side_effect=lambda: signal.raise_signal(signal.SIGTERM))]
        slept: list[float] = []
        previous = signal.getsignal(signal.SIGTERM)
        try:
            code = main([], dump=dump, prune=prune, clock=clock, sleeper=slept.append)
        finally:
            signal.signal(signal.SIGTERM, previous)

        assert code == 0
        assert len(dump.calls) == 1
        assert slept == []


class TestAcceptSuspect:
    def hold_suspect(self, config: SchedulerConfig, label: str = "20261020T030000Z") -> Path:
        suspect_root = config.backup_root / "suspect"
        return write_artifact(
            suspect_root,
            label,
            created_at="2026-10-20T03:00:07Z",
            nodes=30,
            corpus_files=4,
            store_files={**ALL_STORES_PRESENT, "extraction_cache.db": None},
        )

    def test_accepting_a_held_label_moves_it_back_and_records_it_as_the_baseline(
        self, config
    ) -> None:
        seed_success(config, at=START - timedelta(hours=72), nodes=100)
        self.hold_suspect(config)

        code = accept_suspect(config, "20261020T030000Z")

        assert code == 0
        assert (config.backup_root / "20261020T030000Z" / "manifest.json").is_file()
        assert not (config.backup_root / "suspect" / "20261020T030000Z").exists()
        raw = status_json(config)
        assert raw["last_result"] == "ok"
        assert raw["last_success_at"] == "2026-10-20T03:00:07Z"
        assert raw["last_label"] == "20261020T030000Z"
        assert raw["last_success_counts"] == {
            "graph_nodes": 30,
            "vault_corpus_files": 4,
            "stores": ["event_store.db", "vault_sidecar.db"],
        }
        assert list(raw) == ITEM_6_KEYS

    def test_the_accepted_baseline_judges_the_next_dump(self, config, dump, prune, clock) -> None:
        seed_success(config, at=START - timedelta(hours=72), nodes=100)
        self.hold_suspect(config)
        accept_suspect(config, "20261020T030000Z")
        dump.steps = [DumpStep(nodes=30)]
        scheduler = make_scheduler(config, dump, prune, clock)

        scheduler.tick()

        assert status_json(config)["last_result"] == "ok"

    @pytest.mark.parametrize(
        "label",
        [
            pytest.param("20261019T030000Z", id="missing-label"),
            pytest.param("../20261020T030000Z", id="path-traversal"),
            pytest.param("..", id="parent"),
        ],
    )
    def test_a_label_not_held_exits_2_and_changes_nothing(self, config, label: str) -> None:
        seed_success(config, at=START - timedelta(hours=72))
        self.hold_suspect(config)
        before_status = config.status_path.read_bytes()
        before_tree = sorted(str(p) for p in config.backup_root.rglob("*"))

        code = accept_suspect(config, label)

        assert code == 2
        assert config.status_path.read_bytes() == before_status
        assert sorted(str(p) for p in config.backup_root.rglob("*")) == before_tree

    @pytest.mark.parametrize(
        "manifest_text",
        [
            pytest.param("{not json", id="unparsable"),
            pytest.param(
                json.dumps({"layout": BACKUP_LAYOUT, "layout_version": 99, "created_at": "x"}),
                id="unreadable-layout",
            ),
        ],
    )
    def test_an_unreadable_manifest_exits_2_and_changes_nothing(
        self, config, manifest_text: str
    ) -> None:
        seed_success(config, at=START - timedelta(hours=72))
        held = self.hold_suspect(config)
        (held / "manifest.json").write_text(manifest_text, encoding="utf-8")
        before_status = config.status_path.read_bytes()

        code = accept_suspect(config, "20261020T030000Z")

        assert code == 2
        assert held.is_dir()
        assert not (config.backup_root / "20261020T030000Z").exists()
        assert config.status_path.read_bytes() == before_status

    def test_accept_suspect_through_the_cli_makes_no_dump(
        self, config, dump, prune, clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(config.backup_root))
        monkeypatch.setenv("MIST_BACKUP_STATUS_PATH", str(config.status_path))
        self.hold_suspect(config)

        code = main(["--accept-suspect", "20261020T030000Z"], dump=dump, prune=prune, clock=clock)

        assert code == 0
        assert dump.calls == []
        assert prune.calls == []
        assert read_status(config.status_path) is not None
