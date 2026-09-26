"""Operator CLI for the extraction backlog.

    python -m backend.extraction_backlog.admin status
    python -m backend.extraction_backlog.admin retry-dead-letters [--event-id ID]

`status` reads the event store and extraction cache directly (it does not call
the extraction service or need the backend running) and prints the backlog for
the active epoch.

`retry-dead-letters` puts dead-lettered turns (`extraction_failed` skips) back
into the backlog: it deletes the skip row and the turn's applied marker for the
active epoch and restarts its failure count. A running dispatcher picks the
turn up on its next scan (at most `MIST_EXTRACTION_IDLE_POLL_S` later).

A retried turn is applied OUT OF LOG ORDER. Every turn logged after it has
already been applied, and graph outcomes that depend on order (which duplicate
wins dedup, which belief supersedes which) can then differ from a rebuild of
the same log until the next epoch cutover re-extracts it. The command says so
when it runs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TextIO

from backend.event_store.store import EventStore
from backend.knowledge.extraction_cache import ExtractionCache

from .store import BacklogStore

OUT_OF_ORDER_WARNING = (
    "[WARNING] A retried turn is applied OUT OF LOG ORDER: turns logged after it are "
    "already in the graph. Order-dependent results (dedup winners, supersession) may "
    "differ from a rebuild until the next epoch cutover re-extracts the log."
)


def build_store_from_env() -> BacklogStore:
    """Open the production event store and extraction cache named by the environment."""
    from backend.factories import production_cache_path
    from backend.knowledge.config import KnowledgeConfig

    config = KnowledgeConfig.from_env()
    event_store = EventStore(db_path=config.event_store.db_path)
    event_store.initialize()
    cache = ExtractionCache(production_cache_path(config))
    cache.initialize()
    return BacklogStore(event_store, cache)


def _status(store: BacklogStore, out: TextIO) -> int:
    epoch = store.active_epoch()
    if epoch is None:
        print("No epoch in the ledger; nothing to report.", file=out)
        return 1
    scan = store.scan(epoch)
    activation = store.get_activation(epoch)
    print(
        f"epoch {epoch.epoch_id}: extraction_version={epoch.extraction_version} "
        f"model_hash={epoch.model_hash}",
        file=out,
    )
    if activation is None:
        print("activation: not yet activated (no dispatcher has run for this epoch)", file=out)
    else:
        print(
            f"activation: {activation.activated_at} "
            f"(turns_at_activation={activation.turns_at_activation}, "
            f"marked_applied={activation.marked_applied}, "
            f"legacy_unextracted={activation.legacy_unextracted})",
            file=out,
        )
    print(f"backlog_depth={scan.backlog_depth}", file=out)
    print(f"apply_pending={scan.apply_pending}", file=out)
    print(f"dead_lettered={scan.dead_lettered}", file=out)
    print(f"legacy_unextracted={scan.legacy_unextracted}", file=out)
    print(f"oldest_pending={scan.oldest_pending_timestamp or '-'}", file=out)
    for event_id in store.list_dead_letters(epoch):
        print(f"dead_letter event_id={event_id}", file=out)
    return 0


def _retry(store: BacklogStore, event_id: str | None, out: TextIO) -> int:
    epoch = store.active_epoch()
    if epoch is None:
        print("No epoch in the ledger; nothing to retry.", file=out)
        return 1
    targets = [event_id] if event_id is not None else store.list_dead_letters(epoch)
    if not targets:
        print("No dead-lettered turns.", file=out)
        return 0
    retried = 0
    for target in targets:
        if store.retry_dead_letter(target, epoch):
            retried += 1
            print(f"retried event_id={target}", file=out)
        else:
            print(f"skipped event_id={target} (not dead-lettered under the active epoch)", file=out)
    if retried:
        print(OUT_OF_ORDER_WARNING, file=out)
    return 0 if retried == len(targets) else 2


def main(
    argv: Sequence[str] | None = None,
    *,
    store: BacklogStore | None = None,
    out: TextIO | None = None,
) -> int:
    """Run the CLI. `store` and `out` are injectable for tests.

    Returns:
        Process exit code: 0 success, 1 no epoch, 2 some requested turn was not
        dead-lettered.
    """
    parser = argparse.ArgumentParser(prog="python -m backend.extraction_backlog.admin")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="print the backlog for the active epoch")
    retry = sub.add_parser(
        "retry-dead-letters", help="put dead-lettered turns back into the backlog"
    )
    retry.add_argument("--event-id", default=None, help="retry only this turn")
    args = parser.parse_args(argv)

    stream = out if out is not None else sys.stdout
    backlog = store if store is not None else build_store_from_env()
    if args.command == "status":
        return _status(backlog, stream)
    return _retry(backlog, args.event_id, stream)


if __name__ == "__main__":
    sys.exit(main())
