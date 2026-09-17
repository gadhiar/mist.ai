"""Disaster-recovery capture for live MIST.AI state (MIS-140).

A live-safe sibling of `scripts/hydration/`, and NOT a second copy of it. The
hydration snapshot exists to reproduce a test fixture, so it refuses an artifact
whose producer stamps drifted from the tree
(`grep -n "def assert_fresh" scripts/hydration/manifest.py` -> :242). This
package exists to survive the loss of the machine, so it records those stamps
and never gates on them.

Module map:
    `destination`  where a backup may be written, and every refusal
    `stores`       the named SQLite stores, captured through `sqlite3.backup()`
    `manifest`     what an artifact says about itself; the layout version check
    `dump`         the capture command and its CLI
    `target`       where a restore may write: the handshake marker and the
                   typed confirmation token
    `restore`      the destructive leg, its four gates and its CLI
    `prune`        retention by manifest timestamp, and what it refuses to touch
    `errors`       `BackupError` and friends, all under `MistError`

`README.md` beside these modules is the operator runbook: how to take a backup,
what is and is not captured, the restore rehearsal against the dev-hydration
stack, and why no schedule is armed.
"""
