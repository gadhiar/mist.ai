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
    `errors`       `BackupError` and friends, all under `MistError`

The restore leg, the destructive guard and retention are MIS-140 T2 and land in
this package beside these modules.
"""
