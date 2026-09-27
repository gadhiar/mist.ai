"""Instrumentation for the live-path smoke experiment.

Three scripts plus a corpus and a runbook, all driving or measuring the
throwaway stack defined by `docker-compose.live-path-smoke.yml`:

- `turns.json`        -- the five-turn synthetic conversation.
- `drive_turns.py`    -- the WebSocket client that speaks it. The ONLY module
                         here that needs a third-party import (`websockets`),
                         and it imports it lazily so this package stays
                         importable on a host without it.
- `baseline.py`       -- pre/post snapshots of LIVE and dev state, plus a
                         `--compare` mode that adjudicates the deltas.
- `assert_artifacts.py` -- the four artifact assertions (A1..A4), each
                         PASS / FAIL / INCONCLUSIVE with its evidence inline.
- `RUNBOOK.md`        -- the literal command sequence, with abort conditions.

`baseline.py` and `assert_artifacts.py` are pure standard library on purpose:
they run on the HOST during the experiment, and requiring a pip install on the
host to measure a containerised run would be a new failure mode in the
measuring instrument.
"""
