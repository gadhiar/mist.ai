# analyse.py fixture: hand-built, not recorded

Everything under this directory is hand-built, matching the `bench_host.py` results layout
documented in `scripts/model_bench/README.md`. No live bench run, no real llama-server, and no
real layout-perception harness produced any of it.

The per-case values (which layout rows are `correct`, which harness cases score what, which
`vram.csv` rows set the peak) were chosen by hand to drive specific, named branches of
`decision_rules.json`'s rules (R1-R7): a clean pass, a fail, a missing input, an incomplete
coverage count, the `L` clause's thinking branch, the finalist pass superseding an
inconclusive screen pass, the `F` clause's needs-review band, and the `R6` determinism check
failing on one tuning arm. See `tests/unit/model_bench/test_analyse_rules.py` for which arm
exercises which branch.

The bulk of the volume -- the 72/216-row layout `graded.jsonl` files and the harness
`CaseResult` JSONL, which must cover every case in the real
`scripts/eval_harness/tests/*.yaml` files for `run.load_test_files` coverage checks to line up
-- was produced by a short local script (not committed) that applied those hand-chosen values
programmatically rather than requiring each line to be typed by hand. Harness responses use a
generic, deterministic "every entity/relationship type the case's `expected` block asks for,
nothing else" construction, so every harness score in this fixture is either exactly 1.0 or
(for a `finish_reason` deliberately set to `"length"`) 1.0 with truncation flagged separately --
`finish_reason` does not affect `schema_conformance`/`tool_selection` scoring.

One record (`c0/harness/harness/bench-c0.jsonl`, case `simple_usage`) carries the literal
sentinel string `SENTINEL_DO_NOT_LEAK_9f3a21` in `prompt`, `system_prompt`,
`metrics.reasoning_content`, and one entity's `name` inside `response_content`. This fixture
exists to prove that string never reaches `REPORT.md`, `summary.json`, or any `grades/` file
(decision 8: no model text in a public-repo output).
