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

## Plan v2 (2026-09-25): c1-1024 and c0-ctx64k, hand-built to exercise the exploratory rules

Two more arm directories, added by hand (not by the local script above) to exercise
`decision_rules.json`'s exploratory X1/X2/X3 (see `scripts/model_bench/README.md`):

- **`c1-1024/`** -- `layout/screen/graded.jsonl` is c1-512's own 72-row fixture, copied verbatim
  (65/72 correct, `model`/`id` fields relabeled `c1-1024`) rather than generated fresh: the same
  accuracy/timing distribution that makes c1-512 clear R1's bar cleanly also clears X1's identical
  layout_acc threshold and R2's P threshold, so reusing it is a deliberate simplification, not an
  oversight. `ttft.jsonl` reuses c2's three ctx=2048 measured rows (decode_tps median 40 >= X1's D
  threshold of 35). Together these drive X1 to a clean **pass**. `meta.json`'s
  `decision_rules_sha256` is deliberately set to the *v1* sha
  (`f6a42ba8d36084fd5c893ac430294493cd4d1f7cb8da2d40a8f29add49aa3fae`, listed in the current
  `decision_rules.json`'s `supersedes`) rather than the current fixture placeholder sha every other
  arm here uses -- this is what exercises the `[INFO]`, not `[WARN]`, classification path.
- **`c0-ctx64k/`** -- `ttft.jsonl` carries both a ctx=2048 group (decode_tps) and a ctx=65000 group
  (three rows, so `ttft_ms[ctx=65000]` is non-missing), `vram.csv` gives it a peak, and
  `correctness.r1.jsonl` is `c0/correctness.r1.jsonl` copied verbatim -- so X2's
  `correctness_tokens_vs_c0` reads `"identical"` for this arm. No `harness` suite is attached (kept
  out of scope for this fixture pass), so X2's `harness_vs_c0` rows for c0-ctx64k report `null`
  values rather than a computed delta; the other two context arms (`c0-ctx128k`,
  `c0-ctx128k-q4kv`) and the optional `c0-ub1024` arm have no fixture directory at all, so X2's
  `arms` table also exercises the `"present": false` row shape for them.

X1's fail/missing branches, and X3's F_sep evaluated against a genuinely `fail` (not just
`needs-review`) v1 F, are covered by hand-built `RunMetrics`/`ArmMetrics` objects directly in
`test_analyse_rules.py` rather than by more fixture directories here -- the existing convention in
this file (see "hand-built synthetic metric bundles" in that test module's docstring) for branches
a single fixture run cannot exercise simultaneously.
