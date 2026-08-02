# Golden log -- the gap schedule

**Authored 2026-08-02, BEFORE any R1.5 staleness-window constant was consulted.**

This file exists to make that ordering checkable. The elapsed-time distribution below was
fixed first, committed first, and the R1.5 design document was deliberately not read while
choosing it. Anyone auditing this can verify the claim from `git log --diff-filter=A` on this
file versus the commit that introduces a window constant.

---

## 1. Why the ordering matters

Spec 6b: whoever authors this log chooses the gaps, and will naturally author gaps that
straddle whatever staleness window R1.5 already contains. The test then passes by
construction and reads as calibration evidence. Authored data sits closer to real data than
a fixture does, which makes that false confidence more expensive, not less.

So the gaps were chosen from a property that is independent of any particular window value.

## 2. The property the schedule is built on

**No single staleness window may classify every never-restated LEARNING assertion the same
way.**

The corpus therefore carries never-restated LEARNING assertions whose elapsed time to the end
of the log spans two orders of magnitude, 9 days to 310 days:

| Turn id | Asserted | Elapsed to log end | Notes |
|---|---|---|---|
| `stale-learning-haskell` | 2025-09-08 | 310 d | oldest never-restated |
| `stale-learning-ocaml` | 2026-01-15 | 181 d | |
| `ext-26-learning` (kubernetes) | 2026-01-01 | 195 d | from the gold corpus, not authored here |
| `stale-learning-zig` | 2026-04-14 | 92 d | |
| `stale-learning-nim` | 2026-06-14 | 31 d | |
| `stale-learning-lua` | 2026-07-06 | 9 d | newest never-restated |

Whatever constant R1.5 picks, some of these fall on each side of it. A test over this corpus
must state a per-case expectation, and a mis-set window breaks at least one of them. A corpus
whose gaps all sat on one side of the window would pass for any window and prove nothing.

**Log end is `2026-07-15T12:00:00+00:00`**, fixed by the final turn (`log-tail-anchor`, which
asserts no facts). Elapsed times above are measured against it, so they do not move when
wall-clock time passes.

## 3. Controls

These exist so a window that is merely aggressive cannot pass by aging out everything.

| Case | Turn ids | Property |
|---|---|---|
| Negative control A -- restated, must survive | `fresh-learning-swift-1/-2/-3` | LEARNING re-asserted at 2025-09-25, 2026-02-10, 2026-07-08. Last assertion 7 d before log end. |
| Negative control B -- spoken cease | `prior-learning-clojure` then `ext-45-cease-learning` | C3 already closes this edge by cessation. R1.5 must not double-handle it. |
| Re-assertion sequence (spec 4b) | `reassert-uses-postgres-1..4` | The same USES fact at 2025-10-20, 2026-01-25, 2026-05-25, 2026-07-01 -- elapsed 268 / 171 / 51 / 14 d. Each assertion after the first takes the REINFORCE path, which does not advance `recorded_at` -- so only a sequence like this can prove an R1.5 `last_asserted_at` advances. |
| Closed by progression, not by staleness | `ext-06-learning` (rust) then `progression-expert-rust` | LEARNING rust is 298 d old at log end but is closed by EXPERT_IN progression. A staleness pass must not claim it. |
| Closed by stated valid time | `ext-35-validtime-until` (go) | Bounded by `valid_to: 2025-12` in gold. |

## 4. Inter-turn gaps for the gold corpus

The 60 gold records are placed on an authored ladder of day-gaps, cycling:

```
1, 2, 5, 1, 8, 3, 12, 2, 6, 18, 1, 1
```

starting at `2025-09-02T08:00:00+00:00`, which places `ext-60` at `2026-06-28T19:00:00+00:00`
-- a span of just under ten months, satisfying spec 4 requirement 1 (timestamps must span
months). The ladder was applied once and its output written literally into
`turn-schedule.yaml`; that file is the authority from here on, so editing the ladder does not
silently reshuffle every gap.

Time-of-day cycles `08:00` through `19:00` so that no two turns share a timestamp and rowid
order matches timestamp order.

## 5. Close-reason coverage

Spec 4 requirement 4: replay should exercise all five reasons the reconciliation engine can
emit, rather than plain asserts only.

| Reason | Produced by |
|---|---|
| `retract` | `prior-learning-scala` then `ext-48`; also figma, initech, kafka, elixir priors then `ext-49`/`-50`/`-51`/`-52` |
| `cease` | `prior-uses-mongodb` then `ext-43`; also wayfair, clojure, vim, notion |
| `single_supersession` | `prior-works-at-wayfair` closing the open WORKS_AT (cardinality SINGLE) |
| `contradiction` | `ext-07-expert` (EXPERT_IN python) then `contradiction-struggles-python` |
| `progression` | `ext-06-learning` (LEARNING rust) then `progression-expert-rust` (EXPERT_IN progression-supersedes LEARNING) |

Every gold cease and retract is given an authored prior assertion so the close actually fires.
Without a prior, a cease plans `FLAG_AMBIGUOUS / cease_without_prior` instead of a close --
that path is covered by reconciliation unit tests and is deliberately not what this corpus
exercises.

## 6. What this log does NOT establish

It validates mechanism, not calibration. It cannot tell you how long a real person goes silent
about a fact that is still true; that is a behavioural quantity, observable only from
behaviour. Any report citing this log must say so.
