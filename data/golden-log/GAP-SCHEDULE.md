# Golden log -- the gap schedule

**Authored and committed 2026-08-02, before the executor read any R1.5 document.**

## 1. Why the ordering matters, and what is actually claimed here

Spec 6b: whoever authors this log chooses the gaps, and will naturally author gaps that
straddle whatever staleness window R1.5 already contains. The test then passes by
construction and reads as calibration evidence. Authored data sits closer to real data than
a fixture does, which makes that false confidence more expensive, not less.

**Spec 6b's mitigation as literally worded -- "fixed before R1.5's window constant is
chosen" -- was already unachievable when R1.4.5 began.**
`docs/superpowers/specs/2026-08-01-r1.5-staleness-design.md` section 6 predates this file by
a day and already names a candidate: 180 days, borrowed from
`CONFIDENCE_EXTERNAL.decay_half_life_days` rather than derived fresh. It is a candidate
rather than a settled constant, but it existed. And `docs/` is gitignored, so git history
cannot establish ordering against it either -- the audit route the first draft of this file
suggested does not exist.

So the claim made here is the narrower one that is actually true:

1. **Process:** the R1.5 design and plan were deliberately not opened while the schedule was
   being authored. The schedule was committed (`938568c`) before either was read.
2. **Property, not ordering:** the gaps satisfy a rule chosen without reference to any window
   value, stated in section 2 below and machine-checked by
   `tests/unit/golden_log/test_generate.py::TestGapScheduleShape`. That test asserts a
   *spread*, never a threshold, so it cannot be quietly retuned to match a constant later.
3. **Disclosure:** under the 180-day candidate the eight never-restated LEARNING assertions
   split four stale (310, 298, 195, 181 d) and four fresh (137, 92, 31, 9 d). `ocaml` at
   181 d lands one day past 180. That near-boundary is coincidence, not tuning -- 181 falls
   out of the authored ladder -- but it is called out here so nobody reads it as evidence
   the window was validated at its edge.

Point 1 is a process claim and is not independently verifiable. Point 2 is, and is the one
that should carry weight.

---

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
