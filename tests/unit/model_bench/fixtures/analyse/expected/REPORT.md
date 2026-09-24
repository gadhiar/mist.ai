# mist-model-bench analysis report

decision_rules.json sha256: `0cb9a05be5b05e184bdbc85add56f559179d6593b60671df0e772cde9cda21ce`

## decision_rules.json mismatch warnings

- [WARN] arm 'c1-256' meta.decision_rules_sha256='deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef' differs from the analysed decision_rules.json='0cb9a05be5b05e184bdbc85add56f559179d6593b60671df0e772cde9cda21ce'

## Per-arm metrics

### c0-old

| metric | value | n | n_expected | complete | wilson | bootstrap |
|---|---|---|---|---|---|---|
| harness_score[schema_conformance] | 1.0000 | 12 | 12 | True | [0.7575, 1.0000] | [1.0000, 1.0000] |
| truncation_rate[schema_conformance] | 0.0000 | 12 | 12 | True | [0.0000, 0.2425] | [0.0000, 0.0000] |
| decode_tps | n/a | n/a | n/a | n/a | n/a | n/a |
| arm_peak_mib | n/a | n/a | n/a | n/a | n/a | n/a |

### c0

| metric | value | n | n_expected | complete | wilson | bootstrap |
|---|---|---|---|---|---|---|
| harness_score[schema_conformance] | 1.0000 | 12 | 12 | True | [0.7575, 1.0000] | [1.0000, 1.0000] |
| truncation_rate[schema_conformance] | 0.0000 | 12 | 12 | True | [0.0000, 0.2425] | [0.0000, 0.0000] |
| decode_tps | n/a | n/a | n/a | n/a | n/a | n/a |
| arm_peak_mib | n/a | n/a | n/a | n/a | n/a | n/a |

### c0-prod

| metric | value | n | n_expected | complete | wilson | bootstrap |
|---|---|---|---|---|---|---|
| harness_score[schema_conformance_json_object] | 1.0000 | 12 | 12 | True | [0.7575, 1.0000] | [1.0000, 1.0000] |
| harness_score[tool_selection] | 1.0000 | 8 | 8 | True | [0.6756, 1.0000] | [1.0000, 1.0000] |
| truncation_rate[schema_conformance_json_object] | 0.0000 | 12 | 12 | True | [0.0000, 0.2425] | [0.0000, 0.0000] |
| truncation_rate[tool_selection] | 0.0000 | 8 | 8 | True | [0.0000, 0.3244] | [0.0000, 0.0000] |
| decode_tps | n/a | n/a | n/a | n/a | n/a | n/a |
| arm_peak_mib | n/a | n/a | n/a | n/a | n/a | n/a |

### c1-256

| metric | value | n | n_expected | complete | wilson | bootstrap |
|---|---|---|---|---|---|---|
| layout_acc[screen] | 0.6250 | 40 | 72 | False | [0.4703, 0.7578] | [0.2500, 0.8750] |
| layout_mean_completion_tokens[screen] | 220.0000 | 40 | n/a | n/a | n/a | n/a | (per_correct=352.0000) |
| layout_p95_wall_ms[screen] | 7500.0000 | 40 | n/a | n/a | n/a | n/a |
| decode_tps | n/a | n/a | n/a | n/a | n/a | n/a |
| arm_peak_mib | n/a | n/a | n/a | n/a | n/a | n/a |

### c1-512

| metric | value | n | n_expected | complete | wilson | bootstrap |
|---|---|---|---|---|---|---|
| layout_acc[screen] | 0.9028 | 72 | 72 | True | [0.8126, 0.9521] | [0.7361, 1.0000] |
| layout_mean_completion_tokens[screen] | 219.5833 | 72 | n/a | n/a | n/a | n/a | (per_correct=243.2308) |
| layout_p95_wall_ms[screen] | 7500.0000 | 72 | n/a | n/a | n/a | n/a |
| decode_tps | n/a | n/a | n/a | n/a | n/a | n/a |
| arm_peak_mib | n/a | n/a | n/a | n/a | n/a | n/a |

### c2

| metric | value | n | n_expected | complete | wilson | bootstrap |
|---|---|---|---|---|---|---|
| layout_acc[screen] | 0.4167 | 72 | 72 | True | [0.3099, 0.5319] | [0.1667, 0.6667] |
| layout_mean_completion_tokens[screen] | 219.5833 | 72 | n/a | n/a | n/a | n/a | (per_correct=527.0000) |
| layout_p95_wall_ms[screen] | 7500.0000 | 72 | n/a | n/a | n/a | n/a |
| harness_score[schema_conformance_json_object] | 1.0000 | 12 | 12 | True | [0.7575, 1.0000] | [1.0000, 1.0000] |
| harness_score[tool_selection] | 1.0000 | 8 | 8 | True | [0.6756, 1.0000] | [1.0000, 1.0000] |
| truncation_rate[schema_conformance_json_object] | 0.0000 | 12 | 12 | True | [0.0000, 0.2425] | [0.0000, 0.0000] |
| truncation_rate[tool_selection] | 0.0000 | 8 | 8 | True | [0.0000, 0.3244] | [0.0000, 0.0000] |
| decode_tps | 40.0000 | 3 | n/a | n/a | n/a | n/a |
| ttft_ms[ctx=2048] | 111.0000 | 3 | n/a | n/a | n/a | n/a |
| ttft_ms[ctx=4096] | n/a | n/a | n/a | n/a | n/a | n/a |
| arm_peak_mib | 7000.0000 | 5 | n/a | n/a | n/a | n/a |

### c2-think512

| metric | value | n | n_expected | complete | wilson | bootstrap |
|---|---|---|---|---|---|---|
| layout_acc[screen] | 0.9167 | 72 | 72 | True | [0.8299, 0.9612] | [0.7500, 1.0000] |
| layout_mean_completion_tokens[screen] | 340.0000 | 72 | n/a | n/a | n/a | n/a | (per_correct=370.9091) |
| layout_p95_wall_ms[screen] | 7500.0000 | 72 | n/a | n/a | n/a | n/a |
| decode_tps | n/a | n/a | n/a | n/a | n/a | n/a |
| arm_peak_mib | n/a | n/a | n/a | n/a | n/a | n/a |

### c3

| metric | value | n | n_expected | complete | wilson | bootstrap |
|---|---|---|---|---|---|---|
| layout_acc[finalist] | 0.8333 | 216 | 216 | True | [0.7779, 0.8771] | [0.8333, 0.8333] |
| layout_acc[screen] | 0.7500 | 72 | 72 | True | [0.6391, 0.8356] | [0.5000, 1.0000] |
| layout_mean_completion_tokens[finalist] | 219.9074 | 216 | n/a | n/a | n/a | n/a | (per_correct=263.8889) |
| layout_mean_completion_tokens[screen] | 219.5833 | 72 | n/a | n/a | n/a | n/a | (per_correct=292.7778) |
| layout_p95_wall_ms[finalist] | 7500.0000 | 216 | n/a | n/a | n/a | n/a |
| layout_p95_wall_ms[screen] | 7500.0000 | 72 | n/a | n/a | n/a | n/a |
| harness_score[schema_conformance] | 1.0000 | 12 | 12 | True | [0.7575, 1.0000] | [1.0000, 1.0000] |
| harness_score[schema_conformance_json_object] | 1.0000 | 12 | 12 | True | [0.7575, 1.0000] | [1.0000, 1.0000] |
| harness_score[tool_selection] | 1.0000 | 8 | 8 | True | [0.6756, 1.0000] | [1.0000, 1.0000] |
| truncation_rate[schema_conformance] | 0.0833 | 12 | 12 | True | [0.0149, 0.3539] | [0.0000, 0.2500] |
| truncation_rate[schema_conformance_json_object] | 0.1667 | 12 | 12 | True | [0.0470, 0.4480] | [0.0000, 0.4167] |
| truncation_rate[tool_selection] | 0.0000 | 8 | 8 | True | [0.0000, 0.3244] | [0.0000, 0.0000] |
| decode_tps | 44.0000 | 3 | n/a | n/a | n/a | n/a |
| ttft_ms[ctx=2048] | 111.0000 | 3 | n/a | n/a | n/a | n/a |
| ttft_ms[ctx=4096] | n/a | n/a | n/a | n/a | n/a | n/a |
| arm_peak_mib | 8500.0000 | 5 | n/a | n/a | n/a | n/a |

### c3-think512

no meta.json / arm absent

### c4

no meta.json / arm absent

### c4-think512

no meta.json / arm absent

### a1

| metric | value | n | n_expected | complete | wilson | bootstrap |
|---|---|---|---|---|---|---|
| harness_score[schema_conformance] | 1.0000 | 12 | 12 | True | [0.7575, 1.0000] | [1.0000, 1.0000] |
| truncation_rate[schema_conformance] | 0.0000 | 12 | 12 | True | [0.0000, 0.2425] | [0.0000, 0.0000] |
| decode_tps | n/a | n/a | n/a | n/a | n/a | n/a |
| arm_peak_mib | n/a | n/a | n/a | n/a | n/a | n/a |

### a2

| metric | value | n | n_expected | complete | wilson | bootstrap |
|---|---|---|---|---|---|---|
| harness_score[schema_conformance] | 1.0000 | 12 | 12 | True | [0.7575, 1.0000] | [1.0000, 1.0000] |
| truncation_rate[schema_conformance] | 0.0000 | 12 | 12 | True | [0.0000, 0.2425] | [0.0000, 0.0000] |
| decode_tps | n/a | n/a | n/a | n/a | n/a | n/a |
| arm_peak_mib | n/a | n/a | n/a | n/a | n/a | n/a |

### a3

| metric | value | n | n_expected | complete | wilson | bootstrap |
|---|---|---|---|---|---|---|
| harness_score[schema_conformance] | 1.0000 | 12 | 12 | True | [0.7575, 1.0000] | [1.0000, 1.0000] |
| truncation_rate[schema_conformance] | 0.0000 | 12 | 12 | True | [0.0000, 0.2425] | [0.0000, 0.0000] |
| decode_tps | n/a | n/a | n/a | n/a | n/a | n/a |
| arm_peak_mib | n/a | n/a | n/a | n/a | n/a | n/a |

### a4

no meta.json / arm absent

## Session (voice VRAM)

- voice_vram_lower_mib: 2500.0000
- voice_vram_upper_mib: 3700.0000
- total_mib: 12288.0000

## Rules

### R1 `keep_e4b_budget` -- gate

Does Gemma 4 E4B with a thinking budget clear the layout accuracy bar?

verdict: **pass**

| clause | metric | arm | value | threshold | op | verdict | margin | note |
|---|---|---|---|---|---|---|---|---|
| layout_acc_c1_512 | layout_acc | c1-512 | 0.9028 | 0.8500 | >= | pass | within-noise | n/a |

info: `{"c1_256_layout_acc": {"complete": false, "n": 40, "n_expected": 72, "pass_used": "screen", "value": 0.625}}`

### R2 `switch_to_c2` -- gate

Should MIST switch from Gemma 4 E4B to c2?

verdict: **pass**

| clause | metric | arm | value | threshold | op | verdict | margin | note |
|---|---|---|---|---|---|---|---|---|
| L | layout_acc | c2-think512 | 0.9167 | 0.8900 | >= | pass | within-noise | n/a |
| S1 | harness_score | c2 | 1.0000 | 0.8350 | >= | pass | clear | n/a |
| S2 | harness_score | c2 | 1.0000 | 0.8750 | >= | pass | clear | n/a |
| D | decode_tps | c2 | 40.0000 | 35.0000 | >= | pass | n/a | n/a |
| P | layout_p95_wall_ms | c2-think512 | 7500.0000 | 15000 | <= | pass | n/a | n/a |
| F | arm_peak_mib | c2 | 7000.0000 | 12288.0000 | <= | pass | n/a | n/a |

info: `{"candidate": "c2", "thinking_candidate": "c2-think512"}`

### R2 `switch_to_c3` -- gate

Should MIST switch from Gemma 4 E4B to c3?

verdict: **needs-review**

| clause | metric | arm | value | threshold | op | verdict | margin | note |
|---|---|---|---|---|---|---|---|---|
| L | layout_acc | c3 | 0.8333 | 0.7500 | >= | pass | clear | n/a |
| S1 | harness_score | c3 | 1.0000 | 0.8350 | >= | pass | clear | n/a |
| S2 | harness_score | c3 | 1.0000 | 0.8750 | >= | pass | clear | n/a |
| D | decode_tps | c3 | 44.0000 | 35.0000 | >= | pass | n/a | n/a |
| P | layout_p95_wall_ms | c3 | 7500.0000 | 15000 | <= | pass | n/a | n/a |
| F | arm_peak_mib | c3 | 8500.0000 | 12288.0000 | <= | needs-review | n/a | n/a |

info: `{"candidate": "c3", "thinking_candidate": "c3-think512"}`

### R2 `switch_to_c4` -- gate

Should MIST switch from Gemma 4 E4B to c4?

verdict: **missing**

| clause | metric | arm | value | threshold | op | verdict | margin | note |
|---|---|---|---|---|---|---|---|---|
| L | layout_acc | c4 | n/a | 0.7500 | >= | missing | n/a | insufficient data to evaluate L on either branch |
| S1 | harness_score | c4 | n/a | 0.8350 | >= | missing | n/a | no usable harness_score for c4/schema_conformance_json_object |
| S2 | harness_score | c4 | n/a | 0.8750 | >= | missing | n/a | no usable harness_score for c4/tool_selection |
| D | decode_tps | c4 | n/a | 35.0000 | >= | missing | n/a | no ttft.jsonl for this arm |
| P | layout_p95_wall_ms | c4 | n/a | 15000 | <= | missing | n/a | no layout_p95_wall_ms available on the arm used for P |
| F | arm_peak_mib | c4 | n/a | n/a | n/a | missing | n/a | missing inputs: ['arm_peak_mib(c4)'] |

info: `{"candidate": "c4", "thinking_candidate": "c4-think512"}`

### R3 `drop_moe` -- trigger

Is c3's MoE output truncated often enough to drop the MoE candidates?

verdict: **triggered**

| clause | metric | arm | value | threshold | op | verdict | margin | note |
|---|---|---|---|---|---|---|---|---|
| truncation_json_object | truncation_rate | c3 | 0.1667 | 0.1000 | > | triggered | within-noise | n/a |

info: `{"truncation_grammar_schema_conformance": {"complete": true, "n": 12, "n_expected": 12, "value": 0.083333}}`

### R4 `thinking_is_lever` -- trigger

Is thinking mode a large enough lever on layout accuracy at c2 to be load-bearing?

verdict: **triggered**

| clause | metric | arm | value | threshold | op | verdict | margin | note |
|---|---|---|---|---|---|---|---|---|
| layout_acc_c2_off | layout_acc | c2 | 0.4167 | 0.6000 | < | triggered | within-noise | n/a |

### R5 `gtx1070_moves_up` -- trigger

Does the voice VRAM footprint justify moving the GTX 1070 up in priority?

verdict: **triggered**

| clause | metric | arm | value | threshold | op | verdict | margin | note |
|---|---|---|---|---|---|---|---|---|
| voice_vram_lower_bound | voice_vram_lower_mib | n/a | 2500.0000 | 2048 | >= | triggered | n/a | n/a |

### R6 `tuning_gate_a1` -- gate

Does GPU tuning arm a1 (base c0) preserve correctness and stability?

verdict: **pass**

| clause | metric | arm | value | threshold | op | verdict | margin | note |
|---|---|---|---|---|---|---|---|---|
| base_determinism | correctness_tokens | n/a | n/a | n/a | n/a | pass | n/a | base c0 r1 vs r2: identical for all 5 prompts |
| tuned_matches_base | correctness_tokens | n/a | n/a | n/a | n/a | pass | n/a | a1 r1 vs base c0 r1: identical for all 5 prompts |
| harness_ci_containment | harness_score | a1 | 1.0000 | n/a | n/a | pass | n/a | n/a |
| manual_clean | manual | a1 | 0 | 0 | == | pass | n/a | n/a |

info: `{"arm": "a1", "base": "c0"}`

### R6 `tuning_gate_a2` -- gate

Does GPU tuning arm a2 (base c0) preserve correctness and stability?

verdict: **missing**

| clause | metric | arm | value | threshold | op | verdict | margin | note |
|---|---|---|---|---|---|---|---|---|
| base_determinism | correctness_tokens | n/a | n/a | n/a | n/a | pass | n/a | base c0 r1 vs r2: identical for all 5 prompts |
| tuned_matches_base | correctness_tokens | n/a | n/a | n/a | n/a | pass | n/a | a2 r1 vs base c0 r1: identical for all 5 prompts |
| harness_ci_containment | harness_score | a2 | 1.0000 | n/a | n/a | pass | n/a | n/a |
| manual_clean | manual | a2 | n/a | 0 | == | missing | n/a | session/manual.json has no entry for 'a2' |

info: `{"arm": "a2", "base": "c0"}`

### R6 `tuning_gate_a3` -- gate

Does GPU tuning arm a3 (base c3) preserve correctness and stability?

verdict: **fail**

| clause | metric | arm | value | threshold | op | verdict | margin | note |
|---|---|---|---|---|---|---|---|---|
| base_determinism | correctness_tokens | n/a | n/a | n/a | n/a | pass | n/a | base c3 r1 vs r2: identical for all 5 prompts |
| tuned_matches_base | correctness_tokens | n/a | n/a | n/a | n/a | fail | n/a | a3 r1 vs base c3 r1: token mismatch on ['p03'] |
| harness_ci_containment | harness_score | a3 | 1.0000 | n/a | n/a | pass | n/a | n/a |
| manual_clean | manual | a3 | 0 | 0 | == | pass | n/a | n/a |

info: `{"arm": "a3", "base": "c3"}`

### R7 `build_effect` -- informational

What did the pinned build change (c0 vs c0-old)?

verdict: **n/a**

info: `{"deltas": {"decode_tps": {"missing": true}, "harness_score_schema_conformance": {"delta": 0.0, "new": 1.0, "old": 1.0}, "harness_score_schema_conformance_json_object": {"missing": true}, "harness_score_tool_selection": {"missing": true}, "layout_acc": {"missing": true}}, "new_arm": "c0", "old_arm": "c0-old"}`

## Finalist candidates

- c2-think512
- c3

## Coverage per arm

```
{
  "a1": {
    "errors": [],
    "harness": {
      "schema_conformance": {
        "complete": true,
        "n": 12,
        "n_expected": 12
      }
    },
    "layout": {},
    "present": true,
    "suites_completed": [
      "correctness",
      "harness"
    ]
  },
  "a2": {
    "errors": [],
    "harness": {
      "schema_conformance": {
        "complete": true,
        "n": 12,
        "n_expected": 12
      }
    },
    "layout": {},
    "present": true,
    "suites_completed": [
      "correctness",
      "harness"
    ]
  },
  "a3": {
    "errors": [],
    "harness": {
      "schema_conformance": {
        "complete": true,
        "n": 12,
        "n_expected": 12
      }
    },
    "layout": {},
    "present": true,
    "suites_completed": [
      "correctness",
      "harness"
    ]
  },
  "a4": {
    "present": false
  },
  "c0": {
    "errors": [],
    "harness": {
      "schema_conformance": {
        "complete": true,
        "n": 12,
        "n_expected": 12
      }
    },
    "layout": {},
    "present": true,
    "suites_completed": [
      "correctness",
      "harness"
    ]
  },
  "c0-old": {
    "errors": [],
    "harness": {
      "schema_conformance": {
        "complete": true,
        "n": 12,
        "n_expected": 12
      }
    },
    "layout": {},
    "present": true,
    "suites_completed": [
      "harness"
    ]
  },
  "c0-prod": {
    "errors": [],
    "harness": {
      "schema_conformance_json_object": {
        "complete": true,
        "n": 12,
        "n_expected": 12
      },
      "tool_selection": {
        "complete": true,
        "n": 8,
        "n_expected": 8
      }
    },
    "layout": {},
    "present": true,
    "suites_completed": [
      "harness"
    ]
  },
  "c1-256": {
    "errors": [],
    "harness": {},
    "layout": {
      "screen": {
        "complete": false,
        "n": 40,
        "n_expected": 72
      }
    },
    "present": true,
    "suites_completed": [
      "layout"
    ]
  },
  "c1-512": {
    "errors": [],
    "harness": {},
    "layout": {
      "screen": {
        "complete": true,
        "n": 72,
        "n_expected": 72
      }
    },
    "present": true,
    "suites_completed": [
      "layout"
    ]
  },
  "c2": {
    "errors": [],
    "harness": {
      "schema_conformance_json_object": {
        "complete": true,
        "n": 12,
        "n_expected": 12
      },
      "tool_selection": {
        "complete": true,
        "n": 8,
        "n_expected": 8
      }
    },
    "layout": {
      "screen": {
        "complete": true,
        "n": 72,
        "n_expected": 72
      }
    },
    "present": true,
    "suites_completed": [
      "ttft",
      "harness",
      "layout"
    ]
  },
  "c2-think512": {
    "errors": [],
    "harness": {},
    "layout": {
      "screen": {
        "complete": true,
        "n": 72,
        "n_expected": 72
      }
    },
    "present": true,
    "suites_completed": [
      "layout"
    ]
  },
  "c3": {
    "errors": [],
    "harness": {
      "schema_conformance": {
        "complete": true,
        "n": 12,
        "n_expected": 12
      },
      "schema_conformance_json_object": {
        "complete": true,
        "n": 12,
        "n_expected": 12
      },
      "tool_selection": {
        "complete": true,
        "n": 8,
        "n_expected": 8
      }
    },
    "layout": {
      "finalist": {
        "complete": true,
        "n": 216,
        "n_expected": 216
      },
      "screen": {
        "complete": true,
        "n": 72,
        "n_expected": 72
      }
    },
    "present": true,
    "suites_completed": [
      "ttft",
      "correctness",
      "harness",
      "layout"
    ]
  },
  "c3-think512": {
    "present": false
  },
  "c4": {
    "present": false
  },
  "c4-think512": {
    "present": false
  }
}
```

## Missing inputs

- R2/switch_to_c4/D: no ttft.jsonl for this arm
- R2/switch_to_c4/F: missing inputs: ['arm_peak_mib(c4)']
- R2/switch_to_c4/L: insufficient data to evaluate L on either branch
- R2/switch_to_c4/P: no layout_p95_wall_ms available on the arm used for P
- R2/switch_to_c4/S1: no usable harness_score for c4/schema_conformance_json_object
- R2/switch_to_c4/S2: no usable harness_score for c4/tool_selection
- R6/tuning_gate_a2/manual_clean: session/manual.json has no entry for 'a2'

