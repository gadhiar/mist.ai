# Speech bench report, run mh1

Rendered by `speech_bench report` from the rows under `<results-root>/mh1/speech/`, written at git `fixture`.

Each figure gives its median and slow case (the nearest-rank 90th percentile) with its sample count n, or a single value, count or share. Marks set a figure against its target and floor from the reassessment, section 8; they mark, they do not rule. A figure marked derived is computed from the inputs listed under its table, not measured by this bench.

## Targets

| Figure | Target | Floor |
|---|---|---|
| speech first audio | 250 ms | 500 ms |
| final text | 80 ms | 250 ms |
| prefill to first token | 120 ms | 250 ms |
| end-of-turn inference | under 15 ms, on top of 200 ms of silence | - |
| memory headroom | 2 GiB gate | 512 MiB margin |
| chained turn, commands and short answers | 2,000 ms | 3,000 ms |

## Budget: the reassessment, section 8

From the end of speech to first audio, medians in ms. Today, the plan floor and the target are the reassessment, section 8's. Each stage this bench measures is put in once per option; the total adds the fastest option of each stage. Stages 3, 5 and 7 are not measured here and are derived.

| Stage | Today (derived) | Plan floor | Target | Option | This run | Slow case | n | Source | Mark |
|---|---|---|---|---|---|---|---|---|---|
| 1. End of turn | 1,440 (wired path) | 250 ms | 215 ms | smart-turn-v3.2 | 209 ms | 213 ms | 7 | measured, run 4 | target met |
| 2. Speech to text, final | 200 to 600 | 250 ms | 80 ms | whisper-base | 262 ms | 290 ms | 4 | measured, run 3 | over floor |
| 2. Speech to text, final | 200 to 600 | 250 ms | 80 ms | parakeet-gpu (in the total) | 72 ms | 95 ms | 4 | measured, run 3 | target met |
| 2. Speech to text, final | 200 to 600 | 250 ms | 80 ms | parakeet-cpu | 165 ms | 210 ms | 4 | measured, run 3 | floor met |
| 3. Memory fetch | unmeasured | 50 ms | 30 ms | floor | 50 ms | - | 1 | derived | floor met, derived |
| 4. Prefill to first token | about 1,100 (cold, 6,100 tokens) | 250 ms | 120 ms | c0 | 912 ms | 921 ms | 4 | measured, run 2 | over floor |
| 4. Prefill to first token | about 1,100 (cold, 6,100 tokens) | 250 ms | 120 ms | c0-ckpt (in the total) | 128 ms | 142 ms | 4 | measured, run 2 | floor met |
| 5. First speakable words | 220 | 120 ms | 110 ms | c0 | 200 ms | - | 1 | derived | over floor, derived |
| 6. Synthesis to first audio | 1,500 to 1,900 | 500 ms | 250 ms | qwen3 | 282 ms | 301 ms | 2 | measured, run 1 | floor met |
| 6. Synthesis to first audio | 1,500 to 1,900 | 500 ms | 250 ms | turbo | 634 ms | 655 ms | 2 | measured, run 1 | over floor |
| 6. Synthesis to first audio | 1,500 to 1,900 | 500 ms | 250 ms | original | 1,012 ms | 1,044 ms | 2 | measured, run 1 | over floor |
| 6. Synthesis to first audio | 1,500 to 1,900 | 500 ms | 250 ms | pocket (in the total) | 176 ms | 190 ms | 2 | measured, run 1 | target met |
| 7. Send and start playback | about 100 | 100 ms | 60 ms | today | 100 ms | - | 1 | derived | floor met, derived |
| Total, median | about 3,700 to 5,400 | 1,470 ms | about 865 ms | - | 936 ms | - | 7 | derived | floor met, derived |
| Total, slow case | unknown | 2,100 ms | 1,600 ms or less | - | 990 ms | - | 7 | derived | target met, derived |

Derived figures and their inputs:

- memory fetch, carried in: not measured here; the plan floor in the reassessment, section 8: 50 ms.
- first clause at the measured decode rate: run 2 c0 decode rate, median 50.0 tok/s (n=6); a first clause of about 10 tokens.
- send and start playback, carried in: not measured here; today's figure in the reassessment, section 8: about 100 ms.
- total, median: the fastest option of each stage; stage 1: 200 ms of silence plus Smart Turn inference, smart-turn-v3.2; stage 2: final text, parakeet-gpu; stage 3: memory fetch, carried in, floor; stage 4: warm time to first token, turns 2 to 10, c0-ckpt; stage 5: first clause at the measured decode rate, c0; stage 6: first audio, lines, model writing, pocket; stage 7: send and start playback, carried in, today.
- total, slow case: each chosen stage's slow case, summed: an upper bound; stage 1: 200 ms of silence plus Smart Turn inference, smart-turn-v3.2; stage 2: final text, parakeet-gpu; stage 3: memory fetch, carried in, floor; stage 4: warm time to first token, turns 2 to 10, c0-ckpt; stage 5: first clause at the measured decode rate, c0; stage 6: first audio, lines, model writing, pocket; stage 7: send and start playback, carried in, today.

## Levels: what the tier design needs

Each question the tier router and power-level design asks, with the figures that answer it.

### Where the low level runs: 4070 slice, CPU or GTX 1070

Runs: 7, then 8 and 9.

#### Each placement side by side

Chained turn: commands and short answers with endpointing added (derived), against about 2 to 3 s. Tool-call validity: the command prompts, all calls. Memory: the placement's own method. Game cost: the average frame rate of G2 less G0's, from the latest session.

| Model | Placement | chained turn with endpointing | tool-call validity | hand-off validity | memory | load | unload | game cost, run 8 | game cost, run 9 |
|---|---|---|---|---|---|---|---|---|---|
| gemma4-e2b | gpu-full | 866 ms, slow case 904 ms (n=6); target met, derived | 1 of 2 (50%) | 1 of 2 (50%) | 3,100 MiB (n=1) | 6.20 s (n=1) | 1.40 s (n=1) | -9.8 fps (n=3) | not yet run |
| gemma4-e2b | cpu | 2,266 ms, slow case 2,304 ms (n=6); floor met, derived | 1 of 2 (50%) | 1 of 2 (50%) | 4,200 MiB (n=1) | 3.90 s (n=1) | 0.80 s (n=1) | -14.9 fps (n=3) | not yet run |
| gemma4-e2b | gtx1070-shared-idle | 1,524 ms, slow case 1,539 ms (n=3); target met, derived | 1 of 2 (50%) | 1 of 2 (50%) | 352 MiB (n=1) | 14.21 s (n=1) | 2.40 s (n=1) | not measured | not yet run |
| gemma4-e2b | gtx1070-shared-busy | - | 1 of 2 (50%) | 1 of 2 (50%) | 342 MiB (n=1) | 15.00 s (n=1) | 2.61 s (n=1) | not measured | not yet run |
| gemma4-e2b | gtx1070 | - | 1 of 2 (50%) | 1 of 2 (50%) | 3,270 MiB (n=1) | 21.54 s (n=1) | 3.01 s (n=1) | not measured | not yet run |
| e4b-ref | gpu-full | 1,051 ms, slow case 1,066 ms (n=3); target met, derived | 1 of 2 (50%) | 1 of 2 (50%) | 7,400 MiB (n=1) | 6.20 s (n=1) | 1.40 s (n=1) | not measured | not yet run |

Derived figures and their inputs:

- chained turn with endpointing: chained turn, measured: speech to text, the model to its first clause, first audio; 200 ms of silence; Smart Turn inference, run 4 median: 9 ms.

- run 9 not yet run: the game cost per placement waits for it

### Which small model is the interaction tier

Runs: 7.

#### Each model on gpu-full, the same prompts

`e4b-ref` is today's E4B, the reference.

| Model | Placement | command validity | hand-off validity | warm time to first token | decode rate |
|---|---|---|---|---|---|
| gemma4-e2b | gpu-full | 1 of 2 (50%) | 1 of 2 (50%) | 40 ms, slow case 42 ms (n=2); target met | 138.0 tok/s, slow case 139.0 tok/s (n=3) |
| e4b-ref | gpu-full | 1 of 2 (50%) | 1 of 2 (50%) | 86 ms, slow case 90 ms (n=2); target met | 73.0 tok/s, slow case 74.0 tok/s (n=3) |

### Whether the small model can route its own requests up

Runs: 7.

#### Hand-offs on gpu-full

| Model | Placement | hand-off validity | false hand-offs |
|---|---|---|---|
| gemma4-e2b | gpu-full | 1 of 2 (50%) | 1 of 4 (25%) |
| e4b-ref | gpu-full | 1 of 2 (50%) | 1 of 4 (25%) |

### What each tier costs per request

Runs: 2 and 7.

#### Low level on gpu-full

| Model | Placement | warm time to first token | decode rate |
|---|---|---|---|
| gemma4-e2b | gpu-full | 40 ms, slow case 42 ms (n=2); target met | 138.0 tok/s, slow case 139.0 tok/s (n=3) |
| e4b-ref | gpu-full | 86 ms, slow case 90 ms (n=2); target met | 73.0 tok/s, slow case 74.0 tok/s (n=3) |

#### The top tier: E4B as production serves it (run 2, c0)

| Arm | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|
| c0 | cold time to first token | 1,180 ms | 1,180 ms | 1 | no target |
| c0 | warm time to first token, turns 2 to 10 | 912 ms | 921 ms | 4 | over floor |
| c0 | prefill, turns 2 to 10 | 892 ms | 901 ms | 4 | no target |
| c0 | cached share of the prompt, turns 2 to 10 | 0.0% | 0.0% | 4 | no target |
| c0 | decode rate | 50.0 tok/s | 50.0 tok/s | 6 | no target |

### Whether both tiers stay loaded at full level

Runs: 5.

#### E4B with each small model, speech and recognition (run 5)

| Step | Part | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|---|
| e4b+gemma4-e2b | small | peak used | 10,050 MiB | - | 50 | no target |
| e4b+gemma4-e2b | small | headroom | 2,232 MiB | - | 50 | 2 GiB gate met |
| e4b+gemma4-e2b | small | by difference from the desktop step | 8,750 MiB | - | 50 | no target |
| e4b+gemma4-e2b+qwen3+parakeet-gpu | small | peak used | 11,200 MiB | - | 50 | no target |
| e4b+gemma4-e2b+qwen3+parakeet-gpu | small | headroom | 1,082 MiB | - | 50 | 512 MiB margin met |
| e4b+gemma4-e2b+qwen3+parakeet-gpu | small | by difference from the desktop step | 9,900 MiB | - | 50 | no target |
| e4b+gemma4-e2b+turbo+parakeet-gpu | small | peak used | 10,900 MiB | - | 50 | no target |
| e4b+gemma4-e2b+turbo+parakeet-gpu | small | headroom | 1,382 MiB | - | 50 | 512 MiB margin met |
| e4b+gemma4-e2b+turbo+parakeet-gpu | small | by difference from the desktop step | 9,600 MiB | - | 50 | no target |

### What a level change costs

Runs: 7 and 8.

#### Load and unload per model and placement

| Model | Placement | load | unload | memory returned |
|---|---|---|---|---|
| gemma4-e2b | gpu-full | 6.20 s (n=1) | 1.40 s (n=1) | 3,050 MiB (n=1) |
| gemma4-e2b | cpu | 3.90 s (n=1) | 0.80 s (n=1) | - |
| gemma4-e2b | gtx1070-shared-idle | 14.21 s (n=1) | 2.40 s (n=1) | 350 MiB (n=1) |
| gemma4-e2b | gtx1070-shared-busy | 15.00 s (n=1) | 2.61 s (n=1) | 349 MiB (n=1) |
| gemma4-e2b | gtx1070 | 21.54 s (n=1) | 3.01 s (n=1) | 3,268 MiB (n=1) |
| e4b-ref | gpu-full | 6.20 s (n=1) | 1.40 s (n=1) | 7,350 MiB (n=1) |

#### E4B's reload

| Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|
| E4B reload after unserve | 21.40 s | 22.00 s | 3 | no target |

#### E4B's return after a game session

| Session | Run | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|---|
| mh1-game-20261121T190000Z | 8 | E4B's return after the session | 48.20 s | - | 1 | no target |

### What a low-level request costs while the top tier is writing, at full power

Runs: 7.

#### Each small model on gpu-full while E4B streams

| Model | Placement | chained turn with endpointing | E4B decode rate during the turns |
|---|---|---|---|
| gemma4-e2b | gpu-full-e4b-writing | 1,039 ms, slow case 1,054 ms (n=3); target met, derived | 34.5 tok/s, slow case 36.0 tok/s (n=4) |

Derived figures and their inputs:

- chained turn with endpointing: chained turn, measured: speech to text, the model to its first clause, first audio; 200 ms of silence; Smart Turn inference, run 4 median: 9 ms.

#### E4B's decode rate with nothing beside it (run 1 baseline)

| Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|
| E4B decode rate, no speech | 50.0 tok/s | 50.0 tok/s | 4 | no target |

### Whether a busy extraction job on host-1070 slows a low level there

Runs: 7.

#### The shared placement, idle and busy

| Model | Placement | warm time to first token, idle | warm time to first token, busy | decode rate, idle | decode rate, busy | command validity, idle | command validity, busy |
|---|---|---|---|---|---|---|---|
| gemma4-e2b | gtx1070-shared | 223 ms, slow case 233 ms (n=2); floor met | 281 ms, slow case 295 ms (n=2); over floor | 29.0 tok/s, slow case 30.0 tok/s (n=3) | 22.0 tok/s, slow case 23.0 tok/s (n=3) | 1 of 2 (50%) | 1 of 2 (50%) |

### What the low level keeps on the CPU

Runs: 7.

#### Speech on the CPU, alone

| Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|
| Parakeet on the CPU, final text (run 3) | 165 ms | 210 ms | 4 | floor met |
| Pocket TTS first audio, idle (run 1) | 160 ms | 171 ms | 2 | target met |
| Pocket TTS first audio, decoding (run 1) | 176 ms | 190 ms | 2 | target met |

#### Sharing the CPU with the model (run 7, cpu placement)

| Model | Placement | speech to text | speech to first audio | chained turn with endpointing |
|---|---|---|---|---|
| gemma4-e2b | cpu | 252 ms, slow case 265 ms (n=6) | 338 ms, slow case 345 ms (n=6) | 2,266 ms, slow case 2,304 ms (n=6); floor met, derived |

Derived figures and their inputs:

- chained turn with endpointing: chained turn, measured: speech to text, the model to its first clause, first audio; 200 ms of silence; Smart Turn inference, run 4 median: 9 ms.

### What the top tier could become on a 16 to 24 GB card

Runs: 5 (derived).

- See the section on what a 16 to 24 GB card would add, below: every figure there is derived from run 5's memory and run 2's decode rate.

### Whether extraction can stay on host-1070 beside a low level there

Runs: 7.

#### The small model in the 1070's spare memory

Extraction decode given up: what freeing enough of the card to hold the model whole would cost extraction (derived).

| Model | Placement | decode rate | warm time to first token | extraction decode given up |
|---|---|---|---|---|
| gemma4-e2b | gtx1070-shared-idle | 29.0 tok/s, slow case 30.0 tok/s (n=3) | 223 ms, slow case 233 ms (n=2); floor met | 11.2 tok/s (n=1); derived |

Derived figures and their inputs:

- extraction decode given up: memory to free: 2,918 MiB, the model's load alone on the GTX 1070 less its load beside extraction (H6 summary); each ncmoe step frees about 400 MiB and costs about 1.4 tok/s of extraction decode (the host-1070 runbook's own sweep).

## Run 1: speech synthesis

### First audio per engine and condition

Condition `idle`: the bench model idle; `decoding`: the model writing throughout. The gap between them is what a second card would remove.

| Engine | Voice | Condition | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|---|---|
| qwen3 | aiden | idle | first audio, lines | 192 ms | 205 ms | 2 | target met |
| qwen3 | aiden | idle | compute per second of speech | 0.136 | 0.143 | 2 | no target |
| qwen3 | aiden | idle | peak GPU memory reserved | 2,120 MiB | - | 3 | no target |
| qwen3 | aiden | idle | first audio, acknowledgements (kept out of the line figures) | 150 ms | 150 ms | 1 | no target |
| qwen3 | aiden | decoding | first audio, lines | 282 ms | 301 ms | 2 | floor met |
| qwen3 | aiden | decoding | compute per second of speech | 0.199 | 0.210 | 2 | no target |
| qwen3 | aiden | decoding | peak GPU memory reserved | 2,120 MiB | - | 3 | no target |
| qwen3 | aiden | decoding | first audio, acknowledgements (kept out of the line figures) | 240 ms | 240 ms | 1 | no target |
| turbo | builtin | idle | first audio, lines | 440 ms | 461 ms | 2 | floor met |
| turbo | builtin | idle | compute per second of speech | 0.311 | 0.322 | 2 | no target |
| turbo | builtin | idle | peak GPU memory reserved | 2,120 MiB | - | 3 | no target |
| turbo | builtin | idle | first audio, acknowledgements (kept out of the line figures) | 300 ms | 300 ms | 1 | no target |
| turbo | builtin | decoding | first audio, lines | 634 ms | 655 ms | 2 | over floor |
| turbo | builtin | decoding | compute per second of speech | 0.448 | 0.457 | 2 | no target |
| turbo | builtin | decoding | peak GPU memory reserved | 2,120 MiB | - | 3 | no target |
| turbo | builtin | decoding | first audio, acknowledgements (kept out of the line figures) | 480 ms | 480 ms | 1 | no target |
| original | builtin | idle | first audio, lines | 731 ms | 760 ms | 2 | over floor |
| original | builtin | idle | compute per second of speech | 0.517 | 0.531 | 2 | no target |
| original | builtin | idle | peak GPU memory reserved | 2,120 MiB | - | 3 | no target |
| original | builtin | idle | first audio, acknowledgements (kept out of the line figures) | 520 ms | 520 ms | 1 | no target |
| original | builtin | decoding | first audio, lines | 1,012 ms | 1,044 ms | 2 | over floor |
| original | builtin | decoding | compute per second of speech | 0.716 | 0.729 | 2 | no target |
| original | builtin | decoding | peak GPU memory reserved | 2,120 MiB | - | 3 | no target |
| original | builtin | decoding | first audio, acknowledgements (kept out of the line figures) | 760 ms | 760 ms | 1 | no target |
| pocket | alba | idle | first audio, lines | 160 ms | 171 ms | 2 | target met |
| pocket | alba | idle | compute per second of speech | 0.113 | 0.119 | 2 | no target |
| pocket | alba | idle | peak GPU memory reserved | - | - | 0 | no target |
| pocket | alba | idle | first audio, acknowledgements (kept out of the line figures) | 120 ms | 120 ms | 1 | no target |
| pocket | alba | decoding | first audio, lines | 176 ms | 190 ms | 2 | target met |
| pocket | alba | decoding | compute per second of speech | 0.124 | 0.133 | 2 | no target |
| pocket | alba | decoding | peak GPU memory reserved | - | - | 0 | no target |
| pocket | alba | decoding | first audio, acknowledgements (kept out of the line figures) | 131 ms | 131 ms | 1 | no target |

### Load time and the model's decode rate

| Engine | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|
| qwen3 | load | 9.80 s | - | 1 | no target |
| qwen3 | E4B decode rate, no speech | 50.0 tok/s | - | 6 | no target |
| qwen3 | E4B decode rate, speech running | 41.5 tok/s | - | 4 | no target |
| turbo | load | 7.10 s | - | 1 | no target |
| turbo | E4B decode rate, no speech | 50.0 tok/s | - | 6 | no target |
| turbo | E4B decode rate, speech running | 38.0 tok/s | - | 4 | no target |
| original | load | 11.40 s | - | 1 | no target |
| original | E4B decode rate, no speech | 50.0 tok/s | - | 6 | no target |
| original | E4B decode rate, speech running | 35.2 tok/s | - | 4 | no target |
| pocket | load | 2.20 s | - | 1 | no target |
| pocket | E4B decode rate, no speech | 50.0 tok/s | - | 6 | no target |
| pocket | E4B decode rate, speech running | 48.9 tok/s | - | 4 | no target |

## Run 2: prompt cache

### Time to first token and cache reuse per arm

| Arm | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|
| c0 | cold time to first token | 1,180 ms | 1,180 ms | 1 | no target |
| c0 | warm time to first token, turns 2 to 10 | 912 ms | 921 ms | 4 | over floor |
| c0 | prefill, turns 2 to 10 | 892 ms | 901 ms | 4 | no target |
| c0 | cached share of the prompt, turns 2 to 10 | 0.0% | 0.0% | 4 | no target |
| c0 | decode rate | 50.0 tok/s | 50.0 tok/s | 6 | no target |
| c0 | full prompt re-processing logged | 4 | - | 6 | no target |
| c0-ckpt | cold time to first token | 1,175 ms | 1,175 ms | 1 | no target |
| c0-ckpt | warm time to first token, turns 2 to 10 | 128 ms | 142 ms | 4 | floor met |
| c0-ckpt | prefill, turns 2 to 10 | 108 ms | 122 ms | 4 | no target |
| c0-ckpt | cached share of the prompt, turns 2 to 10 | 97.7% | 97.7% | 4 | no target |
| c0-ckpt | decode rate | 51.0 tok/s | 52.0 tok/s | 6 | no target |
| c0-ckpt | full prompt re-processing logged | 0 | - | 6 | no target |

## Run 3: speech to text

### Final text and word error rate per engine

| Engine | Device | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|---|
| whisper-base | cuda | final text | 262 ms | 290 ms | 4 | over floor |
| whisper-base | cuda | word error rate, all words | 8.3% | - | 4 | no target |
| whisper-base | cuda | peak GPU memory reserved | 1,480 MiB | - | 4 | no target |
| parakeet-gpu | cuda | final text | 72 ms | 95 ms | 4 | target met |
| parakeet-gpu | cuda | word error rate, all words | 0.0% | - | 4 | no target |
| parakeet-gpu | cuda | peak GPU memory reserved | 2,310 MiB | - | 4 | no target |
| parakeet-cpu | cpu | final text | 165 ms | 210 ms | 4 | floor met |
| parakeet-cpu | cpu | word error rate, all words | 0.0% | - | 4 | no target |
| parakeet-cpu | cpu | peak GPU memory reserved | - | - | 0 | no target |

## Run 4: end of turn

### Smart Turn v3.2 on the CPU

Inference is marked against 15 ms on top of 200 ms of silence.

| Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|
| inference | 9 ms | 13 ms | 7 | under 15 ms |
| false finished at mid-thought pauses | 1 of 3 (33%) | - | 3 | no target |
| false finished, pause utterances only | 1 of 2 (50%) | - | 2 | no target |
| finished called at true ends | 3 of 4 (75%) | - | 4 | no target |

## Run 5: memory fit

### Peak memory and headroom per combination

Headroom is the card's total less the step's peak, against a 512 MiB margin and a 2 GiB gate.

| Step | Part | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|---|
| e4b | speech | peak used | 8,900 MiB | - | 50 | no target |
| e4b | speech | headroom | 3,382 MiB | - | 50 | 2 GiB gate met |
| e4b | speech | by difference from the desktop step | 7,600 MiB | - | 50 | no target |
| e4b+qwen3+whisper-base | speech | peak used | 10,400 MiB | - | 50 | no target |
| e4b+qwen3+whisper-base | speech | headroom | 1,882 MiB | - | 50 | 512 MiB margin met |
| e4b+qwen3+whisper-base | speech | by difference from the desktop step | 9,100 MiB | - | 50 | no target |
| e4b+original+parakeet-gpu | speech | peak used | 11,382 MiB | - | 50 | no target |
| e4b+original+parakeet-gpu | speech | headroom | 900 MiB | - | 50 | 512 MiB margin met |
| e4b+original+parakeet-gpu | speech | by difference from the desktop step | 10,082 MiB | - | 50 | no target |
| e4b+turbo+nemotron-160 | speech | peak used | - | - | 0 | no target |
| e4b+turbo+nemotron-160 | speech | headroom | - | - | 0 | does not fit |
| e4b+turbo+nemotron-160 | speech | by difference from the desktop step | - | - | 0 | no target |
| e4b+gemma4-e2b | small | peak used | 10,050 MiB | - | 50 | no target |
| e4b+gemma4-e2b | small | headroom | 2,232 MiB | - | 50 | 2 GiB gate met |
| e4b+gemma4-e2b | small | by difference from the desktop step | 8,750 MiB | - | 50 | no target |
| e4b+gemma4-e2b+qwen3+parakeet-gpu | small | peak used | 11,200 MiB | - | 50 | no target |
| e4b+gemma4-e2b+qwen3+parakeet-gpu | small | headroom | 1,082 MiB | - | 50 | 512 MiB margin met |
| e4b+gemma4-e2b+qwen3+parakeet-gpu | small | by difference from the desktop step | 9,900 MiB | - | 50 | no target |
| e4b+gemma4-e2b+turbo+parakeet-gpu | small | peak used | 10,900 MiB | - | 50 | no target |
| e4b+gemma4-e2b+turbo+parakeet-gpu | small | headroom | 1,382 MiB | - | 50 | 512 MiB margin met |
| e4b+gemma4-e2b+turbo+parakeet-gpu | small | by difference from the desktop step | 9,600 MiB | - | 50 | no target |

- e4b+turbo+nemotron-160 does not fit: nemotron-160 ran out of memory: CUDA out of memory

## Run 7: low level, no game

### Time to first token and decode rate

| Model | Placement | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|---|
| gemma4-e2b | gpu-full | cold time to first token, 1,500-token prefix | 210 ms | 210 ms | 1 | no target |
| gemma4-e2b | gpu-full | cold time to first token, 6,100-token prefix | 690 ms | 690 ms | 1 | no target |
| gemma4-e2b | gpu-full | warm time to first token, turns 2 to 10 | 40 ms | 42 ms | 2 | target met |
| gemma4-e2b | gpu-full | decode rate | 138.0 tok/s | 139.0 tok/s | 3 | no target |
| gemma4-e2b | gpu-full | rows cut short by the window | 0 | - | 12 | no target |
| gemma4-e2b | gpu-full | replies still carrying reasoning | 0 | - | 12 | no target |
| gemma4-e2b | cpu | cold time to first token, 1,500-token prefix | 2,400 ms | 2,400 ms | 1 | no target |
| gemma4-e2b | cpu | cold time to first token, 6,100-token prefix | 9,800 ms | 9,800 ms | 1 | no target |
| gemma4-e2b | cpu | warm time to first token, turns 2 to 10 | 326 ms | 341 ms | 2 | over floor |
| gemma4-e2b | cpu | decode rate | 20.0 tok/s | 21.0 tok/s | 3 | no target |
| gemma4-e2b | cpu | rows cut short by the window | 0 | - | 12 | no target |
| gemma4-e2b | cpu | replies still carrying reasoning | 0 | - | 12 | no target |
| gemma4-e2b | gtx1070-shared-idle | cold time to first token, 1,500-token prefix | 1,300 ms | 1,300 ms | 1 | no target |
| gemma4-e2b | gtx1070-shared-idle | cold time to first token, 6,100-token prefix | 5,200 ms | 5,200 ms | 1 | no target |
| gemma4-e2b | gtx1070-shared-idle | warm time to first token, turns 2 to 10 | 223 ms | 233 ms | 2 | floor met |
| gemma4-e2b | gtx1070-shared-idle | decode rate | 29.0 tok/s | 30.0 tok/s | 3 | no target |
| gemma4-e2b | gtx1070-shared-idle | rows cut short by the window | 0 | - | 12 | no target |
| gemma4-e2b | gtx1070-shared-idle | replies still carrying reasoning | 0 | - | 12 | no target |
| gemma4-e2b | gtx1070-shared-busy | cold time to first token, 1,500-token prefix | - | - | 0 | no target |
| gemma4-e2b | gtx1070-shared-busy | cold time to first token, 6,100-token prefix | - | - | 0 | no target |
| gemma4-e2b | gtx1070-shared-busy | warm time to first token, turns 2 to 10 | 281 ms | 295 ms | 2 | over floor |
| gemma4-e2b | gtx1070-shared-busy | decode rate | 22.0 tok/s | 23.0 tok/s | 3 | no target |
| gemma4-e2b | gtx1070-shared-busy | rows cut short by the window | 1 | - | 11 | no target |
| gemma4-e2b | gtx1070-shared-busy | replies still carrying reasoning | 0 | - | 10 | no target |
| gemma4-e2b | gtx1070 | cold time to first token, 1,500-token prefix | 520 ms | 520 ms | 1 | no target |
| gemma4-e2b | gtx1070 | cold time to first token, 6,100-token prefix | 2,100 ms | 2,100 ms | 1 | no target |
| gemma4-e2b | gtx1070 | warm time to first token, turns 2 to 10 | 96 ms | 100 ms | 2 | target met |
| gemma4-e2b | gtx1070 | decode rate | 70.0 tok/s | 71.0 tok/s | 3 | no target |
| gemma4-e2b | gtx1070 | rows cut short by the window | 0 | - | 12 | no target |
| gemma4-e2b | gtx1070 | replies still carrying reasoning | 0 | - | 12 | no target |
| e4b-ref | gpu-full | cold time to first token, 1,500-token prefix | 380 ms | 380 ms | 1 | no target |
| e4b-ref | gpu-full | cold time to first token, 6,100-token prefix | 1,150 ms | 1,150 ms | 1 | no target |
| e4b-ref | gpu-full | warm time to first token, turns 2 to 10 | 86 ms | 90 ms | 2 | target met |
| e4b-ref | gpu-full | decode rate | 73.0 tok/s | 74.0 tok/s | 3 | no target |
| e4b-ref | gpu-full | rows cut short by the window | 0 | - | 12 | no target |
| e4b-ref | gpu-full | replies still carrying reasoning | 0 | - | 12 | no target |

### Tool calls and hand-offs

| Model | Placement | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|---|
| gemma4-e2b | gpu-full | command validity, all calls | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | gpu-full | two-action commands, first call | 1 of 1 (100%) | - | 1 | no target |
| gemma4-e2b | gpu-full | no-tool prompts answered with no call | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | gpu-full | hand-off validity, E01 to E04 | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | gpu-full | false hand-offs on command and no-tool prompts | 1 of 4 (25%) | - | 4 | no target |
| gemma4-e2b | gpu-full | request to the last tool-call fragment | 217 ms | 217 ms | 3 | no target |
| gemma4-e2b | cpu | command validity, all calls | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | cpu | two-action commands, first call | 1 of 1 (100%) | - | 1 | no target |
| gemma4-e2b | cpu | no-tool prompts answered with no call | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | cpu | hand-off validity, E01 to E04 | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | cpu | false hand-offs on command and no-tool prompts | 1 of 4 (25%) | - | 4 | no target |
| gemma4-e2b | cpu | request to the last tool-call fragment | 1,463 ms | 1,463 ms | 3 | no target |
| gemma4-e2b | gtx1070-shared-idle | command validity, all calls | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | gtx1070-shared-idle | two-action commands, first call | 1 of 1 (100%) | - | 1 | no target |
| gemma4-e2b | gtx1070-shared-idle | no-tool prompts answered with no call | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | gtx1070-shared-idle | hand-off validity, E01 to E04 | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | gtx1070-shared-idle | false hand-offs on command and no-tool prompts | 1 of 4 (25%) | - | 4 | no target |
| gemma4-e2b | gtx1070-shared-idle | request to the last tool-call fragment | 1,029 ms | 1,029 ms | 3 | no target |
| gemma4-e2b | gtx1070-shared-busy | command validity, all calls | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | gtx1070-shared-busy | two-action commands, first call | 1 of 1 (100%) | - | 1 | no target |
| gemma4-e2b | gtx1070-shared-busy | no-tool prompts answered with no call | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | gtx1070-shared-busy | hand-off validity, E01 to E04 | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | gtx1070-shared-busy | false hand-offs on command and no-tool prompts | 1 of 4 (25%) | - | 4 | no target |
| gemma4-e2b | gtx1070-shared-busy | request to the last tool-call fragment | 1,322 ms | 1,322 ms | 3 | no target |
| gemma4-e2b | gtx1070 | command validity, all calls | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | gtx1070 | two-action commands, first call | 1 of 1 (100%) | - | 1 | no target |
| gemma4-e2b | gtx1070 | no-tool prompts answered with no call | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | gtx1070 | hand-off validity, E01 to E04 | 1 of 2 (50%) | - | 2 | no target |
| gemma4-e2b | gtx1070 | false hand-offs on command and no-tool prompts | 1 of 4 (25%) | - | 4 | no target |
| gemma4-e2b | gtx1070 | request to the last tool-call fragment | 443 ms | 443 ms | 3 | no target |
| e4b-ref | gpu-full | command validity, all calls | 1 of 2 (50%) | - | 2 | no target |
| e4b-ref | gpu-full | two-action commands, first call | 1 of 1 (100%) | - | 1 | no target |
| e4b-ref | gpu-full | no-tool prompts answered with no call | 1 of 2 (50%) | - | 2 | no target |
| e4b-ref | gpu-full | hand-off validity, E01 to E04 | 1 of 2 (50%) | - | 2 | no target |
| e4b-ref | gpu-full | false hand-offs on command and no-tool prompts | 1 of 4 (25%) | - | 4 | no target |
| e4b-ref | gpu-full | request to the last tool-call fragment | 418 ms | 418 ms | 3 | no target |

### Memory and switching

| Model | Placement | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|---|
| gemma4-e2b | gpu-full | GPU memory by difference | 3,100 MiB | - | 1 | no target |
| gemma4-e2b | gpu-full | container peak host memory | 1,900 MiB | - | 1 | no target |
| gemma4-e2b | gpu-full | load | 6.20 s | - | 1 | no target |
| gemma4-e2b | gpu-full | unload | 1.40 s | - | 1 | no target |
| gemma4-e2b | gpu-full | memory returned on unload | 3,050 MiB | - | 1 | no target |
| gemma4-e2b | cpu | container peak host memory | 4,200 MiB | - | 1 | no target |
| gemma4-e2b | cpu | load | 3.90 s | - | 1 | no target |
| gemma4-e2b | cpu | unload | 0.80 s | - | 1 | no target |
| gemma4-e2b | cpu | memory returned on unload | - | - | 0 | no target |
| gemma4-e2b | gtx1070-shared-idle | GTX 1070 memory taken at load (H6 summary) | 352 MiB | - | 1 | no target |
| gemma4-e2b | gtx1070-shared-idle | GTX 1070 memory free at the peak | 537 MiB | - | 1 | 512 MiB margin met |
| gemma4-e2b | gtx1070-shared-idle | host memory used at the peak | 2,391 MiB | - | 1 | no target |
| gemma4-e2b | gtx1070-shared-idle | load | 14.21 s | - | 1 | no target |
| gemma4-e2b | gtx1070-shared-idle | unload | 2.40 s | - | 1 | no target |
| gemma4-e2b | gtx1070-shared-idle | memory returned on unload | 350 MiB | - | 1 | no target |
| gemma4-e2b | gtx1070-shared-busy | GTX 1070 memory taken at load (H6 summary) | 342 MiB | - | 1 | no target |
| gemma4-e2b | gtx1070-shared-busy | GTX 1070 memory free at the peak | 512 MiB | - | 1 | 512 MiB margin met |
| gemma4-e2b | gtx1070-shared-busy | host memory used at the peak | 2,418 MiB | - | 1 | no target |
| gemma4-e2b | gtx1070-shared-busy | load | 15.00 s | - | 1 | no target |
| gemma4-e2b | gtx1070-shared-busy | unload | 2.61 s | - | 1 | no target |
| gemma4-e2b | gtx1070-shared-busy | memory returned on unload | 349 MiB | - | 1 | no target |
| gemma4-e2b | gtx1070 | GTX 1070 memory taken at load (H6 summary) | 3,270 MiB | - | 1 | no target |
| gemma4-e2b | gtx1070 | GTX 1070 memory free at the peak | 4,570 MiB | - | 1 | 2 GiB gate met |
| gemma4-e2b | gtx1070 | host memory used at the peak | 2,764 MiB | - | 1 | no target |
| gemma4-e2b | gtx1070 | load | 21.54 s | - | 1 | no target |
| gemma4-e2b | gtx1070 | unload | 3.01 s | - | 1 | no target |
| gemma4-e2b | gtx1070 | memory returned on unload | 3,268 MiB | - | 1 | no target |
| minicpm5-2b | host1070-cpu | host memory used at the peak (H6 summary) | 1,854 MiB | - | 1 | no target |
| minicpm5-2b | host1070-cpu | GTX 1070 memory free at the peak | 951 MiB | - | 1 | 512 MiB margin met |
| minicpm5-2b | host1070-cpu | host memory used at the peak | 1,854 MiB | - | 1 | no target |
| minicpm5-2b | host1070-cpu | load | 6.88 s | - | 1 | no target |
| minicpm5-2b | host1070-cpu | unload | 1.10 s | - | 1 | no target |
| minicpm5-2b | host1070-cpu | memory returned on unload | 0 MiB | - | 1 | no target |
| minicpm5-2b | gtx1070 | GTX 1070 memory taken at load (H6 summary) | - | - | 0 | no target |
| minicpm5-2b | gtx1070 | GTX 1070 memory free at the peak | - | - | 0 | no data |
| minicpm5-2b | gtx1070 | host memory used at the peak | - | - | 0 | no target |
| minicpm5-2b | gtx1070 | load | - | - | 0 | no target |
| minicpm5-2b | gtx1070 | unload | - | - | 0 | no target |
| minicpm5-2b | gtx1070 | memory returned on unload | - | - | 0 | no target |
| e4b-ref | gpu-full | GPU memory by difference | 7,400 MiB | - | 1 | no target |
| e4b-ref | gpu-full | container peak host memory | 1,900 MiB | - | 1 | no target |
| e4b-ref | gpu-full | load | 6.20 s | - | 1 | no target |
| e4b-ref | gpu-full | unload | 1.40 s | - | 1 | no target |
| e4b-ref | gpu-full | memory returned on unload | 7,350 MiB | - | 1 | no target |
| gemma4-e2b | gtx1070-shared | layers on the GTX 1070, searched | 9 | - | 4 | no target |
| minicpm5-2b | gtx1070-shared | layers on the GTX 1070, searched | - | - | 0 | no target |

### E4B's reload, the cost of a level change back

| Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|
| E4B reload after unserve | 21.40 s | 22.00 s | 3 | no target |

### Chained turn: Parakeet on the CPU, the model, Pocket TTS on the CPU

The measured turn runs from the end of the recording to first audio. The turn with endpointing adds 200 ms of silence and run 4's median Smart Turn inference (derived), marked against about 2 to 3 s (2,000 ms target, 3,000 ms floor). `gpu-full-e4b-writing` is `gpu-full` while E4B streams a long answer.

| Model | Placement | Turns | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|---|---|
| gemma4-e2b | gpu-full | command and short | chained turn, measured (to first audio) | 658 ms | 695 ms | 6 | no target |
| gemma4-e2b | gpu-full | command and short | chained turn with endpointing | 866 ms | 904 ms | 6 | target met, derived |
| gemma4-e2b | gpu-full | command and short | speech to text | 172 ms | 185 ms | 6 | no target |
| gemma4-e2b | gpu-full | command and short | the model to its first clause | 328 ms | 345 ms | 6 | no target |
| gemma4-e2b | gpu-full | command and short | speech to first audio | 158 ms | 165 ms | 6 | no target |
| gemma4-e2b | gpu-full | command and short | tool call valid | 5 of 6 (83%) | - | 6 | no target |
| gemma4-e2b | gpu-full | command and short | turns that failed | 0 | - | 6 | no target |
| gemma4-e2b | gpu-full | hand-off | chained turn, measured (to first audio) | 594 ms | 601 ms | 2 | no target |
| gemma4-e2b | gpu-full | hand-off | chained turn with endpointing | 802 ms | 810 ms | 2 | target met, derived |
| gemma4-e2b | gpu-full | hand-off | speech to text | 192 ms | 195 ms | 2 | no target |
| gemma4-e2b | gpu-full | hand-off | the model to its first clause | 232 ms | 235 ms | 2 | no target |
| gemma4-e2b | gpu-full | hand-off | speech to first audio | 170 ms | 171 ms | 2 | no target |
| gemma4-e2b | gpu-full | hand-off | tool call valid | 2 of 2 (100%) | - | 2 | no target |
| gemma4-e2b | gpu-full | hand-off | turns that failed | 0 | - | 2 | no target |
| gemma4-e2b | cpu | command and short | chained turn, measured (to first audio) | 2,058 ms | 2,095 ms | 6 | no target |
| gemma4-e2b | cpu | command and short | chained turn with endpointing | 2,266 ms | 2,304 ms | 6 | floor met, derived |
| gemma4-e2b | cpu | command and short | speech to text | 252 ms | 265 ms | 6 | no target |
| gemma4-e2b | cpu | command and short | the model to its first clause | 1,468 ms | 1,485 ms | 6 | no target |
| gemma4-e2b | cpu | command and short | speech to first audio | 338 ms | 345 ms | 6 | no target |
| gemma4-e2b | cpu | command and short | tool call valid | 5 of 6 (83%) | - | 6 | no target |
| gemma4-e2b | cpu | command and short | turns that failed | 1 | - | 7 | no target |
| gemma4-e2b | cpu | hand-off | chained turn, measured (to first audio) | 1,538 ms | 1,545 ms | 2 | no target |
| gemma4-e2b | cpu | hand-off | chained turn with endpointing | 1,746 ms | 1,754 ms | 2 | target met, derived |
| gemma4-e2b | cpu | hand-off | speech to text | 272 ms | 275 ms | 2 | no target |
| gemma4-e2b | cpu | hand-off | the model to its first clause | 916 ms | 919 ms | 2 | no target |
| gemma4-e2b | cpu | hand-off | speech to first audio | 350 ms | 351 ms | 2 | no target |
| gemma4-e2b | cpu | hand-off | tool call valid | 2 of 2 (100%) | - | 2 | no target |
| gemma4-e2b | cpu | hand-off | turns that failed | 0 | - | 2 | no target |
| gemma4-e2b | gtx1070-shared-idle | command and short | chained turn, measured (to first audio) | 1,315 ms | 1,330 ms | 3 | no target |
| gemma4-e2b | gtx1070-shared-idle | command and short | chained turn with endpointing | 1,524 ms | 1,539 ms | 3 | target met, derived |
| gemma4-e2b | gtx1070-shared-idle | command and short | speech to text | 170 ms | 175 ms | 3 | no target |
| gemma4-e2b | gtx1070-shared-idle | command and short | the model to its first clause | 987 ms | 994 ms | 3 | no target |
| gemma4-e2b | gtx1070-shared-idle | command and short | speech to first audio | 158 ms | 161 ms | 3 | no target |
| gemma4-e2b | gtx1070-shared-idle | command and short | tool call valid | 3 of 3 (100%) | - | 3 | no target |
| gemma4-e2b | gtx1070-shared-idle | command and short | turns that failed | 0 | - | 3 | no target |
| gemma4-e2b | gtx1070-shared-idle | hand-off | chained turn, measured (to first audio) | 953 ms | 953 ms | 1 | no target |
| gemma4-e2b | gtx1070-shared-idle | hand-off | chained turn with endpointing | 1,162 ms | 1,162 ms | 1 | target met, derived |
| gemma4-e2b | gtx1070-shared-idle | hand-off | speech to text | 180 ms | 180 ms | 1 | no target |
| gemma4-e2b | gtx1070-shared-idle | hand-off | the model to its first clause | 609 ms | 609 ms | 1 | no target |
| gemma4-e2b | gtx1070-shared-idle | hand-off | speech to first audio | 164 ms | 164 ms | 1 | no target |
| gemma4-e2b | gtx1070-shared-idle | hand-off | tool call valid | 1 of 1 (100%) | - | 1 | no target |
| gemma4-e2b | gtx1070-shared-idle | hand-off | turns that failed | 0 | - | 1 | no target |
| gemma4-e2b | gpu-full-e4b-writing | command and short | chained turn, measured (to first audio) | 830 ms | 845 ms | 3 | no target |
| gemma4-e2b | gpu-full-e4b-writing | command and short | chained turn with endpointing | 1,039 ms | 1,054 ms | 3 | target met, derived |
| gemma4-e2b | gpu-full-e4b-writing | command and short | speech to text | 175 ms | 180 ms | 3 | no target |
| gemma4-e2b | gpu-full-e4b-writing | command and short | the model to its first clause | 462 ms | 469 ms | 3 | no target |
| gemma4-e2b | gpu-full-e4b-writing | command and short | speech to first audio | 193 ms | 196 ms | 3 | no target |
| gemma4-e2b | gpu-full-e4b-writing | command and short | tool call valid | 3 of 3 (100%) | - | 3 | no target |
| gemma4-e2b | gpu-full-e4b-writing | command and short | turns that failed | 0 | - | 3 | no target |
| gemma4-e2b | gpu-full-e4b-writing | command and short | E4B decode rate during the turns | 34.0 tok/s | 35.0 tok/s | 3 | no target |
| gemma4-e2b | gpu-full-e4b-writing | hand-off | chained turn, measured (to first audio) | 678 ms | 678 ms | 1 | no target |
| gemma4-e2b | gpu-full-e4b-writing | hand-off | chained turn with endpointing | 887 ms | 887 ms | 1 | target met, derived |
| gemma4-e2b | gpu-full-e4b-writing | hand-off | speech to text | 185 ms | 185 ms | 1 | no target |
| gemma4-e2b | gpu-full-e4b-writing | hand-off | the model to its first clause | 294 ms | 294 ms | 1 | no target |
| gemma4-e2b | gpu-full-e4b-writing | hand-off | speech to first audio | 199 ms | 199 ms | 1 | no target |
| gemma4-e2b | gpu-full-e4b-writing | hand-off | tool call valid | 1 of 1 (100%) | - | 1 | no target |
| gemma4-e2b | gpu-full-e4b-writing | hand-off | turns that failed | 0 | - | 1 | no target |
| gemma4-e2b | gpu-full-e4b-writing | hand-off | E4B decode rate during the turns | 36.0 tok/s | 36.0 tok/s | 1 | no target |
| e4b-ref | gpu-full | command and short | chained turn, measured (to first audio) | 842 ms | 857 ms | 3 | no target |
| e4b-ref | gpu-full | command and short | chained turn with endpointing | 1,051 ms | 1,066 ms | 3 | target met, derived |
| e4b-ref | gpu-full | command and short | speech to text | 163 ms | 168 ms | 3 | no target |
| e4b-ref | gpu-full | command and short | the model to its first clause | 527 ms | 534 ms | 3 | no target |
| e4b-ref | gpu-full | command and short | speech to first audio | 152 ms | 155 ms | 3 | no target |
| e4b-ref | gpu-full | command and short | tool call valid | 3 of 3 (100%) | - | 3 | no target |
| e4b-ref | gpu-full | command and short | turns that failed | 0 | - | 3 | no target |
| e4b-ref | gpu-full | hand-off | chained turn, measured (to first audio) | 664 ms | 664 ms | 1 | no target |
| e4b-ref | gpu-full | hand-off | chained turn with endpointing | 873 ms | 873 ms | 1 | target met, derived |
| e4b-ref | gpu-full | hand-off | speech to text | 173 ms | 173 ms | 1 | no target |
| e4b-ref | gpu-full | hand-off | the model to its first clause | 333 ms | 333 ms | 1 | no target |
| e4b-ref | gpu-full | hand-off | speech to first audio | 158 ms | 158 ms | 1 | no target |
| e4b-ref | gpu-full | hand-off | tool call valid | 1 of 1 (100%) | - | 1 | no target |
| e4b-ref | gpu-full | hand-off | turns that failed | 0 | - | 1 | no target |

Derived figures and their inputs:

- chained turn with endpointing: chained turn, measured: speech to text, the model to its first clause, first audio; 200 ms of silence; Smart Turn inference, run 4 median: 9 ms.

- host-1070: qwen35-4b on gtx1070-shared not run: model file sha256 differs from the pin file
- host-1070: e4b-ref on gtx1070-shared not run: does not fit beside extraction in host memory
- host-1070: minicpm5-2b on gtx1070-shared not run: does not fit beside extraction
- host-1070 restore: inspect diff empty True, exit code None

## Run 8: low level while a light game runs

### Session mh1-game-20261121T190000Z

No prior target: each condition's frame rate is set beside G0's own spread.

| Session | Condition | Placement | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|---|---|
| mh1-game-20261121T190000Z | G0 | none | average frame rate | 141.2 fps | 142.5 fps | 3 | no target |
| mh1-game-20261121T190000Z | G0 | none | 1% low frame rate | 98.1 fps | 99.0 fps | 3 | no target |
| mh1-game-20261121T190000Z | G0 | none | 99th-percentile frame time | 10 ms | 10 ms | 3 | no target |
| mh1-game-20261121T190000Z | G0 | none | peak GPU memory used | 4,130 MiB | - | 3 | no target |
| mh1-game-20261121T190000Z | G0 | none | CPU load, mean per pass | 30.0% | 31.0% | 3 | no target |
| mh1-game-20261121T190000Z | G0 | none | G0's own spread, highest pass less lowest | 2.7 fps | - | 3 | no target |
| mh1-game-20261121T190000Z | G1-gpu-full | gpu-full | average frame rate | 138.9 fps | 140.1 fps | 3 | no target |
| mh1-game-20261121T190000Z | G1-gpu-full | gpu-full | 1% low frame rate | 95.2 fps | 96.1 fps | 3 | no target |
| mh1-game-20261121T190000Z | G1-gpu-full | gpu-full | 99th-percentile frame time | 10 ms | 11 ms | 3 | no target |
| mh1-game-20261121T190000Z | G1-gpu-full | gpu-full | peak GPU memory used | 7,330 MiB | - | 3 | no target |
| mh1-game-20261121T190000Z | G1-gpu-full | gpu-full | CPU load, mean per pass | 30.0% | 31.0% | 3 | no target |
| mh1-game-20261121T190000Z | G1-gpu-full | gpu-full | average frame rate against G0 | -2.3 fps | - | 3 | no target |
| mh1-game-20261121T190000Z | G2-gpu-full | gpu-full | average frame rate | 131.4 fps | 133.0 fps | 3 | no target |
| mh1-game-20261121T190000Z | G2-gpu-full | gpu-full | 1% low frame rate | 84.7 fps | 86.0 fps | 3 | no target |
| mh1-game-20261121T190000Z | G2-gpu-full | gpu-full | 99th-percentile frame time | 12 ms | 12 ms | 3 | no target |
| mh1-game-20261121T190000Z | G2-gpu-full | gpu-full | peak GPU memory used | 7,480 MiB | - | 3 | no target |
| mh1-game-20261121T190000Z | G2-gpu-full | gpu-full | CPU load, mean per pass | 30.0% | 31.0% | 3 | no target |
| mh1-game-20261121T190000Z | G2-gpu-full | gpu-full | average frame rate against G0 | -9.8 fps | - | 3 | no target |
| mh1-game-20261121T190000Z | G2-gpu-full | gpu-full | chained turn, measured (to first audio) | 728 ms | 750 ms | 6 | no target |
| mh1-game-20261121T190000Z | G2-gpu-full | gpu-full | chained turn with endpointing | 936 ms | 959 ms | 6 | target met, derived |
| mh1-game-20261121T190000Z | G1-cpu | cpu | average frame rate | 139.5 fps | 140.8 fps | 3 | no target |
| mh1-game-20261121T190000Z | G1-cpu | cpu | 1% low frame rate | 95.9 fps | 97.2 fps | 3 | no target |
| mh1-game-20261121T190000Z | G1-cpu | cpu | 99th-percentile frame time | 10 ms | 11 ms | 3 | no target |
| mh1-game-20261121T190000Z | G1-cpu | cpu | peak GPU memory used | 4,180 MiB | - | 3 | no target |
| mh1-game-20261121T190000Z | G1-cpu | cpu | CPU load, mean per pass | 36.0% | 37.0% | 3 | no target |
| mh1-game-20261121T190000Z | G1-cpu | cpu | average frame rate against G0 | -1.7 fps | - | 3 | no target |
| mh1-game-20261121T190000Z | G2-cpu | cpu | average frame rate | 126.3 fps | 128.1 fps | 3 | no target |
| mh1-game-20261121T190000Z | G2-cpu | cpu | 1% low frame rate | 79.4 fps | 81.2 fps | 3 | no target |
| mh1-game-20261121T190000Z | G2-cpu | cpu | 99th-percentile frame time | 13 ms | 13 ms | 3 | no target |
| mh1-game-20261121T190000Z | G2-cpu | cpu | peak GPU memory used | 4,210 MiB | - | 3 | no target |
| mh1-game-20261121T190000Z | G2-cpu | cpu | CPU load, mean per pass | 36.0% | 37.0% | 3 | no target |
| mh1-game-20261121T190000Z | G2-cpu | cpu | average frame rate against G0 | -14.9 fps | - | 3 | no target |
| mh1-game-20261121T190000Z | G2-cpu | cpu | chained turn, measured (to first audio) | 1,978 ms | 2,000 ms | 6 | no target |
| mh1-game-20261121T190000Z | G2-cpu | cpu | chained turn with endpointing | 2,186 ms | 2,209 ms | 6 | floor met, derived |
| mh1-game-20261121T190000Z | restore | none | E4B's return after the session | 48.20 s | - | 1 | no target |

Derived figures and their inputs:

- chained turn with endpointing: chained turn, measured: speech to text, the model to its first clause, first audio; 200 ms of silence; Smart Turn inference, run 4 median: 9 ms.

- mh1-game-20261121T190000Z: Example Game 1.4.2, settings 1440p, high preset, 144 fps cap, no upscaling; frames from presentmon; model gemma4-e2b
- mh1-game-20261121T190000Z G2-cpu pass 2: a short stutter on the second turn

## Run 9: Minecraft with its server on this PC

Not yet run: no game session with its server process on this PC under `game/` yet; `report --replace` renders again when one arrives.

## Blind listening

Unblinded from the key after the ratings were saved; every clip is a stock voice.

### Ratings by engine, setting and voice

| Engine | Setting | Voice | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|---|---|
| qwen3 | default | aiden | naturalness, 1 to 5 | 4.50 | - | 2 | no target |
| qwen3 | default | aiden | carries the line's feeling, 1 to 5 | 4.00 | - | 2 | no target |
| qwen3 | default | aiden | picked as the favourite of its line | 1 | - | 2 | no target |
| qwen3 | default | ryan | naturalness, 1 to 5 | 3.00 | - | 1 | no target |
| qwen3 | default | ryan | carries the line's feeling, 1 to 5 | 3.00 | - | 1 | no target |
| qwen3 | default | ryan | picked as the favourite of its line | 0 | - | 1 | no target |
| turbo | default | builtin | naturalness, 1 to 5 | 4.00 | - | 1 | no target |
| turbo | default | builtin | carries the line's feeling, 1 to 5 | 3.00 | - | 1 | no target |
| turbo | default | builtin | picked as the favourite of its line | 1 | - | 1 | no target |
| original | default | builtin | naturalness, 1 to 5 | 3.00 | - | 2 | no target |
| original | default | builtin | carries the line's feeling, 1 to 5 | 2.50 | - | 2 | no target |
| original | default | builtin | picked as the favourite of its line | 0 | - | 2 | no target |
| original | expressive | builtin | naturalness, 1 to 5 | 4.00 | - | 1 | no target |
| original | expressive | builtin | carries the line's feeling, 1 to 5 | 5.00 | - | 1 | no target |
| original | expressive | builtin | picked as the favourite of its line | 0 | - | 1 | no target |
| pocket | default | alba | naturalness, 1 to 5 | 4.00 | - | 1 | no target |
| pocket | default | alba | carries the line's feeling, 1 to 5 | 4.00 | - | 1 | no target |
| pocket | default | alba | picked as the favourite of its line | 0 | - | 1 | no target |
| pocket | default | fantine | naturalness, 1 to 5 | 4.00 | - | 1 | no target |
| pocket | default | fantine | carries the line's feeling, 1 to 5 | 3.00 | - | 1 | no target |
| pocket | default | fantine | picked as the favourite of its line | 0 | - | 1 | no target |

## What a 16 to 24 GB card would add (derived)

Every figure in this section is derived: run 5's measured memory and run 2's measured decode rate, carried to cards this PC does not have. The card figures are published specifications and November 2026 street prices from price aggregators, plausible only and re-checked before any purchase.

| Card | Memory | Bandwidth | Street price, November 2026 |
|---|---|---|---|
| RTX 4070 SUPER (today) | 12 GB | about 504 GB/s | n/a |
| RTX 5070 Ti | 16 GB | 896 GB/s | about 1,200 to 1,420 USD (list price 749) |
| RTX 5080 | 16 GB | 960 GB/s | about 1,800 to 2,260 USD (list price 999) |
| RTX 4090, used | 24 GB | about 1,008 GB/s | about 1,200 to 1,500 USD |
| RTX 3090, used | 24 GB | 936 GB/s | trackers disagree |

### Decode scaled by bandwidth

| Card | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|
| RTX 5070 Ti | E4B decode rate, scaled by bandwidth | 88.9 tok/s | - | 1 | no target, derived |
| RTX 5080 | E4B decode rate, scaled by bandwidth | 95.2 tok/s | - | 1 | no target, derived |
| RTX 4090, used | E4B decode rate, scaled by bandwidth | 100.0 tok/s | - | 1 | no target, derived |
| RTX 3090, used | E4B decode rate, scaled by bandwidth | 92.9 tok/s | - | 1 | no target, derived |

Derived figures and their inputs:

- E4B decode rate, scaled by bandwidth: run 2 c0 decode rate on the RTX 4070 SUPER, median 50.0 tok/s (n=6); bandwidth 896 GB/s against about 504 GB/s.
- E4B decode rate, scaled by bandwidth: run 2 c0 decode rate on the RTX 4070 SUPER, median 50.0 tok/s (n=6); bandwidth 960 GB/s against about 504 GB/s.
- E4B decode rate, scaled by bandwidth: run 2 c0 decode rate on the RTX 4070 SUPER, median 50.0 tok/s (n=6); bandwidth about 1,008 GB/s against about 504 GB/s.
- E4B decode rate, scaled by bandwidth: run 2 c0 decode rate on the RTX 4070 SUPER, median 50.0 tok/s (n=6); bandwidth 936 GB/s against about 504 GB/s.

### Today's full level on a larger card

| Card | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|
| 16 GB | headroom beside E4B, a small model, speech and recognition | 5,184 MiB | - | 1 | 2 GiB gate met, derived |
| 24 GB | headroom beside E4B, a small model, speech and recognition | 13,376 MiB | - | 1 | 2 GiB gate met, derived |

Derived figures and their inputs:

- headroom beside E4B, a small model, speech and recognition: run 5's heaviest full-level step that fits: e4b+gemma4-e2b+qwen3+parakeet-gpu, peak 11,200 MiB; 16 GB taken as 16,384 MiB.
- headroom beside E4B, a small model, speech and recognition: run 5's heaviest full-level step that fits: e4b+gemma4-e2b+qwen3+parakeet-gpu, peak 11,200 MiB; 24 GB taken as 24,576 MiB.

### A bigger thinking model in E4B's place

Need: the desktop, the small model, speech and recognition as run 5 measured them, the larger model's file, and E4B's cache and buffers taken as the larger model's, which understates a larger model's cache.

| Model | Card | Figure | Median or value | Slow case | n | Mark |
|---|---|---|---|---|---|---|
| gemma-4-26B-A4B-it Q4_K_M | 16 GB | headroom with the larger model in E4B's place | -5,020 MiB | - | 1 | under 512 MiB margin, derived |
| gemma-4-26B-A4B-it Q4_K_M | 24 GB | headroom with the larger model in E4B's place | 3,172 MiB | - | 1 | 2 GiB gate met, derived |
| gemma-4-12B-it Q4_K_M | 16 GB | headroom with the larger model in E4B's place | 3,372 MiB | - | 1 | 2 GiB gate met, derived |
| gemma-4-12B-it Q4_K_M | 24 GB | headroom with the larger model in E4B's place | 11,564 MiB | - | 1 | 2 GiB gate met, derived |

Derived figures and their inputs:

- headroom with the larger model in E4B's place: gemma-4-26B-A4B-it Q4_K_M file size 15,354 MiB (Hugging Face API, read in H9); the desktop, run 5: 1,300 MiB; the small model, speech and recognition beside E4B, run 5 heaviest step less step e4b: 2,300 MiB; E4B's cache and buffers, run 5 step e4b less E4B's file size: 2,450 MiB; 16 GB taken as 16,384 MiB.
- headroom with the larger model in E4B's place: gemma-4-26B-A4B-it Q4_K_M file size 15,354 MiB (Hugging Face API, read in H9); the desktop, run 5: 1,300 MiB; the small model, speech and recognition beside E4B, run 5 heaviest step less step e4b: 2,300 MiB; E4B's cache and buffers, run 5 step e4b less E4B's file size: 2,450 MiB; 24 GB taken as 24,576 MiB.
- headroom with the larger model in E4B's place: gemma-4-12B-it Q4_K_M file size 6,962 MiB (Hugging Face API, read in H9); the desktop, run 5: 1,300 MiB; the small model, speech and recognition beside E4B, run 5 heaviest step less step e4b: 2,300 MiB; E4B's cache and buffers, run 5 step e4b less E4B's file size: 2,450 MiB; 16 GB taken as 16,384 MiB.
- headroom with the larger model in E4B's place: gemma-4-12B-it Q4_K_M file size 6,962 MiB (Hugging Face API, read in H9); the desktop, run 5: 1,300 MiB; the small model, speech and recognition beside E4B, run 5 heaviest step less step e4b: 2,300 MiB; E4B's cache and buffers, run 5 step e4b less E4B's file size: 2,450 MiB; 24 GB taken as 24,576 MiB.

- Docker Desktop passes only NVIDIA GPUs to WSL 2 containers, so an AMD card would not run MIST's stack as built.
