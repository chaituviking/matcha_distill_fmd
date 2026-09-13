# distill-fmd results

12 prompts · CPU (arm64) · torch 2.14.0 · student ckpt: results/student_smoke.pt

## Cost (real numbers — properties of architecture + sampler, valid pre-training)

| config | NFE | params (M) | decoder s/utt | per-NFE ms | decoder RTF | total RTF | peak RSS (MB, CPU)* |
|---|---|---|---|---|---|---|---|
| teacher@32 | 32 | 11.0 | 0.407 | 12.7 | 0.0816 | 0.2095 | 1686 |
| teacher@16 | 16 | 11.0 | 0.202 | 12.6 | 0.0405 | 0.1561 | 1485 |
| teacher@8 | 8 | 11.0 | 0.102 | 12.8 | 0.0205 | 0.1328 | 1798 |
| teacher@4 | 4 | 11.0 | 0.058 | 14.4 | 0.0116 | 0.1320 | 1794 |
| teacher@2 | 2 | 11.0 | 0.030 | 14.9 | 0.0060 | 0.1253 | 1536 |
| student@1 | 1 | 11.0 | 0.016 | 16.3 | 0.0033 | 0.1434 | 1739 |
| student@2 | 2 | 11.0 | 0.035 | 17.4 | 0.0070 | 0.1363 | 1612 |
| student@4 | 4 | 11.0 | 0.064 | 16.1 | 0.0129 | 0.1447 | 1746 |

\* peak process RSS during decode (CPU stand-in for peak VRAM); torch's allocator caches freed memory, so later-run configs can appear cheaper. Params are identical by design: the student is the teacher's architecture — the win is NFE only.

## Quality (teacher rows = real baseline; student rows = **pipeline verification only — student untrained**)

| config | NFE | WER | CER | spk cos vs teacher@32 | MOS proxy (SQUIM) | status |
|---|---|---|---|---|---|---|
| teacher@32 | 32 | n/a | n/a | n/a | n/a | real baseline |
| teacher@16 | 16 | n/a | n/a | n/a | n/a | real baseline |
| teacher@8 | 8 | n/a | n/a | n/a | n/a | real baseline |
| teacher@4 | 4 | n/a | n/a | n/a | n/a | real baseline |
| teacher@2 | 2 | n/a | n/a | n/a | n/a | real baseline |
| student@1 | 1 | n/a | n/a | n/a | n/a | pipeline verification only — student untrained |
| student@2 | 2 | n/a | n/a | n/a | n/a | pipeline verification only — student untrained |
| student@4 | 4 | n/a | n/a | n/a | n/a | pipeline verification only — student untrained |

Columns n/a because unavailable/skipped: asr, spk, mos

## Sanity — one-step student vs teacher (held-out, mel cosine)

On 8 held-out prompts (indices ≥ 4, not in the train_step batch), cosine of each one-step mel against the teacher's 32-step solve:

| one-step predictor | cosine vs teacher@32 |
|---|---|
| **student@1** (consistency map) | **0.9796** |
| naive single Euler step (init baseline) | 0.9786 |
| student − baseline margin | +0.0009 |

The margin is what the consistency objective adds beyond a raw Euler jump. Untrained it is near zero by construction (the student *is* teacher+one-jump at init); a real training run is what opens it — this is a correctly-scoped signal, not a converged claim.
