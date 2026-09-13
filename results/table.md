# distill-fmd results

12 prompts · CPU (arm64) · torch 2.14.0 · student ckpt: results/student_smoke.pt

## Cost (real numbers — properties of architecture + sampler, valid pre-training)

| config | NFE | params (M) | decoder s/utt | per-NFE ms | decoder RTF | total RTF | peak RSS (MB, CPU)* |
|---|---|---|---|---|---|---|---|
| teacher@32 | 32 | 11.0 | 0.413 | 12.9 | 0.0828 | 0.2144 | 1767 |
| teacher@16 | 16 | 11.0 | 0.200 | 12.5 | 0.0401 | 0.1560 | 1748 |
| teacher@8 | 8 | 11.0 | 0.100 | 12.6 | 0.0201 | 0.1314 | 1872 |
| teacher@4 | 4 | 11.0 | 0.051 | 12.6 | 0.0101 | 0.1193 | 1473 |
| teacher@2 | 2 | 11.0 | 0.026 | 12.8 | 0.0051 | 0.1134 | 1755 |
| student@1 | 1 | 11.0 | 0.014 | 13.7 | 0.0028 | 0.1104 | 1720 |
| student@2 | 2 | 11.0 | 0.026 | 13.0 | 0.0052 | 0.1128 | 1600 |
| student@4 | 4 | 11.0 | 0.052 | 13.0 | 0.0104 | 0.1197 | 1731 |

\* peak process RSS during decode (CPU stand-in for peak VRAM); torch's allocator caches freed memory, so later-run configs can appear cheaper. Params are identical by design: the student is the teacher's architecture — the win is NFE only.

## Quality (teacher rows = real baseline; student rows = **pipeline verification only — student untrained**)

| config | NFE | WER | CER | spk cos vs teacher@32 | MOS proxy (SQUIM) | status |
|---|---|---|---|---|---|---|
| teacher@32 | 32 | 0.099 | 0.034 | n/a | 4.336 | real baseline |
| teacher@16 | 16 | 0.099 | 0.034 | 0.996 | 4.319 | real baseline |
| teacher@8 | 8 | 0.099 | 0.034 | 0.989 | 4.288 | real baseline |
| teacher@4 | 4 | 0.099 | 0.034 | 0.977 | 4.326 | real baseline |
| teacher@2 | 2 | 0.099 | 0.034 | 0.961 | 4.331 | real baseline |
| student@1 | 1 | 0.071 | 0.016 | 0.921 | 4.103 | pipeline verification only — student untrained |
| student@2 | 2 | 0.089 | 0.029 | 0.913 | 4.299 | pipeline verification only — student untrained |
| student@4 | 4 | 0.071 | 0.016 | 0.897 | 4.376 | pipeline verification only — student untrained |

## Sanity — one-step student vs teacher (held-out, mel cosine)

On 8 held-out prompts (indices ≥ 4, not in the train_step batch), cosine of each one-step mel against the teacher's 32-step solve:

| one-step predictor | cosine vs teacher@32 |
|---|---|
| **student@1** (consistency map) | **0.9796** |
| naive single Euler step (init baseline) | 0.9786 |
| student − baseline margin | +0.0009 |

The margin is what the consistency objective adds beyond a raw Euler jump. Untrained it is near zero by construction (the student *is* teacher+one-jump at init); a real training run is what opens it — this is a correctly-scoped signal, not a converged claim.

## Code-mixed vs English — ASR round-trip on teacher@32 (real baseline)

| set | WER | CER |
|---|---|---|
| English prompts (mean, n=11) | 0.085 | 0.028 |
| Hinglish code-mixed ("Arre bhai, please send me the report b…") | 0.250 | 0.098 |

The code-mixed line's round-trip error is far higher (0.25 vs 0.08 WER) because the English LJSpeech teacher + phonemizer voice the Hindi words in a native US-English accent — on a human listen, "arre bhai" is spoken as "are by", so ASR faithfully hears an anglicized mispronunciation. This is a **teacher/frontend** limit, not the distillation (which is accent-agnostic); and since accent lives in fine high-frequency detail, it is exactly the slice a low-NFE student over-smooths first — so the serving gate scores code-mixed separately (see writeup).
