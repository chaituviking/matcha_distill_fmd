# Serving design — shipping the distilled decoder

How the one-step student actually ships. Grounded in the harness numbers, the
main constraint is that the decoder win is capped by the vocoder, so the design
is built around that.

```
text ─► [frontend] ─► [AR LM] ─► speech tokens ─► [flow decoder] ─► mel ─► [HiFi-GAN] ─► wav
                                                    │                        │
                                          distilled: N steps → 1    fixed cost, doesn't
                                          ~30x cheaper decoder      shrink with NFE →
                                          (RTF 0.095 → 0.003)       floors total RTF ~0.12
```

## Latency and streaming

Target: time-to-first-audio under ~200 ms on the serving GPU, and total RTF low
enough that one GPU serves several concurrent streams. RTF is a throughput
budget for streaming — a live session needs ~1 audio-second per wall-clock
second, so RTF headroom is what converts into concurrent sessions per device.
(Note: my measured numbers are CPU RTF, for developing the pipeline; production
is GPU and the absolute latency would come from there.)

Streaming is chunked. The LM emits tokens incrementally, so we synthesize in mel
chunks rather than waiting for the full utterance. The one-step student is what
makes this cheap — each chunk is one decoder forward plus one vocoder pass, so
audio comes out at roughly token cadence. The first chunk is cut short (a clause)
to minimize time-to-first-audio; later chunks are longer for prosody. Chunk
boundaries need a few frames of left-context into the decoder and a short vocoder
cross-fade to avoid clicks.

NFE is a per-chunk latency dial: first chunk at NFE 1-2 (people are most
wait-sensitive at the start of a turn), steady-state at NFE 4. Dropping
steady-state NFE globally is the graceful-degradation lever under load.

Batching: requests at different NFE can't share a batch (each needs a different
number of sequential steps), so NFE is a hard partition key; within a group,
bucket by sequence length. Watch queue-wait share of tail latency rather than
GPU utilization — a saturated scheduler shows queued requests with an
under-utilized GPU.

The vocoder is the next bottleneck, and this is measured, not assumed. Once the
decoder is below NFE 4, total RTF floors at ~0.12; at student@1 the decoder is
~3% of total synthesis time (decoder RTF 0.003 against total 0.116) and the
vocoder dominates the rest. So the follow-on work is the vocoder, roughly in
order: (1) a lighter or iSTFT-style vocoder (predict magnitude+phase at frame
rate and let a fixed inverse transform do the upsampling, dropping the
transposed-conv stack); (2) INT8 quantization of the vocoder; (3) only then more
decoder optimization, since it's already cheap. Compile/CUDA-graph tricks need
care — chunked synthesis means varying shapes, and shape-keyed recompiles turn a
per-call win into tail spikes, so pad to shape buckets first.

## The quality gate

A faster-but-worse student must never ship, and the proxies are exactly what a
faster student can quietly regress. The rule: proxies gate iteration, humans gate
promotion. The teacher baseline in the harness shows why proxies aren't enough —
WER is flat from NFE 32 down to 2 while speaker cosine drops steadily, so
intelligibility saturates and what low-NFE loses first is timbre. A WER-only gate
would pass a muffled student.

Promotion, A/B against the current model on the same prompts:
- WER/CER: no regression beyond noise, scored on sliced sets (numbers, named
  entities, code-mixed, long sentences) not just the aggregate — the aggregate
  hides the slices that matter. Code-mixed is called out because the harness
  already shows it's ~3× the English WER.
- Speaker cosine vs the teacher: a hard floor, since it's the earliest-moving
  axis.
- MOS proxy: regression detection only — a drop blocks, a flat/higher score is
  necessary but not sufficient.
- Latency: has to actually hit the target RTF at the production NFE schedule
  under concurrent load, tail included.

Human anchor: a CMOS comparison (student vs incumbent, same utterances, ~20+
listeners) on a fixed set covering the sliced categories and past failure cases.
CMOS because people rank more reliably than they score absolutely. Any CMOS loss
beyond the confidence interval blocks promotion regardless of the proxies. Roll
out via shadow (log, don't serve) → canary on a small slice → full, with rollback
to the warm incumbent.

## The improvement loop

```
  production wav ─► monitors: ASR round-trip · speaker drift · MOS proxy · user flags
        ▲                                          │
        │                                          ▼  (flagged text)
  re-distill student ◄── teacher relabels at high NFE ◄── human check on a sample
        │                                                 (teacher bad? → data/frontend)
        └────► full quality gate + human anchor ────► ship
```

Flagging, cheapest first: (1) automatic — sample production traffic and score ASR
round-trip and MOS proxy off the serving path, plus a speaker-embedding drift
check against the target voice (that catches timbre drift, which WER won't);
(2) user signals (thumbs-down, repeats) mapped to the utterance by request ID;
(3) a periodic human audit, because the automatic tier inherits the proxies'
blind spots.

Relabeling: re-synthesize flagged prompts with the high-NFE teacher to get clean
targets (self-distillation, no new audio). A human listen on a sample decides
whether the teacher itself is good; if the teacher also fails, that's a
data/frontend problem, not distillation.

Re-distill on the accumulated flagged data on a cadence (or when the flagged set
crosses a size threshold), oversampling the slices that failed, then run the full
gate again including the human anchor every cycle. Because the targets come from
the teacher, the student can't beat the teacher and inherits its blind spots, so
a periodic human audit of the teacher on new slices (new languages, accents)
decides when the teacher itself needs upgrading and the student re-distilled from
it.
