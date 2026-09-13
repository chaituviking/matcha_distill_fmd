# Serving design — shipping the distilled decoder

This is how I'd actually ship the one-step student. The thing that shapes the
whole design is something the harness already showed me: cutting decoder steps
only helps up to a point, because the vocoder is a fixed cost that takes over.
So most of the plan is built around that.

```
text ─► [frontend] ─► [AR LM] ─► speech tokens ─► [flow decoder] ─► mel ─► [HiFi-GAN] ─► wav
                                                    │                        │
                                          distilled: N steps → 1    fixed cost, doesn't
                                          ~30x cheaper decoder      shrink with NFE →
                                          (RTF 0.095 → 0.003)       floors total RTF ~0.12
```

## Latency and streaming

The target I'd aim for is time-to-first-audio under about 200 ms on the serving
GPU, and total RTF low enough that one GPU can carry several streams at once. For
streaming, RTF is really a throughput budget: a live session eats about one
second of audio per wall-clock second, so however much RTF headroom I have is how
many concurrent sessions fit on a device. (My measured numbers are CPU RTF, which
I used to build and check the pipeline. Production is GPU and the real latency
would come from there.)

Streaming is chunked. The LM produces tokens as it goes, so I synthesize in mel
chunks instead of waiting for the whole utterance. The one-step student is what
makes chunking cheap, because each chunk is just one decoder forward and one
vocoder pass, so audio comes out at roughly the rate the LM produces tokens. I'd
cut the first chunk short, like a single clause, since that's what decides
time-to-first-audio, and make later chunks longer for prosody once audio is
already playing. Chunk boundaries need a few frames of left context going into
the decoder and a short cross-fade at the vocoder so the seams don't click.

NFE also works as a per-chunk latency knob. First chunk at NFE 1 or 2, since
people notice waiting most at the start of a turn, and steady-state at NFE 4. If
the system is under load, dropping the steady-state NFE for everyone is the
graceful way to degrade, slightly softer audio beats timeouts.

For batching, the one constraint from the architecture is that requests at
different NFE can't go in the same batch, because each one needs a different
number of sequential steps. So NFE is a hard partition key, and inside a group I'd
bucket by sequence length to keep padding down. The signal I'd watch for scaling
is queue-wait as a share of tail latency, not GPU utilization, because a
saturated scheduler shows up as requests waiting in the queue while the GPU still
looks under-used.

The vocoder is the next bottleneck, and I know that from the numbers, not a guess.
Once the decoder drops below NFE 4 the total RTF floors around 0.12, and at
student@1 the decoder is only about 3% of the synthesis time (decoder RTF 0.003
against a total of 0.116), so HiFi-GAN is basically all of it. So the follow-on
work is the vocoder, roughly in this order: first a lighter or iSTFT-style vocoder
(predict magnitude and phase at frame rate and let a fixed inverse transform do
the upsampling, which drops the transposed-conv stack), then INT8 quantization of
whatever vocoder ships, and only after that more decoder optimization since it's
already cheap. Compile and CUDA-graph tricks I'd apply carefully, because chunked
synthesis means shapes keep changing, and shape-keyed recompiles turn a per-call
speedup into tail spikes, so I'd pad to a few shape buckets first.

## The quality gate

The rule I care about here is that a faster-but-worse student never ships, and
the tricky part is that the cheap metrics are exactly the ones a faster student
can quietly regress. So the way I split it is: the automatic proxies decide which
candidates are worth looking at, and humans decide what actually ships. The
teacher baseline in my harness is the reason I don't trust proxies alone, WER
stays flat from NFE 32 down to 2 while speaker cosine keeps dropping, which means
intelligibility saturates and the first thing low-NFE loses is timbre. A student
gated on WER alone would sail through sounding muffled.

For promotion I'd A/B the candidate against the current model on the same prompts
and look at:

- WER/CER, no regression past noise, but scored on sliced sets (numbers, named
  entities, code-mixed, long sentences) rather than one aggregate, because the
  aggregate hides the slices that actually matter. I'd call out code-mixed
  specifically since the harness already shows it running about 3x the English
  WER.
- Speaker cosine against the teacher, as a hard floor, since that's the axis that
  moves first.
- MOS proxy, for catching regressions only. A drop blocks it; a flat or higher
  score is necessary but not enough on its own.
- Latency, it has to actually hit the target RTF at the production NFE schedule
  under real concurrent load, tail included.

On top of the proxies there's a human anchor, a CMOS comparison of student vs the
current model on the same utterances with around 20 listeners, on a fixed
set that covers the sliced categories and old failure cases. I'd use CMOS rather
than plain MOS because people rank two clips against each other far more reliably
than they score one in isolation. Any CMOS loss beyond the confidence interval
blocks promotion no matter how good the proxies look. Rollout goes shadow (run it
on real traffic but log instead of serve), then a small canary slice, then full,
and rollback is just flipping back to the previous model which stays warm.

## The improvement loop

```
  production wav ─► monitors: ASR round-trip · speaker drift · MOS proxy · user flags
        ▲                                          │
        │                                          ▼  (flagged text)
  re-distill student ◄── teacher relabels at high NFE ◄── human check on a sample
        │                                                 (teacher bad? → data/frontend)
        └────► full quality gate + human anchor ────► ship
```

Flagging I'd do in tiers, cheapest first. The automatic tier samples production
traffic and scores ASR round-trip and the MOS proxy off the serving path, plus a
speaker-embedding drift check against the target voice, which is the one that
catches timbre drift that WER won't. Then user signals, thumbs-down or repeats,
mapped back to the exact utterance by request ID. And a periodic human audit of a
random sample, because the automatic tier inherits the same blind spots as the
proxies it's built on.

For relabeling, I re-synthesize the flagged prompts with the teacher at high NFE
to get clean targets, which is self-distillation again so it needs no new
recordings. A human listens to a sample of those to decide whether the teacher
itself is actually good; if the teacher fails too, that's a data or frontend
problem, not a distillation one.

Then I re-distill on the accumulated flagged data on a cadence, or whenever the
flagged set gets big enough, oversampling the slices that failed, and run the full
gate again including the human anchor every cycle, not just once. One thing to
watch: since the targets come from the teacher, the student can never beat the
teacher and it inherits the teacher's blind spots, so a periodic human audit of
the teacher on new slices (new languages, new accents) is what decides when the
teacher itself needs upgrading and the student re-distilled from the new one.

