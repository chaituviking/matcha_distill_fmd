# Teacher comparison — Matcha-TTS vs CosyVoice 2 flow decoder

Both are OT-CFM (flow-matching) mel decoders driven by an Euler ODE solver, so
the same distillation machinery *should* apply to either. This documents what
actually happened when CosyVoice 2's decoder was wired in as a second teacher,
and why Matcha remains the substrate for the distillation itself.

All numbers are CPU (arm64 laptop), measured by `eval_harness.py` over the same
12 prompts. The Matcha numbers come from the main repo (`results/table.md`). The
CosyVoice run and its code live in the separate experiment folder
`distill-fmd(cosyvoice)/` (`teacher_cosyvoice.py`, and
`results_cosyvoice/table.md` from `eval_harness.py --teacher cosyvoice`, cost
only — see the quality caveat below). `cosy_notes.txt` here is the short version.

## Cost — the decoder that gets distilled

| metric (teacher decoder) | Matcha-TTS | CosyVoice 2 | ratio |
|---|---|---|---|
| estimator params | 11.0 M | 71.3 M | **6.5× larger** |
| per-NFE latency | ~13 ms | ~440–575 ms | **~35–45× heavier per step** |
| decoder s/utt @ NFE 32 | 0.420 s | 14.14 s | 34× |
| decoder s/utt @ NFE 2 | 0.027 s | 1.15 s | 43× |
| decoder RTF @ NFE 32 | 0.084 | 0.738 | — |
| decoder RTF @ NFE 2 | 0.0053 | 0.060 | — |
| total RTF @ NFE 32 (incl. vocoder) | 0.215 | 0.772 | — |
| peak RSS during decode | ~1.7 GB | ~6.2 GB | 3.7× |
| output sample rate | 22.05 kHz | 24 kHz | — |

**Reading it.** CosyVoice 2's flow decoder is a much bigger network (71 M vs
11 M params) and costs ~35–45× more per function evaluation. At its typical
NFE it is **slower than real time on CPU** (decoder RTF 0.74 at NFE 32), where
Matcha is comfortably faster than real time. This is exactly why the NFE axis
matters *more* for CosyVoice — but also why it is the wrong thing to distill on
a laptop with no GPU: a single training step needs teacher forward passes, and
at ~0.5 s per NFE those are ~35× more expensive than Matcha's.

(Per-NFE latency drifting up slightly at *lower* NFE — 442 ms at NFE 32 vs
577 ms at NFE 2 — is fixed per-utterance overhead, e.g. tensor allocation,
being divided by fewer steps, plus CPU jitter; the per-step compute itself is
constant. Peak-RSS wobble is torch's caching allocator, noted in the tables.)

## Distillability — why Matcha is the substrate, not CosyVoice

The consistency student (`student.py`) copies `teacher.decoder.estimator` and
calls it as `estimator(x, mask, mu, t, None)` — i.e. it supplies only `x` and
`mu` (160 mel-channels) with no speaker/extra conditioning.

- **Matcha** — its estimator is built for exactly that: `in_channels = 160`
  (`x`+`mu`), `spks=None` for the single-speaker LJSpeech model. The student
  forward runs unchanged.
- **CosyVoice 2** — its estimator (`CausalConditionalDecoder`) is built for
  `in_channels = 320`: the concatenation of `x`(80) + `mu`(80) + `spks`(80) +
  `cond`(80). The captured conditioning confirms this:

  ```
  mu   : (2, 80, T)   token-derived features
  mask : (2, 1,  T)
  spks : (2, 80)      speaker xvector, normalized + affine-projected
  cond : (2, 80, T)   prompt-feat in a zeros tensor (zero-shot conditioning)
  ```
  (batch = 2 is CosyVoice's classifier-free-guidance conditional/uncond pair.)

  Called the way `student.py` calls it, the packed input is only 160 channels,
  so the first convolution fails:

  ```
  Given groups=1, weight of size [256, 320, 3], expected input[2, 160, 472]
  to have 320 channels, but got 160 channels instead
  ```

  Bridging this needs `spks`/`cond` to flow through the `Cond` object and the
  student call — i.e. edits to the frozen `student.py`/`losses.py`. **That is
  the concrete "extraction cost" the project README refers to**, now measured
  rather than asserted.

## Quality — not reported per-NFE for CosyVoice (and why)

Matcha's quality-vs-NFE is real (`results/table.md`): its native sampler *is*
fixed-step linear Euler, so the harness's `decode()` reproduces it faithfully.

CosyVoice's is **not** reported across NFE, on purpose. `teacher_cosyvoice.decode()`
is a simplified linear-Euler loop written to *time the estimator* — it is not
CosyVoice 2's production sampler, which uses classifier-free guidance, a cosine
timestep schedule, and a causal prompt cache. Audio from the simplified loop is
therefore degraded and unrepresentative (a quick ASR check gave WER ~0.9 even
at NFE 32, its best setting — a sampler artefact, not the model's quality).
Reporting those as "CosyVoice quality" would be exactly the kind of unfaithful
number the harness is built to avoid. Faithful CosyVoice output only comes from
its native `inference_zero_shot` (fixed at 10 NFE), which is used here solely to
capture conditioning. Cost is the distillation axis, and cost *is* measured
faithfully (estimator call timing is independent of the t-schedule).

## Bottom line

CosyVoice 2's decoder loads, runs, and its conditioning is captured — but it is
6–7× bigger, ~35–45× more expensive per NFE, needs 320-channel conditioning the
frozen student can't supply, and is too heavy to distill or faithfully evaluate
on a CPU laptop. Matcha-TTS is the identical flow-matching substrate at a
fraction of the cost and with a student-compatible estimator, so the
distillation and its honest eval stay on Matcha; CosyVoice stands as the "what
production really looks like" cost reference.
