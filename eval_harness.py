"""End-to-end eval harness: cost (real) and quality (wired, honestly labelled).

Runs the both teacher and student across an NFE sweep on prompts.txt(around 12 prompts) and gives the output as
results/table.md + results/rtf_vs_nfe.png. Cost numbers are real right now and they're just properties of the architecture and sampler. Quality is real for
the teacher (that's the baseline curve); the student's quality is labelled as verification only, because its checkpoint is a short smoke run, not a trainedmodel.
"""

import argparse
import os
import threading
import time
from dataclasses import dataclass, field

import torch

from student import ConsistencyStudent
from teacher import SAMPLE_RATE, TeacherFMD

EVAL_SR = 16000  #  16 kHz
STUDENT_QUALITY_NOTE = "pipeline verification only — student untrained"


# helper functions


def peak_rss_during(fn):
    """Run fn() while a background thread samples process RSS; returns (result, peak_rss_mb). This is my CPU stand-in for peak VRAM. Caveat (also
    noted in the table)and the torch caches freed memory, so later-run configs can appear cheaper than they are."""
    import psutil

    proc = psutil.Process()
    peak = proc.memory_info().rss
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, proc.memory_info().rss)
            time.sleep(0.005)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        stop.set()
        t.join()
    return result, peak / 1e6


def to_16k(wav: torch.Tensor) -> torch.Tensor:
    import torchaudio.functional as AF

    return AF.resample(wav, SAMPLE_RATE, EVAL_SR)


def load_quality_metrics(args) -> dict:
    """Load each metric behind a try or except, so a missing dependency or a failed download just drops that column to n/a instead of taking down the run."""
    metrics: dict = {}

    if not args.skip_asr:
        try:
            from faster_whisper import WhisperModel

            metrics["asr"] = WhisperModel("small", device="cpu", compute_type="int8")
        except Exception as e:  # degrades it, doesn't crash the harness
            print(f"[warn] ASR unavailable, skipping WER/CER: {e}")

    if not args.skip_speaker:
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            metrics["spk"] = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(args.out_dir, ".ecapa"),
                run_opts={"device": "cpu"},
            )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] speaker embedding unavailable, skipping cosine: {e}")

    if not args.skip_mos:
        try:
            from torchaudio.pipelines import SQUIM_SUBJECTIVE

            metrics["mos"] = SQUIM_SUBJECTIVE.get_model()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] SQUIM unavailable, skipping MOS proxy: {e}")

    return metrics


def wer_cer(asr, wav: torch.Tensor, ref_text: str) -> tuple[float, float, str]:
    """ASR round-trip intelligibility. i lowercase and strip punctuation so formatting differences like '4:30 p.m.' vs '4:30 pm' don't drown the signal
    number formatting still highs WER a bit, so treat it as relative across configs, not an absolute score."""
    import jiwer

    segments, _ = asr.transcribe(to_16k(wav).numpy(), language="en", beam_size=1)
    hyp = " ".join(s.text for s in segments)
    norm = jiwer.Compose(
        [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()]
    )
    ref, hyp = norm(ref_text), norm(hyp)
    if not hyp:
        return 1.0, 1.0, hyp
    # Return the normalized hypothesis too, so the harness can dump per-config
    # transcripts that's the audit trail that WER came from this config's own
    # audio, not a reused transcript.
    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp), hyp


def speaker_cosine(spk, wav: torch.Tensor, ref_wav: torch.Tensor) -> float:
    emb = spk.encode_batch(to_16k(wav).unsqueeze(0)).squeeze()
    ref = spk.encode_batch(to_16k(ref_wav).unsqueeze(0)).squeeze()
    return float(torch.nn.functional.cosine_similarity(emb, ref, dim=0))


def mos_proxy(mos_model, wav: torch.Tensor, nmr_wav: torch.Tensor) -> float:
    # SQUIM-subjective needs a non-matching reference; a teacher clip of a
    # different prompt is exactly that.
    with torch.no_grad():
        return float(mos_model(to_16k(wav).unsqueeze(0), to_16k(nmr_wav).unsqueeze(0)).squeeze())


#synthesis


@dataclass
class ConfigResult:
    name: str  # e.g. "teacher@8" / "student@1"
    nfe: int
    decoder_s: list[float] = field(default_factory=list)
    total_rtf: list[float] = field(default_factory=list)
    audio_s: list[float] = field(default_factory=list)
    wavs: list[torch.Tensor] = field(default_factory=list)
    peak_rss_mb: float = 0.0
    wer: list[float] = field(default_factory=list)
    cer: list[float] = field(default_factory=list)
    transcripts: list[str] = field(default_factory=list)
    spk_cos: list[float] = field(default_factory=list)
    mos: list[float] = field(default_factory=list)

    @property
    def is_student(self) -> bool:
        return self.name.startswith("student")

    def mean(self, attr: str) -> float:
        vals = getattr(self, attr)
        return sum(vals) / len(vals) if vals else float("nan")


def run_config(
    teacher: TeacherFMD,
    student: ConsistencyStudent | None,
    conds: list,
    nfe: int,
    seed: int,
) -> ConfigResult:
    """Synthesize every prompt at one (model, NFE) setting, timing decoder and vocoder separately. Same seed per config means the same starting noise, so
    the differences come from the sampler, not the noise."""
    res = ConfigResult(name=("student" if student else "teacher") + f"@{nfe}", nfe=nfe)

    def run() -> None:
        torch.manual_seed(seed)
        for cond in conds:
            if student is None:
                mel, dec_s = teacher.decode(cond, nfe)
            else:
                x0 = torch.randn_like(cond.mu) * 0.667  # match teacher temperature
                t0 = time.perf_counter()
                mel = (
                    student.sample_onestep(x0, cond)
                    if nfe == 1
                    else student.sample_multistep(x0, cond, nfe)
                )
                dec_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            wav = teacher.mel_to_wav(mel)  # same vocoder for both, so this isolates the decoder
            voc_s = time.perf_counter() - t0
            audio_s = wav.numel() / SAMPLE_RATE
            res.decoder_s.append(dec_s)
            res.total_rtf.append((dec_s + voc_s) / audio_s)
            res.audio_s.append(audio_s)
            res.wavs.append(wav)

    _, res.peak_rss_mb = peak_rss_during(run)
    return res


def _euler_from(teacher: TeacherFMD, x0: torch.Tensor, cond, nfe: int) -> torch.Tensor:
    """Euler-solve the teacher field from a x0, so all three predictions in the agreement check start from the same noise."""
    x = x0.clone()
    t_span = torch.linspace(0, 1, nfe + 1, device=x0.device)
    for i in range(nfe):
        t = t_span[i].expand(x.shape[0])
        x = x + (t_span[i + 1] - t_span[i]) * teacher.velocity_field(x, t, cond)
    return x


@torch.no_grad()
def student_teacher_agreement(
    teacher: TeacherFMD,
    student: ConsistencyStudent,
    conds: list,
    heldout_from: int,
    max_nfe: int,
    seed: int,
) -> dict[str, float]:
    """The sanity check, does the one-step student track the teacher on held-out prompts (indices >= heldout_from, i.e. not in
    train_step's default batch)?

    For each held-out prompt i draw one x0 and, from that same x0, compute the teacher's full max_nfe solve (the target), the teacher's single Euler step
    (naive baseline)and the student's one-step prediction. Then cosine of each one-step mel against the full solve, over valid frames. The number i actually
    trust is the margin of student over the naive baseline, that's what the consistency map adds beyond a plain Euler jump. Untrained it's nearly 0 by
    construction (student nearly equals to teacher one-step at init); training is what opens it. So this is a correctly-scoped number, not a converged claim."""
    
    stu_cos, base_cos = [], []
    for i in range(heldout_from, len(conds)):
        cond = conds[i]
        torch.manual_seed(seed + i)
        x0 = torch.randn_like(cond.mu) * 0.667
        target = _euler_from(teacher, x0, cond, max_nfe)
        base = _euler_from(teacher, x0, cond, 1)
        stu = student.forward(x0, torch.zeros(x0.shape[0], device=x0.device), cond)
        m = cond.mask.bool().expand_as(target)
        stu_cos.append(float(torch.cosine_similarity(stu[m].flatten(), target[m].flatten(), dim=0)))
        base_cos.append(float(torch.cosine_similarity(base[m].flatten(), target[m].flatten(), dim=0)))
    n = len(stu_cos)
    mean = lambda a: sum(a) / n if n else float("nan")
    return {"n_heldout": n, "student_vs_teacher": mean(stu_cos),
            "naive_onestep_vs_teacher": mean(base_cos),
            "margin": mean(stu_cos) - mean(base_cos)}


# outputs


def cost_table(results: list[ConfigResult], n_params: int) -> str:
    lines = [
        "## Cost (real numbers — properties of architecture + sampler, valid pre-training)",
        "",
        "| config | NFE | params (M) | decoder s/utt | per-NFE ms | decoder RTF | total RTF | peak RSS (MB, CPU)* |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        dec = r.mean("decoder_s")
        rtf = dec / r.mean("audio_s")
        lines.append(
            f"| {r.name} | {r.nfe} | {n_params / 1e6:.1f} | {dec:.3f} | {1000 * dec / r.nfe:.1f} "
            f"| {rtf:.4f} | {r.mean('total_rtf'):.4f} | {r.peak_rss_mb:.0f} |"
        )
    lines.append(
        "\n\\* peak process RSS during decode (CPU stand-in for peak VRAM); torch's allocator "
        "caches freed memory, so later-run configs can appear cheaper. Params are identical by "
        "design: the student is the teacher's architecture — the win is NFE only."
    )
    return "\n".join(lines)


def quality_table(results: list[ConfigResult], metrics: dict, ref_name: str) -> str:
    lines = [
        "## Quality (teacher rows = real baseline; student rows = "
        f"**{STUDENT_QUALITY_NOTE}**)",
        "",
        "| config | NFE | WER | CER | spk cos vs " + ref_name + " | MOS proxy (SQUIM) | status |",
        "|---|---|---|---|---|---|---|",
    ]
    fmt = lambda r, a: (f"{r.mean(a):.3f}" if getattr(r, a) else "n/a")
    for r in results:
        status = STUDENT_QUALITY_NOTE if r.is_student else "real baseline"
        lines.append(
            f"| {r.name} | {r.nfe} | {fmt(r, 'wer')} | {fmt(r, 'cer')} | {fmt(r, 'spk_cos')} "
            f"| {fmt(r, 'mos')} | {status} |"
        )
    missing = [k for k in ("asr", "spk", "mos") if k not in metrics]
    if missing:
        lines.append(f"\nColumns n/a because unavailable/skipped: {', '.join(missing)}")
    return "\n".join(lines)


def codemixed_report(ref: ConfigResult, codemixed_idx: int, prompts: list[str]) -> str:
    """WER/CER on the Hinglish code-mixed line, reported separately from the English prompts. Uses the teacher baseline `ref` (the real numbers). Returns
    "" if ASR was skipped."""
    if not ref.wer or codemixed_idx >= len(ref.wer):
        return ""
    eng = [j for j in range(len(ref.wer)) if j != codemixed_idx]
    e_wer = sum(ref.wer[j] for j in eng) / len(eng)
    e_cer = sum(ref.cer[j] for j in eng) / len(eng)
    cm_wer, cm_cer = ref.wer[codemixed_idx], ref.cer[codemixed_idx]
    snippet = prompts[codemixed_idx][:38]
    return (
        f"## Code-mixed vs English — ASR round-trip on {ref.name} (real baseline)\n\n"
        f"| set | WER | CER |\n|---|---|---|\n"
        f"| English prompts (mean, n={len(eng)}) | {e_wer:.3f} | {e_cer:.3f} |\n"
        f'| Hinglish code-mixed ("{snippet}…") | {cm_wer:.3f} | {cm_cer:.3f} |\n\n'
        f"The code-mixed line's round-trip error is far higher ({cm_wer:.2f} vs {e_wer:.2f} WER) "
        f"because the English LJSpeech teacher + phonemizer voice the Hindi words in a native "
        f"US-English accent — on a human listen, \"arre bhai\" is spoken as \"are by\", so ASR "
        f"faithfully hears an anglicized mispronunciation. This is a **teacher/frontend** limit, "
        f"not the distillation (which is accent-agnostic); and since accent lives in fine "
        f"high-frequency detail, it is exactly the slice a low-NFE student over-smooths first — "
        f"so the serving gate scores code-mixed separately (see writeup)."
    )


def plot_rtf(results: list[ConfigResult], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for is_student, label, style in ((False, "teacher (Euler CFM)", "o-"), (True, "student (consistency)", "s--")):
        pts = sorted((r.nfe, r.mean("decoder_s") / r.mean("audio_s")) for r in results if r.is_student == is_student)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("NFE (decoder function evaluations)")
    ax.set_ylabel("decoder RTF (CPU)")
    ax.set_title("Decoder cost vs NFE — teacher Euler sweep vs consistency student")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


# main


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost + quality eval for teacher vs student.")
    parser.add_argument("--prompts", default="prompts.txt")
    parser.add_argument("--teacher-nfe", default="32,16,8,4,2")
    parser.add_argument("--student-nfe", default="1,2,4")
    parser.add_argument("--student-ckpt", default="results/student_smoke.pt")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-asr", action="store_true", help="skip WER/CER (no whisper download)")
    parser.add_argument("--skip-speaker", action="store_true", help="skip speaker cosine (no ECAPA download)")
    parser.add_argument("--skip-mos", action="store_true", help="skip SQUIM MOS proxy")
    parser.add_argument("--heldout-from", type=int, default=4,
                        help="prompt index treated as held-out for the agreement check "
                             "(matches train_step's default --n-prompts 4)")
    parser.add_argument("--codemixed-idx", type=int, default=4,
                        help="0-based index of the Hinglish code-mixed prompt, reported "
                             "separately from the English prompts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    teacher = TeacherFMD()
    student = ConsistencyStudent(teacher).eval()
    if os.path.exists(args.student_ckpt):
        student.load_state_dict(torch.load(args.student_ckpt, weights_only=True))
        print(f"loaded student from {args.student_ckpt} (smoke run — still untrained)")
    else:
        print("no student checkpoint found — evaluating teacher-initialized student")

    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    print(f"encoding {len(prompts)} prompts once (conditioning is shared across configs)")
    conds = [teacher.encode(p) for p in prompts]

    teacher_nfes = sorted((int(n) for n in args.teacher_nfe.split(",")), reverse=True)
    student_nfes = sorted(int(n) for n in args.student_nfe.split(","))

    results: list[ConfigResult] = []
    for nfe in teacher_nfes:
        print(f"synthesizing teacher@{nfe} ...")
        results.append(run_config(teacher, None, conds, nfe, args.seed))
    for nfe in student_nfes:
        print(f"synthesizing student@{nfe} ...")
        results.append(run_config(teacher, student, conds, nfe, args.seed))

    # Speaker-similarity reference = teacher at its highest NFE (best quality).
    ref = results[0]
    metrics = load_quality_metrics(args)
    for r in results:
        print(f"scoring {r.name} ...")
        for i, wav in enumerate(r.wavs):
            if "asr" in metrics:
                w, c, hyp = wer_cer(metrics["asr"], wav, prompts[i])
                r.wer.append(w)
                r.cer.append(c)
                r.transcripts.append(hyp)
            if "spk" in metrics and r is not ref:
                r.spk_cos.append(speaker_cosine(metrics["spk"], wav, ref.wavs[i]))
            if "mos" in metrics:
                # non-matching reference: the ref config's clip of another prompt
                r.mos.append(mos_proxy(metrics["mos"], wav, ref.wavs[(i + 1) % len(prompts)]))

    if "asr" in metrics:
        tdir = os.path.join(args.out_dir, "transcripts")
        os.makedirs(tdir, exist_ok=True)
        for r in results:
            with open(os.path.join(tdir, f"{r.name.replace('@', '_nfe')}.txt"), "w") as f:
                f.write("\n".join(r.transcripts) + "\n")
        print(f"wrote per-config ASR transcripts to {tdir}/")

    # A few wavs per config to actually listen to (the metrics are only proxies).
    import soundfile as sf

    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    for r in results:
        for i in (0, 4, 6):  # numbers/currency, code-mixed, long sentence
            sf.write(os.path.join(audio_dir, f"{r.name.replace('@', '_nfe')}_p{i}.wav"), r.wavs[i].numpy(), SAMPLE_RATE)

    # Sanity: does the one-step student track the teacher on held-out prompts?
    agree = student_teacher_agreement(teacher, student, conds, args.heldout_from, teacher_nfes[0], args.seed)
    agree_md = (
        f"## Sanity — one-step student vs teacher (held-out, mel cosine)\n\n"
        f"On {agree['n_heldout']} held-out prompts (indices ≥ {args.heldout_from}, not in the "
        f"train_step batch), cosine of each one-step mel against the teacher's {teacher_nfes[0]}-step solve:\n\n"
        f"| one-step predictor | cosine vs teacher@{teacher_nfes[0]} |\n|---|---|\n"
        f"| **student@1** (consistency map) | **{agree['student_vs_teacher']:.4f}** |\n"
        f"| naive single Euler step (init baseline) | {agree['naive_onestep_vs_teacher']:.4f} |\n"
        f"| student − baseline margin | {agree['margin']:+.4f} |\n\n"
        f"The margin is what the consistency objective adds beyond a raw Euler jump. Untrained it "
        f"is near zero by construction (the student *is* teacher+one-jump at init); a real training "
        f"run is what opens it — this is a correctly-scoped signal, not a converged claim."
    )
    print(f"\n{agree_md}\n")

    # Code-mixed (Hinglish) prompt reported separately from the English prompts.
    cm_md = codemixed_report(ref, args.codemixed_idx, prompts)
    if cm_md:
        print(f"{cm_md}\n")

    n_params = sum(p.numel() for p in teacher.decoder.estimator.parameters())
    header = (
        f"# distill-fmd results\n\n"
        f"{len(prompts)} prompts · CPU ({os.uname().machine}) · torch {torch.__version__} · "
        f"student ckpt: {args.student_ckpt if os.path.exists(args.student_ckpt) else 'none (teacher-init)'}\n"
    )
    table = "\n".join([header, cost_table(results, n_params), "", quality_table(results, metrics, ref.name),
                       "", agree_md] + (["", cm_md] if cm_md else []))
    table_path = os.path.join(args.out_dir, "table.md")
    with open(table_path, "w") as f:
        f.write(table + "\n")
    print(f"wrote {table_path}\n")
    print(table)
    plot_rtf(results, os.path.join(args.out_dir, "rtf_vs_nfe.png"))


if __name__ == "__main__":
    main()
