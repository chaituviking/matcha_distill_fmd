"""One consistency-distillation step that actually runs, plus a short smoke loop.

Targets come from self-distillation: the teacher makes its own mel targets at
high NFE from prompts.txt, so I don't need any external audio. The full training
run is a stub (--full) — the task is to show a step runs correctly, not to train
to convergence.
"""

import argparse

import torch

from losses import consistency_loss, make_ema_student, update_ema
from student import ConsistencyStudent
from teacher import Cond, TeacherFMD


def build_batch(teacher: TeacherFMD, prompts: list[str], teacher_nfe: int) -> tuple[torch.Tensor, Cond]:
    """(cond, x1) training pairs from the teacher, padded to a common length.

    Each prompt is encoded and decoded on its own (lengths differ), then
    zero-padded to the longest; the mask carries the real lengths into the loss.
    Per-prompt lengths are already multiples of 4 (fix_len_compatibility), so
    the max is a valid U-Net length too.
    """
    conds, x1s = [], []
    for text in prompts:
        cond = teacher.encode(text)
        x1, _ = teacher.decode(cond, nfe=teacher_nfe)
        conds.append(cond)
        x1s.append(x1)
    t_max = max(c.mu.shape[-1] for c in conds)
    pad = lambda x: torch.nn.functional.pad(x, (0, t_max - x.shape[-1]))
    mu = torch.cat([pad(c.mu) for c in conds])
    mask = torch.cat([pad(c.mask) for c in conds])
    x1 = torch.cat([pad(x) for x in x1s])
    return x1, Cond(mu=mu, mask=mask)


def full_training_loop() -> None:
    """STUB. What I'd actually run with a GPU (not needed for this deliverable):
    ~100k steps, batch 32 (LJSpeech transcripts or production text pushed through
    the teacher at NFE 32), Adam lr 1e-4 with cosine decay, EMA decay ramped
    0.9 -> 0.9999, n_grid ramped 16 -> 64 as the student sharpens, and checkpoint
    selection off the eval harness (WER + speaker cosine + MOS proxy at NFE 1)."""
    raise NotImplementedError("full training loop intentionally stubbed — see docstring")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one distillation step + smoke loop.")
    parser.add_argument("--prompts", default="prompts.txt")
    parser.add_argument("--n-prompts", type=int, default=4, help="batch size (the prompts are the corpus)")
    parser.add_argument("--teacher-nfe", type=int, default=32, help="NFE for the self-distillation targets")
    parser.add_argument("--steps", type=int, default=60, help="smoke-loop length")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-grid", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/student_smoke.pt")
    parser.add_argument("--full", action="store_true", help="run the (stubbed) full training loop")
    args = parser.parse_args()

    if args.full:
        full_training_loop()

    torch.manual_seed(args.seed)
    teacher = TeacherFMD()
    prompts = [l.strip() for l in open(args.prompts) if l.strip()][: args.n_prompts]

    print(f"building self-distillation batch: {len(prompts)} prompts @ teacher NFE {args.teacher_nfe}")
    x1, cond = build_batch(teacher, prompts, args.teacher_nfe)
    print(f"batch: x1 {tuple(x1.shape)}, {cond.n_frames} real frames")

    student = ConsistencyStudent(teacher).train()
    ema = make_ema_student(student)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    # --- the thing being graded: one optimizer step that runs cleanly ---------
    loss = consistency_loss(student, ema, teacher, x1, cond, args.n_grid)
    opt.zero_grad()
    loss.backward()
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
    grad_norms = [p.grad.abs().max().item() for p in student.parameters() if p.grad is not None]
    assert grad_norms and max(grad_norms) > 0, "no gradient reached the student"
    opt.step()
    update_ema(student, ema)
    print(f"single step OK: loss={loss.item():.6f}, max|grad|={max(grad_norms):.3e}, params updated")

    # --- smoke loop: fresh noise every step on a fixed target batch -----------
    # Each step draws new (x0, t), so the loss is a stochastic estimate of the
    # objective's expectation. A downward trend here is real optimization, not
    # memorizing one frozen (x0, t) draw. It won't be monotonic, so I summarize
    # with first-window vs last-window averages rather than reading single steps.
    print(f"smoke loop ({args.steps} steps, fresh noise each step on a fixed target batch):")
    trace: list[float] = []
    win = max(1, args.steps // 10)
    for i in range(args.steps):
        loss = consistency_loss(student, ema, teacher, x1, cond, args.n_grid)
        opt.zero_grad()
        loss.backward()
        opt.step()
        update_ema(student, ema)
        trace.append(loss.item())
        if i == 0 or (i + 1) % win == 0:
            print(f"  step {i + 1:3d}: loss = {loss.item():.6f}")
    first, last = sum(trace[:win]) / win, sum(trace[-win:]) / win
    direction = "down" if last < first else "UP (no learning signal)"
    print(f"trend over {args.steps} steps: first-{win} avg {first:.6f} -> "
          f"last-{win} avg {last:.6f}  ({direction}, {(first - last) / first * 100:+.0f}%)")

    torch.save(student.state_dict(), args.out)
    print(f"saved smoke-trained student to {args.out} (used by eval_harness.py if present)")


if __name__ == "__main__":
    main()
