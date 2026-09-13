"""Consistency-distillation loss for the flow-matching decoder.

The idea was two neighbouring points on the same teacher trajectory (one teacher Euler step apart) should map to the same endpoint. 
As the grid gets finer, the thing that satisfies this is the teacher's own ODE solution but reachable in a single student pass.
"""

import copy

import torch
import torch.nn.functional as F

from student import ConsistencyStudent
from teacher import Cond, TeacherFMD


def make_ema_student(student: ConsistencyStudent) -> ConsistencyStudent:
    """Frozen copy of the student used as the distillation target."""
    ema = copy.deepcopy(student)
    ema.requires_grad_(False)
    ema.eval()
    return ema


@torch.no_grad()
def update_ema(student: ConsistencyStudent, ema: ConsistencyStudent, decay: float = 0.99) -> None:
    """EMA update. Simplification vs the full method: constant decay, no schedule
    ramp (added in the README honest note)."""
    for p_ema, p in zip(ema.parameters(), student.parameters()):
        p_ema.lerp_(p, 1.0 - decay)


def consistency_loss(
    student: ConsistencyStudent,
    ema_student: ConsistencyStudent,
    teacher: TeacherFMD,
    x1: torch.Tensor,
    cond: Cond,
    n_grid: int = 16,
) -> torch.Tensor:
    """One consistency-distillation term on a batch of teacher outputs x1.

    Steps:
      1. Draw noise x0 and a per-sample grid index n, giving neighbouring times
         t_n = n/n_grid and t_{n+1} = (n+1)/n_grid.
      2. Put x_tn on the OT path between x0 and x1 (a lerp — no ODE solve; valid
         because Matcha trained on exactly this path).
      3. Take one Euler step of the TEACHER field to reach x_tn1. This is the
         only place the teacher's knowledge of the real trajectory comes in.
      4. Pull the trainable student at the noisier point (x_tn, t_n) toward the
         frozen EMA target at the nearer-data point (x_tn1, t_{n+1}). Huber
         instead of MSE so mel transients (plosive bursts) don't dominate.

    Simplifications vs the full method (also in the README): constant EMA decay,
    uniform grid sampling (no weighting toward hard timesteps), fixed n_grid.
    """
    b = x1.shape[0]
    device = x1.device

    x0 = torch.randn_like(x1)
    n = torch.randint(0, n_grid, (b,), device=device)
    t_n = n.float() / n_grid
    t_n1 = (n.float() + 1) / n_grid

    x_tn = teacher.trajectory_point(x0, x1, t_n)

    with torch.no_grad():  # teacher is a fixed target generator, never trained
        v = teacher.velocity_field(x_tn, t_n, cond)
    x_tn1 = x_tn + (t_n1 - t_n).view(-1, 1, 1) * v

    # Direction matters here, and it's easy to get backwards. x_tn1 is one Euler
    # step closer to the data (t=1), where the boundary f(x,1)=x pins the target
    # to the true endpoint. So the frozen EMA reads that nearer-data point, and
    # the trainable student learns to match it from the noisier point x_tn —
    # trust flows backward from the boundary. Flip it and the trainable side sits
    # at t=1 where c_out(1)=0 kills the gradient, so nothing ever ties the loss
    # to real data. Hence: trainable = noisier point, stopgrad target = nearer data.
    pred = student(x_tn, t_n, cond)
    with torch.no_grad():
        target = ema_student(x_tn1, t_n1, cond)

    # Mask out padding, normalize by real frame count so the loss is comparable
    # across prompt lengths.
    per_elem = F.huber_loss(pred, target, reduction="none") * cond.mask
    return per_elem.sum() / (cond.mask.sum() * x1.shape[1])
