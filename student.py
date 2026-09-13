"""The consistency student: same network as the teacher, but it predicts the mel
endpoint in one shot instead of stepping the ODE.

The axis I'm optimizing is NFE, not size, so the student reuses the teacher's
architecture and starts from its weights. One forward pass replaces the teacher's
N Euler steps, and params/memory stay the same by design.
"""

import copy

import torch
from torch import nn

from teacher import Cond, TeacherFMD


class ConsistencyStudent(nn.Module):
    """f(x_t, t, cond) -> x1_hat, i.e. the predicted endpoint of the OT path.

    I parameterize it with the consistency-model boundary condition, in the
    flow-matching form:

        f(x, t) = c_skip(t) * x + c_out(t) * F_theta(x, t),
        c_skip(t) = 1,  c_out(t) = 1 - t

    That gives c_skip(1)=1, c_out(1)=0, so f(x1, 1) = x1 exactly — the boundary
    holds by construction, no extra penalty term. Both coefficients are simple
    in t. I picked this form over the Karras sigma schedules for one reason
    specific to my setup: F_theta starts as a copy of the teacher's velocity
    net, and x + (1-t) * v(x, t) is exactly one Euler jump from t to 1. So at
    init the student already equals "teacher, extrapolated one step to the end,"
    and training only has to fix the curvature of the path.
    """

    def __init__(self, teacher: TeacherFMD) -> None:
        super().__init__()
        # Own trainable copy of the teacher's estimator. The teacher's own copy
        # stays frozen.
        self.net = copy.deepcopy(teacher.decoder.estimator)
        self.sigma_min = teacher.sigma_min

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: Cond) -> torch.Tensor:
        """x_t: (B, 80, T), t: (B,) in [0, 1] -> x1_hat: (B, 80, T)."""
        c_out = (1.0 - t).view(-1, 1, 1)
        return x_t + c_out * self.net(x_t, cond.mask, cond.mu, t, None)

    # ---------------------------------------------------------------- sampling

    @torch.no_grad()
    def sample_onestep(self, x0: torch.Tensor, cond: Cond) -> torch.Tensor:
        """NFE=1: jump from noise (t=0) straight to the predicted endpoint."""
        t = torch.zeros(x0.shape[0], device=x0.device)
        return self.forward(x0, t, cond)

    @torch.no_grad()
    def sample_multistep(self, x0: torch.Tensor, cond: Cond, steps: int) -> torch.Tensor:
        """Standard multi-step consistency sampling: predict the endpoint, add
        fresh noise back onto the OT path at a later t, predict again. Each extra
        step spends one NFE to recover detail — the re-noise/re-predict cycle
        lets the model fix its own one-step over-smoothing. steps == NFE."""
        b = x0.shape[0]
        t_grid = torch.linspace(0, 1, steps + 1, device=x0.device)[:-1]
        x1_hat = self.forward(x0, t_grid[0].expand(b), cond)
        for t_next in t_grid[1:]:
            z = torch.randn_like(x1_hat)
            # Same interpolation as teacher.trajectory_point; target here is the
            # current endpoint estimate.
            x_t = (1 - (1 - self.sigma_min) * t_next) * z + t_next * x1_hat
            x1_hat = self.forward(x_t, t_next.expand(b), cond)
        return x1_hat
