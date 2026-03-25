"""Curriculum-Aware Dual-rate Normalizer (CADN)

Per-branch dual-rate EMA normalizer with drift detection.
Replaces SKRL's RunningStandardScaler for curriculum RL where
observation distributions shift across stages.

Obs layout (139D):
  [0:4]   ego    — accel, vel, omega, radius
  [4:6]   goal   — x_g, y_g in body frame
  [6:78]  lidar  — 72-bin VLP-16 2D projection
  [78:138] obs   — top-k obstacles × 6D
  [138:139] time — remaining ratio
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CurriculumAwareDualRateNormalizer(nn.Module):
    """Per-branch dual-rate EMA normalizer with drift detection."""

    def __init__(
        self,
        size: int,
        alpha_fast: float = 0.01,
        alpha_slow: float = 0.0005,
        tau_lo: float = 0.2,
        tau_hi: float = 0.5,
        clip_threshold: float = 5.0,
        epsilon: float = 1e-8,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.size = size
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.tau_lo = tau_lo
        self.tau_hi = tau_hi
        self.clip_threshold = clip_threshold
        self.epsilon = epsilon

        # Fast channel
        self.register_buffer("mu_f", torch.zeros(size, dtype=torch.float64, device=device))
        self.register_buffer("var_f", torch.ones(size, dtype=torch.float64, device=device))
        # Slow channel
        self.register_buffer("mu_s", torch.zeros(size, dtype=torch.float64, device=device))
        self.register_buffer("var_s", torch.ones(size, dtype=torch.float64, device=device))

        self.register_buffer("_initialized", torch.tensor(False, device=device))

    def forward(
        self, x: torch.Tensor, train: bool = False, inverse: bool = False
    ) -> torch.Tensor:
        if train:
            self._update(x)

        mu_act, var_act = self._active_stats()

        if inverse:
            return x * torch.sqrt(var_act + self.epsilon).float() + mu_act.float()

        return torch.clamp(
            (x - mu_act.float()) / torch.sqrt(var_act.float() + self.epsilon),
            -self.clip_threshold,
            self.clip_threshold,
        )

    @torch.no_grad()
    def _update(self, x: torch.Tensor):
        batch_mean = x.double().mean(dim=0)
        batch_var = x.double().var(dim=0, correction=0)

        if not self._initialized:
            self.mu_f.copy_(batch_mean)
            self.var_f.copy_(batch_var + self.epsilon)
            self.mu_s.copy_(batch_mean)
            self.var_s.copy_(batch_var + self.epsilon)
            self._initialized.fill_(True)
            return

        # Fast channel
        self.mu_f.lerp_(batch_mean, self.alpha_fast)
        self.var_f.lerp_((batch_mean - self.mu_f).pow(2), self.alpha_fast)
        # Slow channel
        self.mu_s.lerp_(batch_mean, self.alpha_slow)
        self.var_s.lerp_((batch_mean - self.mu_s).pow(2), self.alpha_slow)

    def _active_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.drift_score()
        w = max(0.0, min(1.0, (delta - self.tau_lo) / (self.tau_hi - self.tau_lo + 1e-12)))
        mu = w * self.mu_f + (1 - w) * self.mu_s
        var = w * self.var_f + (1 - w) * self.var_s
        return mu, var

    def drift_score(self) -> float:
        """Compute drift intensity between fast and slow channels."""
        diff = (self.mu_f - self.mu_s).norm()
        denom = math.sqrt(self.size) * (self.var_s.sqrt().norm() + self.epsilon)
        return (diff / denom).item()

    def blend_weight(self) -> float:
        """Current fast/slow blend weight (0=slow, 1=fast)."""
        delta = self.drift_score()
        return max(0.0, min(1.0, (delta - self.tau_lo) / (self.tau_hi - self.tau_lo + 1e-12)))


class PerBranchCADN(nn.Module):
    """Wraps 3 CADN instances matching the VLP16 139D obs layout.

    Compatible with SKRL's state_preprocessor interface:
      __call__(x, train=False) → normalized tensor
    """

    def __init__(self, device: torch.device | None = None):
        super().__init__()
        # State: ego(4) + goal(2) + time(1) = 7D — 中等漂移
        self.cadn_s = CurriculumAwareDualRateNormalizer(
            size=7, alpha_fast=0.008, alpha_slow=0.0003,
            tau_lo=0.25, tau_hi=0.55, device=device,
        )
        # LiDAR: 72D — 低漂移
        self.cadn_l = CurriculumAwareDualRateNormalizer(
            size=72, alpha_fast=0.005, alpha_slow=0.0002,
            tau_lo=0.3, tau_hi=0.6, device=device,
        )
        # Obstacles: 60D — 高漂移（課程切換時有效維度跳變）
        self.cadn_o = CurriculumAwareDualRateNormalizer(
            size=60, alpha_fast=0.01, alpha_slow=0.0005,
            tau_lo=0.2, tau_hi=0.5, epsilon=1e-6, device=device,
        )

    def forward(
        self, x: torch.Tensor, train: bool = False, inverse: bool = False
    ) -> torch.Tensor:
        # Split by obs layout: ego(4) + goal(2) + lidar(72) + obs(60) + time(1)
        ego = x[:, 0:4]
        goal = x[:, 4:6]
        lidar = x[:, 6:78]
        obs = x[:, 78:138]
        time_r = x[:, 138:139]

        # Combine state dims for single normalizer
        state_cat = torch.cat([ego, goal, time_r], dim=-1)  # 7D

        state_n = self.cadn_s(state_cat, train=train, inverse=inverse)
        lidar_n = self.cadn_l(lidar, train=train, inverse=inverse)
        obs_n = self.cadn_o(obs, train=train, inverse=inverse)

        # Reconstruct original layout
        ego_n = state_n[:, 0:4]
        goal_n = state_n[:, 4:6]
        time_n = state_n[:, 6:7]
        return torch.cat([ego_n, goal_n, lidar_n, obs_n, time_n], dim=-1)

    def get_diagnostics(self) -> dict[str, float]:
        """Return drift diagnostics for WandB logging."""
        return {
            "cadn/drift_state": self.cadn_s.drift_score(),
            "cadn/drift_lidar": self.cadn_l.drift_score(),
            "cadn/drift_obstacle": self.cadn_o.drift_score(),
            "cadn/blend_weight_state": self.cadn_s.blend_weight(),
            "cadn/blend_weight_lidar": self.cadn_l.blend_weight(),
            "cadn/blend_weight_obstacle": self.cadn_o.blend_weight(),
            "cadn/obs_mu_fast_norm": self.cadn_o.mu_f.norm().item(),
            "cadn/obs_var_fast_mean": self.cadn_o.var_f.mean().item(),
        }
