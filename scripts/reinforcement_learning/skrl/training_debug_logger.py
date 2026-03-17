"""Training Metrics Logger for Isaac Lab RL.

Collects per-step metrics that SKRL does NOT provide natively:
  action/  — raw vs applied actions, saturation, safety override, scatter plot
  robot/   — velocity, progress-to-goal, obstacle distance, LiDAR

Metrics that SKRL already logs (we do NOT duplicate):
  Reward / *              — SKRL tracks instantaneous + total reward
  Info / Episode_Reward/* — SKRL tracks per-term episode reward averages
  Info / Episode_Termination/* — SKRL tracks termination reasons
  Episode / *             — SKRL tracks episode lengths
  Loss / *, Policy / *    — SKRL tracks PPO internals
  nav/*                   — WandBSequentialTrainer tracks running success/collision/timeout

Design:
  - `step()` is called EVERY env step — must be FAST (batched GPU→CPU)
  - `get_and_reset()` returns aggregated stats and clears buffers
  - Zero modification to Isaac Lab core code; reads env internals via public APIs
  - All GPU reductions batched via torch.stack().cpu() to minimize CUDA syncs

Usage in WandBSequentialTrainer:
    debug_logger = TrainingDebugLogger(env)
    ...
    debug_logger.step(actions, rewards, terminated, truncated, infos)
    ...
    metrics = debug_logger.get_and_reset()
    wandb.log(metrics, step=timestep)
"""

from __future__ import annotations

from typing import Optional

import torch
import numpy as np


class TrainingDebugLogger:
    """Collects per-step training diagnostics and returns aggregated metrics."""

    def __init__(
        self,
        env,
        log_interval: int = 128,
    ):
        """
        Args:
            env: The SKRL-wrapped environment (AACIsaacLabWrapper).
                 We unwrap to get the Isaac Lab ManagerBasedRLEnv.
            log_interval: Expected number of steps between get_and_reset() calls
                          (used only for buffer pre-allocation hint, not enforced).
        """
        self._wrapped_env = env
        self._isaac_env = self._unwrap_to_isaac(env)
        self._log_interval = log_interval
        self._step_count = 0

        # ── One-time diagnostics flags ──
        self._diag_printed = False
        self._action_error_printed = False
        self._motion_error_printed = False
        self._lidar_error_printed = False

        # ── Action pipeline accumulators ──
        self._raw_action_mean: list[float] = []
        self._raw_action_std: list[float] = []
        self._applied_action_mean: list[float] = []
        self._applied_action_std: list[float] = []
        self._action_saturation_rate: list[float] = []
        self._action_safety_override_rate: list[float] = []

        # Acceleration scatter data (GPU tensor buffer — no per-step CPU transfer)
        self._scatter_buf: Optional[torch.Tensor] = None  # [max_samples, 2], lazily initialized on GPU
        self._scatter_count: int = 0
        self._accel_scatter_max_samples: int = 5000

        # ── Robot state accumulators ──
        self._lin_vel_mean: list[float] = []
        self._lin_vel_max: list[float] = []
        self._ang_vel_mean: list[float] = []
        self._ang_vel_max: list[float] = []
        self._progress_per_step: list[float] = []  # delta distance to goal
        self._min_obstacle_dist: list[float] = []
        # LiDAR diagnostics
        self._lidar_min_raw: list[float] = []
        self._lidar_max_range_hit_ratio: list[float] = []

        # ── H1/H2/H3 Debug: reward & termination diagnostics ──
        self._terminated_count: int = 0
        self._truncated_count: int = 0
        self._total_step_count: int = 0
        self._progress_reward_at_reset: list[float] = []  # H1 check: should be ~0
        self._reward_summary_printed: bool = False

        # Print unwrap result
        if self._isaac_env is not None:
            print(f"[DebugLogger] Isaac Lab env unwrap successful: {type(self._isaac_env).__name__}")
        else:
            print(f"[DebugLogger] WARNING: Isaac Lab env unwrap FAILED!")
            print(f"[DebugLogger]   Wrapper chain: {self._get_wrapper_chain(env)}")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def step(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict,
    ) -> None:
        """Called every env step. All tensors are [num_envs, ...].

        This must be FAST — only batched scalar reductions on GPU tensors.
        """
        self._step_count += 1

        # ── One-time diagnostic print ──
        if not self._diag_printed:
            self._print_diagnostic(actions, rewards, terminated, truncated, infos)
            self._diag_printed = True

        with torch.no_grad():
            dones = (terminated.view(-1) | truncated.view(-1))

            # ── H1/H2 Debug: termination stats ──
            self._terminated_count += terminated.view(-1).sum().item()
            self._truncated_count += truncated.view(-1).sum().item()
            self._total_step_count += terminated.numel()

            # ── H1 Debug: check progress reward at episode reset ──
            self._check_progress_reward_at_reset()

            # ── H3 Debug: one-time reward term summary ──
            if not self._reward_summary_printed:
                self._print_reward_summary()
                self._reward_summary_printed = True

            # ── Action pipeline ──
            self._collect_action_stats(actions)

            # ── Robot state ──
            self._collect_motion_stats(dones)

    def on_ppo_update(self, metrics: dict[str, float]) -> None:
        """Called after each PPO _update. No-op: SKRL logs Loss/Policy natively."""
        pass

    def get_accel_scatter_table(self):
        """Build a wandb.Table from GPU scatter buffer (single CPU transfer at flush).

        Returns:
            wandb.Table with columns ["linear_acceleration", "angular_acceleration"],
            or None if no samples collected or wandb not available.
        """
        if self._scatter_count == 0 or self._scatter_buf is None:
            return None
        try:
            import wandb
        except ImportError:
            return None

        # Single GPU→CPU transfer of all scatter samples at flush time
        data_cpu = self._scatter_buf[:self._scatter_count].cpu().tolist()
        table = wandb.Table(
            columns=["linear_acceleration", "angular_acceleration"],
            data=data_cpu,
        )
        return table

    def get_and_reset(self) -> dict[str, float]:
        """Return aggregated metrics dict (flat keys for WandB) and clear buffers.

        Only outputs metrics that SKRL does NOT provide:
          action/  — policy output and applied action stats
          robot/   — velocity, progress, obstacle distance, LiDAR
        """
        m: dict[str, float] = {}

        # ── Action pipeline ──
        if self._raw_action_mean:
            m["action/raw_mean"] = np.mean(self._raw_action_mean)
            m["action/raw_std"] = np.mean(self._raw_action_std)
        if self._applied_action_mean:
            m["action/applied_mean"] = np.mean(self._applied_action_mean)
            m["action/applied_std"] = np.mean(self._applied_action_std)
        if self._action_saturation_rate:
            m["action/saturation_rate"] = np.mean(self._action_saturation_rate)
        if self._action_safety_override_rate:
            m["action/safety_override_rate"] = np.mean(self._action_safety_override_rate)

        # ── Robot state ──
        if self._lin_vel_mean:
            m["robot/speed_mean"] = np.mean(self._lin_vel_mean)
            m["robot/speed_max"] = np.mean(self._lin_vel_max)
        if self._ang_vel_mean:
            m["robot/angular_vel_mean"] = np.mean(self._ang_vel_mean)
            m["robot/angular_vel_max"] = np.mean(self._ang_vel_max)
        if self._progress_per_step:
            m["robot/progress_per_step"] = np.mean(self._progress_per_step)
        if self._min_obstacle_dist:
            m["robot/min_obstacle_dist_mean"] = np.mean(self._min_obstacle_dist)
            m["robot/min_obstacle_dist_min"] = np.min(self._min_obstacle_dist)
        if self._lidar_min_raw:
            m["robot/lidar_min_raw"] = np.mean(self._lidar_min_raw)
        if self._lidar_max_range_hit_ratio:
            m["robot/lidar_no_hit_ratio"] = np.mean(self._lidar_max_range_hit_ratio)

        # ── H1/H2 Debug: termination & progress reset stats ──
        if self._total_step_count > 0:
            m["debug/terminated_rate"] = self._terminated_count / self._total_step_count
            m["debug/truncated_rate"] = self._truncated_count / self._total_step_count
        if self._progress_reward_at_reset:
            m["debug/progress_reward_at_reset_mean"] = np.mean(self._progress_reward_at_reset)
            m["debug/progress_reward_at_reset_max"] = np.max(np.abs(self._progress_reward_at_reset))

        # Reset all buffers
        self._reset()
        return m

    # ──────────────────────────────────────────────────────────────────────
    # One-time diagnostic print
    # ──────────────────────────────────────────────────────────────────────

    def _print_diagnostic(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict,
    ) -> None:
        """Print one-time diagnostic to verify tensor shapes and env structure."""
        print("\n" + "=" * 70)
        print("[DebugLogger] === ONE-TIME DIAGNOSTIC (step 1) ===")
        print("=" * 70)
        print(f"  actions:    shape={actions.shape}, dtype={actions.dtype}, device={actions.device}")
        print(f"  rewards:    shape={rewards.shape}, dtype={rewards.dtype}")
        print(f"  terminated: shape={terminated.shape}, sum={terminated.sum().item()}")
        print(f"  truncated:  shape={truncated.shape}, sum={truncated.sum().item()}")
        print(f"  infos keys: {list(infos.keys())[:20]}")
        if "log" in infos:
            log_keys = list(infos["log"].keys())
            print(f"  infos['log'] keys ({len(log_keys)}): {log_keys[:15]}")

        isaac_env = self._isaac_env
        if isaac_env is not None:
            print(f"\n  Isaac env type: {type(isaac_env).__name__}")
            if hasattr(isaac_env, 'num_envs'):
                print(f"  num_envs: {isaac_env.num_envs}")

            try:
                robot = isaac_env.scene["robot"]
                speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
                print(f"  speed: min={speed.min().item():.4f}, max={speed.max().item():.4f}")
            except Exception as e:
                print(f"  robot access FAILED: {e}")

            try:
                action_term = list(isaac_env.action_manager._terms.values())[0]
                print(f"  action_term type: {type(action_term).__name__}")
            except Exception as e:
                print(f"  action_manager access FAILED: {e}")

            try:
                sensors = isaac_env.scene.sensors if hasattr(isaac_env.scene, 'sensors') else {}
                sensor = sensors.get("lidar", None)
                if sensor is not None:
                    print(f"  lidar.ray_hits_w: shape={sensor.data.ray_hits_w.shape}")
                    if hasattr(sensor.cfg, 'max_distance'):
                        print(f"  lidar max_distance: {sensor.cfg.max_distance}")
            except Exception as e:
                print(f"  lidar access FAILED: {e}")

        print("=" * 70 + "\n")

    # ──────────────────────────────────────────────────────────────────────
    # Internal collection helpers
    # ──────────────────────────────────────────────────────────────────────

    def _collect_action_stats(self, actions: torch.Tensor) -> None:
        """Collect raw (policy output) and applied (post-processing) action stats.

        Uses batched GPU→CPU transfers to minimize CUDA synchronization points.
        """
        with torch.no_grad():
            # Raw actions — batch 4 reductions into 1 CPU transfer
            a = actions.float().view(-1)
            raw_batch = torch.stack([
                a.mean(), a.std() if a.numel() > 1 else a.new_zeros(()), a.min(), a.max()
            ]).cpu()
            self._raw_action_mean.append(raw_batch[0].item())
            self._raw_action_std.append(raw_batch[1].item())

        isaac_env = self._isaac_env
        if isaac_env is None:
            return

        try:
            action_term = list(isaac_env.action_manager._terms.values())[0]

            # Applied actions + saturation — batch into 1 CPU transfer
            pa = action_term.processed_actions.float().view(-1)
            gpu_scalars = [pa.mean(), pa.std() if pa.numel() > 1 else pa.new_zeros(())]

            has_saturation = hasattr(action_term, '_current_velocity') and hasattr(action_term, 'cfg')
            if has_saturation:
                vel = action_term._current_velocity
                max_v = action_term.cfg.max_linear_velocity
                if vel.numel() > 0:
                    gpu_scalars.append(((vel >= max_v - 1e-4) | (vel <= 1e-4)).float().mean())

            has_override = hasattr(action_term, '_applied_accelerations') and hasattr(action_term, '_raw_actions')
            if has_override:
                raw = action_term._raw_actions
                applied_a = action_term._applied_accelerations[:, 0]
                if raw.numel() > 0:
                    idx_a = raw[:, 0].round().long().clamp(0, action_term.cfg.num_accel_bins - 1)
                    a_raw = action_term.accel_table[idx_a]
                    gpu_scalars.append(((a_raw.abs() > 1e-6) & (applied_a.abs() < 1e-6)).float().mean())

            # Single GPU→CPU transfer
            applied_batch = torch.stack(gpu_scalars).cpu()
            self._applied_action_mean.append(applied_batch[0].item())
            self._applied_action_std.append(applied_batch[1].item())
            idx = 2
            if has_saturation and vel.numel() > 0:
                self._action_saturation_rate.append(applied_batch[idx].item())
                idx += 1
            if has_override and raw.numel() > 0:
                self._action_safety_override_rate.append(applied_batch[idx].item())

            # Scatter data — GPU tensor buffer (no per-step CPU transfer)
            if hasattr(action_term, '_applied_accelerations'):
                aa = action_term._applied_accelerations  # [N, 2]
                remaining = self._accel_scatter_max_samples - self._scatter_count
                if aa.numel() > 0 and remaining > 0:
                    if self._scatter_buf is None:
                        self._scatter_buf = torch.zeros(
                            self._accel_scatter_max_samples, 2, device=aa.device
                        )
                    n_sample = min(aa.shape[0], 32, remaining)
                    indices = torch.randperm(aa.shape[0], device=aa.device)[:n_sample]
                    end = self._scatter_count + n_sample
                    self._scatter_buf[self._scatter_count:end] = aa[indices]
                    self._scatter_count = end

        except Exception as e:
            if not self._action_error_printed:
                print(f"[DebugLogger] _collect_action_stats FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self._action_error_printed = True

    def _collect_motion_stats(self, dones: torch.Tensor) -> None:
        """Collect robot velocity, progress-to-goal, obstacle distance."""
        isaac_env = self._isaac_env
        if isaac_env is None:
            return

        try:
            # Robot velocity — batch 4 reductions into 1 CPU transfer
            robot = isaac_env.scene["robot"]
            lin_vel = torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0)
            ang_vel = torch.nan_to_num(robot.data.root_ang_vel_w[:, 2], nan=0.0)
            speed = torch.norm(lin_vel, dim=1)
            ang_abs = ang_vel.abs()

            vel_batch = torch.stack([
                speed.mean(), speed.max(), ang_abs.mean(), ang_abs.max()
            ]).cpu()
            self._lin_vel_mean.append(vel_batch[0].item())
            self._lin_vel_max.append(vel_batch[1].item())
            self._ang_vel_mean.append(vel_batch[2].item())
            self._ang_vel_max.append(vel_batch[3].item())

            # Progress per step (distance reduction to goal)
            robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)
            goal_pos = None
            if hasattr(isaac_env, '_local_goal_world') and isaac_env._local_goal_world is not None:
                goal_pos = isaac_env._local_goal_world
            else:
                try:
                    goal_pos = isaac_env.command_manager.get_command("goal_command")[:, :2]
                except (KeyError, AttributeError):
                    pass

            if goal_pos is not None:
                d_curr = torch.norm(goal_pos - robot_pos, dim=1)

                if hasattr(self, '_prev_goal_dist_debug'):
                    valid_mask = ~dones
                    progress = self._prev_goal_dist_debug - d_curr
                    valid_mask = valid_mask & (progress.abs() < 2.0)
                    if valid_mask.any():
                        self._progress_per_step.append(progress[valid_mask].mean().item())

                self._prev_goal_dist_debug = d_curr.clone()

        except Exception as e:
            if not self._motion_error_printed:
                print(f"[DebugLogger] _collect_motion_stats FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self._motion_error_printed = True
            return

        # LiDAR obstacle distance (separate try block)
        try:
            self._collect_lidar_stats(isaac_env)
        except Exception as e:
            if not self._lidar_error_printed:
                print(f"[DebugLogger] _collect_lidar_stats FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self._lidar_error_printed = True

    def _collect_lidar_stats(self, isaac_env) -> None:
        """Collect LiDAR-based obstacle distance metrics."""
        sensors = isaac_env.scene.sensors if hasattr(isaac_env.scene, 'sensors') else {}
        sensor = sensors.get("lidar", None)
        if sensor is None:
            return

        hits = sensor.data.ray_hits_w  # [N, num_rays, 3]
        pos = sensor.data.pos_w        # [N, 3]

        dists = torch.norm(hits[:, :, :2] - pos[:, :2].unsqueeze(1), dim=-1)  # [N, num_rays]
        max_range = sensor.cfg.max_distance if hasattr(sensor.cfg, 'max_distance') else 20.0
        invalid = ~torch.isfinite(dists) | (dists > max_range - 0.01)
        dists_clamped = torch.where(invalid, torch.tensor(max_range, device=dists.device), dists)

        # Batch 3 lidar reductions into 1 CPU transfer
        d_min_per_env = dists_clamped.min(dim=1)[0]
        dists_finite = torch.where(torch.isfinite(dists), dists, torch.tensor(max_range, device=dists.device))
        lidar_batch = torch.stack([
            d_min_per_env.mean(), dists_finite.min(), invalid.float().mean()
        ]).cpu()
        self._min_obstacle_dist.append(lidar_batch[0].item())
        self._lidar_min_raw.append(lidar_batch[1].item())
        self._lidar_max_range_hit_ratio.append(lidar_batch[2].item())

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap_to_isaac(env):
        """Walk the wrapper chain to find the Isaac Lab ManagerBasedRLEnv."""
        current = env
        for _ in range(20):
            if hasattr(current, 'reward_manager'):
                return current
            if hasattr(current, '_unwrapped'):
                unwrapped = current._unwrapped
                if hasattr(unwrapped, 'reward_manager'):
                    return unwrapped
            if hasattr(current, 'unwrapped'):
                unwrapped = current.unwrapped
                if hasattr(unwrapped, 'reward_manager'):
                    return unwrapped
            if hasattr(current, '_env'):
                current = current._env
            elif hasattr(current, 'env'):
                current = current.env
            else:
                break
        return None

    @staticmethod
    def _get_wrapper_chain(env) -> str:
        """Return a string showing the wrapper chain for debugging."""
        chain = []
        current = env
        for _ in range(20):
            chain.append(type(current).__name__)
            if hasattr(current, '_env'):
                current = current._env
            elif hasattr(current, 'env'):
                current = current.env
            else:
                break
        return " -> ".join(chain)

    def _check_progress_reward_at_reset(self) -> None:
        """H1 验证: 检查 episode reset 后第一步的 progress reward 是否异常。

        如果 H1 修复生效，reset 后第一步的 progress reward 应接近 0。
        如果看到大数值 (>0.5)，表示仍有跨 episode 状态泄漏。
        """
        isaac_env = self._isaac_env
        if isaac_env is None:
            return
        try:
            just_reset = isaac_env.episode_length_buf == 0
            if not just_reset.any():
                return

            # 检查 progress_to_goal 的缓存状态
            if hasattr(isaac_env, '_previous_goal_distance'):
                robot = isaac_env.scene["robot"]
                robot_pos = robot.data.root_pos_w[:, :2]
                if hasattr(isaac_env, '_local_goal_world') and isaac_env._local_goal_world is not None:
                    goal_pos = isaac_env._local_goal_world
                else:
                    goal_pos = isaac_env.command_manager.get_command("goal_command")[:, :2]
                d_curr = torch.norm(goal_pos - robot_pos, dim=1)
                d_prev = isaac_env._previous_goal_distance
                # 只检查刚 reset 的 env
                delta = (d_prev[just_reset] - d_curr[just_reset]).abs()
                if delta.numel() > 0:
                    self._progress_reward_at_reset.append(delta.mean().item())

            # 同样检查 potential_progress_reward 的缓存
            if hasattr(isaac_env, '_prev_goal_dist'):
                robot = isaac_env.scene["robot"]
                robot_pos = robot.data.root_pos_w[:, :2]
                if hasattr(isaac_env, '_local_goal_world') and isaac_env._local_goal_world is not None:
                    goal_pos = isaac_env._local_goal_world
                else:
                    goal_pos = isaac_env.command_manager.get_command("goal_command")[:, :2]
                d_curr = torch.norm(goal_pos - robot_pos, dim=1)
                d_prev = isaac_env._prev_goal_dist
                delta = (d_prev[just_reset] - d_curr[just_reset]).abs()
                if delta.numel() > 0:
                    self._progress_reward_at_reset.append(delta.mean().item())
        except Exception:
            pass  # Non-critical debug metric

    def _print_reward_summary(self) -> None:
        """H3 验证: 打印当前启用的 reward terms 和权重。"""
        isaac_env = self._isaac_env
        if isaac_env is None:
            return
        try:
            rm = isaac_env.reward_manager
            print("\n" + "=" * 70)
            print("[DebugLogger] === ACTIVE REWARD TERMS (H3 Check) ===")
            print("=" * 70)
            for term_name, term_cfg in zip(rm._term_names, rm._term_cfgs):
                weight = getattr(term_cfg, 'weight', '?')
                func = getattr(term_cfg, 'func', None)
                func_name = func.__name__ if callable(func) else str(func)
                is_goal = any(kw in term_name.lower() or kw in func_name.lower()
                              for kw in ['goal', 'progress', 'reaching', 'heading', 'velocity_toward', 'potential'])
                marker = " ← GOAL-RELATED" if is_goal else ""
                print(f"  {term_name:30s} w={weight:>8} func={func_name}{marker}")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"[DebugLogger] Reward summary failed: {e}")

    def _reset(self) -> None:
        """Clear all per-window accumulators."""
        self._step_count = 0

        self._raw_action_mean.clear()
        self._raw_action_std.clear()
        self._applied_action_mean.clear()
        self._applied_action_std.clear()
        self._action_saturation_rate.clear()
        self._action_safety_override_rate.clear()

        self._scatter_count = 0
        # Keep _scatter_buf allocated (reuse GPU memory)

        self._lin_vel_mean.clear()
        self._lin_vel_max.clear()
        self._ang_vel_mean.clear()
        self._ang_vel_max.clear()
        self._progress_per_step.clear()
        self._min_obstacle_dist.clear()
        self._lidar_min_raw.clear()
        self._lidar_max_range_hit_ratio.clear()

        # H1/H2 debug counters
        self._terminated_count = 0
        self._truncated_count = 0
        self._total_step_count = 0
        self._progress_reward_at_reset.clear()
