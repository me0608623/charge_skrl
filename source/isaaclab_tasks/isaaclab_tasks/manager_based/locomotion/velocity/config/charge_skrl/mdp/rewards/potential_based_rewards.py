"""Potential-Based Reward Shaping for Phase 0

Implements principled reward functions based on:
  1. Potential-Based Shaping: R = Phi(s') - Phi(s), guarantees policy invariance
  2. Decoupled collision penalty: warning zone [min_dist, warn_dist] separate from termination
  3. Terminal reward dominance: reaching_goal=+100, collision=-100

Sign convention:
  All functions return NON-NEGATIVE values (or values with clear physical meaning).
  The SIGN is controlled by the weight in RewardsCfgPhase0.
  IsaacLab RewardManager computes: reward = func(...) * weight * dt

References:
  - Ng et al. (1999) "Policy invariance under reward transformations"
  - NavRL (Xu et al., 2025) for log-distance safety
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_lidar_min_distance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Helper: compute per-env minimum LiDAR 2D distance.

    Uses the same 2D projection as collision_occurred / collision_penalty
    for consistency across the reward system.

    Returns:
        [num_envs] minimum 2D distance to nearest obstacle per env.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    sensor_pos_2d = sensor.data.pos_w[:, :2]  # [num_envs, 2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]  # [num_envs, num_rays, 2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(
        distances_2d,
        nan=sensor.cfg.max_distance,
        posinf=sensor.cfg.max_distance,
    )
    return torch.min(distances_2d, dim=1)[0]  # [num_envs]


def potential_progress_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Potential-Based Reward Shaping for goal progress.

    Math:
        Phi(s) = -d(robot, goal)
        R(t) = Phi(s_{t+1}) - Phi(s_t)
             = -d_{t+1} + d_t
             = d_t - d_{t+1}          (positive when approaching)

    Properties:
        - Telescoping sum: sum(R) = Phi(s_final) - Phi(s_0) = d_0 - d_final
        - Bounded total: max = d_0 (initial distance, typically 5-14m)
        - Uses env._local_goal_world (fixes CRITICAL-2 target split)
        - Stores prev distance in env._prev_goal_dist per env

    Returns:
        [num_envs] -- positive when approaching goal, negative when retreating.
        Typical range per step: [-0.03, +0.03] at 50Hz with max 1.5m/s.
        Weight should be POSITIVE (e.g., +12.0).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device

    # Robot position (2D) — protect against NaN from physics instability
    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)  # [num_envs, 2]

    # Goal position -- unified source (fixes CRITICAL-2)
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        goal_pos = env._local_goal_world  # [num_envs, 2]
    else:
        goal_pos = env.command_manager.get_command("goal_command")[:, :2]

    # Current distance to goal
    d_curr = torch.norm(goal_pos - robot_pos, dim=1)  # [num_envs]

    # Initialize prev distance on first call or after episode reset
    if not hasattr(env, "_prev_goal_dist") or env._prev_goal_dist is None:
        env._prev_goal_dist = d_curr.clone()
        return torch.zeros(env.num_envs, device=device)

    # Detect episode resets: episode_length_buf == 0 means just reset
    # After reset, prev distance is stale -- reinitialize to current
    just_reset = env.episode_length_buf == 0
    env._prev_goal_dist[just_reset] = d_curr[just_reset]

    d_prev = env._prev_goal_dist

    # Potential-based shaping: R = d_prev - d_curr
    reward = d_prev - d_curr  # [num_envs]

    # Update stored distance for next step
    env._prev_goal_dist = d_curr.clone()

    # Safety: clamp to prevent physics explosions
    reward = reward.clamp(-1.0, 1.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    return reward


def near_obstacle_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    safe_distance: float = 1.2,
    alpha: float = 4.0,
) -> torch.Tensor:
    """Linear repulsive penalty when LiDAR min distance enters safety zone.

    Math:
        d_min = min(LiDAR readings) per env
        if d_min >= safe_distance:  penalty = 0
        else: penalty = alpha * (safe_distance - d_min)

    Properties:
        - Linear ramp: smooth gradient signal, no dead zone within safety zone
        - Returns non-negative values; weight should be NEGATIVE in config
        - Max penalty = alpha * safe_distance (when d_min = 0)

    Returns:
        [num_envs] in [0, alpha * safe_distance].
    """
    d_min = _get_lidar_min_distance(env, sensor_cfg)  # [num_envs]
    penalty = (alpha * (safe_distance - d_min)).clamp(min=0.0)
    return penalty


def exponential_obstacle_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    safe_distance: float = 1.0,
    steepness: float = 6.0,
) -> torch.Tensor:
    """Exponential repulsive penalty — strong gradient near obstacles.

    Math:
        d_min = min(LiDAR readings) per env
        if d_min >= safe_distance:  penalty = 0
        else:
            x = (safe_distance - d_min) / safe_distance   ∈ [0, 1]
            penalty = (exp(steepness * x) - 1) / (exp(steepness) - 1)

    Properties:
        - Exponential ramp: mild penalty at edge of safety zone, extreme near collision
        - Returns [0, 1]: weight should be NEGATIVE (e.g., -5.0)
        - steepness=6: at d_min=0 → penalty=1.0, at d_min=0.5*safe → penalty≈0.05
        - Provides strong gradient signal for the agent to stay centered in corridors

    Returns:
        [num_envs] in [0, 1].
    """
    d_min = _get_lidar_min_distance(env, sensor_cfg)  # [num_envs]

    # Normalized penetration into safety zone: 0 at edge, 1 at d_min=0
    x = ((safe_distance - d_min) / safe_distance).clamp(0.0, 1.0)

    # Exponential ramp normalized to [0, 1]
    exp_norm = 1.0 / (math.exp(steepness) - 1.0)
    penalty = (torch.exp(steepness * x) - 1.0) * exp_norm

    return penalty


def smooth_collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    warn_dist: float = 0.8,
    min_dist: float = 0.3,
) -> torch.Tensor:
    """Smooth repulsive collision penalty (decoupled from termination).

    Math:
        d_min = min(LiDAR readings) per env
        if d_min >= warn_dist:  penalty = 0
        elif d_min <= min_dist: penalty = 1.0 (maximum)
        else: penalty = ((warn_dist - d_min) / (warn_dist - min_dist))^2

    Properties:
        - Warning zone: [min_dist, warn_dist] = [0.3m, 0.8m]
        - Smooth quadratic ramp (no discontinuous jump)
        - Returns [0, 1] -- always non-negative
        - Weight should be NEGATIVE in config (e.g., -3.0)
        - Decoupled: termination at 0.3m, penalty starts at 0.8m

    Returns:
        [num_envs] in [0, 1].
    """
    d_min = _get_lidar_min_distance(env, sensor_cfg)  # [num_envs]

    # Quadratic ramp in warning zone
    range_size = warn_dist - min_dist  # 0.5m
    normalized = ((warn_dist - d_min) / (range_size + 1e-6)).clamp(0.0, 1.0)
    penalty = normalized ** 2  # [num_envs], in [0, 1]

    return penalty


def near_obstacle_speed_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    danger_distance: float = 1.5,
    safe_distance: float = 3.0,
) -> torch.Tensor:
    """近障時高速懲罰 — 教會 agent 在靠近障礙物時減速。

    Math:
        d_min = min(LiDAR readings)
        speed = ||v_xy||
        if d_min >= safe_distance: penalty = 0
        else:
            proximity = clamp((safe_distance - d_min) / (safe_distance - danger_distance), 0, 1)
            penalty = proximity * speed

    原理:
        - 遠離障礙物時不受限（proximity=0）
        - 進入 [danger, safe] 區間時，速度越快懲罰越大
        - 這比純 exponential_obstacle_penalty 更有行為意義：
          不是「靠近就懲罰」，而是「靠近且快速才懲罰」
        - 鼓勵 agent 在窄道中降速通過而非全速衝撞

    Returns:
        [num_envs] in [0, ~1.0]. Weight should be NEGATIVE (e.g., -2.0).
    """
    d_min = _get_lidar_min_distance(env, sensor_cfg)  # [num_envs]
    robot: Articulation = env.scene[robot_cfg.name]
    speed = torch.norm(
        torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0), dim=1
    )  # [num_envs]

    # Proximity factor: 0 when far, 1 when at danger_distance
    range_size = safe_distance - danger_distance
    proximity = ((safe_distance - d_min) / (range_size + 1e-6)).clamp(0.0, 1.0)

    # Penalty = proximity × speed (normalized by max_speed ~1.0 m/s)
    penalty = proximity * speed.clamp(max=2.0)

    return penalty


_collision_diag_count = 0

def collision_terminal_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    threshold: float = 0.3,
) -> torch.Tensor:
    """Binary collision terminal penalty (one-shot).

    Returns 1.0 when any LiDAR reading <= threshold, else 0.0.
    Weight should be NEGATIVE (e.g., -100.0).

    This is aligned with the termination threshold (0.3m) so that
    the terminal penalty fires on the same step as episode termination.

    Returns:
        [num_envs] -- 0.0 or 1.0.
    """
    global _collision_diag_count
    d_min = _get_lidar_min_distance(env, sensor_cfg)  # [num_envs]

    # 啟動診斷（僅前 5 步）
    if _collision_diag_count < 5:
        _collision_diag_count += 1
        colliding = (d_min <= threshold).sum().item()
        print(
            f"[碰撞偵測 #{_collision_diag_count}] "
            f"碰撞門檻={threshold:.2f}m | "
            f"最近障礙物距離 — 第0環境={d_min[0].item():.3f} 最近={d_min.min().item():.3f} 平均={d_min.mean().item():.3f} | "
            f"碰撞中={colliding}/{d_min.shape[0]}環境",
            flush=True,
        )

    return (d_min <= threshold).float()


def per_step_time_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Constant per-step time penalty to encourage efficiency.

    Returns 1.0 every step. Weight should be negative (e.g., -0.03).
    At 50Hz x 20s = 1000 steps, total = -0.03 * 1000 * dt ~ -0.6.

    Returns:
        [num_envs] -- always 1.0.
    """
    return torch.ones(env.num_envs, device=env.device)


def acceleration_squared_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """加速度平方懲罰：||a_t||² = ||v_t - v_{t-1}||²。

    懲罰劇烈的速度變化，保護馬達並確保 LiDAR/相機數據穩定。
    使用機器人實際線速度 (root_lin_vel_w) 計算差分。

    Math:
        a_t = v_t - v_{t-1}  (2D 線速度變化)
        r_a = ||a_t||²       (平方 L2 範數)

    Weight 應為負值（例如 -0.25，使得 -0.25 × dt(0.2) ≈ -0.05 per step）。

    Returns:
        [num_envs] — 非負值。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    vel_xy = robot.data.root_lin_vel_w[:, :2]  # [N, 2]

    if not hasattr(env, "_prev_lin_vel") or env._prev_lin_vel is None:
        env._prev_lin_vel = vel_xy.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Reset 後的 env 重新初始化
    just_reset = env.episode_length_buf == 0
    env._prev_lin_vel[just_reset] = vel_xy[just_reset]

    accel = vel_xy - env._prev_lin_vel           # [N, 2]
    penalty = (accel ** 2).sum(dim=1)             # [N] = ||a_t||²

    env._prev_lin_vel = vel_xy.clone()
    return penalty


def discrete_acceleration_squared_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """離散動作空間專用加速度平方懲罰：|applied_a|²。

    直接從 DiscreteDifferentialDriveAction.applied_accelerations 讀取
    飽和裁切後的加速度，避免從物理速度差分計算（可能有模擬器雜訊）。

    Math:
        r_a = |applied_a|²  (裁切後的線加速度平方)

    Weight 應為負值（例如 -0.25，使得 -0.25 × dt(0.2) ≈ -0.05 per step）。

    Returns:
        [num_envs] — 非負值。
    """
    term = list(env.action_manager._terms.values())[0]
    if hasattr(term, "applied_accelerations"):
        applied_a = term.applied_accelerations[:, 0]  # [N] 裁切後的加速度
    else:
        # Fallback: 使用 processed_actions（連續動作空間相容）
        applied_a = term.processed_actions[:, 0]
    penalty = applied_a ** 2
    # Protect against NaN
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=10.0, neginf=0.0)
    return penalty.clamp(0.0, 10.0)


def angular_velocity_squared_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """角速度平方懲罰：|ω_t|²。

    鼓勵機器人走直線，只有在必要避障時才轉彎。
    比單純獎勵直線行駛更穩健，主動懲罰不必要的抖動。

    Math:
        r_ω = |ω_z|²  (yaw 軸角速度平方)

    Weight 應為負值（例如 -0.25，使得 -0.25 × dt(0.2) ≈ -0.05 per step）。

    Returns:
        [num_envs] — 非負值。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # Protect against NaN from physics instabilities (collision frames)
    omega_z = torch.nan_to_num(robot.data.root_ang_vel_w[:, 2], nan=0.0)  # [N]
    penalty = omega_z ** 2
    return penalty.clamp(0.0, 10.0)


def velocity_too_low_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_velocity: float = 0.05,
) -> torch.Tensor:
    """靜止懲罰：當機器人幾乎不動時給予懲罰。

    防止 agent 學會「不動 = 安全」的退化策略。
    當線速度 < min_velocity 時返回 1.0，否則返回 0.0。

    Math:
        penalty = 1.0 if ||v_xy|| < min_velocity else 0.0

    Weight 應為負值（例如 -0.3），懲罰不動行為。
    在 episode 長度 100 步 × dt=0.2 下：
        G(不動) += -0.3 × 100 × 0.2 = -6.0（額外靜止懲罰）

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        min_velocity: 最低速度閾值 (m/s)

    Returns:
        [num_envs] — 0.0 或 1.0
    """
    robot: Articulation = env.scene[robot_cfg.name]
    vel_xy = robot.data.root_lin_vel_w[:, :2]  # [N, 2]
    speed = torch.norm(vel_xy, dim=-1)  # [N]
    # NaN 保護：NaN 速度視為靜止
    speed = torch.nan_to_num(speed, nan=0.0)
    return (speed < min_velocity).float()


__all__ = [
    "potential_progress_reward",
    "near_obstacle_penalty",
    "exponential_obstacle_penalty",
    "smooth_collision_penalty",
    "collision_terminal_penalty",
    "per_step_time_penalty",
    "acceleration_squared_penalty",
    "discrete_acceleration_squared_penalty",
    "angular_velocity_squared_penalty",
    "velocity_too_low_penalty",
]
