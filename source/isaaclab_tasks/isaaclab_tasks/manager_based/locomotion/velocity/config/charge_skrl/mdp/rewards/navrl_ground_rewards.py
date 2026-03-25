"""NavRL-Ground v1 — 地面差速機器人版 NavRL 獎勵

設計原則（vs 舊版 NavRL）：
1. toward-goal 永不完全歸零（soft gate with beta floor）
2. progress 永不翻負（只縮放，不 sign flip）
3. static safety 使用 LiDAR 72 bins 方向性（不只 bottom-K scalar）
4. dynamic safety 獨立使用 obstacle state（不混回 d_safe）

8 項 reward：
  r_goal_velocity:  朝目標速度 × soft gate（beta=0.2 floor）
  r_progress:       PBRS 進度 × soft scale（gamma=0.3 floor）
  r_static_safety:  72-bin log clearance + 前向堵塞懲罰
  r_dynamic_safety: per-obstacle log clearance + closing risk
  r_smooth:         控制平滑 (dv² + dw²)
  r_time:           每步小時間懲罰
  r_goal:           到達目標終端獎勵（沿用 reaching_goal）
  r_collision:      碰撞終端懲罰（沿用 collision_terminal_penalty）

References:
  - NavRL (Xu et al., 2025) for reward decomposition
  - Ng et al. (1999) for PBRS policy invariance
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# Helper: LiDAR 2D distances + bottom-K
# ============================================================================

def _get_lidar_2d_distances(env, sensor_cfg):
    """取得 LiDAR 每條 ray 的 2D 距離 [N, num_rays]。"""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    return torch.nan_to_num(distances, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)


def _get_d_safe(env, sensor_cfg, body_radius=0.35, bottom_k=10):
    """bottom-K mean - body_radius → d_safe [N]。"""
    distances = _get_lidar_2d_distances(env, sensor_cfg)
    k = min(bottom_k, distances.shape[1])
    bottom_vals = torch.topk(distances, k=k, dim=1, largest=False).values
    return (bottom_vals.mean(dim=1) - body_radius).clamp(min=1e-4)


def _get_goal_pos(env):
    """統一取得 goal 位置 [N, 2]。"""
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        return env._local_goal_world
    return env.command_manager.get_command("goal_command")[:, :2]


# ============================================================================
# 1. Goal Velocity Reward — soft gate, 永不歸零
# ============================================================================

def goal_velocity_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    body_radius: float = 0.35,
    bottom_k: int = 10,
    v_max: float = 1.0,
    min_goal_dist: float = 0.5,
    use_soft_gate: bool = True,
    gate_beta: float = 0.2,
    gate_dmin: float = 0.6,
    gate_dmax: float = 2.0,
) -> torch.Tensor:
    """朝目標方向的速度獎勵 — soft gate 永不完全關閉。

    Math:
        v_toward = dot(vel_2d, goal_dir)
        raw = clamp(v_toward / v_max, -1, 1)
        soft_gate = beta + (1-beta) * clamp((d_safe - dmin)/(dmax - dmin), 0, 1)
        reward = raw * soft_gate * (goal_dist > min_goal_dist)

    與舊版差異：soft_gate 最低 = beta (0.2)，永不歸零。
    舊版 v_gate 在 d_safe < 1.0 時 = 0（完全關閉 goal attraction）。

    Returns: [N] in [-1, 1]
    """
    robot: Articulation = env.scene[robot_cfg.name]

    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)
    vel_2d = torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0)
    goal_pos = _get_goal_pos(env)

    diff = goal_pos - robot_pos
    goal_dist = torch.norm(diff, dim=1, keepdim=True).clamp(min=1e-6)
    goal_dir = diff / goal_dist

    v_toward = (vel_2d * goal_dir).sum(dim=1)
    raw = (v_toward / v_max).clamp(-1.0, 1.0)

    far_enough = (goal_dist.squeeze(1) > min_goal_dist).float()
    raw = raw * far_enough

    if use_soft_gate:
        d_safe = _get_d_safe(env, sensor_cfg, body_radius, bottom_k)
        gate = gate_beta + (1.0 - gate_beta) * (
            (d_safe - gate_dmin) / (gate_dmax - gate_dmin + 1e-6)
        ).clamp(0.0, 1.0)
        raw = raw * gate

    return torch.nan_to_num(raw, nan=0.0)


# ============================================================================
# 2. Goal Progress Reward — soft scale, 永不翻負
# ============================================================================

def goal_progress_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    body_radius: float = 0.35,
    bottom_k: int = 10,
    progress_clip: float = 1.0,
    use_soft_scale: bool = True,
    scale_gamma: float = 0.3,
    scale_dmin: float = 0.5,
    scale_dmax: float = 2.0,
) -> torch.Tensor:
    """PBRS 進度獎勵 — soft scale, 永不翻負。

    Math:
        progress = clamp(d_prev - d_curr, -clip, clip)
        scale = gamma + (1-gamma) * clamp((d_safe - dmin)/(dmax - dmin), 0, 1)
        reward = progress * scale

    與舊版差異：scale 最低 = gamma (0.3)，永不翻負。
    舊版 safe_progress 在 d_safe < 0.8 時 gate = -0.5（前進扣分、撤退加分）。

    Returns: [N] in [-1, 1]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device

    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)
    goal_pos = _get_goal_pos(env)
    d_curr = torch.norm(goal_pos - robot_pos, dim=1)

    # 初始化
    if not hasattr(env, "_prev_goal_dist_ground") or env._prev_goal_dist_ground is None:
        env._prev_goal_dist_ground = d_curr.clone()
        return torch.zeros(env.num_envs, device=device)

    just_reset = env.episode_length_buf == 0
    env._prev_goal_dist_ground[just_reset] = d_curr[just_reset]

    progress = (env._prev_goal_dist_ground - d_curr).clamp(-progress_clip, progress_clip)
    env._prev_goal_dist_ground = d_curr.clone()

    if use_soft_scale:
        d_safe = _get_d_safe(env, sensor_cfg, body_radius, bottom_k)
        scale = scale_gamma + (1.0 - scale_gamma) * (
            (d_safe - scale_dmin) / (scale_dmax - scale_dmin + 1e-6)
        ).clamp(0.0, 1.0)
        progress = progress * scale

    return torch.nan_to_num(progress, nan=0.0)


# ============================================================================
# 3. Static Safety Reward — 72 bins 方向性 + 前向堵塞懲罰
# ============================================================================

def static_safety_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    body_radius: float = 0.35,
    a_global: float = 1.0,
    a_front_block: float = 1.0,
    front_half_angle_deg: float = 20.0,
    front_nearest_k: int = 5,
    front_warn_dist: float = 1.2,
) -> torch.Tensor:
    """LiDAR 72-bin 靜態安全獎勵 — 全域 log clearance + 前向堵塞懲罰。

    (A) 全域: mean(log(max(bin_dist - body_radius, eps))) 對 72 bins
        → 正值 = 遠離障礙物, 負值 = 靠近
    (B) 前向堵塞: -max(0, warn_dist - d_front) * forward_speed
        → 正前方被堵 + 高速前進 → 大懲罰

    Returns: [N] (可正可負)
    """
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device
    N = env.num_envs

    # 取 raw LiDAR → min-pool to 72 bins
    distances = _get_lidar_2d_distances(env, sensor_cfg)  # [N, num_rays]
    num_rays = distances.shape[1]
    num_bins = 72
    rays_per_bin = num_rays // num_bins

    if rays_per_bin > 1:
        trimmed = distances[:, :num_bins * rays_per_bin]
        binned = trimmed.reshape(N, num_bins, rays_per_bin).min(dim=2).values  # [N, 72]
    else:
        binned = distances[:, :num_bins]

    # (A) 全域 log clearance
    clearances = (binned - body_radius).clamp(min=1e-3)  # [N, 72]
    r_global = torch.log(clearances).mean(dim=1)  # [N]
    r_global = r_global.clamp(-6.0, 3.0)

    # (B) 前向堵塞懲罰
    # bin 0 = -180°, bin 36 = 0° (正前方), bin 72 = +180°
    # ±20° = bins (36 - 20/5) to (36 + 20/5) = bins 32 to 40
    half_bins = int(front_half_angle_deg / 5.0)
    center = num_bins // 2  # 36
    start = max(0, center - half_bins)
    end = min(num_bins, center + half_bins)
    front_sector = binned[:, start:end]  # [N, ~8]

    # 取前方最近 k 條的平均
    k = min(front_nearest_k, front_sector.shape[1])
    d_front = torch.topk(front_sector, k=k, dim=1, largest=False).values.mean(dim=1)  # [N]

    # 前向速度（body frame x 軸）
    quat = robot.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    vel_2d = torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0)
    forward_speed = (vel_2d[:, 0] * torch.cos(yaw) + vel_2d[:, 1] * torch.sin(yaw)).clamp(min=0.0)

    r_front_block = -(front_warn_dist - d_front).clamp(min=0.0) * forward_speed  # [N] ≤ 0

    reward = a_global * r_global + a_front_block * r_front_block
    return torch.nan_to_num(reward, nan=0.0)


# ============================================================================
# 4. Dynamic Safety Reward — per-obstacle log clearance + closing risk
# ============================================================================

def dynamic_safety_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_radius: float = 0.35,
    max_obstacles: int = 10,
    mode: str = "log_distance",
    risk_sigma: float = 1.0,
    b_log: float = 1.0,
    b_risk: float = 1.0,
) -> torch.Tensor:
    """動態障礙物安全獎勵 — 獨立使用 obstacle state。

    Mode A (log_distance):
        mean_visible(log(clearance_i))
    Mode B (closing_risk):
        b_log * log_term - b_risk * mean_visible(risk_i)
        risk_i = max(0, closing_rate_i) * exp(-clearance_i / sigma)

    與舊版差異：不使用 d_safe scalar，直接從 obstacle state 計算。

    Returns: [N]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device
    N = env.num_envs

    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)  # [N, 2]
    robot_vel = torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0)  # [N, 2]

    # 收集所有可見障礙物
    all_clearance = torch.full((N, max_obstacles), 10.0, device=device)
    all_risk = torch.zeros(N, max_obstacles, device=device)
    all_visible = torch.zeros(N, max_obstacles, dtype=torch.bool, device=device)
    num_found = 0

    # 障礙物尺寸
    obs_sizes = getattr(env, "_obstacle_sizes", None)

    for i in range(max_obstacles):
        obs_name = f"obstacle_{i}"
        if obs_name not in env.scene.keys():
            continue

        obs_entity = env.scene[obs_name]
        pos_w = obs_entity.data.root_pos_w
        visible = pos_w[:, 2] > 0.0  # Z > 0 = visible

        obs_pos = torch.nan_to_num(pos_w[:, :2], nan=0.0)  # [N, 2]

        # 障礙物半徑
        if obs_sizes is not None and i < len(obs_sizes):
            obs_r = obs_sizes[i] / 2.0  # size → radius
        else:
            obs_r = 0.3

        # 相對位置、距離、clearance
        rel_pos = obs_pos - robot_pos  # [N, 2]
        dist_center = torch.norm(rel_pos, dim=1).clamp(min=1e-4)  # [N]
        clearance = (dist_center - body_radius - obs_r).clamp(min=1e-3)  # [N]

        all_clearance[:, num_found] = clearance
        all_visible[:, num_found] = visible

        # Closing risk
        if mode == "closing_risk":
            # 障礙物相對速度
            if hasattr(env, "_obstacle_velocities") and env._obstacle_velocities is not None:
                try:
                    obs_vel = env._obstacle_velocities[:, i, :2]
                except (IndexError, TypeError):
                    obs_vel = torch.zeros(N, 2, device=device)
            else:
                try:
                    obs_vel = obs_entity.data.root_lin_vel_w[:, :2]
                except (AttributeError, IndexError):
                    obs_vel = torch.zeros(N, 2, device=device)

            rel_vel = obs_vel - robot_vel  # [N, 2] 相對速度
            unit_pos = rel_pos / dist_center.unsqueeze(1)
            # closing_rate > 0 = 彼此接近
            closing_rate = -(unit_pos * rel_vel).sum(dim=1)  # [N]
            risk = closing_rate.clamp(min=0.0) * torch.exp(-clearance / risk_sigma)
            all_risk[:, num_found] = risk

        num_found += 1

    if num_found == 0:
        return torch.zeros(N, device=device)

    # 裁剪到實際數量
    clearance = all_clearance[:, :num_found]  # [N, F]
    visible = all_visible[:, :num_found]  # [N, F]
    risk = all_risk[:, :num_found]  # [N, F]

    # Log clearance（只對 visible 障礙物平均）
    log_clearance = torch.log(clearance)  # [N, F]
    # 不可見障礙物設為 0（不影響平均）
    log_clearance = log_clearance * visible.float()
    vis_count = visible.float().sum(dim=1).clamp(min=1.0)  # [N]
    r_log = log_clearance.sum(dim=1) / vis_count  # [N]
    r_log = r_log.clamp(-6.0, 3.0)

    if mode == "log_distance":
        reward = r_log
    else:
        # Closing risk（只對 visible 平均）
        risk = risk * visible.float()
        r_risk = risk.sum(dim=1) / vis_count
        reward = b_log * r_log - b_risk * r_risk

    return torch.nan_to_num(reward, nan=0.0)


# ============================================================================
# 5. Control Smoothness Penalty — dv² + dw²
# ============================================================================

def control_smoothness_penalty(
    env: ManagerBasedRLEnv,
    smooth_v_coeff: float = 1.0,
    smooth_w_coeff: float = 1.0,
) -> torch.Tensor:
    """控制平滑懲罰 — 懲罰加速度和角速度的變化量。

    Math:
        dv = curr_linear_accel - prev_linear_accel
        dw = curr_angular_vel - prev_angular_vel
        penalty = smooth_v * dv² + smooth_w * dw²

    返回非負值，由 weight 負號控制懲罰。

    Returns: [N] ≥ 0
    """
    device = env.device
    N = env.num_envs

    # 讀取當前 action
    term = list(env.action_manager._terms.values())[0]
    if hasattr(term, "applied_accelerations"):
        curr_a = term.applied_accelerations[:, 0]  # [N] 線加速度
        curr_w = term.applied_accelerations[:, 1]  # [N] 角速度
    else:
        curr_a = term.processed_actions[:, 0]
        curr_w = term.processed_actions[:, 1] if term.processed_actions.shape[1] > 1 else torch.zeros(N, device=device)

    curr = torch.stack([curr_a, curr_w], dim=1)  # [N, 2]

    # 初始化
    if not hasattr(env, "_prev_actions_ground") or env._prev_actions_ground is None:
        env._prev_actions_ground = curr.clone()
        return torch.zeros(N, device=device)

    just_reset = env.episode_length_buf == 0
    env._prev_actions_ground[just_reset] = curr[just_reset]

    dv = curr[:, 0] - env._prev_actions_ground[:, 0]
    dw = curr[:, 1] - env._prev_actions_ground[:, 1]

    env._prev_actions_ground = curr.clone()

    penalty = smooth_v_coeff * dv ** 2 + smooth_w_coeff * dw ** 2
    return torch.nan_to_num(penalty, nan=0.0).clamp(0.0, 10.0)


def alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """NavRL 風格存活獎勵：每步無條件 +1.0。

    配合 γ=0.99 折扣產生時間壓力：停在原地也有 +1.0，
    但朝目標走有 +1.0 + r_vel > +1.0，所以前進永遠比不動好。

    Returns: [N] = 1.0
    """
    return torch.ones(env.num_envs, device=env.device)


__all__ = [
    "goal_velocity_reward",
    "goal_progress_reward",
    "static_safety_reward",
    "dynamic_safety_reward",
    "control_smoothness_penalty",
    "alive_reward",
]
