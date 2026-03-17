"""目標相關獎勵函數

VLP16 訓練使用的函數:
    reaching_goal: 到達目標時的一次性獎勵 (weight=+250)

其他函數保留供未來使用，但目前 VLP16 配置未啟用。

參數:
    threshold: float = 0.5  — 判定到達的距離閾值 [m] (= ROBOT_BODY_RADIUS)
    body_radius: float = 0.5 — 機器人半徑 [m]，距離修正用

物理意義:
    Terminal reward 是 RL 的「指南針」。+250 的權重確保
    一次成功到達 >> 多步 dense reward 累積，防止 agent
    學到「在目標附近繞圈」的退化策略。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import _check_reward_term


def velocity_toward_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_dist: float = 1.0) -> torch.Tensor:
    """朝向目標的速度獎勵
    
    獎勵機器人「朝著目標移動」的速度分量。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        min_dist: 最小距離（米），低於此距離不給獎勵（避免衝撞）
    
    Returns:
        shape [num_envs]：獎勵值 [0, ~1.5]
        值越大表示越快朝目標移動
    
    為什麼用速度獎勵？
    - 鼓勵機器人「積極移動」
    - 避免機器人「磨蹭」（慢慢走）
    - 比單純的距離獎勵更直接
    """
    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    # 🔥 關鍵修復：優先使用規劃器的局部目標
    if hasattr(env, '_local_goal_world'):
        goal_pos_w = env._local_goal_world
    else:
        goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_vel_w = asset.data.root_lin_vel_w[:, :2]  # 只要 XY 速度（忽略 Z）
    # shape: [num_envs, 2]：[vx, vy] 在世界座標系
    
    # ------------------------------------------------------------------------
    # 步驟 1：計算目標方向（歸一化單位向量）
    # ------------------------------------------------------------------------
    goal_direction = goal_pos_w[:, :2] - robot_pos_w[:, :2]
    # 從機器人指向目標的向量
    # 例如：robot=[0,0], goal=[3,4] → direction=[3,4]
    
    goal_distance = torch.norm(goal_direction, dim=1)
    # 計算距離：sqrt(3^2 + 4^2) = 5.0 米
    
    goal_direction = goal_direction / (goal_distance.unsqueeze(1) + 1e-6)
    # 歸一化為單位向量（長度=1）
    # +1e-6 防止除以零
    # 例如：[3,4] / 5.0 = [0.6, 0.8]（單位向量）
    
    # ------------------------------------------------------------------------
    # 步驟 2：計算速度在目標方向上的投影
    # ------------------------------------------------------------------------
    velocity_projection = torch.sum(robot_vel_w * goal_direction, dim=1)
    # 向量點積（dot product）：v · d = |v| |d| cos(θ)
    # 當 d 是單位向量時：v · d = 速度在 d 方向的分量
    # 
    # 例子：
    #   robot_vel = [1.0, 0.5]（往東北移動）
    #   goal_direction = [1.0, 0.0]（目標在正東）
    #   projection = 1.0*1.0 + 0.5*0.0 = 1.0 m/s
    #   （機器人以 1 m/s 的速度朝目標移動）
    
    velocity_projection = torch.clamp(velocity_projection, min=0.0)
    # 只獎勵正向速度（朝目標移動）
    # 如果背離目標（負值）→ 設為 0（不給獎勵，但不懲罰）
    
    # ------------------------------------------------------------------------
    # 步驟 3：距離閘門（避免衝撞）
    # ------------------------------------------------------------------------
    gated_velocity = torch.where(
        goal_distance > min_dist,  # 條件：距離 > 1.0 米
        velocity_projection,  # True：給速度獎勵
        torch.zeros_like(velocity_projection),  # False：不給獎勵
    )
    # 為什麼需要這個？
    # - 防止機器人「衝撞」目標（撞過頭）
    # - 在目標附近（<1米）需要減速，不應獎勵高速
    
    # 安全處理：清理 NaN/Inf 並限制範圍
    gated_velocity = torch.nan_to_num(gated_velocity, nan=0.0, posinf=0.0, neginf=0.0)
    gated_velocity = torch.clamp(gated_velocity, 0.0, 10.0)  # 限制到合理範圍
    
    # 檢查 reward term（防止 PPO std>=0 錯誤）
    gated_velocity = _check_reward_term("velocity_toward_goal", gated_velocity, env, raise_on_error=True)
    
    return gated_velocity
    # 返回值範圍：[0, max_linear_velocity]
    # 例如：[0, 1.5] m/s


def velocity_toward_goal_smooth(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    slow_distance: float = 1.0,
    stop_distance: float = 0.28,
    max_reward_speed: float = 1.5,
    min_reward_speed: float = 0.3,
) -> torch.Tensor:
    """漸進式速度獎勵：距離越近，獎勵的速度越低
    
    這個函數鼓勵機器人在接近目標時自動減速，避免衝撞。
    獎勵機制：
    - 距離 > slow_distance：獎勵高速（max_reward_speed）
    - stop_distance < 距離 < slow_distance：線性插值期望速度
    - 距離 < stop_distance：不給獎勵（避免衝撞）
    
    獎勵計算：
    - 使用指數衰減：速度越接近「期望速度」，獎勵越高
    - 完美匹配期望速度 → 獎勵 1.0
    - 偏離期望速度 → 獎勵指數衰減
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        slow_distance: 開始減速的距離（米），低於此距離開始降低期望速度
        stop_distance: 停止獎勵的距離（米），低於此距離不給獎勵
        max_reward_speed: 遠距離的目標速度（m/s），距離 > slow_distance 時的期望速度
        min_reward_speed: 近距離的目標速度（m/s），距離 = stop_distance 時的期望速度
    
    Returns:
        shape [num_envs]：獎勵值 [0, 1]
        1.0 = 完美匹配期望速度
        0.0 = 距離太近或速度偏離太大
    
    設計理念：
    - 遠距離：鼓勵快速接近（1.5 m/s）
    - 中距離：鼓勵適度減速（0.3-1.5 m/s 之間）
    - 近距離：鼓勵低速接近（0.3 m/s）
    - 極近距離：停止獎勵（避免衝撞）
    
    例子：
        距離 = 3.0m → 期望速度 1.5 m/s → 以 1.5 m/s 前進得最高獎勵
        距離 = 1.0m → 期望速度 0.3 m/s → 開始自動減速
        距離 = 0.5m → 期望速度 0.15 m/s → 繼續減速
        距離 = 0.28m → 期望速度 0 m/s → 停止獎勵
    """
    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    # 🔥 關鍵修復：優先使用規劃器的局部目標
    if hasattr(env, '_local_goal_world'):
        goal_pos_w = env._local_goal_world
    else:
        goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_vel_w = asset.data.root_lin_vel_w[:, :2]  # 只要 XY 速度（忽略 Z）
    
    # ------------------------------------------------------------------------
    # 步驟 1：計算目標方向和距離
    # ------------------------------------------------------------------------
    goal_direction = goal_pos_w[:, :2] - robot_pos_w[:, :2]
    goal_distance = torch.norm(goal_direction, dim=1)
    goal_direction = goal_direction / (goal_distance.unsqueeze(1) + 1e-6)
    
    # ------------------------------------------------------------------------
    # 步驟 2：計算速度在目標方向上的投影
    # ------------------------------------------------------------------------
    velocity_projection = torch.sum(robot_vel_w * goal_direction, dim=1)
    velocity_projection = torch.clamp(velocity_projection, min=0.0)  # 只獎勵正向速度
    
    # ------------------------------------------------------------------------
    # 步驟 3：計算「期望速度」（距離加權）
    # ------------------------------------------------------------------------
    # 遠距離：期望高速（max_reward_speed）
    # 近距離：期望低速（min_reward_speed）
    # 中間距離：線性插值
    target_speed = torch.where(
        goal_distance > slow_distance,
        # 距離 > slow_distance：期望高速
        torch.full_like(goal_distance, max_reward_speed),
        torch.where(
            goal_distance > stop_distance,
            # stop_distance < 距離 < slow_distance：線性插值
            min_reward_speed + (max_reward_speed - min_reward_speed) * 
            ((goal_distance - stop_distance) / (slow_distance - stop_distance + 1e-6)),
            # 距離 < stop_distance：期望速度為 0（停止）
            torch.zeros_like(goal_distance)
        )
    )
    
    # ------------------------------------------------------------------------
    # 步驟 4：計算獎勵（越接近目標速度越好）
    # ------------------------------------------------------------------------
    # 使用指數衰減：完美匹配 target_speed 時獎勵 = 1.0
    # 偏離越大，獎勵越小
    speed_error = torch.abs(velocity_projection - target_speed)
    # 使用 exp(-error) 作為獎勵，error=0 時 reward=1.0，error 越大 reward 越小
    # 為了讓獎勵更平滑，使用 exp(-error / scale)，scale 控制衰減速度
    scale = max_reward_speed * 0.5  # 衰減尺度：當誤差 = scale 時，獎勵約為 0.6
    reward = torch.exp(-speed_error / (scale + 1e-6))
    
    # ------------------------------------------------------------------------
    # 步驟 5：距離閘門（避免衝撞）
    # ------------------------------------------------------------------------
    reward = torch.where(
        goal_distance > stop_distance,  # 距離 > stop_distance 才給獎勵
        reward,
        torch.zeros_like(reward)  # 距離太近：不給獎勵
    )
    
    # 安全處理：清理 NaN/Inf 並限制範圍
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    reward = torch.clamp(reward, 0.0, 1.0)
    
    # 檢查 reward term（防止 PPO std>=0 錯誤）
    reward = _check_reward_term("velocity_toward_goal_smooth", reward, env, raise_on_error=True)
    
    return reward


def progress_to_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """目標進度獎勵（使用距離差，傳統且有效）

    🔥 Bug 修復：改用「距離差」計算進度，而非嚴苛的「面向且移動」條件！

    舊版問題：
    - reward = heading_factor * velocity_factor
    - 只要車子稍微「沒有對準目標」或「側向滑行」，reward 直接變 0
    - 導致車子「原地發呆」或「不敢避障轉彎」

    新版實現：
    - reward = previous_distance - current_distance
    - 直接獎勵「距離減少」，不管車頭朝哪裡
    - 鼓勵「繞路避障」和「側向移動」
    - 這是傳統 navigation reward，被證明非常有效

    Args:
        env: 環境實例
        asset_cfg: 資產配置(引用機器人)

    Returns:
        shape [num_envs]：獎勵值
        正值 = 靠近目標（好！）
        負值 = 遠離目標（不好）
        0 = 距離不變（原地轉圈、側移）
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 🔥 關鍵修復：優先使用規劃器的局部目標
    if hasattr(env, '_local_goal_world'):
        goal_pos_w = env._local_goal_world
    else:
        goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]

    # 🔥 重要：清理所有輸入數據的 NaN/Inf
    goal_pos_w = torch.nan_to_num(goal_pos_w, nan=0.0, posinf=100.0, neginf=-100.0)
    robot_pos_w = torch.nan_to_num(robot_pos_w, nan=0.0, posinf=100.0, neginf=-100.0)

    # 計算當前距離
    current_distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
    current_distance = torch.nan_to_num(current_distance, nan=10.0, posinf=10.0, neginf=0.0)

    # 獲取上一步的距離（如果沒有記錄，初始化為當前距離）
    if not hasattr(env, '_previous_goal_distance') or env._previous_goal_distance is None:
        env._previous_goal_distance = current_distance.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # 🔥 H1 修復：偵測 episode reset，避免跨 episode 獎勵洩漏
    # episode_length_buf == 0 表示該 env 剛 reset，prev distance 是上一輪殘留
    just_reset = env.episode_length_buf == 0
    env._previous_goal_distance[just_reset] = current_distance[just_reset]

    previous_distance = env._previous_goal_distance

    # 🔥 關鍵改進：獎勵「距離減少量」
    # reward = 上一步距離 - 當前距離
    # 正值 = 靠近了（好！）
    # 負值 = 遠離了（不好，但允許「先退後進」）
    # 0 = 距離不變（原地轉圈、純側移）
    reward = previous_distance - current_distance

    # 更新記錄
    env._previous_goal_distance = current_distance.clone()

    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
    reward = torch.clamp(reward, -10.0, 10.0)

    # 檢查 reward term（防止 PPO std>=0 錯誤）
    reward = _check_reward_term("progress_to_goal", reward, env, raise_on_error=True)

    return reward

_reaching_diag_count = 0

def reaching_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    body_radius: float = 0.0,
) -> torch.Tensor:
    """到達目標獎勵

    當機器人到達目標時給予大獎勵。

    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        threshold: 距離閾值（米），低於此距離算「到達」

    Returns:
        shape [num_envs]：0 或 1
        1 = 到達目標
        0 = 未到達

    這是最重要的獎勵！
    - 配合大權重（weight=100），成為主要目標
    - 稀疏獎勵（大部分時間是 0，到達時才給）
    """
    global _reaching_diag_count

    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    # 🔥 關鍵修復：優先使用規劃器的局部目標
    if hasattr(env, '_local_goal_world'):
        goal_pos_w = env._local_goal_world
    else:
        goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]

    # 計算距離（扣除機器人體半徑，使用「身體到目標」距離）
    distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
    if body_radius > 0.0:
        distance = torch.clamp(distance - body_radius, min=0.0)

    # 安全處理：清理 NaN/Inf
    distance = torch.nan_to_num(distance, nan=100.0, posinf=100.0, neginf=0.0)
    distance = torch.clamp(distance, 0.0, 100.0)

    # 啟動診斷（僅前 5 步）
    if _reaching_diag_count < 5:
        _reaching_diag_count += 1
        reached = (distance < threshold).sum().item()
        print(
            f"[目標到達 #{_reaching_diag_count}] "
            f"到達門檻={threshold:.2f}m 扣除半徑={body_radius:.2f}m | "
            f"距離 — 第0環境={distance[0].item():.3f} 最近={distance.min().item():.3f} 平均={distance.mean().item():.3f} | "
            f"已到達={reached}/{distance.shape[0]}環境",
            flush=True,
        )

    # 判斷是否到達
    reward = (distance < threshold).float()

    # 檢查 reward term（防止 PPO std>=0 錯誤）
    reward = _check_reward_term("reaching_goal", reward, env, raise_on_error=True)

    return reward


def approaching_goal_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    close_threshold: float = 2.0,
    decay_scale: float = 0.5,
) -> torch.Tensor:
    """接近目標時給予額外獎勵（鼓勵 agent 敢於靠近）
    
    這個函數鼓勵 agent 接近目標，即使還沒到達。
    使用指數衰減：距離越近，獎勵越高。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        close_threshold: 接近閾值（米），進入此距離內開始給獎勵
        decay_scale: 衰減尺度係數（0-1），控制指數衰減的陡峭程度
                     0.5 = 較平緩的衰減（推薦）
                     1.0 = 較陡峭的衰減
    
    Returns:
        shape [num_envs]：獎勵值 [0, 1]
        1.0 = 距離為 0（已到達）
        0.0 = 距離 >= close_threshold（太遠）
    
    設計理念：
    - 鼓勵 agent 敢於接近目標，而不是「遠離一切」
    - 與 progressive_collision 形成平衡：可以接近障礙物，但不要太近
    - 幫助 agent 克服「恐懼」，積極嘗試到達目標
    """
    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    # 🔥 關鍵修復：優先使用規劃器的局部目標
    if hasattr(env, '_local_goal_world'):
        goal_pos_w = env._local_goal_world
    else:
        goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]
    
    # 計算距離
    goal_distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
    
    # 計算衰減尺度：使用 close_threshold * decay_scale 作為分母
    # decay_scale=0.5 時：
    #   距離 = 2.0m → exp(-2.0/1.0) = exp(-2) ≈ 0.14
    #   距離 = 1.0m → exp(-1.0/1.0) = exp(-1) ≈ 0.37
    #   距離 = 0.5m → exp(-0.5/1.0) = exp(-0.5) ≈ 0.61
    #   距離 = 0.0m → exp(0) = 1.0
    # 
    # 這比之前更平緩：在閾值邊界（2.0m）處獎勵約 0.14，而不是 0.37
    # 這樣可以避免衰減過於陡峭導致 agent 難以學習
    scale = close_threshold * decay_scale
    
    # 距離閘門：只對接近的環境給獎勵
    # 距離 < close_threshold 時，使用指數衰減計算獎勵
    # 距離 >= close_threshold 時，獎勵為 0
    reward = torch.where(
        goal_distance < close_threshold,
        torch.exp(-goal_distance / (scale + 1e-6)),  # 指數衰減：距離越近獎勵越高
        torch.zeros_like(goal_distance)  # 距離太遠：不給獎勵
    )
    
    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    reward = torch.clamp(reward, 0.0, 1.0)
    
    # 檢查 reward term
    reward = _check_reward_term("approaching_goal_bonus", reward, env, raise_on_error=True)
    
    return reward


_heading_diag_count = 0

def heading_to_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """朝向目標獎勵

    獎勵機器人「面向」目標的角度。這個獎勵鼓勵機器人先對準目標方向，再加速前進。

    Returns:
        shape [num_envs]：cos(angle) 範圍 [-1, 1]
        1.0 = 正對目標（角度 0°）
        0.0 = 側對目標（角度 90°）
        -1.0 = 背對目標（角度 180°）

    設計理念：
    - 在接近目標時，朝向獎勵更重要（先對準方向）
    - 在遠離目標時，速度獎勵更重要（快速接近）
    """
    global _heading_diag_count

    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    # 🔥 關鍵修復：優先使用規劃器的局部目標
    if hasattr(env, '_local_goal_world'):
        goal_pos_w = env._local_goal_world
    else:
        goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_quat_w = asset.data.root_quat_w

    # Protect against NaN from physics instability (collision frames)
    robot_pos_w = torch.nan_to_num(robot_pos_w, nan=0.0)
    robot_quat_w = torch.nan_to_num(robot_quat_w, nan=0.0)
    # Fix degenerate quaternions (norm≈0 after NaN replacement) → identity [1,0,0,0]
    quat_norm = torch.norm(robot_quat_w, dim=1, keepdim=True)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand_as(robot_quat_w)
    robot_quat_w = torch.where(quat_norm > 0.5, robot_quat_w, identity_quat)

    # 計算目標方向（世界座標系）
    goal_direction_w = goal_pos_w - robot_pos_w
    goal_distance = torch.norm(goal_direction_w[:, :2], dim=1)
    goal_direction_w = goal_direction_w / (torch.norm(goal_direction_w, dim=1, keepdim=True) + 1e-6)
    # 歸一化單位向量

    # 創建機器人前方向量（本地座標系）
    num_envs = robot_quat_w.shape[0]
    forward_vec = torch.zeros(num_envs, 3, device=env.device)
    forward_vec[:, 0] = 1.0  # X 軸是前方

    # 轉換到世界座標系
    forward_w = math_utils.quat_apply(robot_quat_w, forward_vec)

    # 計算夾角的余弦值（向量點積）
    heading_reward = torch.sum(forward_w[:, :2] * goal_direction_w[:, :2], dim=1)
    # cos(0°) = 1.0（正對）
    # cos(90°) = 0.0（側對）
    # cos(180°) = -1.0（背對）

    # 啟動診斷（僅前 5 步）
    if _heading_diag_count < 5:
        _heading_diag_count += 1
        print(
            f"[朝向獎勵 #{_heading_diag_count}] "
            f"目標距離={goal_distance[0].item():.3f}m 朝向餘弦={heading_reward[0].item():.4f} | "
            f"全環境平均={heading_reward.mean().item():.4f}",
            flush=True,
        )

    # 只獎勵正向朝向（不懲罰背對，因為速度獎勵會處理）
    heading_reward = torch.clamp(heading_reward, min=0.0)

    # 安全處理
    heading_reward = torch.nan_to_num(heading_reward, nan=0.0, posinf=1.0, neginf=0.0)
    heading_reward = torch.clamp(heading_reward, 0.0, 1.0)

    return heading_reward


def heading_to_goal_distance_weighted(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    close_distance: float = 2.0,
    far_distance: float = 5.0,
) -> torch.Tensor:
    """朝向目標獎勵（根據距離加權）
    
    根據距離目標的遠近，動態調整朝向獎勵的權重：
    - 接近目標時（< close_distance）：強烈獎勵朝向目標
    - 遠離目標時（> far_distance）：較少獎勵朝向（速度更重要）
    - 中間距離：線性插值
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        close_distance: 接近距離閾值（米），低於此距離時朝向獎勵權重最大
        far_distance: 遠距離閾值（米），高於此距離時朝向獎勵權重最小
    
    Returns:
        shape [num_envs]：加權後的朝向獎勵 [0, 1]
    """
    # 獲取朝向獎勵
    heading_reward = heading_to_goal(env, asset_cfg)

    # 獲取距離
    asset: Articulation = env.scene[asset_cfg.name]
    # 🔥 關鍵修復：優先使用規劃器的局部目標
    if hasattr(env, '_local_goal_world'):
        goal_pos_w = env._local_goal_world
    else:
        goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]
    goal_distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
    
    # 計算距離權重（線性插值）
    # 距離 < close_distance → 權重 = 1.0（最大）
    # 距離 > far_distance → 權重 = 0.2（最小）
    # 中間距離 → 線性插值
    weight = torch.clamp(
        (far_distance - goal_distance) / (far_distance - close_distance + 1e-6),
        min=0.2,
        max=1.0
    )
    
    # 應用權重
    weighted_reward = heading_reward * weight
    
    # 安全處理
    weighted_reward = torch.nan_to_num(weighted_reward, nan=0.0, posinf=1.0, neginf=0.0)
    weighted_reward = torch.clamp(weighted_reward, 0.0, 1.0)
    
    return weighted_reward


def alignment_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    misalignment_penalty_scale: float = 0.5,
    high_speed_threshold: float = 1.0,
) -> torch.Tensor:
    """對齊獎勵 (Alignment)：車頭朝向目標的獎勵
    
    核心設計：
    1. 車頭朝向與目標方向的夾角越小，獎勵越高
    2. 若車頭未對準目標但 Speed_x 很高，給予懲罰（防止衝過頭）
    
    設計理念：
    - 由於車輛不能側移，「車頭朝向目標」變得至關重要
    - 在 ROS 系統測試中，為了讓 Agent 準確抵達，加入了「比例控制器」概念
    - 在 RL 獎勵中，加入對齊獎勵可以引導 Agent 先對準方向再加速
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        misalignment_penalty_scale: 未對準時的懲罰係數（默認 0.5）
        high_speed_threshold: 高速閾值（m/s），超過此速度且未對準時給予懲罰（默認 1.0）
    
    Returns:
        shape [num_envs]：獎勵值 [-penalty, 1.0]
        - 正對目標（0°）→ +1.0
        - 側對目標（90°）→ 0.0
        - 背對目標（180°）→ 0.0（不懲罰，因為速度獎勵會處理）
        - 未對準但高速前進 → 負懲罰（防止衝過頭）
    """
    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    # 🔥 關鍵修復：優先使用規劃器的局部目標
    if hasattr(env, '_local_goal_world'):
        goal_pos_w = env._local_goal_world
    else:
        goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_quat_w = asset.data.root_quat_w
    velocity_b = asset.data.root_lin_vel_b  # 機器人本體座標系速度
    
    # 計算目標方向（世界座標系）
    goal_direction_w = goal_pos_w - robot_pos_w
    goal_distance = torch.norm(goal_direction_w[:, :2], dim=1)
    goal_direction_w = goal_direction_w / (torch.norm(goal_direction_w, dim=1, keepdim=True) + 1e-6)
    # 歸一化單位向量
    
    # 創建機器人前方向量（本地座標系）
    num_envs = robot_quat_w.shape[0]
    forward_vec = torch.zeros(num_envs, 3, device=env.device)
    forward_vec[:, 0] = 1.0  # X 軸是前方
    
    # 轉換到世界座標系
    forward_w = math_utils.quat_apply(robot_quat_w, forward_vec)
    
    # 計算夾角的余弦值（向量點積）
    # cos(0°) = 1.0（正對）
    # cos(90°) = 0.0（側對）
    # cos(180°) = -1.0（背對）
    alignment_cosine = torch.sum(forward_w[:, :2] * goal_direction_w[:, :2], dim=1)
    
    # 基礎對齊獎勵：只獎勵正向朝向（不懲罰背對）
    alignment_reward_base = torch.clamp(alignment_cosine, min=0.0)
    
    # 獲取前進速度（本體座標系 X 軸）
    speed_x = velocity_b[:, 0]
    speed_x_forward = torch.clamp(speed_x, min=0.0)  # 只考慮正向速度
    
    # 計算未對準角度（0 到 1，1 表示完全未對準）
    misalignment = 1.0 - alignment_reward_base  # 0.0 = 完全對準，1.0 = 完全未對準
    
    # 懲罰條件：未對準（misalignment > 0.3，約 72°）且高速前進（speed_x > threshold）
    # 這防止 Agent「衝過頭」：高速前進但方向不對
    high_speed_mask = speed_x_forward > high_speed_threshold
    misaligned_mask = misalignment > 0.3  # 約 72° 以上視為未對準
    penalty_mask = high_speed_mask & misaligned_mask
    
    # 計算懲罰：未對準程度 × 速度超出閾值的部分 × 懲罰係數
    speed_excess = torch.clamp(speed_x_forward - high_speed_threshold, min=0.0)
    misalignment_penalty = misalignment * speed_excess * misalignment_penalty_scale
    
    # 只在懲罰條件下應用懲罰
    penalty = torch.where(
        penalty_mask,
        misalignment_penalty,
        torch.zeros_like(misalignment_penalty)
    )
    
    # 總獎勵：對齊獎勵 - 未對準懲罰
    reward = alignment_reward_base - penalty
    
    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-10.0)
    reward = torch.clamp(reward, -10.0, 1.0)  # 限制範圍
    
    # 檢查 reward term
    reward = _check_reward_term("alignment_reward", reward, env, raise_on_error=True)
    
    return reward


def velocity_toward_goal_distance_weighted(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_dist: float = 0.5,
    close_distance: float = 2.0,
    far_distance: float = 5.0,
) -> torch.Tensor:
    """朝向目標的速度獎勵（根據距離動態調整）
    
    根據距離目標的遠近，動態調整速度獎勵：
    - 接近目標時（< close_distance）：降低速度獎勵權重（優先朝向）
    - 遠離目標時（> far_distance）：提高速度獎勵權重（快速接近）
    - 中間距離：線性插值
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        min_dist: 最小距離（米），低於此距離不給速度獎勵
        close_distance: 接近距離閾值（米）
        far_distance: 遠距離閾值（米）
    
    Returns:
        shape [num_envs]：加權後的速度獎勵
    """
    # 獲取原始速度獎勵
    base_reward = velocity_toward_goal(env, asset_cfg, min_dist=min_dist)
    
    # 獲取距離
    asset: Articulation = env.scene[asset_cfg.name]
    # 🔥 關鍵修復：優先使用規劃器的局部目標
    if hasattr(env, '_local_goal_world'):
        goal_pos_w = env._local_goal_world
    else:
        goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]
    goal_distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
    
    # 計算距離權重（線性插值）
    # 距離 < close_distance → 權重 = 0.3（降低速度獎勵，優先朝向）
    # 距離 > far_distance → 權重 = 1.0（提高速度獎勵，快速接近）
    # 中間距離 → 線性插值
    weight = torch.clamp(
        (goal_distance - close_distance) / (far_distance - close_distance + 1e-6),
        min=0.3,
        max=1.0
    )
    
    # 應用權重
    weighted_reward = base_reward * weight
    
    # 安全處理
    weighted_reward = torch.nan_to_num(weighted_reward, nan=0.0, posinf=10.0, neginf=0.0)
    weighted_reward = torch.clamp(weighted_reward, 0.0, 10.0)
    
    # 檢查 reward term
    weighted_reward = _check_reward_term("velocity_toward_goal_distance_weighted", weighted_reward, env, raise_on_error=True)
    
    return weighted_reward


def reverse_toward_goal_distance_weighted(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_dist: float = 0.28,
    close_distance: float = 2.0,
    far_distance: float = 5.0,
) -> torch.Tensor:
    """倒車靠近目標的速度獎勵（根據距離動態調整）
    
    當機器人「背對」目標時，鼓勵用倒車靠近目標。
    - 背對目標：朝向角度 > 90°（dot < 0）
    - 只獎勵倒車方向的速度分量
    - 接近目標時權重較高（避免繞圈）
    """
    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    # 🔥 關鍵修復：優先使用規劃器的局部目標
    if hasattr(env, '_local_goal_world'):
        goal_pos_w = env._local_goal_world
    else:
        goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_quat_w = asset.data.root_quat_w
    robot_vel_w = asset.data.root_lin_vel_w[:, :2]
    
    # 目標方向（世界座標系）
    goal_direction = goal_pos_w[:, :2] - robot_pos_w[:, :2]
    goal_distance = torch.norm(goal_direction, dim=1)
    goal_direction = goal_direction / (goal_distance.unsqueeze(1) + 1e-6)
    
    # 機器人前方向量（世界座標系）
    num_envs = robot_quat_w.shape[0]
    forward_vec = torch.zeros(num_envs, 3, device=env.device)
    forward_vec[:, 0] = 1.0
    forward_w = math_utils.quat_apply(robot_quat_w, forward_vec)
    
    # 朝向判斷（dot < 0 表示背對目標）
    heading_dot = torch.sum(forward_w[:, :2] * goal_direction, dim=1)
    is_facing_away = heading_dot < 0.0
    
    # 速度在目標方向的投影（正值=前進，負值=倒車靠近）
    velocity_projection = torch.sum(robot_vel_w * goal_direction, dim=1)
    reverse_speed = torch.clamp(-velocity_projection, min=0.0)
    
    # 距離閘門（避免已到達仍拿獎勵）
    reverse_speed = torch.where(
        goal_distance > min_dist,
        reverse_speed,
        torch.zeros_like(reverse_speed),
    )
    
    # 只有背對目標時才啟用倒車獎勵
    reverse_speed = torch.where(
        is_facing_away,
        reverse_speed,
        torch.zeros_like(reverse_speed),
    )
    
    # 距離權重（接近目標更重要）
    weight = torch.clamp(
        (far_distance - goal_distance) / (far_distance - close_distance + 1e-6),
        min=0.2,
        max=1.0,
    )
    weighted_reward = reverse_speed * weight
    
    # 安全處理
    weighted_reward = torch.nan_to_num(weighted_reward, nan=0.0, posinf=10.0, neginf=0.0)
    weighted_reward = torch.clamp(weighted_reward, 0.0, 10.0)
    
    # 檢查 reward term
    weighted_reward = _check_reward_term("reverse_toward_goal_distance_weighted", weighted_reward, env, raise_on_error=True)
    
    return weighted_reward


def velocity_toward_goal_dynamic_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_dist: float = 0.5,
    safe_clearance: float = 1.5,
) -> torch.Tensor:
    """動態門控速度獎勵：根據障礙物距離調整獎勵（Phase 2 專屬）
    
    核心邏輯：
    - 障礙物 > safe_clearance → 正常速度獎勵（權重 1.0）
    - 障礙物 < safe_clearance → 降低速度獎勵（權重 0.2-1.0）
    - 這會讓 Agent 在接近障礙物時「自動減速」
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        sensor_cfg: 傳感器配置（引用雷達）
        min_dist: 最小距離（米），低於此距離不給獎勵
        safe_clearance: 安全距離（米），低於此距離降低速度獎勵
    
    Returns:
        shape [num_envs]：加權後的速度獎勵
    """
    # 獲取基礎速度獎勵
    base_reward = velocity_toward_goal(env, asset_cfg, min_dist=min_dist)
    
    # 獲取最近障礙物距離
    from isaaclab.sensors import RayCaster
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)
    min_distance = torch.min(distances_2d, dim=1)[0]
    
    # 計算速度權重（障礙物越近權重越低）
    # > safe_clearance: 權重 1.0
    # < safe_clearance: 權重線性衰減到 0.2
    weight = torch.clamp(
        (min_distance - 0.5) / (safe_clearance - 0.5 + 1e-6),
        min=0.2,
        max=1.0
    )
    
    # 應用權重
    weighted_reward = base_reward * weight
    
    # 安全處理
    weighted_reward = torch.nan_to_num(weighted_reward, nan=0.0, posinf=10.0, neginf=0.0)
    weighted_reward = torch.clamp(weighted_reward, 0.0, 10.0)
    weighted_reward = _check_reward_term("velocity_toward_goal_dynamic_gated", weighted_reward, env, raise_on_error=True)
    
    return weighted_reward
