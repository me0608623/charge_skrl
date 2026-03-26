"""安全相關獎勵/偵測函數

VLP16 訓練使用的函數:
    collision_occurred: LiDAR 碰撞偵測 (用於 TerminationsCfg)
        → 判定 min(LiDAR 2D 距離) < threshold
    collision_contact_occurred: PhysX 接觸感測器碰撞偵測

參數:
    sensor_cfg: SceneEntityCfg — LiDAR 感測器配置 ("lidar")
    threshold: float = 1.0    — 碰撞距離閾值 [m] (ROBOT_BODY_RADIUS + COLLISION_BUFFER)

碰撞偵測原理:
    1. 取 LiDAR 所有射線的 hit_points_w (世界座標)
    2. 計算 2D 歐式距離 (忽略 Z 軸高度差)
    3. 取 min → d_min
    4. d_min <= threshold → 判定碰撞

注意: collision_occurred 是終止條件用的函數，不是獎勵函數。
      碰撞的獎勵懲罰由 potential_based_rewards.collision_terminal_penalty 處理。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster, ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import _check_reward_term


def obstacle_avoidance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    safe_distance: float = 1.0,
    collision_distance: float = 0.3,
) -> torch.Tensor:
    """避障獎勵（未使用）
    
    這個函數沒有在 charge_env_cfg.py 中使用，但保留以供參考。
    
    根據最近障礙物的距離給予獎勵：
    - 距離 > safe_distance：獎勵 +1
    - 距離 < collision_distance：懲罰 -1
    - 中間：線性插值
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    distances = sensor.data.ray_hits_w[..., 0]
    min_distance = torch.min(distances, dim=1)[0]
    # 找出 360 條射線中最近的障礙物
    
    reward = torch.clamp(
        (min_distance - collision_distance) / (safe_distance - collision_distance),
        min=-1.0,
        max=1.0,
    )
    # 線性映射：[0.3, 1.0] → [-1, 1]
    return reward


def collision_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """碰撞懲罰（二元懲罰）
    
    當機器人過於接近障礙物時給予懲罰。
    這是一個二元懲罰：要麼發生碰撞風險（1.0），要麼沒有（0.0）。
    
    注意：使用 2D 平面距離（忽略高度），與 collision_occurred 保持一致。
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用雷達）
        threshold: 碰撞閾值（米），低於此距離算「碰撞」（使用 2D 平面距離）
    
    Returns:
        shape [num_envs]：0 或 1
        1 = 發生碰撞（懲罰！）
        0 = 安全
    """
    # 獲取雷達數據
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取感測器位置和射線碰撞點
    sensor_pos = sensor.data.pos_w  # [num_envs, 3] 感測器位置（世界座標系）
    hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3] 射線碰撞點（世界座標系）
    
    # ========================================================================
    # 使用 2D 平面距離（忽略高度 Z），與 collision_occurred 保持一致
    # ========================================================================
    sensor_pos_2d = sensor_pos[:, :2]  # [num_envs, 2] 只取 X, Y 座標
    hit_points_2d = hit_points[:, :, :2]  # [num_envs, num_rays, 2] 只取 X, Y 座標
    
    # 計算 2D 平面距離（從雷達投影到地面的點到障礙物的水平距離）
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    # shape: [num_envs, num_rays]
    
    # 處理無效碰撞（射線沒打到任何東西，距離為 inf）
    # 將 inf 替換為 max_distance，這樣不會誤判為碰撞
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)
    
    # 找出每個環境中最近的障礙物距離（2D 平面距離）
    min_distance = torch.min(distances_2d, dim=1)[0]  # [num_envs]
    # torch.min(tensor, dim=1) 返回 (values, indices)
    # [0] 取 values（最小距離）
    
    # 安全處理：確保距離在合理範圍
    min_distance = torch.clamp(min_distance, 0.0, sensor.cfg.max_distance)
    
    # 判斷是否碰撞：最近距離 < 閾值
    reward = (min_distance < threshold).float()
    # min_distance < 0.6 米 → 1.0（碰撞）
    # 配合 weight=-200，變成 -200 分懲罰（極重懲罰！）
    
    # 檢查 reward term（防止 PPO std>=0 錯誤）
    reward = _check_reward_term("collision_penalty", reward, env, raise_on_error=True)
    
    return reward


def progressive_collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = None,
    safe_distance: float = 1.0,
    danger_distance: float = 0.75,
    collision_distance: float = 0.5,
    use_directional_penalty: bool = True,
    use_nonlinear_penalty: bool = True,
) -> torch.Tensor:
    """漸進式碰撞懲罰（改進版：方向性資訊 + 非線性懲罰）
    
    核心改進：
    1. 方向性懲罰：只有「朝障礙物前進」才重罰
       - 距離近 + 朝向障礙物的速度大 → 懲罰急遽增加
       - 距離近但側向移動 → 懲罰變小
    2. 非線性懲罰：在危險區域（0.6m 以內）使用平方函數
       - 讓 0.6m 以內「非常不舒服」
       - 直接瓦解「一頭撞上去」策略
    
    根據機器人與障礙物的距離給予漸進式懲罰：
    - 距離 >= safe_distance：無懲罰（0.0）
    - safe_distance > 距離 >= danger_distance：輕微懲罰（0.0 到 0.5）
    - danger_distance > 距離 >= collision_distance：中等懲罰（0.5 到 1.0，非線性）
    - 距離 < collision_distance：最大懲罰（1.0，非線性）
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用雷達）
        asset_cfg: 資產配置（引用機器人，用於計算速度方向）
        safe_distance: 安全距離（米），超過此距離無懲罰
        danger_distance: 危險距離（米），低於此距離開始中等懲罰
        collision_distance: 碰撞距離（米），低於此距離最大懲罰
        use_directional_penalty: 是否使用方向性懲罰（默認 True）
        use_nonlinear_penalty: 是否使用非線性懲罰（默認 True）
    
    Returns:
        shape [num_envs]：懲罰值 [0, 1+]
        0 = 安全距離，無懲罰
        >1 = 朝障礙物高速前進時，懲罰會超過 1.0
    """
    # 獲取雷達數據
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取感測器位置和射線碰撞點
    sensor_pos = sensor.data.pos_w  # [num_envs, 3]
    hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3]
    
    # 使用 2D 平面距離
    sensor_pos_2d = sensor_pos[:, :2]
    hit_points_2d = hit_points[:, :, :2]
    
    # 計算 2D 平面距離
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)
    
    # 找出最近距離和對應的障礙物位置
    min_distances, min_indices = torch.min(distances_2d, dim=1)  # [num_envs]
    min_distance = torch.clamp(min_distances, 0.0, sensor.cfg.max_distance)
    
    # 獲取最近障礙物的位置（用於計算方向）
    batch_indices = torch.arange(len(min_indices), device=env.device)
    nearest_hit_points = hit_points_2d[batch_indices, min_indices]  # [num_envs, 2]
    
    # ========================================================================
    # 計算基礎距離懲罰（非線性版本）
    # ========================================================================
    penalty = torch.zeros_like(min_distance)
    
    if use_nonlinear_penalty:
        # 非線性懲罰：使用平方函數，讓危險區域「非常不舒服」
        # 區域 1：danger_distance 到 collision_distance（中等懲罰，平方）
        mask1 = (min_distance < danger_distance) & (min_distance >= collision_distance)
        if mask1.any():
            # 線性插值後平方：讓懲罰曲線更陡峭
            linear_penalty = 0.5 + 0.5 * (1.0 - (min_distance[mask1] - collision_distance) / (danger_distance - collision_distance + 1e-6))
            penalty[mask1] = linear_penalty ** 2  # 平方：0.5² 到 1.0²
        
        # 區域 2：safe_distance 到 danger_distance（輕微懲罰，保持線性）
        mask2 = (min_distance < safe_distance) & (min_distance >= danger_distance)
        if mask2.any():
            penalty[mask2] = 0.5 * (1.0 - (min_distance[mask2] - danger_distance) / (safe_distance - danger_distance + 1e-6))
        
        # 區域 3：距離 < collision_distance（最大懲罰，平方 + 反比）
        mask3 = min_distance < collision_distance
        if mask3.any():
            # 使用反比函數：距離越近，懲罰急遽增加
            # penalty = 1.0 + (collision_distance / distance) - 1.0
            # 當 distance = collision_distance 時，penalty = 1.0
            # 當 distance → 0 時，penalty → ∞（但會被 clamp）
            inverse_penalty = 1.0 + (collision_distance / (min_distance[mask3] + 1e-6) - 1.0)
            penalty[mask3] = torch.clamp(inverse_penalty, 1.0, 5.0)  # 限制最大懲罰為 5.0
    else:
        # 原始線性版本（向後兼容）
        mask1 = (min_distance < danger_distance) & (min_distance >= collision_distance)
        if mask1.any():
            penalty[mask1] = 0.5 + 0.5 * (1.0 - (min_distance[mask1] - collision_distance) / (danger_distance - collision_distance + 1e-6))
        
        mask2 = (min_distance < safe_distance) & (min_distance >= danger_distance)
        if mask2.any():
            penalty[mask2] = 0.5 * (1.0 - (min_distance[mask2] - danger_distance) / (safe_distance - danger_distance + 1e-6))
        
        mask3 = min_distance < collision_distance
        if mask3.any():
            penalty[mask3] = 1.0
    
    # ========================================================================
    # 方向性懲罰：只有「朝障礙物前進」才重罰
    # ========================================================================
    if use_directional_penalty and asset_cfg is not None:
        # 獲取機器人速度和位置
        asset: Articulation = env.scene[asset_cfg.name]
        robot_pos_w = asset.data.root_pos_w[:, :2]  # [num_envs, 2]
        robot_vel_w = asset.data.root_lin_vel_w[:, :2]  # [num_envs, 2] 機器人速度（世界座標）
        
        # 計算從機器人到最近障礙物（射線碰撞點）的方向向量
        # 注意：使用射線碰撞點代表障礙物位置（最小改動原則）
        obstacle_direction = nearest_hit_points - robot_pos_w  # [num_envs, 2]
        obstacle_distance = torch.norm(obstacle_direction, dim=1)  # [num_envs]
        obstacle_direction_unit = obstacle_direction / (obstacle_distance.unsqueeze(1) + 1e-6)  # 歸一化
        
        # 計算機器人速度在「朝向障礙物」方向上的投影
        # 正值 = 朝障礙物前進，負值 = 遠離障礙物
        velocity_toward_obstacle = torch.sum(robot_vel_w * obstacle_direction_unit, dim=1)  # [num_envs]
        velocity_toward_obstacle = torch.clamp(velocity_toward_obstacle, min=0.0)  # 只考慮朝障礙物前進的速度
        
        # 計算方向性懲罰係數（非線性：平方函數）
        # 速度越大，懲罰係數越大（最大 3.0 倍，使用平方讓效果更明顯）
        # 側向移動（速度小）時，懲罰係數接近 1.0
        max_speed = 1.5  # 最大速度（m/s）
        speed_ratio = velocity_toward_obstacle / (max_speed + 1e-6)  # [0, 1]
        # 使用平方函數：讓高速朝障礙物前進的懲罰急遽增加
        directional_multiplier = 1.0 + 2.0 * (speed_ratio ** 2)  # [1.0, 3.0]
        
        # 只在危險區域應用方向性懲罰（距離 < danger_distance）
        danger_mask = min_distance < danger_distance
        if danger_mask.any():
            penalty[danger_mask] = penalty[danger_mask] * directional_multiplier[danger_mask]
    
    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=5.0, neginf=0.0)
    penalty = torch.clamp(penalty, 0.0, 5.0)  # 限制最大懲罰為 5.0
    
    # 檢查 reward term
    penalty = _check_reward_term("progressive_collision_penalty", penalty, env, raise_on_error=True)
    
    return penalty


def safe_navigation_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    comfort_distance: float = 1.5,
) -> torch.Tensor:
    """安全導航獎勵：獎勵與障礙物保持安全距離（Phase 2 專屬）
    
    設計理念：
    - 最近障礙物 > comfort_distance → 獎勵 +1.0
    - 最近障礙物 < comfort_distance → 線性衰減到 0
    - 這會讓 Agent 「主動繞路」而非「擦邊通過」
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        sensor_cfg: 傳感器配置（引用雷達）
        comfort_distance: 舒適距離（米），超過此距離給滿分獎勵
    
    Returns:
        shape [num_envs]：獎勵值 [0, 1]
    """
    # 獲取雷達數據
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取最近障礙物距離（2D）
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)
    min_distance = torch.min(distances_2d, dim=1)[0]
    
    # 計算獎勵（線性衰減）
    reward = torch.clamp(min_distance / comfort_distance, 0.0, 1.0)
    
    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    reward = torch.clamp(reward, 0.0, 1.0)
    reward = _check_reward_term("safe_navigation_bonus", reward, env, raise_on_error=True)
    
    return reward


def speed_control_near_obstacles(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    warning_distance: float = 1.5,
    max_safe_speed: float = 0.5,
) -> torch.Tensor:
    """接近障礙物時的速度控制獎勵
    
    設計理念：
    - 當機器人接近障礙物時（距離 < warning_distance），鼓勵降低速度
    - 距離越近，速度應該越慢
    - 如果速度超過 max_safe_speed，給予懲罰
    - 如果速度低於 max_safe_speed，給予獎勵
    
    這會讓機器人在接近障礙物時主動減速，提高安全性。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        sensor_cfg: 傳感器配置（引用雷達）
        warning_distance: 警告距離（米），低於此距離開始速度控制
        max_safe_speed: 最大安全速度（m/s），接近障礙物時不應超過此速度
    
    Returns:
        shape [num_envs]：獎勵值 [-1, 1]
        - 接近障礙物且速度過快 → 負值（懲罰）
        - 接近障礙物且速度適中 → 正值（獎勵）
        - 遠離障礙物 → 0（無影響）
    """
    # 獲取機器人速度（機器人座標系）
    robot: Articulation = env.scene[asset_cfg.name]
    robot_vel_b = robot.data.root_lin_vel_b  # [num_envs, 3]
    robot_speed = torch.norm(robot_vel_b[:, :2], dim=1)  # [num_envs] 2D 速度大小
    
    # 獲取雷達數據
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取最近障礙物距離（2D）
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)
    min_distance = torch.min(distances_2d, dim=1)[0]  # [num_envs]
    
    # 計算速度控制獎勵
    # 只在接近障礙物時（距離 < warning_distance）才生效
    close_to_obstacle = min_distance < warning_distance
    
    # 計算速度偏差（相對於最大安全速度）
    speed_excess = robot_speed - max_safe_speed  # [num_envs]
    
    # 計算距離權重（距離越近，影響越大）
    # 距離 = warning_distance → 權重 = 0
    # 距離 = 0 → 權重 = 1
    distance_weight = torch.clamp(
        1.0 - (min_distance / warning_distance),
        0.0, 1.0
    )  # [num_envs]
    
    # 計算獎勵
    # 如果速度超過安全速度 → 懲罰（負值）
    # 如果速度低於安全速度 → 獎勵（正值）
    # 獎勵值 = -speed_excess * distance_weight
    # 例如：速度 = 1.0 m/s, max_safe_speed = 0.5 m/s
    #      speed_excess = 0.5
    #      距離 = 1.0 米, warning_distance = 2.0 米
    #      distance_weight = 0.5
    #      獎勵 = -0.5 * 0.5 = -0.25（懲罰）
    reward = -speed_excess * distance_weight
    
    # 只在接近障礙物時生效
    reward = torch.where(close_to_obstacle, reward, torch.zeros_like(reward))
    
    # 歸一化到 [-1, 1] 範圍
    # 假設最大速度約為 1.5 m/s，則最大 speed_excess = 1.0
    # 所以 reward 範圍約為 [-1.0, 1.0]
    reward = torch.clamp(reward, -1.0, 1.0)
    
    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    reward = _check_reward_term("speed_control_near_obstacles", reward, env, raise_on_error=True)
    
    return reward


def collision_occurred(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """終止條件：發生碰撞
    
    當機器人撞到障礙物時終止 episode（失敗）。
    
    注意：使用 2D 平面距離（忽略高度），因為：
    1. 雷達安裝在機器人上方 0.5 米處
    2. 使用 3D 距離會導致距離偏大（考慮了高度差）
    3. 2D 距離更能反映機器人底盤與障礙物的實際距離
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用雷達）
        threshold: 碰撞閾值（米），使用 2D 平面距離
    
    Returns:
        shape [num_envs]：布林張量
        True = 碰撞，終止
        False = 安全，繼續
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取感測器位置和射線碰撞點
    sensor_pos = sensor.data.pos_w  # [num_envs, 3] 感測器位置（世界座標系）
    hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3] 射線碰撞點（世界座標系）
    
    # ========================================================================
    # 使用 2D 平面距離（忽略高度 Z），因為：
    # 1. 雷達安裝在機器人上方 0.5 米處
    # 2. 我們關心的是機器人底盤與障礙物的水平距離
    # 3. 使用 3D 距離會因為高度差導致距離偏大
    # ========================================================================
    sensor_pos_2d = sensor_pos[:, :2]  # [num_envs, 2] 只取 X, Y 座標
    hit_points_2d = hit_points[:, :, :2]  # [num_envs, num_rays, 2] 只取 X, Y 座標
    
    # 計算 2D 平面距離（從雷達投影到地面的點到障礙物的水平距離）
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    # shape: [num_envs, num_rays]
    
    # 處理無效碰撞（射線沒打到任何東西，距離為 inf）
    # 將 inf 替換為 max_distance，這樣不會誤判為碰撞
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)
    
    # 找出每個環境中最近的障礙物距離（2D 平面距離）
    min_distance = torch.min(distances_2d, dim=1)[0]  # [num_envs]
    
    # 判斷是否碰撞：最近距離 < 閾值
    return min_distance < threshold
    # 最近障礙物 < threshold 米 → True（碰撞，終止）


def collision_contact_occurred(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """終止條件：接觸感測器檢測到碰撞
    
    當機器人的碰撞框與障礙物的碰撞框接觸時終止 episode（失敗）。
    這比雷達距離判定更準確，因為它使用物理引擎的真實碰撞檢測。
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用接觸感測器）
    
    Returns:
        shape [num_envs]：布林張量
        True = 檢測到碰撞接觸，終止
        False = 無碰撞，繼續
    
    優勢：
    - 使用真實的碰撞框（物理引擎的碰撞檢測）
    - 比雷達距離判定更準確（不會因為雷達安裝位置產生誤差）
    - 可以檢測機器人任何部件與障礙物的碰撞（不只是底盤）
    
    注意：
    - 接觸感測器必須在場景配置中啟用
    - 必須設置 filter_prim_paths_expr 來過濾障礙物
    """
    # 獲取接觸感測器
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 獲取接觸力數據
    # force_matrix_w: shape [num_envs, num_bodies, num_filtered_objects, 3]
    # - num_envs: 環境數量
    # - num_bodies: 每個感測器的機器人部件數量（例如：底盤、輪子等）
    # - num_filtered_objects: 過濾的物體數量（障礙物數量）
    # - 3: 力的 XYZ 分量（世界座標系）
    contact_forces = sensor.data.force_matrix_w
    
    # 如果沒有設置過濾器，force_matrix_w 會是 None
    if contact_forces is None:
        # 沒有過濾器：無法檢測與障礙物的碰撞
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    
    # 計算每個環境的總接觸力大小
    # 對所有機器人部件和所有障礙物的接觸力求和
    total_force = torch.sum(contact_forces, dim=(1, 2))  # [num_envs, 3]：所有部件與所有障礙物的總接觸力
    force_magnitude = torch.norm(total_force, dim=1)  # [num_envs]：總接觸力的大小
    
    # 判斷是否碰撞：接觸力 > 0 表示有接觸
    # 使用一個小的閾值（0.1 N）來避免數值誤差
    collision_threshold = 0.1  # 接觸力閾值：0.1 牛頓
    return force_magnitude > collision_threshold
    # 接觸力 > 0.1 N → True（碰撞，終止）


def obstacle_proximity_termination(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.45,
    max_obstacles: int = 100,
) -> torch.Tensor:
    """終止條件：robot center 距 obstacle center 過近（直接位置計算）

    不依賴 LiDAR 射線或 PhysX 接觸力，直接比較 XY 平面距離。
    對 kinematic obstacles 和任何高度的障礙物都有效。

    碰撞判定: dist(robot, obstacle) < threshold + obstacle_radius
    threshold=0.45m = robot_body_radius(0.35) + buffer(0.10)
    """
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]  # [N, 2]
    collision = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    obs_sizes = getattr(env, "_obstacle_sizes", None)

    for i in range(max_obstacles):
        obs_name = f"obstacle_{i}"
        if obs_name not in env.scene.keys():
            continue
        obs_entity = env.scene[obs_name]
        obs_pos = obs_entity.data.root_pos_w  # [N, 3]
        visible = obs_pos[:, 2] > 0.0  # Z > 0 = 可見
        if not visible.any():
            continue
        dist = torch.norm(robot_pos - obs_pos[:, :2], dim=1)  # [N]
        obs_r = obs_sizes[i] / 2.0 if obs_sizes is not None and i < len(obs_sizes) else 0.3
        collision = collision | (visible & (dist < threshold + obs_r))

    return collision


def wall_collision_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    boundary: float = 5.0,
    robot_radius: float = 0.28,
    safe_distance: float = 1.0,
) -> torch.Tensor:
    """漸進式牆壁碰撞懲罰：距離牆壁越近，懲罰越大
    
    根據機器人與場景邊界（牆壁）的距離給予漸進式懲罰。
    這會鼓勵機器人遠離牆壁，避免撞牆。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        boundary: 邊界距離（米），默認 5.0 米
        robot_radius: 機器人半徑（米），默認 0.28 米
        safe_distance: 安全距離（米），低於此距離開始懲罰
    
    Returns:
        shape [num_envs]：懲罰值 [0, 1]
        0 = 遠離牆壁，無懲罰
        1 = 撞牆，最大懲罰
    
    設計理念：
    - 距離 >= safe_distance：無懲罰（0.0）
    - 距離 < safe_distance：線性增加懲罰
    - 距離 <= 0：最大懲罰（1.0，撞牆）
    
    注意：
    - 使用 2D 平面距離（忽略高度）
    - 考慮機器人半徑，避免機器人邊緣觸牆時才開始懲罰
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 獲取機器人位置（世界座標）
    robot_pos_w = asset.data.root_pos_w[:, :2]  # [num_envs, 2]
    
    # 獲取環境原點（用於計算相對位置）
    env_origins = env.scene.env_origins[:, :2]  # [num_envs, 2]
    
    # 計算相對於環境原點的位置
    robot_pos_local = robot_pos_w - env_origins  # [num_envs, 2]
    
    # 計算到各牆壁的距離
    # 東西牆（X 方向）
    dist_to_east_wall = boundary - robot_pos_local[:, 0]  # 正值 = 在邊界內
    dist_to_west_wall = boundary + robot_pos_local[:, 0]  # 正值 = 在邊界內
    # 南北牆（Y 方向）
    dist_to_north_wall = boundary - robot_pos_local[:, 1]
    dist_to_south_wall = boundary + robot_pos_local[:, 1]
    
    # 取最小距離（減去機器人半徑）
    min_dist_x = torch.min(dist_to_east_wall, dist_to_west_wall) - robot_radius
    min_dist_y = torch.min(dist_to_north_wall, dist_to_south_wall) - robot_radius
    min_wall_dist = torch.min(min_dist_x, min_dist_y)
    
    # 計算懲罰（距離越近懲罰越大）
    # 距離 >= safe_distance → 懲罰 0
    # 距離 <= 0 → 懲罰 1（撞牆）
    # 中間線性插值
    penalty = torch.clamp(1.0 - min_wall_dist / safe_distance, 0.0, 1.0)
    
    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=1.0, neginf=0.0)
    penalty = _check_reward_term("wall_collision_penalty", penalty, env, raise_on_error=True)

    return penalty


def log_distance_safety_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    top_k: int = 5,
    max_obstacles: int = 10,
    max_distance: float = 8.0,
) -> torch.Tensor:
    """NavRL 對數距離安全獎勵

    公式: reward = mean(log(clamp(distance - radius/2, 1e-6, max_dist))) over Top-K

    特性:
    - 遠離障礙物: log(大值) -> 正獎勵
    - 靠近障礙物: log(小值) -> 強負信號（平滑梯度，不是懸崖）
    - 只計算可見障礙物（Z > 0）
    - 無可見障礙物的環境返回 0

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        top_k: 選取最近的 K 個障礙物計算獎勵
        max_obstacles: 場景中最大障礙物數量
        max_distance: 最大距離（用於 clamp 上界）

    Returns:
        [num_envs] 獎勵值
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    robot_pos_xy = robot.data.root_pos_w[:, :2]  # [num_envs, 2]

    # 收集障礙物位置
    all_pos = torch.zeros(num_envs, max_obstacles, 3, device=device)
    all_size = torch.zeros(num_envs, max_obstacles, device=device)

    num_found = 0
    for i in range(max_obstacles):
        obstacle_name = f"obstacle_{i}"
        # 🔥 修正：InteractiveScene 只支援 scene["name"]，不支援 hasattr/getattr
        if obstacle_name not in env.scene.keys():
            continue
        obstacle = env.scene[obstacle_name]
        all_pos[:, num_found, :] = obstacle.data.root_pos_w[:, :3]
        num_found += 1

    if num_found == 0:
        return torch.zeros(num_envs, device=device)

    all_pos = all_pos[:, :num_found, :]
    all_size = all_size[:, :num_found]

    # 讀取大小
    if hasattr(env, "_obstacle_sizes") and env._obstacle_sizes is not None:
        sizes_raw = env._obstacle_sizes  # list[float] 或 Tensor
        sizes = torch.as_tensor(sizes_raw, device=device, dtype=torch.float32)
        n_copy = min(num_found, len(sizes))
        all_size[:, :n_copy] = sizes[:n_copy].unsqueeze(0).expand(num_envs, -1)

    # 可見性（Z > 0）
    visible = all_pos[:, :, 2] > 0.0  # [num_envs, num_found]

    # 距離計算
    rel_pos = all_pos[:, :, :2] - robot_pos_xy.unsqueeze(1)  # [num_envs, num_found, 2]
    distances = torch.norm(rel_pos, dim=2)  # [num_envs, num_found]

    # 不可見的設為大值
    large_val = max_distance * 10.0
    masked_distances = torch.where(visible, distances, torch.full_like(distances, large_val))

    # Top-K 最近
    actual_k = min(top_k, num_found)
    topk_dist, topk_idx = torch.topk(masked_distances, k=actual_k, dim=1, largest=False)
    topk_valid = topk_dist < large_val  # [num_envs, actual_k]

    # 收集 Top-K 的半徑
    topk_radius = torch.gather(all_size, 1, topk_idx)  # [num_envs, actual_k]

    # 計算 log(distance - radius/2)
    effective_dist = topk_dist - topk_radius / 2.0
    effective_dist = effective_dist.clamp(min=1e-6, max=max_distance)
    log_dist = torch.log(effective_dist)  # [num_envs, actual_k]

    # 只對有效的（可見的）障礙物取平均
    # 將無效的 log_dist 設為 0 不參與平均
    log_dist = torch.where(topk_valid, log_dist, torch.zeros_like(log_dist))
    valid_count = topk_valid.float().sum(dim=1).clamp(min=1.0)  # [num_envs]
    reward = log_dist.sum(dim=1) / valid_count  # [num_envs]

    # 無可見障礙物的環境返回 0
    has_visible = topk_valid.any(dim=1)
    reward = torch.where(has_visible, reward, torch.zeros_like(reward))

    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-5.0)
    reward = reward.clamp(-5.0, 5.0)
    reward = _check_reward_term("log_distance_safety_reward", reward, env, raise_on_error=True)

    return reward
