"""環境重置事件 — 安全位置重置與障礙物管理

VLP16 訓練使用的函數:
    reset_root_state_random_safe: 安全隨機位置重置 (EventTerm mode="reset")

其他函數:
    hide_unused_obstacles: 隱藏多餘障礙物到 Z=-10
    reset_root_state_fixed_per_env: 固定位置重置 (debug 用)
    reset_obstacles_with_adaptive_params: 自適應課程重置 (已移至 temp/)

reset_root_state_random_safe 參數:
    pose_range: dict — 位置/角度採樣範圍
        x: (-5.0, 5.0), y: (-5.0, 5.0), yaw: (-π, π)
    velocity_range: dict — 初始速度範圍
        x: (-0.5, 0.5), y: (-0.15, 0.15)

安全機制:
    1. 隨機採樣位置 (x, y, yaw)
    2. check_wall_proximity_batch: 確保不在牆內或太近牆壁
    3. 障礙物距離檢查: 確保不與障礙物重疊
    4. 最多 50 次重試，失敗則使用原點 (0,0)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

from .obstacles import reset_obstacles

# 導入 Isaac Lab 的工具函數
import isaaclab.utils.math as math_utils

# 迷宮牆壁 proximity 檢查（原 core/wall_layout.py → mdp/wall_layout.py）
from ..wall_layout import (
    get_wall_tensors,
    check_wall_proximity_batch,
    check_wall_proximity_perenv,
    get_combined_wall_data,
)


def hide_unused_obstacles(env, env_ids=None):
    """隱藏未使用的障礙物（將它們移動到場景外）
    
    此函數可以在 interval 事件中調用，確保障礙物始終在正確位置。
    使用了「強制隱藏」機制，不依賴讀取當前座標。
    
    Args:
        env: 環境實例
        env_ids: 環境 ID 列表，如果為 None 則處理所有環境
    """
    # 調試：確認函數被調用
    if not hasattr(env, "_hide_call_count"):
        env._hide_call_count = 0
        print(f"[DEBUG hide_unused_obstacles] 函數首次被調用")
    env._hide_call_count += 1
    
    # ⚠️ 確保課程學習已初始化（優先設置 env._num_obstacles）
    if not hasattr(env, "_num_obstacles") or env._num_obstacles is None:
        # 課程學習已移至 temp/，VLP16 不使用此路徑
        env._num_obstacles = 3  # 預設值
    
    adaptive_num_obstacles = env._num_obstacles
    
    # 調試：顯示當前難度
    if env._hide_call_count <= 5 or env._hide_call_count % 100 == 0:
        print(f"[DEBUG hide_unused_obstacles] Call #{env._hide_call_count}, adaptive_num_obstacles = {adaptive_num_obstacles}")
    
    # 獲取總障礙物數量
    from .state import get_obstacle_sizes
    all_sizes = get_obstacle_sizes()
    max_obstacles = len(all_sizes) if all_sizes is not None else 10
    
    # 調試：檢查 max_obstacles 是否正確
    if not hasattr(env, "_logged_max_obstacles"):
        if all_sizes is not None:
            print(f"[DEBUG] all_sizes = {all_sizes}")
        print(f"[DEBUG] max_obstacles = {max_obstacles}, adaptive_num_obstacles = {adaptive_num_obstacles}")
        
        # Deep Introspection
        print("--- Scene Introspection ---")
        try:
            print(f"env.scene type: {type(env.scene)}")
            if hasattr(env.scene, "keys"):
                print(f"env.scene keys: {list(env.scene.keys())}")
            if hasattr(env.scene, "rigid_objects"):
                print(f"env.scene.rigid_objects keys: {list(env.scene.rigid_objects.keys())}")
            
            # Check for pattern matching of attributes
            all_attrs = dir(env.scene)
            matches = [a for a in all_attrs if "obstacle" in a.lower()]
            print(f"Attributes matching 'obstacle': {matches}")
        except Exception as e:
            print(f"Introspection failed: {e}")
        print("---------------------------")
        
        env._logged_max_obstacles = True
    
    # 如果所有障礙物都使用，不需要隱藏
    if adaptive_num_obstacles >= max_obstacles:
        return
    
    # 準備環境 ID 列表
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    
    # 準備「強制隱藏位姿」
    all_env_origins = env.scene.env_origins[all_env_ids]
    hidden_pose = torch.zeros((len(all_env_ids), 7), device=env.device)
    hidden_pose[:, :2] = all_env_origins[:, :2] # 設為環境中心
    hidden_pose[:, 2] = -16.0                  # 強制埋入地下 16 米
    hidden_pose[:, 3] = 1.0                    # 身份旋轉
    
    # 💡 關鍵新增：檢測難度變化，主動顯示/隱藏障礙物
    previous_difficulty = getattr(env, "_last_hide_difficulty", 3)
    difficulty_increased = adaptive_num_obstacles > previous_difficulty
    
    # 調試：顯示難度比較
    if env._hide_call_count <= 5:
        print(f"[DEBUG] previous_difficulty = {previous_difficulty}, adaptive_num_obstacles = {adaptive_num_obstacles}, difficulty_increased = {difficulty_increased}")
    
    # 如果難度增加，將新啟用的障礙物移回地面（隨機位置）
    if difficulty_increased:
        print(f"[Obstacle Show] 難度提升: {previous_difficulty} → {adaptive_num_obstacles} | 顯示新障礙物")
        # 為新啟用的障礙物生成隨機位置
        for i in range(previous_difficulty, adaptive_num_obstacles):
            obstacle_name = f"obstacle_{i}"
            # 修正：使用字典方式訪問場景實體
            if obstacle_name in env.scene.keys():
                print(f"[DEBUG] Showing obstacle {obstacle_name}")
                obstacle = env.scene[obstacle_name]
                # 生成隨機位置（在場景範圍內）
                boundary = 8.0  # 場景邊界
                # 安全邊距 = 用戶請求 (1.0m) + 障礙物半徑 (0.5m) = 1.5m
                # 這樣確保障礙物邊緣距離牆壁至少 1.0m
                safe_margin = 1.5
                spawn_range = boundary - safe_margin
                
                # 為所有環境生成隨機位置
                # 🛡️ 實施拒絕採樣：檢查與機器人的距離，避免貼臉生成
                pos = torch.zeros((len(all_env_ids), 3), device=env.device)
                pos[:, 2] = 0.5  # 地面高度
                
                # 獲取機器人位置
                robot_pos = None
                if "robot" in env.scene.keys():
                    robot_pos = env.scene["robot"].data.root_pos_w[all_env_ids]
                
                min_robot_dist = 2.5  # 最小安全距離（比一般的 1.5m 碰撞距離更保守）
                max_attempts = 10
                
                # 首次生成
                rand_x = torch.rand(len(all_env_ids), device=env.device) * (2 * spawn_range) - spawn_range
                rand_y = torch.rand(len(all_env_ids), device=env.device) * (2 * spawn_range) - spawn_range
                pos[:, 0] = all_env_origins[:, 0] + rand_x
                pos[:, 1] = all_env_origins[:, 1] + rand_y
                
                if robot_pos is not None:
                    # 循環檢查與重採樣
                    for attempt in range(max_attempts):
                        # 計算與機器人的距離 (XY 平面)
                        dist = torch.norm(pos[:, :2] - robot_pos[:, :2], dim=-1)
                        invalid_mask = dist < min_robot_dist
                        
                        if not invalid_mask.any():
                            break  # 所有位置都安全
                        
                        # 重新生成不安全的位置
                        num_invalid = invalid_mask.sum().item()
                        if attempt == max_attempts - 1:
                            # 最後一次嘗試若仍失敗，打印警告（但仍使用該位置，避免死循環）
                            if env._hide_call_count <= 20: # 防止日誌氾濫
                                print(f"[Adaptive Curriculum] Warning: {num_invalid} obstacles spawned close to robot after {max_attempts} attempts.")
                        else:
                            # 重採樣
                            new_rand_x = torch.rand(num_invalid, device=env.device) * (2 * spawn_range) - spawn_range
                            new_rand_y = torch.rand(num_invalid, device=env.device) * (2 * spawn_range) - spawn_range
                            
                            # 更新位置
                            pos[invalid_mask, 0] = all_env_origins[invalid_mask, 0] + new_rand_x
                            pos[invalid_mask, 1] = all_env_origins[invalid_mask, 1] + new_rand_y
                
                quat = torch.zeros((len(all_env_ids), 4), device=env.device)
                quat[:, 0] = 1.0  # 無旋轉
                
                obstacle.write_root_pose_to_sim(
                    torch.cat([pos, quat], dim=-1),
                    env_ids=all_env_ids
                )
    
    # 執行所有未使用項目的強制隱藏
    num_hidden = 0
    for i in range(adaptive_num_obstacles, max_obstacles):
        obstacle_name = f"obstacle_{i}"
        # 修正：使用字典方式訪問場景實體
        if obstacle_name in env.scene.keys():
            obstacle = env.scene[obstacle_name]
            obstacle.write_root_pose_to_sim(hidden_pose, env_ids=all_env_ids)
            num_hidden += 1
        else:
             # Only log once per obstacle
             if env._hide_call_count <= 5:
                 print(f"[DEBUG] Could not find {obstacle_name} to hide (keys: {list(env.scene.keys())})")
            
    # 更新記錄的難度
    env._last_hide_difficulty = adaptive_num_obstacles
    
    # 調試日誌（僅在難度變化或初始運行時輸出，避免洗屏）
    if not hasattr(env, "_last_logged_difficulty") or env._last_logged_difficulty != adaptive_num_obstacles:
        print(f"[Obstacle Hiding] 目標數量: {adaptive_num_obstacles} | 強制隱藏: {num_hidden} 個物體 (Z = -16m)")
        env._last_logged_difficulty = adaptive_num_obstacles


def reset_obstacles_with_adaptive_params(
    env,
    env_ids,
    speed_range: float = 0.0,
    min_speed: float = 0.0,
    min_robot_distance: float = 3.5,
    min_goal_distance: float = 2.0,
    min_obstacle_spacing: float = 2.0,
    max_spawn_attempts: int = 50,
    boundary: float = 8.0,
):
    """重置障礙物位置（Phase 2.5：使用自適應課程學習參數）
    
    此函數會根據自適應課程學習的結果動態調整參數。
    策略：場景中創建 MAX_OBSTACLES 個障礙物，但根據難度等級只啟用前 N 個。
    
    Args:
        env: 環境實例
        env_ids: 需要重置的環境 ID 列表
        speed_range: 速度範圍（靜態障礙物不使用）
        min_speed: 最小速度（靜態障礙物不使用）
        min_robot_distance: 機器人與障礙物最小距離（會被動態調整）
        min_goal_distance: 目標與障礙物最小距離（會被動態調整）
        min_obstacle_spacing: 障礙物之間最小間距（會被動態調整）
        max_spawn_attempts: 最大生成嘗試次數
        boundary: 場景邊界
    """
    # 初始化課程學習（如果尚未初始化）
    if not hasattr(env, "_adaptive_curriculum_params") or not env._adaptive_curriculum_params:
        # 課程學習已移至 temp/，VLP16 不使用此路徑
        env._num_obstacles = 3  # 預設值
    
    # 確保 env._num_obstacles 已設置（初始化時設置為 3）
    if not hasattr(env, "_num_obstacles") or env._num_obstacles is None:
        env._num_obstacles = 3  # 初始難度：3 個障礙物
    
    # 從環境中獲取課程學習參數（由 adaptive_curriculum_update 設置）
    if hasattr(env, "_adaptive_curriculum_params") and env._adaptive_curriculum_params:
        # 使用課程學習更新後的參數
        difficulty_params = env._adaptive_curriculum_params
        adaptive_min_robot_distance = difficulty_params.get("min_robot_distance", min_robot_distance)
        adaptive_min_goal_distance = difficulty_params.get("min_goal_distance", min_goal_distance)
        adaptive_min_obstacle_spacing = difficulty_params.get("min_obstacle_spacing", min_obstacle_spacing)
        adaptive_num_obstacles = difficulty_params.get("num_obstacles", env._num_obstacles)
    else:
        # 沒有課程學習參數時，使用初始參數
        adaptive_min_robot_distance = min_robot_distance
        adaptive_min_goal_distance = min_goal_distance
        adaptive_min_obstacle_spacing = min_obstacle_spacing
        adaptive_num_obstacles = env._num_obstacles  # 使用已設置的值（應該是 3）
    
    # 檢測障礙物數量是否發生變化（在更新之前讀取）
    previous_num_obstacles = getattr(env, "_num_obstacles", None)
    difficulty_changed = (previous_num_obstacles is not None and 
                          previous_num_obstacles != adaptive_num_obstacles)
    
    # 更新環境中的障礙物數量記錄
    env._num_obstacles = adaptive_num_obstacles
    
    # 準備環境 ID 列表（用於移動未使用的障礙物）
    if env_ids is None:
        env_ids_list = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, torch.Tensor):
        env_ids_list = env_ids
    else:
        env_ids_list = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    
    
    # ⚠️ 如果難度發生變化，重置所有環境的障礙物位置，確保新的配置生效
    curriculum_triggered_reset = getattr(env, "_curriculum_triggered_reset", False)
    if difficulty_changed or curriculum_triggered_reset:
        reset_env_ids = torch.arange(env.num_envs, device=env.device) # Use all_env_ids
        if curriculum_triggered_reset:
            env._curriculum_triggered_reset = False
        # 調試日誌
        if not hasattr(env, "_last_reset_difficulty") or env._last_reset_difficulty != adaptive_num_obstacles:
            print(f"[Adaptive Reset] 難度變化: {previous_num_obstacles} → {adaptive_num_obstacles} | 重置所有環境的障礙物")
            env._last_reset_difficulty = adaptive_num_obstacles
    else:
        reset_env_ids = env_ids_list
    
    # 執行使用的障礙物重置（1 ~ N）
    reset_obstacles(
        env=env,
        env_ids=reset_env_ids,
        speed_range=speed_range,
        min_speed=min_speed,
        min_robot_distance=adaptive_min_robot_distance,
        min_goal_distance=adaptive_min_goal_distance,
        min_obstacle_spacing=adaptive_min_obstacle_spacing,
        max_spawn_attempts=max_spawn_attempts,
        boundary=boundary,
    )
    
    # ⚠️ 執行無條件強制隱藏邏輯，確保未使用障礙物不在場景內
    # 💡 關鍵：必須在 reset_obstacles 之後調用，否則新啟用的障礙物會被誤隱藏
    hide_unused_obstacles(env)


def reset_root_state_fixed_per_env(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: "SceneEntityCfg" = None,
):
    """重置Agent到固定位置（每個環境編號保持固定）
    
    此函數確保：
    - 不同環境編號的Agent位置不同（隨機分配）
    - 相同環境編號的Agent位置在訓練過程中保持固定
    - 提升訓練穩定性和可重現性
    
    設計理念：
    為了讀取數值和訓練的方便性，在訓練的100次過程中固定事件的機率。
    即使每個RL環境中的Agent位置隨機分配，相同編號的環境內的Agent位置
    依然需要保持固定。設定RL環境中的Agent位置不一致，不僅提升訓練效果，
    還能增強訓練的穩定性。
    
    Args:
        env: 環境實例
        env_ids: 需要重置的環境ID列表
        pose_range: 位置和朝向範圍字典
            - "x", "y", "z": 位置範圍 (min, max)
            - "roll", "pitch", "yaw": 朝向範圍 (min, max)
        velocity_range: 速度範圍字典
            - "x", "y", "z": 線速度範圍 (min, max)
            - "roll", "pitch", "yaw": 角速度範圍 (min, max)
        asset_cfg: 資產配置（默認為 "robot"）
    """
    # 設置默認資產名稱
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg
        asset_cfg = SceneEntityCfg("robot")
    
    # 獲取資產
    asset = env.scene[asset_cfg.name]
    device = asset.device
    
    # 初始化固定位置存儲（如果尚未初始化）
    if not hasattr(env, "_fixed_robot_poses"):
        # 為所有環境預分配固定位置和朝向
        # shape: [num_envs, 7] (x, y, z, qw, qx, qy, qz)
        env._fixed_robot_poses = torch.zeros(env.num_envs, 7, device=device)
        # shape: [num_envs, 6] (vx, vy, vz, wx, wy, wz)
        env._fixed_robot_velocities = torch.zeros(env.num_envs, 6, device=device)
        # 標記哪些環境已經初始化
        env._fixed_robot_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    
    # 獲取默認根狀態
    root_states = asset.data.default_root_state[env_ids].clone()
    
    # 處理需要初始化的環境（第一次調用）
    uninitialized_mask = ~env._fixed_robot_initialized[env_ids]
    
    if uninitialized_mask.any():
        num_uninitialized = uninitialized_mask.sum().item()
        
        # 生成隨機位置和朝向
        # 位置範圍
        pos_range_list = [
            pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]
        ]
        pos_ranges = torch.tensor(pos_range_list, device=device)
        pos_samples = math_utils.sample_uniform(
            pos_ranges[:, 0], 
            pos_ranges[:, 1], 
            (num_uninitialized, 3), 
            device=device
        )
        
        # 朝向範圍
        ori_range_list = [
            pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]
        ]
        ori_ranges = torch.tensor(ori_range_list, device=device)
        ori_samples = math_utils.sample_uniform(
            ori_ranges[:, 0], 
            ori_ranges[:, 1], 
            (num_uninitialized, 3), 
            device=device
        )
        
        # 速度範圍
        vel_range_list = [
            velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        vel_ranges = torch.tensor(vel_range_list, device=device)
        vel_samples = math_utils.sample_uniform(
            vel_ranges[:, 0], 
            vel_ranges[:, 1], 
            (num_uninitialized, 6), 
            device=device
        )
        
        # ----------------------------------------------------------------
        # 牆壁 + 障礙物 proximity 拒絕採樣
        # agent 必須離牆壁和障礙物都至少 SPAWN_SAFE_DIST
        # ----------------------------------------------------------------
        ROBOT_RADIUS = 0.35
        SPAWN_SAFE_DIST = 1.0 + ROBOT_RADIUS  # 1.35m: robot body 邊緣離障礙物表面至少 1.0m

        pos_xy = pos_samples[:, :2].clone()
        x_range = pos_ranges[0]  # (min, max) for x
        y_range = pos_ranges[1]  # (min, max) for y

        uninitialized_ids = env_ids[uninitialized_mask]
        wall_c, wall_s, wall_mask = get_combined_wall_data(env)
        wc = wall_c[uninitialized_ids]       # [n_reset, W, 2]
        ws = wall_s[uninitialized_ids]       # [n_reset, W, 2]
        wm = wall_mask[uninitialized_ids]    # [n_reset, W]

        # 收集所有障礙物位置（per-env local coords）
        obs_positions = []  # list of [N, 2] tensors
        obs_radii = []
        obs_sizes = getattr(env, "_obstacle_sizes", None)
        for i in range(20):
            obs_name = f"obstacle_{i}"
            if obs_name not in env.scene.keys():
                continue
            obs_entity = env.scene[obs_name]
            obs_pos_w = obs_entity.data.root_pos_w[uninitialized_ids]  # [n_reset, 3]
            # 只考慮 visible 的障礙物 (z > 0)
            visible = obs_pos_w[:, 2] > 0.0
            # 轉換為 env-local 座標
            env_origins_xy = env.scene.env_origins[uninitialized_ids, :2]
            obs_local = obs_pos_w[:, :2] - env_origins_xy  # [n_reset, 2]
            r = obs_sizes[i] / 2.0 if obs_sizes is not None and i < len(obs_sizes) else 0.3
            obs_positions.append((obs_local, visible, r))

        def _check_too_close(xy):
            """檢查 xy 是否離牆壁或障礙物太近"""
            bad = check_wall_proximity_perenv(xy, wc, ws, wm, SPAWN_SAFE_DIST)
            for obs_local, visible, r in obs_positions:
                dist = torch.norm(xy - obs_local, dim=1)
                bad = bad | (visible & (dist < SPAWN_SAFE_DIST + r))
            return bad

        too_close = _check_too_close(pos_xy)
        for _ in range(50):
            if not too_close.any():
                break
            n = too_close.sum().item()
            pos_xy[too_close, 0] = torch.rand(n, device=device) * (x_range[1] - x_range[0]) + x_range[0]
            pos_xy[too_close, 1] = torch.rand(n, device=device) * (y_range[1] - y_range[0]) + y_range[0]
            too_close = _check_too_close(pos_xy)

        pos_samples[:, :2] = pos_xy

        # 計算世界座標位置（相對於環境原點）
        uninitialized_env_ids = env_ids[uninitialized_mask]
        default_positions = root_states[uninitialized_mask, 0:3]
        env_origins = env.scene.env_origins[uninitialized_env_ids]
        world_positions = default_positions + env_origins + pos_samples
        
        # 計算四元數朝向
        orientations_delta = math_utils.quat_from_euler_xyz(
            ori_samples[:, 0], 
            ori_samples[:, 1], 
            ori_samples[:, 2]
        )
        default_orientations = root_states[uninitialized_mask, 3:7]
        world_orientations = math_utils.quat_mul(default_orientations, orientations_delta)
        
        # 計算速度（相對於默認速度）
        default_velocities = root_states[uninitialized_mask, 7:13]
        world_velocities = default_velocities + vel_samples
        
        # 保存到固定位置存儲（使用環境相對座標，不包含環境原點）
        # 注意：我們保存的是相對於環境原點的偏移，這樣在重置時可以正確應用
        env._fixed_robot_poses[uninitialized_env_ids, 0:3] = pos_samples
        env._fixed_robot_poses[uninitialized_env_ids, 3:7] = world_orientations
        env._fixed_robot_velocities[uninitialized_env_ids] = world_velocities
        env._fixed_robot_initialized[uninitialized_env_ids] = True
    
    # 為所有請求的環境應用固定位置
    # 獲取保存的位置和朝向
    saved_positions_offset = env._fixed_robot_poses[env_ids, 0:3]  # 相對於環境原點的偏移
    saved_orientations = env._fixed_robot_poses[env_ids, 3:7]
    saved_velocities = env._fixed_robot_velocities[env_ids]
    
    # 計算世界座標位置
    default_positions = root_states[:, 0:3]
    env_origins = env.scene.env_origins[env_ids]
    world_positions = default_positions + env_origins + saved_positions_offset
    
    # 應用到模擬器
    asset.write_root_pose_to_sim(
        torch.cat([world_positions, saved_orientations], dim=-1),
        env_ids=env_ids
    )
    asset.write_root_velocity_to_sim(saved_velocities, env_ids=env_ids)


def reset_root_state_random_safe(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: "SceneEntityCfg" = None,
):
    """每次 reset 都重新隨機位置（含牆壁碰撞檢查）

    與 reset_root_state_fixed_per_env 的差異：
    - fixed_per_env: 每個 env_id 的位置固定不變（訓練穩定性）
    - random_safe: 每次 reset 都重新隨機化（更多 diversity）

    兩者都包含牆壁 proximity 拒絕採樣，避免 robot 生成在牆內。
    """
    if hasattr(env, "_fixed_robot_initialized"):
        env._fixed_robot_initialized[env_ids] = False
    reset_root_state_fixed_per_env(env, env_ids, pose_range, velocity_range, asset_cfg)
