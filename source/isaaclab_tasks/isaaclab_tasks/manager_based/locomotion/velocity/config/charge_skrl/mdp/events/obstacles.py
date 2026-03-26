"""障礙物重置與動態移動事件

VLP16 訓練使用的函數:
    move_obstacles_vectorized: GPU 向量化動態障礙物移動 (EventTerm interval=0.2s)

其他函數 (reset_obstacles 等) 被 mixed_parallel.py 內部呼叫。

move_obstacles_vectorized 參數:
    move_dt: float = 0.2     — 每次移動間隔 [s] (匹配 env dt)
    speed_min/max: float     — 障礙物速度範圍 [m/s]
    goal_reach_threshold: float = 0.5 — 到達目標後重新採樣
    area_limit: float = 6.0  — 活動區域半徑 [m]
    bound_limit: float = 7.0 — 絕對邊界 [m]

移動邏輯:
    每個動態障礙物有目標點，向目標直線移動。
    到達後隨機新目標。超出邊界反彈。
    所有計算 GPU 向量化 (無 for loop over obstacles)。
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 障礙物狀態管理（原 core/state.py → events/state.py）
from .state import (
    get_obstacle_num as _get_obstacle_num,
    get_obstacle_sizes as _get_obstacle_sizes,
)

# 迷宮牆壁 proximity 檢查（原 core/wall_layout.py → mdp/wall_layout.py）
from ..wall_layout import get_wall_tensors, check_wall_proximity_batch


def reset_obstacles(
    env,
    env_ids,
    speed_range: float = 0.5,
    min_speed: float = 0.05,
    min_robot_distance: float = 1.5,
    min_goal_distance: float = 1.0,
    min_obstacle_spacing: float = 1.0,
    max_spawn_attempts: int = 50,
    boundary: float = 5.0,  # 場景邊界（米），默認 5.0 米
):
    """重置障礙物位置（Phase 1：完全靜態，含碰撞檢查）

    在環境重置時，將所有障礙物隨機重新擺放到新位置。
    Phase 1 設計：障礙物完全靜態（速度設為 0），只在每個 episode 開始時移動。
    
    2024-01 修正：加入碰撞檢查，確保障礙物不會與機器人、目標或其他障礙物重疊。

    Args:
        env: 環境實例
        env_ids: 需要重置的環境 ID 列表
                 例如：[0, 5, 12] = 第 0、5、12 個環境需要重置
        speed_range: 速度範圍（Phase 1 不使用，保留用於 Phase 2）
        min_speed: 最小速度（Phase 1 不使用，保留用於 Phase 2）
        min_robot_distance: 障礙物與機器人的最小距離（米）
        min_goal_distance: 障礙物與目標的最小距離（米）
        min_obstacle_spacing: 障礙物之間的最小距離（米）
        max_spawn_attempts: 每個障礙物最大嘗試生成次數
        boundary: 場景邊界（米），默認 5.0 米（Phase 1/2），Phase 3 為 8.0 米

    Phase 1 設計說明：
    - 障礙物設為 kinematic（固定不動）
    - 速度設為 0（與物理引擎一致）
    - 只在環境重置時隨機擺放位置
    - 避免「瞬移」問題（不會在 episode 中間突然移動）
    - 碰撞檢查確保合理的初始配置

    Phase 2 升級（動態障礙物）：
    - 將 rigid_props.kinematic_enabled 設為 False
    - 使用 move_obstacles 事件進行連續移動
    - 此時速度才會被實際使用
    """
    # reset 模式可能傳入 env_ids=None，代表所有環境
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    num_resets = len(env_ids)  # 需要重置的環境數量
    device = env.device  # 設備（CPU 或 GPU）

    # 快取迷宮牆壁張量
    wall_c, wall_s = get_wall_tensors(device)

    # 遍歷所有可能的障礙物
    # Phase 2.5：優先使用環境中已設置的 _num_obstacles（由課程學習設置）
    # 只有在未設置時才使用全局配置
    num_obstacles = getattr(env, "_num_obstacles", None)
    if num_obstacles is None:
        num_obstacles = _get_obstacle_num()
        env._num_obstacles = num_obstacles
    # 注意：如果 env._num_obstacles 已經設置（例如由 reset_obstacles_with_adaptive_params 設置），
    # 則使用該值，不覆蓋它
    if not hasattr(env, "_obstacle_sizes"):
        env._obstacle_sizes = _get_obstacle_sizes()

    # 初始化障礙物速度緩存（用於觀測）
    if not hasattr(env, "_obstacle_velocities") or env._obstacle_velocities.shape[1] != num_obstacles:
        env._obstacle_velocities = torch.zeros(env.num_envs, num_obstacles, 2, device=device)

    # Phase 1：速度設為 0（障礙物完全靜態）
    # 這樣觀測中的速度資訊會正確反映物理引擎的狀態（靜止）
    env._obstacle_velocities[env_ids, :, :] = 0.0

    # ------------------------------------------------------------------------
    # 獲取機器人和目標位置（用於碰撞檢查）
    # ------------------------------------------------------------------------
    robot_pos_xy = env.scene["robot"].data.root_pos_w[env_ids, :2]  # [num_resets, 2]
    
    # 嘗試獲取目標位置（可能還未初始化）
    try:
        goal_pos = env.command_manager.get_command("goal_command")
        goal_pos_xy = goal_pos[env_ids, :2]  # [num_resets, 2]
        has_goal = True
    except (AttributeError, KeyError, IndexError):
        goal_pos_xy = None
        has_goal = False

    # 儲存已放置的障礙物位置（用於障礙物間碰撞檢查）
    # shape: [num_resets, num_obstacles, 2]
    placed_positions = torch.full(
        (num_resets, num_obstacles, 2), float('inf'), device=device, dtype=torch.float32
    )

    # ------------------------------------------------------------------------
    # 分層採樣設計：確保障礙物均勻分布在場景各區域
    # ------------------------------------------------------------------------
    # 將場景分成 4 個象限，每個象限分配固定數量的障礙物
    # 這樣可以避免障礙物集中在某一側
    #
    # 象限劃分（以原點為中心）：
    #   象限 0: X > 0, Y > 0（右上）
    #   象限 1: X < 0, Y > 0（左上）
    #   象限 2: X < 0, Y < 0（左下）
    #   象限 3: X > 0, Y < 0（右下）
    #
    # 邊界值：使用傳入的 boundary 參數（但留安全邊距，實際生成範圍 boundary - safe_margin）
    # ------------------------------------------------------------------------
    
    # boundary 已作為參數傳入（默認 5.0 米，Phase 3 為 8.0 米）
    # boundary 已作為參數傳入（默認 5.0 米，Phase 3 為 8.0 米）
    # 安全邊距 = 用戶請求 (1.0m) + 障礙物半徑 (0.5m) = 1.5m
    safe_margin = 1.5
    spawn_range = boundary - safe_margin  # 實際生成範圍
    
    # 定義 4 個象限的範圍
    # 格式：(x_min, x_max, y_min, y_max)
    quadrants = [
        (0.3, spawn_range, 0.3, spawn_range),      # 象限 0: 右上（避開原點附近）
        (-spawn_range, -0.3, 0.3, spawn_range),    # 象限 1: 左上
        (-spawn_range, -0.3, -spawn_range, -0.3),  # 象限 2: 左下
        (0.3, spawn_range, -spawn_range, -0.3),    # 象限 3: 右下
    ]
    
    # 計算每個象限分配多少障礙物
    # 例如：10 個障礙物 → 每象限 2-3 個
    obstacles_per_quadrant = num_obstacles // 4  # 基礎數量
    remainder = num_obstacles % 4  # 餘數分配給前幾個象限
    
    # 建立障礙物到象限的映射
    obstacle_to_quadrant = []
    for q in range(4):
        count = obstacles_per_quadrant + (1 if q < remainder else 0)
        obstacle_to_quadrant.extend([q] * count)
    
    for i in range(num_obstacles):
        obstacle_name = f"obstacle_{i}"

        # 檢查障礙物是否存在
        # 🔥 修正：InteractiveScene 只支援 scene["name"]，不支援 hasattr/getattr
        if obstacle_name not in env.scene.keys():
            continue

        # ------------------------------------------------------------------------
        # 取得此障礙物應該生成的象限
        # ------------------------------------------------------------------------
        quadrant_idx = obstacle_to_quadrant[i] if i < len(obstacle_to_quadrant) else i % 4
        x_min, x_max, y_min, y_max = quadrants[quadrant_idx]

        # ------------------------------------------------------------------------
        # 生成隨機位置（含碰撞檢查）
        # ------------------------------------------------------------------------
        pos = torch.zeros(num_resets, 3, device=device, dtype=torch.float32)
        pos[:, 2] = 0.5  # Z 座標：固定高度

        # 追蹤哪些環境還需要找到有效位置
        needs_position = torch.ones(num_resets, dtype=torch.bool, device=device)
        
        for attempt in range(max_spawn_attempts):
            if not needs_position.any():
                break  # 所有環境都找到有效位置了
            
            # 為需要位置的環境生成隨機候選位置（在指定象限內）
            num_need = needs_position.sum().item()
            
            # 在指定象限範圍內生成隨機位置
            candidate_x = torch.rand(num_need, device=device) * (x_max - x_min) + x_min
            candidate_y = torch.rand(num_need, device=device) * (y_max - y_min) + y_min
            
            # 暫存候選位置
            pos[needs_position, 0] = candidate_x
            pos[needs_position, 1] = candidate_y
            
            # ------------------------------------------------------------------------
            # 碰撞檢查 1：與機器人距離
            # ------------------------------------------------------------------------
            dist_to_robot = torch.norm(pos[:, :2] - robot_pos_xy, dim=1)
            valid_robot = dist_to_robot >= min_robot_distance
            
            # ------------------------------------------------------------------------
            # 碰撞檢查 2：與目標距離
            # ------------------------------------------------------------------------
            if has_goal:
                dist_to_goal = torch.norm(pos[:, :2] - goal_pos_xy, dim=1)
                valid_goal = dist_to_goal >= min_goal_distance
            else:
                valid_goal = torch.ones(num_resets, dtype=torch.bool, device=device)
            
            # ------------------------------------------------------------------------
            # 碰撞檢查 3：與其他已放置障礙物的距離
            # ------------------------------------------------------------------------
            valid_obstacles = torch.ones(num_resets, dtype=torch.bool, device=device)
            for j in range(i):  # 只檢查已放置的障礙物
                dist_to_other = torch.norm(pos[:, :2] - placed_positions[:, j, :], dim=1)
                valid_obstacles &= dist_to_other >= min_obstacle_spacing
            
            # ------------------------------------------------------------------------
            # 碰撞檢查 4：與內部迷宮牆壁距離
            # ------------------------------------------------------------------------
            clear_of_walls = ~check_wall_proximity_batch(pos[:, :2], wall_c, wall_s, 0.8)

            # ------------------------------------------------------------------------
            # 綜合判斷：所有檢查都通過才算有效
            # ------------------------------------------------------------------------
            valid_this_attempt = valid_robot & valid_goal & valid_obstacles & clear_of_walls & needs_position
            
            # 更新 needs_position：有效的不再需要
            needs_position = needs_position & ~valid_this_attempt
        
        # 如果達到最大嘗試次數仍有環境找不到有效位置
        # 允許在整個場景範圍內重新嘗試（降級策略）
        if needs_position.any():
            num_failed = needs_position.sum().item()
            # 使用全場景範圍重新生成
            fallback_x = torch.rand(num_failed, device=device) * (2 * spawn_range) - spawn_range
            fallback_y = torch.rand(num_failed, device=device) * (2 * spawn_range) - spawn_range
            pos[needs_position, 0] = fallback_x
            pos[needs_position, 1] = fallback_y
        
        # 記錄已放置的位置
        placed_positions[:, i, :] = pos[:, :2]

        # ------------------------------------------------------------------------
        # 生成姿態（不旋轉）
        # ------------------------------------------------------------------------
        quat = torch.zeros(num_resets, 4, device=device, dtype=torch.float32)
        quat[:, 0] = 1.0  # (w=1, x=0, y=0, z=0) = 不旋轉

        # ------------------------------------------------------------------------
        # 應用到模擬器
        # ------------------------------------------------------------------------
        env.scene[obstacle_name].write_root_pose_to_sim(
            torch.cat([pos, quat], dim=-1),  # 拼接位置和姿態 [7]
            env_ids=env_ids  # 只更新指定的環境
        )
        # write_root_pose_to_sim：直接設定物體的位置和姿態
        # env_ids：指定哪些環境需要更新（避免影響其他環境）


def move_obstacles(
    env,
    env_ids,
    move_dt: float = 0.2,
    speed_range: float = 0.5,
    min_speed: float = 0.05,
    speed_jitter: float = 0.05,
    area_limit: float = 6.0,
):
    """動態移動障礙物（interval 事件）
    
    透過「連續速度 + 小擾動」讓障礙物平滑移動。
    
    Args:
        env: 環境實例
        env_ids: 需要更新的環境 ID（可能是 None、張量或列表）
        move_dt: 每次更新的時間步長（秒）
        speed_range: 最大速度（m/s）
        min_speed: 最小速度（m/s）
        speed_jitter: 速度小擾動（m/s）
        area_limit: 障礙物活動範圍限制（正負範圍）
    """
    # 處理 env_ids 參數（interval 模式可能傳入 None 或張量）
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, (list, tuple)):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor([env_ids], device=env.device, dtype=torch.long)
    
    num_resets = len(env_ids)
    device = env.device
    
    num_obstacles = getattr(env, "_num_obstacles", None)
    if num_obstacles is None:
        num_obstacles = _get_obstacle_num()
    if not hasattr(env, "_obstacle_velocities") or env._obstacle_velocities.shape[1] != num_obstacles:
        env._obstacle_velocities = torch.zeros(env.num_envs, num_obstacles, 2, device=device)
    
    for i in range(num_obstacles):
        obstacle_name = f"obstacle_{i}"
        # 🔥 修正：InteractiveScene 只支援 scene["name"]
        try:
            obstacle = env.scene[obstacle_name]
        except KeyError:
            continue
        
        # 取出當前位置與速度
        pos = obstacle.data.root_pos_w[env_ids, :3].clone()
        vel = env._obstacle_velocities[env_ids, i, :].clone()
        
        # 如果速度為 0（初始狀態），初始化隨機速度
        speed = torch.linalg.norm(vel, dim=1, keepdim=True)
        zero_speed_mask = (speed.squeeze(1) < 1e-6)
        if zero_speed_mask.any():
            # 為速度為 0 的障礙物初始化隨機速度
            num_zero = zero_speed_mask.sum().item()
            angle = torch.rand(num_zero, device=device) * 2.0 * math.pi - math.pi  # [-π, π]
            init_speed = torch.rand(num_zero, device=device) * (speed_range - min_speed) + min_speed
            vel[zero_speed_mask, 0] = init_speed * torch.cos(angle)
            vel[zero_speed_mask, 1] = init_speed * torch.sin(angle)
        
        # 速度小擾動（讓方向慢慢變化）
        if speed_jitter > 0.0:
            vel += (torch.rand(num_resets, 2, device=device) - 0.5) * 2.0 * speed_jitter
        
        # 速度限制
        speed = torch.linalg.norm(vel, dim=1, keepdim=True)
        speed = torch.clamp(speed, min_speed, speed_range)
        vel = vel / (torch.linalg.norm(vel, dim=1, keepdim=True) + 1e-6) * speed
        
        # 平滑位移
        pos[:, 0:2] += vel * move_dt
        
        # 邊界反彈
        hit_x = (pos[:, 0] <= -area_limit) | (pos[:, 0] >= area_limit)
        hit_y = (pos[:, 1] <= -area_limit) | (pos[:, 1] >= area_limit)
        vel[hit_x, 0] *= -1.0
        vel[hit_y, 1] *= -1.0
        
        pos[:, 0] = torch.clamp(pos[:, 0], -area_limit, area_limit)
        pos[:, 1] = torch.clamp(pos[:, 1], -area_limit, area_limit)
        
        # 取出當前姿態（保持不變）
        if hasattr(obstacle.data, "root_quat_w"):
            quat = obstacle.data.root_quat_w[env_ids, :4].clone()
        else:
            quat = torch.zeros(num_resets, 4, device=device, dtype=torch.float32)
            quat[:, 0] = 1.0
        
        # 應用到模擬器
        obstacle.write_root_pose_to_sim(
            torch.cat([pos, quat], dim=-1),
            env_ids=env_ids,
        )
        
        # 回寫速度
        env._obstacle_velocities[env_ids, i, :] = vel


def move_obstacles_goal_directed(
    env,
    env_ids,
    move_dt: float = 0.2,
    speed_min: float = 0.3,
    speed_max: float = 1.2,
    goal_reach_threshold: float = 0.5,
    speed_resample_steps: int = 10,
    area_limit: float = 6.0,
    max_obstacles: int = 10,
):
    """Goal-directed 障礙物移動（NavRL 風格）

    取代隨機漫步，讓障礙物朝隨機目標位置移動：
    - 每個障礙物持有一個隨機目標位置
    - 到達目標 0.5m 內自動換新目標
    - 每 speed_resample_steps 步重新採樣速度大小
    - 只影響 dynamic 環境（_env_difficulty == 2）

    重要：只對 dynamic 環境讀寫 sim 位置，避免覆蓋 reset 事件對
    static/empty 環境的寫入（read/write buffer 一步延遲問題）。

    Args:
        env: 環境實例
        env_ids: 需要更新的環境 ID
        move_dt: 時間步長（秒）
        speed_min: 最小速度 (m/s)
        speed_max: 最大速度 (m/s)
        goal_reach_threshold: 到達目標的距離閾值 (m)
        speed_resample_steps: 每隔多少步重新採樣速度
        area_limit: 障礙物目標生成範圍
        max_obstacles: 場景中最大障礙物數量
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, (list, tuple)):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor([env_ids], device=env.device, dtype=torch.long)

    num_envs_total = env.num_envs
    device = env.device

    # 🔥 DEBUG: 追蹤此函數的呼叫頻率（僅前 5 次）
    if not hasattr(env, "_move_goal_directed_call_count"):
        env._move_goal_directed_call_count = 0
    env._move_goal_directed_call_count += 1
    _debug = env._move_goal_directed_call_count <= 5

    if _debug:
        print(f"[DEBUG] move_obstacles_goal_directed called (#{env._move_goal_directed_call_count}), "
              f"env_ids={len(env_ids)}")

    # 只處理 dynamic 環境（difficulty == 2）
    if not hasattr(env, "_env_difficulty"):
        if _debug:
            print(f"[DEBUG]   ⚠️ _env_difficulty not set, returning early")
        return
    difficulty = env._env_difficulty[env_ids]  # [len(env_ids)]
    is_dynamic = (difficulty == 2)
    if not is_dynamic.any():
        if _debug:
            print(f"[DEBUG]   No dynamic envs in env_ids, returning early")
        return

    # 🔥 關鍵：只取 dynamic 環境的 IDs，避免讀寫 static/empty 環境的位置
    # 原因：interval 事件與 reset 事件在同一步執行，
    # root_pos_w (read buffer) 尚未反映 reset 事件的 write，
    # 若對全部 env 讀寫會把 reset 寫入的 Z=0.5 覆蓋回 Z=-10
    dyn_env_ids = env_ids[is_dynamic]  # 只有 dynamic 環境
    D = len(dyn_env_ids)

    if _debug:
        print(f"[DEBUG]   Dynamic envs: {D}/{len(env_ids)}, dyn_env_ids={dyn_env_ids[:5].tolist()}")

    num_obstacles = getattr(env, "_num_obstacles", None)
    if num_obstacles is None:
        num_obstacles = _get_obstacle_num()

    # 初始化 goal-directed 狀態
    if not hasattr(env, "_obstacle_goals"):
        env._obstacle_goals = torch.zeros(
            num_envs_total, max_obstacles, 2, device=device
        )
        env._obstacle_goals[:, :, 0] = (
            torch.rand(num_envs_total, max_obstacles, device=device) * 2 * area_limit
            - area_limit
        )
        env._obstacle_goals[:, :, 1] = (
            torch.rand(num_envs_total, max_obstacles, device=device) * 2 * area_limit
            - area_limit
        )

    if not hasattr(env, "_obstacle_goal_speeds"):
        env._obstacle_goal_speeds = (
            torch.rand(num_envs_total, max_obstacles, device=device)
            * (speed_max - speed_min)
            + speed_min
        )

    if not hasattr(env, "_obstacle_step_counter"):
        env._obstacle_step_counter = torch.zeros(
            num_envs_total, max_obstacles, device=device, dtype=torch.long
        )

    # 確保速度緩存存在
    if not hasattr(env, "_obstacle_velocities") or env._obstacle_velocities.shape[1] != max_obstacles:
        env._obstacle_velocities = torch.zeros(
            num_envs_total, max_obstacles, 2, device=device
        )

    # 遞增步數計數器
    env._obstacle_step_counter[dyn_env_ids] += 1

    for i in range(min(num_obstacles, max_obstacles)):
        obstacle_name = f"obstacle_{i}"
        # 🔥 修正：InteractiveScene 只支援 scene["name"]，不支援 hasattr/getattr
        if obstacle_name not in env.scene.keys():
            continue

        obstacle = env.scene[obstacle_name]

        # 只讀取 dynamic 環境的位置
        pos = obstacle.data.root_pos_w[dyn_env_ids, :3].clone()  # [D, 3]

        # 可見障礙物（Z > 0）
        visible = pos[:, 2] > 0.0  # [D]
        if not visible.any():
            continue

        # 取出目標和速度（只取 dynamic 環境）
        goals = env._obstacle_goals[dyn_env_ids, i, :]  # [D, 2]
        speeds = env._obstacle_goal_speeds[dyn_env_ids, i]  # [D]
        step_counts = env._obstacle_step_counter[dyn_env_ids, i]  # [D]

        # --- 檢查是否到達目標，需要換新目標 ---
        dist_to_goal = torch.norm(pos[:, :2] - goals, dim=1)  # [D]
        reached_goal = visible & (dist_to_goal < goal_reach_threshold)

        if reached_goal.any():
            num_reached = reached_goal.sum().item()
            new_goals_x = (
                torch.rand(num_reached, device=device) * 2 * area_limit - area_limit
            )
            new_goals_y = (
                torch.rand(num_reached, device=device) * 2 * area_limit - area_limit
            )
            goals[reached_goal, 0] = new_goals_x
            goals[reached_goal, 1] = new_goals_y
            env._obstacle_goals[dyn_env_ids[reached_goal], i, :] = goals[reached_goal]

        # --- 每 speed_resample_steps 步重新採樣速度 ---
        should_resample = visible & (step_counts % speed_resample_steps == 0)
        if should_resample.any():
            num_resample = should_resample.sum().item()
            new_speeds = (
                torch.rand(num_resample, device=device) * (speed_max - speed_min)
                + speed_min
            )
            speeds[should_resample] = new_speeds
            env._obstacle_goal_speeds[dyn_env_ids[should_resample], i] = new_speeds

        # --- 計算朝目標的速度向量 ---
        direction = goals - pos[:, :2]  # [D, 2]
        dist = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)  # [D, 1]
        direction_unit = direction / dist  # [D, 2]
        vel = direction_unit * speeds.unsqueeze(1)  # [D, 2]

        # 不可見的障礙物速度為 0
        vel[~visible] = 0.0

        # 更新位置（只更新可見障礙物的 XY）
        pos[visible, 0:2] += vel[visible] * move_dt

        # 邊界反彈（到邊界後換目標）
        hit_boundary = (
            (pos[:, 0] <= -area_limit)
            | (pos[:, 0] >= area_limit)
            | (pos[:, 1] <= -area_limit)
            | (pos[:, 1] >= area_limit)
        )
        hit_and_visible = hit_boundary & visible
        if hit_and_visible.any():
            pos[hit_and_visible, 0] = pos[hit_and_visible, 0].clamp(
                -area_limit, area_limit
            )
            pos[hit_and_visible, 1] = pos[hit_and_visible, 1].clamp(
                -area_limit, area_limit
            )
            num_hit = hit_and_visible.sum().item()
            env._obstacle_goals[dyn_env_ids[hit_and_visible], i, 0] = (
                torch.rand(num_hit, device=device) * 2 * (area_limit - 1.0)
                - (area_limit - 1.0)
            )
            env._obstacle_goals[dyn_env_ids[hit_and_visible], i, 1] = (
                torch.rand(num_hit, device=device) * 2 * (area_limit - 1.0)
                - (area_limit - 1.0)
            )

        # 取出當前姿態
        if hasattr(obstacle.data, "root_quat_w"):
            quat = obstacle.data.root_quat_w[dyn_env_ids, :4].clone()
        else:
            quat = torch.zeros(D, 4, device=device, dtype=torch.float32)
            quat[:, 0] = 1.0

        # 🔥 只寫入 dynamic 環境（不觸碰 static/empty 環境的 write buffer）
        obstacle.write_root_pose_to_sim(
            torch.cat([pos, quat], dim=-1),
            env_ids=dyn_env_ids,
        )

        # 🔥 DEBUG: 顯示前 2 個障礙物的 Z 值
        if _debug and i < 2:
            print(f"[DEBUG]   obstacle_{i}: read Z=[{pos[:,2].min().item():.1f}, {pos[:,2].max().item():.1f}], "
                  f"visible={visible.sum().item()}/{D}")

        # 回寫速度緩存（只更新 dynamic 環境）
        env._obstacle_velocities[dyn_env_ids, i, :] = vel


def enforce_obstacle_bounds(
    obs_positions: torch.Tensor,   # [D, N, 3] world coordinates
    obs_velocities: torch.Tensor,  # [D, N, 2] (vx, vy)
    env_origins: torch.Tensor,     # [D, 3]
    visible_mask: torch.Tensor,    # [D, N] bool
    bound_limit: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized geofencing: keep obstacles within per-env bounds.

    Converts world positions to local frame, reflects velocity on boundary
    hits, clamps positions, and converts back to world frame.
    All operations are fully vectorized (zero for-loops).

    Args:
        obs_positions: [D, N, 3] obstacle world positions (modified in-place).
        obs_velocities: [D, N, 2] obstacle velocities (modified in-place).
        env_origins: [D, 3] world-frame origin of each environment.
        visible_mask: [D, N] boolean mask — only visible obstacles are processed.
        bound_limit: Half-width of the allowed region in local frame.

    Returns:
        Tuple of (reflected_x, reflected_y) boolean masks [D, N] indicating
        which obstacles had their velocity reflected on each axis.
    """
    # World → local
    rel_xy = obs_positions[:, :, :2] - env_origins[:, None, :2]  # [D, N, 2]

    # Per-axis boundary detection (only for visible obstacles)
    hit_x = (rel_xy[:, :, 0].abs() > bound_limit) & visible_mask  # [D, N]
    hit_y = (rel_xy[:, :, 1].abs() > bound_limit) & visible_mask  # [D, N]

    # Velocity reflection
    obs_velocities[:, :, 0] = torch.where(hit_x, -obs_velocities[:, :, 0], obs_velocities[:, :, 0])
    obs_velocities[:, :, 1] = torch.where(hit_y, -obs_velocities[:, :, 1], obs_velocities[:, :, 1])

    # Position clamp in local frame
    rel_xy[:, :, 0].clamp_(-bound_limit, bound_limit)
    rel_xy[:, :, 1].clamp_(-bound_limit, bound_limit)

    # Local → world
    obs_positions[:, :, :2] = rel_xy + env_origins[:, None, :2]

    return hit_x, hit_y


def move_obstacles_vectorized(
    env,
    env_ids,
    move_dt: float = 0.2,
    speed_min: float = 0.3,
    speed_max: float = 1.2,
    goal_reach_threshold: float = 0.5,
    speed_resample_steps: int = 10,
    area_limit: float = 6.0,
    max_obstacles: int = 10,
    bound_limit: float = 7.0,
):
    """Vectorized goal-directed obstacle movement with correct geofencing.

    Three-phase architecture:
      Phase 1 (GATHER): Read per-asset state into batched tensors [D, N, ...].
      Phase 2 (COMPUTE): Fully vectorized movement, geofencing, goal logic.
      Phase 3 (SCATTER): Write batched tensors back to per-asset sim buffers.

    Critical bug fix over move_obstacles_goal_directed():
      Goals are stored in LOCAL frame. Direction = goals - local_xy (both local).
      Boundary checks compare local_xy against bound_limit. This eliminates the
      world/local frame mismatch that caused wall penetration.

    Only affects dynamic environments (_env_difficulty == 2).

    Args:
        env: Environment instance.
        env_ids: Environment IDs to update.
        move_dt: Time step per update (seconds).
        speed_min: Minimum obstacle speed (m/s).
        speed_max: Maximum obstacle speed (m/s).
        goal_reach_threshold: Distance threshold to consider goal reached (m).
        speed_resample_steps: Re-sample speeds every N steps.
        area_limit: Goal generation range in local frame.
        max_obstacles: Maximum obstacle count in scene.
        bound_limit: Geofencing half-width in local frame.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, (list, tuple)):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor([env_ids], device=env.device, dtype=torch.long)

    device = env.device
    num_envs_total = env.num_envs

    # Process dynamic (difficulty == 2) and mixed (difficulty == 3) environments
    if not hasattr(env, "_env_difficulty"):
        return
    difficulty = env._env_difficulty[env_ids]
    is_dynamic = (difficulty == 2) | (difficulty == 3)
    if not is_dynamic.any():
        return

    dyn_env_ids = env_ids[is_dynamic]
    D = len(dyn_env_ids)

    num_obstacles = getattr(env, "_num_obstacles", None)
    if num_obstacles is None:
        num_obstacles = _get_obstacle_num()
    N = min(num_obstacles, max_obstacles)

    # ------------------------------------------------------------------
    # Lazy-init persistent state
    # ------------------------------------------------------------------
    if not hasattr(env, "_obstacle_goals_local"):
        env._obstacle_goals_local = torch.zeros(
            num_envs_total, max_obstacles, 2, device=device
        )
        # Initialize with random local-frame goals
        env._obstacle_goals_local[:, :, 0] = (
            torch.rand(num_envs_total, max_obstacles, device=device) * 2 * area_limit - area_limit
        )
        env._obstacle_goals_local[:, :, 1] = (
            torch.rand(num_envs_total, max_obstacles, device=device) * 2 * area_limit - area_limit
        )

    if not hasattr(env, "_obstacle_goal_speeds"):
        env._obstacle_goal_speeds = (
            torch.rand(num_envs_total, max_obstacles, device=device)
            * (speed_max - speed_min) + speed_min
        )

    if not hasattr(env, "_obstacle_step_counter"):
        env._obstacle_step_counter = torch.zeros(
            num_envs_total, max_obstacles, device=device, dtype=torch.long
        )

    if not hasattr(env, "_obstacle_velocities") or env._obstacle_velocities.shape[1] != max_obstacles:
        env._obstacle_velocities = torch.zeros(
            num_envs_total, max_obstacles, 2, device=device
        )

    # Increment step counter for dynamic envs
    env._obstacle_step_counter[dyn_env_ids] += 1

    # ------------------------------------------------------------------
    # Phase 1: GATHER — read per-asset state into [D, N, ...] tensors
    # ------------------------------------------------------------------
    all_pos = torch.zeros(D, N, 3, device=device)
    all_quat = torch.zeros(D, N, 4, device=device)
    all_quat[:, :, 0] = 1.0  # default identity quaternion

    for i in range(N):
        obstacle_name = f"obstacle_{i}"
        if obstacle_name not in env.scene.keys():
            continue
        obstacle = env.scene[obstacle_name]
        all_pos[:, i, :] = obstacle.data.root_pos_w[dyn_env_ids, :3]
        if hasattr(obstacle.data, "root_quat_w"):
            all_quat[:, i, :] = obstacle.data.root_quat_w[dyn_env_ids, :4]

    # ------------------------------------------------------------------
    # Phase 2: COMPUTE — fully vectorized, zero for-loops
    # ------------------------------------------------------------------
    env_origins = env.scene.env_origins[dyn_env_ids]  # [D, 3]

    # Visibility: Z > 0 means the obstacle is active
    visible = all_pos[:, :, 2] > 0.0  # [D, N]

    # World → local position
    local_xy = all_pos[:, :, :2] - env_origins[:, None, :2]  # [D, N, 2]

    # Slice goals and speeds for dynamic envs
    goals = env._obstacle_goals_local[dyn_env_ids, :N, :]  # [D, N, 2] LOCAL frame
    speeds = env._obstacle_goal_speeds[dyn_env_ids, :N]     # [D, N]
    step_counts = env._obstacle_step_counter[dyn_env_ids, :N]  # [D, N]

    # Direction to goal (both in LOCAL frame — the critical bug fix)
    direction = goals - local_xy  # [D, N, 2]
    dist_to_goal = torch.norm(direction, dim=2).clamp(min=1e-6)  # [D, N]
    direction_unit = direction / dist_to_goal.unsqueeze(2)  # [D, N, 2]

    # Velocity = normalized direction * speed
    vel = direction_unit * speeds.unsqueeze(2)  # [D, N, 2]

    # Zero velocity for invisible obstacles
    vel[~visible] = 0.0

    # Mixed mode (difficulty == 3): 只移動 index >= num_obstacles_static 的障礙物
    # 前 num_obstacles_static 個是靜態的，不應該移動
    if hasattr(env, "_env_difficulty"):
        mixed_envs = (env._env_difficulty[dyn_env_ids] == 3)  # [D]
        if mixed_envs.any():
            num_static = getattr(env, "_num_obstacles_static_mixed", 0)
            if num_static > 0 and num_static < N:
                # 對 mixed env 的前 num_static 個障礙物清零速度
                vel[mixed_envs, :num_static, :] = 0.0

    # Position update (world frame, only visible)
    all_pos[:, :, 0:2] += vel * move_dt * visible.unsqueeze(2).float()

    # Internal maze wall bounce: revert displacement if new position is inside a wall
    wall_c, wall_s = get_wall_tensors(device)
    local_new = all_pos[:, :, :2] - env_origins[:, None, :2]  # [D, N, 2]
    in_wall = check_wall_proximity_batch(
        local_new.reshape(-1, 2), wall_c, wall_s, 0.3
    ).reshape(D, N) & visible

    if in_wall.any():
        # Undo displacement and reverse velocity
        all_pos[:, :, 0:2] -= vel * move_dt * in_wall.unsqueeze(2).float()
        vel[in_wall] *= -1.0

    # Geofencing (operates in-place on all_pos and vel)
    reflected_x, reflected_y = enforce_obstacle_bounds(
        all_pos, vel, env_origins, visible, bound_limit
    )

    # For reflected obstacles, sample new LOCAL-frame goals
    reflected = reflected_x | reflected_y  # [D, N]
    num_reflected = reflected.sum().item()
    if num_reflected > 0:
        # Generate new goals within a smaller range (bound_limit - 1.0) to avoid immediate re-trigger
        goal_range = bound_limit - 1.0
        new_goals = torch.zeros(D, N, 2, device=device)
        new_goals[:, :, 0] = torch.rand(D, N, device=device) * 2 * goal_range - goal_range
        new_goals[:, :, 1] = torch.rand(D, N, device=device) * 2 * goal_range - goal_range
        goals[reflected] = new_goals[reflected]

    # Goal reaching: where dist_to_goal < threshold, sample new LOCAL goals
    reached = visible & (dist_to_goal < goal_reach_threshold)  # [D, N]
    num_reached = reached.sum().item()
    if num_reached > 0:
        goal_range = bound_limit - 1.0
        new_goals = torch.zeros(D, N, 2, device=device)
        new_goals[:, :, 0] = torch.rand(D, N, device=device) * 2 * goal_range - goal_range
        new_goals[:, :, 1] = torch.rand(D, N, device=device) * 2 * goal_range - goal_range
        goals[reached] = new_goals[reached]

    # Speed resampling every speed_resample_steps
    should_resample = visible & (step_counts % speed_resample_steps == 0)  # [D, N]
    num_resample = should_resample.sum().item()
    if num_resample > 0:
        new_speeds = torch.rand(D, N, device=device) * (speed_max - speed_min) + speed_min
        speeds[should_resample] = new_speeds[should_resample]

    # Write back goals and speeds to persistent state
    env._obstacle_goals_local[dyn_env_ids, :N, :] = goals
    env._obstacle_goal_speeds[dyn_env_ids, :N] = speeds

    # Write back velocity cache (for observations)
    env._obstacle_velocities[dyn_env_ids, :N, :] = vel

    # ------------------------------------------------------------------
    # Phase 3: SCATTER — write [D, N, ...] tensors back to per-asset sim
    # ------------------------------------------------------------------
    for i in range(N):
        obstacle_name = f"obstacle_{i}"
        if obstacle_name not in env.scene.keys():
            continue
        pose = torch.cat([all_pos[:, i, :], all_quat[:, i, :]], dim=-1)  # [D, 7]
        env.scene[obstacle_name].write_root_pose_to_sim(pose, env_ids=dyn_env_ids)
