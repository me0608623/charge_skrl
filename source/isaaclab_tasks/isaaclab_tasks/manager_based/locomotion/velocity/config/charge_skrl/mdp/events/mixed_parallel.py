"""混合難度環境配置 — 取代傳統分階段課程學習

VLP16 訓練使用的函數:
    randomize_obstacles_by_difficulty: 每次 reset 時配置障礙物 (startup + reset)

策略: 在同一批次 (1024 envs) 中同時訓練不同難度：
    - 50% empty (無障礙物): 學習基本導航
    - 30% static (靜態障礙物): 學習避障
    - 20% dynamic (動態障礙物): 學習動態避障

參數:
    empty_ratio: float = 0.5       — 空環境比例
    static_ratio: float = 0.3     — 靜態環境比例
    dynamic_ratio: float = 0.2    — 動態環境比例
    num_obstacles_static: int = 5  — 靜態場景障礙物數量
    num_obstacles_dynamic: int = 8 — 動態場景障礙物數量
    active_obstacle_ratio: float = 0.25 — 動態障礙物中實際移動的比例
    min_robot_distance: float = 1.5 — 障礙物與機器人最小距離 [m]
    min_obstacle_spacing: float = 1.0 — 障礙物間最小間距 [m]

優勢 vs 分階段:
    1. 不需要手動設計課程學習的升級條件
    2. Agent 始終接觸到各種難度，不會遺忘簡單場景
    3. Hard envs 提供挑戰，easy envs 提供正回饋（穩定訓練）
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

from ..wall_layout import check_wall_proximity_perenv, get_combined_wall_data


def randomize_obstacles_by_difficulty(
    env,
    env_ids,
    # 難度分佈參數
    empty_ratio: float = 0.5,      # 50% 空環境（無障礙物）
    static_ratio: float = 0.3,     # 30% 靜態障礙物
    dynamic_ratio: float = 0.2,    # 20% 動態障礙物
    mixed_ratio: float = 0.0,      # 混合模式：前 N_s 個靜態 + 後 N_d 個動態
    # 障礙物配置
    num_obstacles_static: int = 5,   # 靜態環境的障礙物數量
    num_obstacles_dynamic: int = 8,  # 動態環境的障礙物數量
    max_obstacles: int = 10,         # 場景中最大障礙物數量
    # 移動參數（僅動態環境使用）
    speed_range: float = 0.5,
    min_speed: float = 0.05,
    # 碰撞檢查參數
    min_robot_distance: float = 1.5,
    min_goal_distance: float = 1.0,
    min_obstacle_spacing: float = 1.0,
    max_spawn_attempts: int = 50,
    boundary: float = 7.5,  # Phase 0 房間半徑 8m，留 0.5m 邊距
    # Active density masking: fraction of dynamic obstacles that are active
    active_obstacle_ratio: float = 1.0,
    # 🔥 Debug 模式
    debug: bool = False,
):
    """混合平行環境障礙物隨機化事件（論文核心亮點實作）

    在同一個批次中建立不同難度的混合場景：
    - Group 1 (Empty, 50%): 所有障礙物 Z = -10.0，速度 V = 0.0
    - Group 2 (Static, 30%): 前 num_obstacles_static 個障礙物 Z = 0.5，速度 V = 0.0
    - Group 3 (Dynamic, 20%): 前 num_obstacles_dynamic 個障礙物 Z = 0.5，隨機速度

    🔥 關鍵實現：
    1. 使用 PyTorch Tensor 操作，避免 Python for 迴圈遍歷環境
    2. 切片邏輯：split1 = int(0.5 * N), split2 = int(0.8 * N)
    3. 地底遮蔽：Z < 0 的障礙物在 Critic 觀測中被設為 0.0

    Args:
        env: 環境實例
        env_ids: 需要重置的環境 ID 列表
        empty_ratio: 空環境比例（默認 0.5 = 50%）
        static_ratio: 靜態障礙物比例（默認 0.3 = 30%）
        dynamic_ratio: 動態障礙物比例（默認 0.2 = 20%）
        num_obstacles_static: 靜態環境的障礙物數量
        num_obstacles_dynamic: 動態環境的障礙物數量
        max_obstacles: 場景中最大障礙物數量（用於觀測 padding）
        speed_range: 動態障礙物最大速度 (m/s)
        min_speed: 動態障礙物最小速度 (m/s)
        min_robot_distance: 障礙物與機器人最小距離 (m)
        min_goal_distance: 障礙物與目標最小距離 (m)
        min_obstacle_spacing: 障礙物之間最小距離 (m)
        max_spawn_attempts: 每個障礙物最大嘗試生成次數
        boundary: 場景邊界（米）
    """
    # ========================================================================
    # 參數處理與初始化
    # ========================================================================
    # 首次呼叫時 log 障礙物配置
    if not hasattr(env, "_obstacle_cfg_logged"):
        env._obstacle_cfg_logged = True
        mr = mixed_ratio if mixed_ratio else 0.0
        print(f"[randomize_obstacles] config: "
              f"static={num_obstacles_static} dynamic={num_obstacles_dynamic} "
              f"max_obstacles={max_obstacles} "
              f"ratios: E={empty_ratio:.0%} S={static_ratio:.0%} D={dynamic_ratio:.0%} M={mr:.0%}")

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    N = len(env_ids)  # 重置環境的數量
    device = env.device

    # ========================================================================
    # 🔥 DEBUG: 確認事件被觸發
    # ========================================================================
    if debug:
        print(f"[DEBUG] randomize_obstacles_by_difficulty CALLED!")
        print(f"[DEBUG]   - Resetting {N} environments")
        print(f"[DEBUG]   - env_ids: {env_ids[:10].tolist() if len(env_ids) > 10 else env_ids.tolist()}{'...' if len(env_ids) > 10 else ''}")

    # 初始化障礙物元數據
    num_obstacles = getattr(env, "_num_obstacles", None)
    if num_obstacles is None:
        num_obstacles = _get_obstacle_num()
        env._num_obstacles = num_obstacles

    if not hasattr(env, "_obstacle_sizes"):
        env._obstacle_sizes = _get_obstacle_sizes()

    # 初始化障礙物速度緩存 [num_envs, max_obstacles, 2]
    if not hasattr(env, "_obstacle_velocities") or env._obstacle_velocities.shape[1] != max_obstacles:
        env._obstacle_velocities = torch.zeros(env.num_envs, max_obstacles, 2, device=device)

    # 初始化環境難度標記
    if not hasattr(env, "_env_difficulty"):
        env._env_difficulty = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    if not hasattr(env, "_env_num_visible_obstacles"):
        env._env_num_visible_obstacles = torch.zeros(env.num_envs, device=device, dtype=torch.long)

    # Initialize active obstacle mask (for density masking)
    if not hasattr(env, "_obstacle_active_mask"):
        env._obstacle_active_mask = torch.ones(
            env.num_envs, max_obstacles, device=device, dtype=torch.bool
        )

    # ========================================================================
    # 第一步：按比例分配難度級別（probabilistic，適用於任意 N）
    # ========================================================================
    # 使用隨機採樣而非確定性切分，避免 N 較小時整數截斷導致比例失真
    # （例如 N=1 時 int(0.2*1)=0, int(0.7*1)=0 → 全部變 dynamic）
    # 難度分級: 0=empty, 1=static, 2=dynamic, 3=mixed (static+dynamic 同場)
    rand = torch.rand(N, device=device)
    t_e = empty_ratio
    t_s = t_e + static_ratio
    t_d = t_s + dynamic_ratio
    difficulty = torch.where(
        rand < t_e, torch.zeros(N, device=device, dtype=torch.long),
        torch.where(rand < t_s, torch.ones(N, device=device, dtype=torch.long),
            torch.where(rand < t_d, torch.full((N,), 2, device=device, dtype=torch.long),
                torch.full((N,), 3, device=device, dtype=torch.long))))

    # 🔥 DEBUG: 顯示分配結果
    if debug:
        print(f"[DEBUG] Difficulty (N={N}): "
              f"E={(difficulty==0).sum().item()} S={(difficulty==1).sum().item()} "
              f"D={(difficulty==2).sum().item()} M={(difficulty==3).sum().item()}")

    # 記錄每個環境的難度級別
    env._env_difficulty[env_ids] = difficulty

    # ========================================================================
    # 第三步：獲取機器人和目標位置（用於碰撞檢查）
    # ========================================================================
    robot_pos_xy = env.scene["robot"].data.root_pos_w[env_ids, :2]

    try:
        goal_pos = env.command_manager.get_command("goal_command")
        goal_pos_xy = goal_pos[env_ids, :2]
        has_goal = True
    except (AttributeError, KeyError, IndexError):
        goal_pos_xy = None
        has_goal = False

    # 獲取環境原點（用於將局部座標轉換為世界座標）
    env_origins = env.scene.env_origins[env_ids]  # [N, 3]

    # ========================================================================
    # 第四步：批量準備障礙物位置和速度張量
    # ========================================================================
    # 場景參數
    safe_margin = 1.5
    spawn_range = boundary - safe_margin
    HIDDEN_Z = -10.0
    VISIBLE_Z = 0.5

    # 定義 4 個象限（用於分層採樣，確保障礙物分布均勻）
    quadrants = [
        (0.3, spawn_range, 0.3, spawn_range),      # 第一象限
        (-spawn_range, -0.3, 0.3, spawn_range),    # 第二象限
        (-spawn_range, -0.3, -spawn_range, -0.3),  # 第三象限
        (0.3, spawn_range, -spawn_range, -0.3),    # 第四象限
    ]

    # 創建難度掩碼
    is_empty = (difficulty == 0)
    is_static = (difficulty == 1)
    is_dynamic = (difficulty == 2)
    is_mixed = (difficulty == 3)
    num_obstacles_mixed = num_obstacles_static + num_obstacles_dynamic

    # ========================================================================
    # Active density masking for dynamic environments
    # ========================================================================
    num_dyn = is_dynamic.sum().item()
    if num_dyn > 0 and active_obstacle_ratio < 1.0:
        num_active = max(1, int(num_obstacles_dynamic * active_obstacle_ratio))
        # Random scores to select which obstacles are active per dynamic env
        rand_scores = torch.rand(num_dyn, num_obstacles_dynamic, device=device)
        _, active_indices = rand_scores.topk(num_active, dim=1)
        # Build full-width mask [num_dyn, max_obstacles]
        dyn_active_mask = torch.zeros(num_dyn, max_obstacles, device=device, dtype=torch.bool)
        dyn_active_mask.scatter_(1, active_indices, True)
        dyn_env_ids_local = torch.arange(N, device=device)[is_dynamic]
        env._obstacle_active_mask[env_ids[dyn_env_ids_local]] = dyn_active_mask
    else:
        # All obstacles active (default)
        env._obstacle_active_mask[env_ids[is_dynamic]] = True

    # Mixed 模式: 記錄靜態障礙物數量，供 move_obstacles_vectorized 判斷
    if is_mixed.any():
        env._num_obstacles_static_mixed = num_obstacles_static

    # ========================================================================
    # 第五步：遍歷所有障礙物，設置位置和速度
    # ========================================================================
    for i in range(max_obstacles):
        obstacle_name = f"obstacle_{i}"
        # 🔥 修正：InteractiveScene 只支援 scene["name"]，不支援 hasattr/getattr
        try:
            obstacle = env.scene[obstacle_name]
        except KeyError:
            continue

        # -------------------------------------------------------------------
        # 根據障礙物索引 i 和難度級別決定可見性
        # 🔥 關鍵修正：基於障礙物索引 i 判斷可見性，而非環境索引
        # -------------------------------------------------------------------
        # Empty 環境：所有障礙物都隱藏
        # Static 環境：只有前 num_obstacles_static 個障礙物可見
        # Dynamic 環境：只有前 num_obstacles_dynamic 個障礙物可見

        should_show_in_static = (i < num_obstacles_static)
        should_show_in_dynamic = env._obstacle_active_mask[env_ids, i]  # [N] tensor
        should_show_in_mixed = (i < num_obstacles_mixed)

        # 計算可見性掩碼
        visible_mask = (
            (is_static & should_show_in_static)
            | (is_dynamic & should_show_in_dynamic)
            | (is_mixed & should_show_in_mixed)
        )

        # 準備位置張量 [N, 3]
        pos = torch.zeros(N, 3, device=device, dtype=torch.float32)
        quat = torch.zeros(N, 4, device=device, dtype=torch.float32)
        quat[:, 0] = 1.0  # 單位四元數

        # 設置 Z 座標（隱藏或可見）
        pos[:, 2] = torch.where(visible_mask,
                                torch.full((N,), VISIBLE_Z, device=device),
                                torch.full((N,), HIDDEN_Z, device=device))

        # 🔥 DEBUG: 顯示 Z 座標分配（僅顯示第一個障礙物）
        if debug and i == 0:
            num_visible = visible_mask.sum().item()
            print(f"[DEBUG] obstacle_{i} Z assignment: {num_visible}/{N} visible at Z={VISIBLE_Z}, {N-num_visible} hidden at Z={HIDDEN_Z}")

        # -------------------------------------------------------------------
        # 對可見的障礙物進行隨機位置採樣（含碰撞檢查）
        # -------------------------------------------------------------------
        if visible_mask.any():
            # 獲取此障礙物應該在的象限
            quadrant_idx = i % 4
            x_min, x_max, y_min, y_max = quadrants[quadrant_idx]

            # 生成隨機 XY 座標（局部座標）
            rand_x = torch.rand(N, device=device) * (x_max - x_min) + x_min
            rand_y = torch.rand(N, device=device) * (y_max - y_min) + y_min

            pos[:, 0] = rand_x
            pos[:, 1] = rand_y

            # 碰撞檢查（拒絕採樣）— 含牆壁近接檢查
            needs_resample = visible_mask.clone()

            # 取得 per-env 牆壁數據（局部座標）
            wall_c_all, wall_s_all, wall_mask_all = get_combined_wall_data(env)
            wc = wall_c_all[env_ids]       # [N, W, 2]
            ws = wall_s_all[env_ids]       # [N, W, 2]
            wm = wall_mask_all[env_ids]    # [N, W]

            for attempt in range(max_spawn_attempts):
                if not needs_resample.any():
                    break

                # 檢查與機器人的距離
                dist_to_robot = torch.norm(pos[:, :2] - robot_pos_xy, dim=1)
                valid_robot = dist_to_robot >= min_robot_distance

                # 檢查與目標的距離
                if has_goal:
                    dist_to_goal = torch.norm(pos[:, :2] - goal_pos_xy, dim=1)
                    valid_goal = dist_to_goal >= min_goal_distance
                else:
                    valid_goal = torch.ones(N, dtype=torch.bool, device=device)

                # 檢查與其他已放置障礙物的距離
                valid_obstacles = torch.ones(N, dtype=torch.bool, device=device)
                for j in range(i):
                    other_name = f"obstacle_{j}"
                    try:
                        other_obstacle = env.scene[other_name]
                        other_pos = other_obstacle.data.root_pos_w[env_ids, :2]
                        dist_to_other = torch.norm(pos[:, :2] - other_pos, dim=1)
                        # 允許部分重疊但不允許完全重疊（中心距 < 0.3m 才算重疊）
                        valid_obstacles &= dist_to_other >= 0.3
                    except KeyError:
                        pass

                # 檢查與牆壁的距離（障礙物不應在牆壁內部或太近）
                # pos[:, :2] 是局部座標（尚未加 env_origins）
                valid_walls = ~check_wall_proximity_perenv(pos[:, :2], wc, ws, wm, 0.8)

                # 綜合判斷
                all_valid = valid_robot & valid_goal & valid_obstacles & valid_walls & needs_resample
                needs_resample = needs_resample & ~all_valid

                # 對無效位置重新採樣
                if needs_resample.any():
                    num_resample = needs_resample.sum().item()
                    new_x = torch.rand(num_resample, device=device) * (x_max - x_min) + x_min
                    new_y = torch.rand(num_resample, device=device) * (y_max - y_min) + y_min
                    pos[needs_resample, 0] = new_x
                    pos[needs_resample, 1] = new_y

            # 降級策略：最終仍失敗的位置使用隨機座標
            if needs_resample.any():
                num_failed = needs_resample.sum().item()
                fallback_x = torch.rand(num_failed, device=device) * (2 * spawn_range) - spawn_range
                fallback_y = torch.rand(num_failed, device=device) * (2 * spawn_range) - spawn_range
                pos[needs_resample, 0] = fallback_x
                pos[needs_resample, 1] = fallback_y

        # -------------------------------------------------------------------
        # 🔥 關鍵修正：加上環境原點（轉換為世界座標）
        # -------------------------------------------------------------------
        # 只有可見的障礙物需要加上環境原點
        # 隱藏的障礙物保持在 (0, 0, -10.0) 相對於環境中心
        world_pos = pos.clone()
        world_pos[visible_mask, :2] += env_origins[visible_mask, :2]

        # -------------------------------------------------------------------
        # 設置障礙物速度（僅動態環境的可見障礙物）
        # -------------------------------------------------------------------
        vel = torch.zeros(N, 6, device=device, dtype=torch.float32)  # [vx, vy, vz, wx, wy, wz]

        # Dynamic: 所有可見障礙物都移動
        # Mixed: 前 num_obstacles_static 個靜止，後面的才移動
        mixed_should_move = is_mixed & visible_mask & (i >= num_obstacles_static)
        should_have_velocity = (is_dynamic & visible_mask) | mixed_should_move

        if should_have_velocity.any():
            num_vel = should_have_velocity.sum().item()
            # 生成隨機速度（XY 平面）
            angle = torch.rand(num_vel, device=device) * 2.0 * math.pi - math.pi
            speed = torch.rand(num_vel, device=device) * (speed_range - min_speed) + min_speed

            vel[should_have_velocity, 0] = speed * torch.cos(angle)  # vx
            vel[should_have_velocity, 1] = speed * torch.sin(angle)  # vy

        # 更新速度緩存（只用於觀測函數）
        env._obstacle_velocities[env_ids, i, :] = vel[:, :2]

        # -------------------------------------------------------------------
        # 🔥 關鍵修正：寫入物理引擎（位置 + 速度）
        # -------------------------------------------------------------------
        if debug and i == 0:
            print(f"[DEBUG] Writing obstacle_{i} to sim:")
            print(f"[DEBUG]   - world_pos shape: {world_pos.shape}")
            print(f"[DEBUG]   - world_pos Z range: [{world_pos[:, 2].min().item():.2f}, {world_pos[:, 2].max().item():.2f}]")
            print(f"[DEBUG]   - env_ids: {env_ids[:10].tolist() if len(env_ids) > 10 else env_ids.tolist()}{'...' if len(env_ids) > 10 else ''}")

        obstacle.write_root_pose_to_sim(
            torch.cat([world_pos, quat], dim=-1),
            env_ids=env_ids,
        )
        obstacle.write_root_velocity_to_sim(vel, env_ids=env_ids)

        # 🔥 DEBUG: 驗證寫入後的位置
        if debug and i < 3:
            num_vis = visible_mask.sum().item()
            z_vals = world_pos[:, 2]
            print(f"[DEBUG] obstacle_{i}: visible={num_vis}/{N}, "
                  f"Z_written=[{z_vals.min().item():.1f}, {z_vals.max().item():.1f}], "
                  f"type={type(obstacle).__name__}")
            if num_vis > 0:
                vis_xy = world_pos[visible_mask, :2]
                print(f"[DEBUG]   visible XY range: "
                      f"X=[{vis_xy[:,0].min().item():.2f}, {vis_xy[:,0].max().item():.2f}], "
                      f"Y=[{vis_xy[:,1].min().item():.2f}, {vis_xy[:,1].max().item():.2f}]")
            # 回讀 data buffer 驗證（write_root_pose_to_sim 會同步更新 data buffer）
            read_back_pos = obstacle.data.root_pos_w[env_ids, :3]
            rb_z_min = read_back_pos[:, 2].min().item()
            rb_z_max = read_back_pos[:, 2].max().item()
            print(f"[DEBUG]   read-back Z from data buffer: [{rb_z_min:.1f}, {rb_z_max:.1f}]")
            # 比較寫入值和回讀值
            z_diff = (world_pos[:, 2] - read_back_pos[:, 2]).abs().max().item()
            if z_diff > 0.01:
                print(f"[DEBUG]   ⚠️ WARNING: Z mismatch! max_diff={z_diff:.3f}")
                print(f"[DEBUG]   ⚠️ write_root_pose_to_sim may not be working correctly!")
            else:
                print(f"[DEBUG]   ✓ Z values match (diff={z_diff:.6f})")

    # ========================================================================
    # 第六步：記錄每個環境的可見障礙物數量
    # ========================================================================
    # 使用張量操作，避免 Python for 循環
    # Compute per-env active counts for dynamic envs from the mask
    dynamic_active_counts = env._obstacle_active_mask[env_ids].sum(dim=1).long()  # [N]
    mixed_counts = torch.full((N,), num_obstacles_mixed, device=device, dtype=torch.long)
    env._env_num_visible_obstacles[env_ids] = torch.where(
        is_empty, torch.zeros(N, device=device, dtype=torch.long),
        torch.where(is_static, torch.full((N,), num_obstacles_static, device=device, dtype=torch.long),
            torch.where(is_dynamic, dynamic_active_counts, mixed_counts)))

    # 🔥 DEBUG: 最終統計
    if debug:
        num_empty = is_empty.sum().item()
        num_static = is_static.sum().item()
        num_dynamic = is_dynamic.sum().item()
        num_mixed = is_mixed.sum().item()
        print(f"[DEBUG] === SUMMARY ===")
        print(f"[DEBUG] Empty: {num_empty}, Static: {num_static} ({num_obstacles_static}ea), "
              f"Dynamic: {num_dynamic} ({num_obstacles_dynamic}ea), "
              f"Mixed: {num_mixed} ({num_obstacles_static}s+{num_obstacles_dynamic}d={num_obstacles_mixed}ea)")
        print(f"[DEBUG] Dynamic environments: {num_dynamic} ({num_obstacles_dynamic} obstacles each)")
        print(f"[DEBUG] =================")


def get_env_difficulty(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """獲取環境難度級別

    Args:
        env: 環境實例
        env_ids: 環境 ID 列表，None 表示所有環境

    Returns:
        難度級別張量：0=empty, 1=static, 2=dynamic
    """
    if env_ids is None:
        env_ids = slice(None)  # 所有環境

    if not hasattr(env, "_env_difficulty"):
        # 默认全部為 static
        return torch.zeros(env.num_envs if env_ids is None else len(env_ids),
                          device=env.device, dtype=torch.long)

    return env._env_difficulty[env_ids]


def get_env_num_visible_obstacles(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """獲取環境的可見障礙物數量

    Args:
        env: 環境實例
        env_ids: 環境 ID 列表，None 表示所有環境

    Returns:
        可見障礙物數量張量
    """
    if env_ids is None:
        env_ids = slice(None)

    if not hasattr(env, "_env_num_visible_obstacles"):
        # 默認全部障礙物可見
        num_obstacles = getattr(env, "_num_obstacles", _get_obstacle_num())
        return torch.full((env.num_envs if env_ids is None else len(env_ids),),
                         num_obstacles, device=env.device, dtype=torch.long)

    return env._env_num_visible_obstacles[env_ids]


__all__ = [
    "randomize_obstacles_by_difficulty",
    "get_env_difficulty",
    "get_env_num_visible_obstacles",
]
