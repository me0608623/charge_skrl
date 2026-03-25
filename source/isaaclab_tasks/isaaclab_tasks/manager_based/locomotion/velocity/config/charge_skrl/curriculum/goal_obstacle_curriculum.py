"""Goal-Obstacle 聯動課程學習 — 資料驅動多版本設計

支援多個 curriculum version，透過 CLI --curriculum_version 切換：
- baseline_v1: 原始 v12 線性 8 階段（每階 +1S+1D, -1G）
- goal_first_v1: 先導航再避障（Stage 1-4 無 dynamic）

獎勵函數在所有階段、所有版本完全不變。課程只改變環境參數。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# Curriculum Version Configs — 資料驅動
# ============================================================================

def _make_stage(
    num_goals, n_static, n_dynamic, min_walls, max_walls,
    gamma, episode_s, empty_ratio,
    upgrade_sr, upgrade_max_cr=0.40, upgrade_max_to=0.30,
    upgrade_min_dyn_sr=0.0, min_stage_updates=50,
    downgrade_sr=0.15, downgrade_min_cr=0.70, downgrade_min_to=0.65,
    name="",
    reward_weights=None,
):
    """建立單一 stage config dict。

    reward_weights: dict of {reward_term_name: weight} 用於 stage-dependent 獎勵權重。
                   None = 不修改權重（使用 config 預設值）。
    """
    if n_static + n_dynamic > 0:
        remaining = round(1.0 - empty_ratio, 2)
        total_obs = n_static + n_dynamic
        s_ratio = round(remaining * n_static / total_obs, 2) if total_obs > 0 else 0.0
        d_ratio = round(remaining - s_ratio, 2)
    else:
        s_ratio = 0.0
        d_ratio = 0.0

    stage = {
        "name": name,
        "num_goals": num_goals,
        "goal_distance": (2.0, 13.0),
        "num_obstacles_static": n_static,
        "num_obstacles_dynamic": n_dynamic,
        "empty_ratio": empty_ratio,
        "static_ratio": s_ratio,
        "dynamic_ratio": d_ratio,
        "gamma": gamma,
        "episode_length_s": float(episode_s),
        "min_walls": min_walls,
        "max_walls": max_walls,
        "upgrade_sr": upgrade_sr,
        "upgrade_max_cr": upgrade_max_cr,
        "upgrade_max_to": upgrade_max_to,
        "upgrade_min_dyn_sr": upgrade_min_dyn_sr,
        "min_stage_updates": min_stage_updates,
        "downgrade_sr": downgrade_sr,
        "downgrade_min_cr": downgrade_min_cr,
        "downgrade_min_to": downgrade_min_to,
    }
    if reward_weights is not None:
        stage["reward_weights"] = reward_weights
    return stage


CURRICULUM_CONFIGS = {
    # ==================================================================
    # baseline_v1: 精確重現原 v12 線性 8 階段
    # 每階 -1G, +1S, +1D。Stage 2 就同時有 static+dynamic。
    # ==================================================================
    "baseline_v1": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "stages": [
            _make_stage(8, 0, 0, 0, 2, 0.990, 45, 1.00,
                        upgrade_sr=0.72, upgrade_max_cr=1.0, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=50,
                        downgrade_sr=0.0, downgrade_min_cr=1.0, downgrade_min_to=1.0,
                        name="純導航"),
            _make_stage(7, 1, 1, 0, 3, 0.991, 51, 0.87,
                        upgrade_sr=0.65, min_stage_updates=65, name="1S+1D"),
            _make_stage(6, 2, 2, 1, 4, 0.993, 56, 0.74,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=80, name="2S+2D"),
            _make_stage(5, 3, 3, 1, 5, 0.994, 61, 0.61,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=95, name="3S+3D"),
            _make_stage(4, 4, 4, 2, 6, 0.996, 67, 0.48,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=110, name="4S+4D"),
            _make_stage(3, 5, 5, 2, 7, 0.997, 72, 0.35,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=125, name="5S+5D"),
            _make_stage(2, 6, 6, 3, 8, 0.998, 78, 0.22,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=140, name="6S+6D"),
            _make_stage(1, 7, 7, 4, 8, 0.998, 90, 0.15,
                        upgrade_sr=1.0, upgrade_max_cr=0.0, upgrade_max_to=0.0,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=0,
                        downgrade_sr=0.15, name="終極挑戰"),
        ],
    },

    # ==================================================================
    # goal_first_v1: 先導航再避障
    #
    # | Stage | 學習重點         | Goals | Static | Dynamic | Walls | SR門檻 |
    # |-------|-----------------|-------|--------|---------|-------|--------|
    # | 1     | 純 goal-reaching | 6~8   | 0      | 0       | 0~1   | >85%   |
    # | 2     | 近距離 + 少量牆   | 5~7   | 0      | 0       | 1~2   | >85%   |
    # | 3     | 開始學繞路        | 4~6   | 1~2    | 0       | 2~3   | >80%   |
    # | 4     | 增加靜態擁擠度    | 3~5   | 2~3    | 0       | 3~4   | >80%   |
    # | 5     | 輕度動態干擾      | 3~4   | 3~4    | 1~2     | 3~4   | >75%   |
    # | 6     | 中度動態避障      | 2~3   | 4~5    | 2~3     | 4~5   | >75%   |
    # | 7     | 高擁擠 + 多障礙   | 1~2   | 5~6    | 4~5     | 5~6   | >70%   |
    # | 8     | 最終挑戰          | 1     | 6~7    | 6~7     | 6~8   | 固定    |
    #
    # 表中 "~" 表示 mixed_parallel 的隨機採樣範圍，
    # 下面 num_obstacles_static/dynamic 設為該範圍的中位數或上限。
    # ==================================================================
    # ==================================================================
    # goal_first_v1: 先導航再避障 + stage-dependent reward weights
    #
    # Stage-dependent 權重設計理念：
    # - Phase A (1-2): 無障礙物 → safety 權重低，goal 權重高
    # - Phase B (3-4): 有 static → 提高 static_safety
    # - Phase C (5-8): 有 dynamic → 提高 dynamic_safety
    #
    # | Stage | vel | prog | ss  | ds  | 理由 |
    # |-------|-----|------|-----|-----|------|
    # | 1-2   | 4.0 | 5.0  | 0.5 | 0   | 專注 goal-reaching |
    # | 3-4   | 3.0 | 4.0  | 2.0 | 0.5 | 開始學 static 避障 |
    # | 5-6   | 2.0 | 3.0  | 2.0 | 2.0 | 加入 dynamic |
    # | 7-8   | 2.0 | 3.0  | 2.0 | 2.0 | 全功能 |
    # ==================================================================
    "goal_first_v1": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "stages": [
            # --- Phase A: Goal-reaching (Stage 1-2) ---
            _make_stage(7, 0, 0, 0, 1, 0.990, 45, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=50,
                        downgrade_sr=0.0, downgrade_min_cr=1.0, downgrade_min_to=1.0,
                        name="goal_reaching_open",
                        reward_weights={"goal_velocity": 4.0, "goal_progress": 5.0,
                                        "static_safety": 0.5, "dynamic_safety": 0.0}),
            _make_stage(6, 0, 0, 1, 2, 0.990, 50, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=60,
                        downgrade_sr=0.30, downgrade_min_cr=1.0, downgrade_min_to=0.80,
                        name="goal_walls",
                        reward_weights={"goal_velocity": 4.0, "goal_progress": 5.0,
                                        "static_safety": 0.5, "dynamic_safety": 0.0}),

            # --- Phase B: Static obstacles (Stage 3-4) ---
            _make_stage(5, 2, 0, 2, 3, 0.992, 55, 0.55,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=70,
                        name="static_intro",
                        reward_weights={"goal_velocity": 3.0, "goal_progress": 4.0,
                                        "static_safety": 2.0, "dynamic_safety": 0.5}),
            _make_stage(4, 3, 0, 3, 4, 0.993, 60, 0.40,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=85,
                        name="static_dense",
                        reward_weights={"goal_velocity": 3.0, "goal_progress": 4.0,
                                        "static_safety": 2.0, "dynamic_safety": 0.5}),

            # --- Phase C: Dynamic obstacles (Stage 5-8) ---
            _make_stage(4, 3, 1, 3, 4, 0.994, 65, 0.30,
                        upgrade_sr=0.75, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=100,
                        name="dynamic_intro",
                        reward_weights={"goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
            _make_stage(3, 4, 3, 4, 5, 0.995, 72, 0.20,
                        upgrade_sr=0.75, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=115,
                        name="dynamic_medium",
                        reward_weights={"goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
            _make_stage(2, 5, 4, 5, 6, 0.997, 80, 0.15,
                        upgrade_sr=0.70, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=130,
                        name="crowded",
                        reward_weights={"goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
            _make_stage(1, 7, 7, 6, 8, 0.998, 90, 0.15,
                        upgrade_sr=1.0, upgrade_max_cr=0.0, upgrade_max_to=0.0,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=0,
                        downgrade_sr=0.15, name="final_challenge",
                        reward_weights={"goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
        ],
    },

    # ==================================================================
    # goal_first_v2: 密度自適應權重 + 開放式 12 Stage (MAX_OBSTACLES=20)
    #
    # 設計原則:
    # 1. ss/ds weight 跟障礙物密度掛鉤（少障礙=低權重，密集=高權重）
    # 2. vel/prog weight 前期高後期低（先學 goal-reaching）
    # 3. 開放式：Stage 8 不是終點，可繼續增加到 Stage 12
    #
    # 公式: ss_w = ss_base × max(n_static/10, 0.1)
    #        ds_w = ds_base × max(n_dynamic/10, 0.0)
    #        vel_w = lerp(5.0, 2.0, stage_progress)
    #        prog_w = lerp(6.0, 3.0, stage_progress)
    # ==================================================================
    "goal_first_v2": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "stages": [
            # --- Phase A: Goal-reaching (Stage 1-2) ---
            # ss=0.1(floor), ds=0, vel=5, prog=6
            _make_stage(7, 0, 0, 0, 1, 0.990, 45, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=50,
                        downgrade_sr=0.0, downgrade_min_cr=1.0, downgrade_min_to=1.0,
                        name="goal_open",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 6.0,
                                        "static_safety": 0.2, "dynamic_safety": 0.0}),
            _make_stage(6, 0, 0, 1, 2, 0.990, 50, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=60,
                        downgrade_sr=0.30, downgrade_min_cr=1.0, downgrade_min_to=0.80,
                        name="goal_walls",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 6.0,
                                        "static_safety": 0.2, "dynamic_safety": 0.0}),

            # --- Phase B: Static obstacles (Stage 3-5) ---
            # ss 隨密度上升: 2/10=0.4, 4/10=0.8, 6/10=1.2
            _make_stage(5, 2, 0, 2, 3, 0.992, 55, 0.55,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=70,
                        name="static_light",
                        reward_weights={"goal_velocity": 4.5, "goal_progress": 5.5,
                                        "static_safety": 0.4, "dynamic_safety": 0.0}),
            _make_stage(4, 4, 0, 3, 4, 0.993, 60, 0.40,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=85,
                        name="static_medium",
                        reward_weights={"goal_velocity": 4.0, "goal_progress": 5.0,
                                        "static_safety": 0.8, "dynamic_safety": 0.0}),
            _make_stage(3, 6, 0, 3, 5, 0.994, 65, 0.30,
                        upgrade_sr=0.78, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=100,
                        name="static_dense",
                        reward_weights={"goal_velocity": 3.5, "goal_progress": 4.5,
                                        "static_safety": 1.2, "dynamic_safety": 0.0}),

            # --- Phase C: Dynamic obstacles (Stage 6-8) ---
            # reaching_goal 隨 γ/episode 增大: 1000 確保 V(reach) > V(stay)
            _make_stage(3, 5, 2, 4, 5, 0.995, 72, 0.20,
                        upgrade_sr=0.75, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=115,
                        name="dynamic_intro",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 3.0, "goal_progress": 4.0,
                                        "static_safety": 1.0, "dynamic_safety": 0.4}),
            _make_stage(2, 6, 4, 5, 6, 0.996, 78, 0.15,
                        upgrade_sr=0.72, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=130,
                        name="dynamic_medium",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 2.5, "goal_progress": 3.5,
                                        "static_safety": 1.2, "dynamic_safety": 0.8}),
            _make_stage(1, 7, 6, 6, 7, 0.997, 85, 0.15,
                        upgrade_sr=0.70, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=140,
                        name="crowded",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.4, "dynamic_safety": 1.2}),

            # --- Phase D: Open-ended (Stage 9-12, 超過 10 個障礙物) ---
            # reaching_goal=1500: γ=0.998 + 450 步 → V(stay) 可達 240+
            _make_stage(1, 8, 8, 7, 8, 0.998, 90, 0.15,
                        upgrade_sr=0.68, upgrade_max_cr=0.40, upgrade_max_to=0.35,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=150,
                        name="dense_9",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.6, "dynamic_safety": 1.6}),
            _make_stage(1, 9, 9, 7, 8, 0.998, 90, 0.10,
                        upgrade_sr=0.65, upgrade_max_cr=0.45, upgrade_max_to=0.35,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=160,
                        name="dense_10",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.8, "dynamic_safety": 1.8}),
            _make_stage(1, 10, 10, 8, 8, 0.998, 90, 0.10,
                        upgrade_sr=0.60, upgrade_max_cr=0.45, upgrade_max_to=0.40,
                        upgrade_min_dyn_sr=0.25, min_stage_updates=170,
                        name="dense_11",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
            _make_stage(1, 10, 10, 8, 8, 0.998, 90, 0.10,
                        upgrade_sr=1.0, upgrade_max_cr=0.0, upgrade_max_to=0.0,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=0,
                        downgrade_sr=0.15, name="ultimate",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
        ],
    },

    # ==================================================================
    # goal_first_v2_b: v2 基礎 + max_walls≤3 + gap reward weights
    #
    # 設計理由:
    # 1. max_walls 上限 3 — 減少長牆對繞行學習的干擾
    # 2. 加入 heading_to_gap / forward_clearance 的 stage-dependent 權重
    # 3. gap 權重在 Phase B/C 較高（需要繞行時），Phase A/D 較低
    # ==================================================================
    "goal_first_v2_b": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "stages": [
            # --- Phase A: Goal-reaching (Stage 1-2) — 無障礙，gap reward 極低 ---
            _make_stage(7, 0, 0, 0, 1, 0.990, 45, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=50,
                        downgrade_sr=0.0, downgrade_min_cr=1.0, downgrade_min_to=1.0,
                        name="goal_open",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 6.0,
                                        "static_safety": 0.2, "dynamic_safety": 0.0,
                                        "heading_to_gap": 0.0, "forward_clearance": 0.0}),
            _make_stage(6, 0, 0, 1, 2, 0.990, 50, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=60,
                        downgrade_sr=0.30, downgrade_min_cr=1.0, downgrade_min_to=0.80,
                        name="goal_walls",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 6.0,
                                        "static_safety": 0.2, "dynamic_safety": 0.0,
                                        "heading_to_gap": 0.5, "forward_clearance": 0.3}),

            # --- Phase B: Static obstacles (Stage 3-5) — gap reward 升高 ---
            _make_stage(5, 2, 0, 1, 2, 0.992, 55, 0.55,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=70,
                        name="static_light",
                        reward_weights={"goal_velocity": 4.5, "goal_progress": 5.5,
                                        "static_safety": 0.4, "dynamic_safety": 0.0,
                                        "heading_to_gap": 1.5, "forward_clearance": 1.0}),
            _make_stage(4, 4, 0, 2, 3, 0.993, 60, 0.40,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=85,
                        name="static_medium",
                        reward_weights={"goal_velocity": 4.0, "goal_progress": 5.0,
                                        "static_safety": 0.8, "dynamic_safety": 0.0,
                                        "heading_to_gap": 1.5, "forward_clearance": 1.0}),
            _make_stage(3, 6, 0, 2, 3, 0.994, 65, 0.30,
                        upgrade_sr=0.78, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=100,
                        name="static_dense",
                        reward_weights={"goal_velocity": 3.5, "goal_progress": 4.5,
                                        "static_safety": 1.2, "dynamic_safety": 0.0,
                                        "heading_to_gap": 1.5, "forward_clearance": 1.0}),

            # --- Phase C: Dynamic obstacles (Stage 6-8) — gap + dynamic ---
            _make_stage(3, 5, 2, 2, 3, 0.995, 72, 0.20,
                        upgrade_sr=0.75, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=115,
                        name="dynamic_intro",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 3.0, "goal_progress": 4.0,
                                        "static_safety": 1.0, "dynamic_safety": 0.4,
                                        "heading_to_gap": 1.5, "forward_clearance": 1.0}),
            _make_stage(2, 6, 4, 2, 3, 0.996, 78, 0.15,
                        upgrade_sr=0.72, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=130,
                        name="dynamic_medium",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 2.5, "goal_progress": 3.5,
                                        "static_safety": 1.2, "dynamic_safety": 0.8,
                                        "heading_to_gap": 1.5, "forward_clearance": 1.0}),
            _make_stage(1, 7, 6, 3, 3, 0.997, 85, 0.15,
                        upgrade_sr=0.70, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=140,
                        name="crowded",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.4, "dynamic_safety": 1.2,
                                        "heading_to_gap": 1.5, "forward_clearance": 1.0}),

            # --- Phase D: Open-ended (Stage 9-12) ---
            _make_stage(1, 8, 8, 3, 3, 0.998, 90, 0.15,
                        upgrade_sr=0.68, upgrade_max_cr=0.40, upgrade_max_to=0.35,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=150,
                        name="dense_9",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.6, "dynamic_safety": 1.6,
                                        "heading_to_gap": 1.0, "forward_clearance": 0.5}),
            _make_stage(1, 9, 9, 3, 3, 0.998, 90, 0.10,
                        upgrade_sr=0.65, upgrade_max_cr=0.45, upgrade_max_to=0.35,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=160,
                        name="dense_10",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.8, "dynamic_safety": 1.8,
                                        "heading_to_gap": 1.0, "forward_clearance": 0.5}),
            _make_stage(1, 10, 10, 3, 3, 0.998, 90, 0.10,
                        upgrade_sr=0.60, upgrade_max_cr=0.45, upgrade_max_to=0.40,
                        upgrade_min_dyn_sr=0.25, min_stage_updates=170,
                        name="dense_11",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0,
                                        "heading_to_gap": 1.0, "forward_clearance": 0.5}),
            _make_stage(1, 10, 10, 3, 3, 0.998, 90, 0.10,
                        upgrade_sr=1.0, upgrade_max_cr=0.0, upgrade_max_to=0.0,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=0,
                        downgrade_sr=0.15, name="ultimate",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0,
                                        "heading_to_gap": 1.0, "forward_clearance": 0.5}),
        ],
    },

    # ==================================================================
    # goal_first_v3: v2 基礎 + Stage 3-6 局部提高 static_safety
    #
    # 修正: v2 Stage 3-6 的安全側 (ss) 太弱，agent 偏向硬闖。
    # 只改 Stage 3-6，其餘 stage 與 v2 完全相同，確保可診斷。
    #
    # Stage 3: ss 0.4→0.8  | Stage 4: ss 0.8→1.4
    # Stage 5: ss 1.2→1.8  | Stage 6: ss 1.0→1.6
    # ==================================================================
    "goal_first_v3": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "stages": [
            # --- Phase A: Goal-reaching (Stage 1-2) — 與 v2 相同 ---
            _make_stage(7, 0, 0, 0, 1, 0.990, 45, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=50,
                        downgrade_sr=0.0, downgrade_min_cr=1.0, downgrade_min_to=1.0,
                        name="goal_open",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 6.0,
                                        "static_safety": 0.2, "dynamic_safety": 0.0}),
            _make_stage(6, 0, 0, 1, 2, 0.990, 50, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=60,
                        downgrade_sr=0.30, downgrade_min_cr=1.0, downgrade_min_to=0.80,
                        name="goal_walls",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 6.0,
                                        "static_safety": 0.2, "dynamic_safety": 0.0}),

            # --- Phase B: Static obstacles (Stage 3-5) — ss 局部提高 ---
            # v2: 0.4, 0.8, 1.2 → v3: 0.8, 1.4, 1.8
            _make_stage(5, 2, 0, 2, 3, 0.992, 55, 0.55,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=70,
                        name="static_light",
                        reward_weights={"goal_velocity": 4.5, "goal_progress": 5.5,
                                        "static_safety": 0.8, "dynamic_safety": 0.0}),
            _make_stage(4, 4, 0, 3, 4, 0.993, 60, 0.40,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=85,
                        name="static_medium",
                        reward_weights={"goal_velocity": 4.0, "goal_progress": 5.0,
                                        "static_safety": 1.4, "dynamic_safety": 0.0}),
            _make_stage(3, 6, 0, 3, 5, 0.994, 65, 0.30,
                        upgrade_sr=0.78, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=100,
                        name="static_dense",
                        reward_weights={"goal_velocity": 3.5, "goal_progress": 4.5,
                                        "static_safety": 1.8, "dynamic_safety": 0.0}),

            # --- Phase C: Dynamic obstacles (Stage 6-8) — Stage 6 ss 提高，7-8 同 v2 ---
            _make_stage(3, 5, 2, 4, 5, 0.995, 72, 0.20,
                        upgrade_sr=0.75, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=115,
                        name="dynamic_intro",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 3.0, "goal_progress": 4.0,
                                        "static_safety": 1.6, "dynamic_safety": 0.4}),
            # Stage 7-8: 與 v2 相同
            _make_stage(2, 6, 4, 5, 6, 0.996, 78, 0.15,
                        upgrade_sr=0.72, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=130,
                        name="dynamic_medium",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 2.5, "goal_progress": 3.5,
                                        "static_safety": 1.2, "dynamic_safety": 0.8}),
            _make_stage(1, 7, 6, 6, 7, 0.997, 85, 0.15,
                        upgrade_sr=0.70, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=140,
                        name="crowded",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.4, "dynamic_safety": 1.2}),

            # --- Phase D: Open-ended (Stage 9-12) — 與 v2 相同 ---
            _make_stage(1, 8, 8, 7, 8, 0.998, 90, 0.15,
                        upgrade_sr=0.68, upgrade_max_cr=0.40, upgrade_max_to=0.35,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=150,
                        name="dense_9",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.6, "dynamic_safety": 1.6}),
            _make_stage(1, 9, 9, 7, 8, 0.998, 90, 0.10,
                        upgrade_sr=0.65, upgrade_max_cr=0.45, upgrade_max_to=0.35,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=160,
                        name="dense_10",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.8, "dynamic_safety": 1.8}),
            _make_stage(1, 10, 10, 8, 8, 0.998, 90, 0.10,
                        upgrade_sr=0.60, upgrade_max_cr=0.45, upgrade_max_to=0.40,
                        upgrade_min_dyn_sr=0.25, min_stage_updates=170,
                        name="dense_11",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
            _make_stage(1, 10, 10, 8, 8, 0.998, 90, 0.10,
                        upgrade_sr=1.0, upgrade_max_cr=0.0, upgrade_max_to=0.0,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=0,
                        downgrade_sr=0.15, name="ultimate",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
        ],
    },
}


def _load_stages(version: str) -> tuple[dict, int, dict]:
    """根據版本載入 stages → (stages_dict, max_stage, version_config)"""
    if version not in CURRICULUM_CONFIGS:
        print(f"[Curriculum] WARNING: unknown version '{version}', fallback to baseline_v1")
        version = "baseline_v1"

    config = CURRICULUM_CONFIGS[version]
    stages = {}
    for i, stage_cfg in enumerate(config["stages"]):
        stages[i + 1] = stage_cfg
    return stages, len(config["stages"]), config


# ============================================================================
# 全局 STAGES（預設 baseline_v1 用於向後相容）
# ============================================================================
STAGES, MAX_STAGE, _DEFAULT_CONFIG = _load_stages("baseline_v1")


# ============================================================================
# 主函數
# ============================================================================

def goal_obstacle_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    window_size: int = 2000,
    min_stage_episodes: int = 5000,
    initial_stage: int = 1,
    curriculum_version: str = "baseline_v1",
) -> dict[str, float]:
    """資料驅動課程學習 — 支援多版本切換。

    Args:
        curriculum_version: "baseline_v1" (原 v12) 或 "goal_first_v1" (先導航再避障)
    """
    global STAGES, MAX_STAGE

    if not hasattr(env, "_goal_obs_curriculum"):
        # 根據 version 載入 stages
        stages, max_stage, ver_config = _load_stages(curriculum_version)
        STAGES = stages
        MAX_STAGE = max_stage
        upgrade_pass_required = ver_config.get("upgrade_pass_required", 5)

        n = env.num_envs
        effective_window = max(window_size, n * 5)
        effective_min = max(min_stage_episodes, n * 8)
        env._goal_obs_curriculum = {
            "stage": initial_stage,
            "outcome_window": deque(maxlen=effective_window),
            "effective_window_size": effective_window,
            "min_stage_episodes": effective_min,
            "total_episodes": 0,
            "stage_episodes": 0,
            "stage_transitions": 0,
            "upgrade_pass_count": 0,
            "upgrade_pass_required": upgrade_pass_required,
            "curriculum_version": curriculum_version,
            "clear_window_on_promote": ver_config.get("clear_window_on_promote", True),
        }
        _apply_stage(env, initial_stage)
        s = STAGES[initial_stage]
        print(
            f"\n{'='*70}\n"
            f"[Curriculum {curriculum_version}] 初始化 — Stage {initial_stage}: "
            f"{s.get('name', '')}\n"
            f"  num_envs={n}  |  窗口={effective_window}  |  "
            f"最低停留={effective_min} episodes\n"
            f"  Goals={s['num_goals']}  "
            f"障礙物={s['num_obstacles_static']}S+{s['num_obstacles_dynamic']}D  "
            f"牆壁={s['min_walls']}-{s['max_walls']}\n"
            f"  γ={s['gamma']}  episode={s['episode_length_s']}s\n"
            f"  升級: SR>{s['upgrade_sr']:.0%}"
            f"{'  CR<' + format(s['upgrade_max_cr'], '.0%') if s['upgrade_max_cr'] < 1.0 else ''}"
            f"  TO<{s['upgrade_max_to']:.0%}"
            f"  需連續通過 {upgrade_pass_required} 次\n"
            f"{'='*70}",
            flush=True,
        )

    state = env._goal_obs_curriculum
    upgrade_pass_required = state.get("upgrade_pass_required", 5)

    if env_ids is not None and not isinstance(env_ids, slice):
        _collect_episode_results(env, env_ids, state)

    window = state["outcome_window"]
    effective_window = state["effective_window_size"]

    # 計算整體 rates
    if len(window) >= 10:
        total = len(window)
        success_rate = sum(1 for o, _ in window if o == 1) / total
        collision_rate = sum(1 for o, _ in window if o == -1) / total
        timeout_rate = sum(1 for o, _ in window if o == 0) / total
    else:
        success_rate = 0.0
        collision_rate = 0.0
        timeout_rate = 0.0

    # dynamic-only SR
    dyn_outcomes = [(o, t) for o, t in window if t == 2]
    dynamic_sr = sum(1 for o, _ in dyn_outcomes if o == 1) / max(len(dyn_outcomes), 1)

    current_stage = state["stage"]
    stage_cfg = STAGES[current_stage]

    approx_rollout_cycles = state["stage_episodes"] // max(env.num_envs, 1)
    min_stage_updates = stage_cfg.get("min_stage_updates", 0)

    can_transition = (
        len(window) >= effective_window
        and state["stage_episodes"] >= state["min_stage_episodes"]
        and approx_rollout_cycles >= min_stage_updates
    )

    if can_transition:
        up_sr = stage_cfg["upgrade_sr"]
        up_cr = stage_cfg["upgrade_max_cr"]
        up_to = stage_cfg["upgrade_max_to"]
        up_dyn_sr = stage_cfg.get("upgrade_min_dyn_sr", 0.0)
        down_sr = stage_cfg["downgrade_sr"]
        down_cr = stage_cfg["downgrade_min_cr"]
        down_to = stage_cfg["downgrade_min_to"]

        total = len(window)
        success_count = sum(1 for o, _ in window if o == 1)
        collision_count = sum(1 for o, _ in window if o == -1)
        timeout_count = sum(1 for o, _ in window if o == 0)

        ver = state["curriculum_version"]

        def _debug_log(direction: str, old_stage: int, new_stage: int):
            s = STAGES[new_stage]
            print(
                f"\n{'='*70}\n"
                f"[Curriculum {ver}] {direction} Stage {old_stage} → "
                f"Stage {new_stage}: {s.get('name', '')}\n"
                f"  SR={success_rate:.4f} CR={collision_rate:.4f} TO={timeout_rate:.4f} "
                f"dynSR={dynamic_sr:.4f}\n"
                f"  累計 {state['total_episodes']} ep, "
                f"第 {state['stage_transitions']} 次轉換\n"
                f"  Goals={s['num_goals']}  "
                f"障礙物={s['num_obstacles_static']}S+{s['num_obstacles_dynamic']}D  "
                f"牆壁={s['min_walls']}-{s['max_walls']}\n"
                f"  γ={s['gamma']}  episode={s['episode_length_s']}s\n"
                f"{'='*70}",
                flush=True,
            )

        # 升級檢查
        sr_ok = success_rate > up_sr
        cr_ok = collision_rate < up_cr
        to_ok = timeout_rate < up_to
        dyn_ok = dynamic_sr > up_dyn_sr if (up_dyn_sr > 0 and len(dyn_outcomes) > 0) else True
        all_pass = sr_ok and cr_ok and to_ok and dyn_ok

        if all_pass and current_stage < MAX_STAGE:
            state["upgrade_pass_count"] += 1
            if state["upgrade_pass_count"] >= upgrade_pass_required:
                old = current_stage
                current_stage += 1
                state["stage"] = current_stage
                if state.get("clear_window_on_promote", True):
                    state["outcome_window"].clear()
                state["stage_episodes"] = 0
                state["stage_transitions"] += 1
                state["upgrade_pass_count"] = 0
                _apply_stage(env, current_stage)
                _debug_log(f"▲ 升級", old, current_stage)
                stage_cfg = STAGES[current_stage]
            else:
                print(
                    f"[Curriculum {ver}] 升級待確認 "
                    f"{state['upgrade_pass_count']}/{upgrade_pass_required} — "
                    f"SR={success_rate:.3f} CR={collision_rate:.3f} TO={timeout_rate:.3f}",
                    flush=True,
                )
        else:
            if state["upgrade_pass_count"] > 0:
                print(
                    f"[Curriculum {ver}] 升級計數歸零 "
                    f"(was {state['upgrade_pass_count']}/{upgrade_pass_required})",
                    flush=True,
                )
            state["upgrade_pass_count"] = 0

            # 降級
            if current_stage > 1:
                reason = None
                if success_rate < down_sr:
                    reason = f"SR={success_rate:.4f}<{down_sr}"
                elif collision_rate > down_cr:
                    reason = f"CR={collision_rate:.4f}>{down_cr}"
                elif timeout_rate > down_to:
                    reason = f"TO={timeout_rate:.4f}>{down_to}"

                if reason is not None:
                    old = current_stage
                    current_stage -= 1
                    state["stage"] = current_stage
                    if state.get("clear_window_on_promote", True):
                        state["outcome_window"].clear()
                    state["stage_episodes"] = 0
                    state["stage_transitions"] += 1
                    state["upgrade_pass_count"] = 0
                    _apply_stage(env, current_stage)
                    _debug_log(f"▼ 降級({reason})", old, current_stage)
                    stage_cfg = STAGES[current_stage]

    up_sr_target = stage_cfg["upgrade_sr"]
    up_cr_target = stage_cfg["upgrade_max_cr"]
    up_to_target = stage_cfg["upgrade_max_to"]

    return {
        "stage": float(current_stage),
        "stage_name": stage_cfg.get("name", ""),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "dynamic_sr": dynamic_sr,
        "num_goals": float(stage_cfg["num_goals"]),
        "num_obstacles_static": float(stage_cfg["num_obstacles_static"]),
        "num_obstacles_dynamic": float(stage_cfg["num_obstacles_dynamic"]),
        "min_walls": float(stage_cfg["min_walls"]),
        "max_walls": float(stage_cfg["max_walls"]),
        "gamma": float(stage_cfg["gamma"]),
        "episode_length_s": float(stage_cfg["episode_length_s"]),
        "num_episodes": float(state["total_episodes"]),
        "stage_episodes": float(state["stage_episodes"]),
        "window_fill": float(len(window)) / float(effective_window),
        "upgrade_pass_count": float(state["upgrade_pass_count"]),
        "approx_rollout_cycles": float(approx_rollout_cycles),
        "upgrade_sr_target": up_sr_target,
        "upgrade_cr_target": up_cr_target,
        "upgrade_to_target": up_to_target,
        "sr_gap": success_rate - up_sr_target,
        "cr_gap": up_cr_target - collision_rate,
        "to_gap": up_to_target - timeout_rate,
    }


def _stage_label(stage: int) -> str:
    return str(stage)


def _phase_name(stage: int) -> str:
    cfg = STAGES[stage]
    name = cfg.get("name", "")
    g = cfg["num_goals"]
    s = cfg["num_obstacles_static"]
    d = cfg["num_obstacles_dynamic"]
    label = f"{g}G/{s}S+{d}D"
    return f"{name} ({label})" if name else label


def _collect_episode_results(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    state: dict,
):
    """收集 per-env episode 結局：(outcome, env_type)"""
    import torch

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    if ids.numel() == 0:
        return

    difficulty = None
    if hasattr(env, '_env_difficulty') and env._env_difficulty is not None:
        difficulty = env._env_difficulty

    try:
        tm = env.termination_manager
        goal_buf = None
        coll_bufs = []
        for name in tm._term_names:
            if "goal_reached" in name and goal_buf is None:
                goal_buf = tm.get_term(name)
            elif "collision" in name:
                coll_bufs.append(tm.get_term(name))

        for idx in ids:
            eid = idx.item() if isinstance(idx, torch.Tensor) else int(idx)
            if goal_buf is not None and goal_buf[eid].item():
                outcome = 1
            elif any(buf[eid].item() for buf in coll_bufs):
                outcome = -1
            else:
                outcome = 0

            env_type = int(difficulty[eid].item()) if difficulty is not None else -1
            state["outcome_window"].append((outcome, env_type))
            state["total_episodes"] += 1
            state["stage_episodes"] += 1

    except Exception:
        n = ids.numel() if hasattr(ids, 'numel') else len(ids)
        for _ in range(n):
            state["outcome_window"].append((0, -1))
            state["total_episodes"] += 1
            state["stage_episodes"] += 1


def _apply_stage(env: ManagerBasedRLEnv, stage: int):
    """套用指定階段的環境參數 + reward 權重。"""
    cfg = STAGES[stage]

    try:
        cmd = env.command_manager.get_term("goal_command")
        cmd.cfg.num_goals = cfg["num_goals"]
        cmd.cfg.ranges.distance = cfg["goal_distance"]
        cmd.cfg.num_obstacles = cfg["num_obstacles_static"] + cfg["num_obstacles_dynamic"]
    except Exception:
        pass

    try:
        evt = env.event_manager
        for name in ["randomize_obstacles", "randomize_obstacles_startup"]:
            try:
                ec = evt.get_term_cfg(name)
                ec.params["empty_ratio"] = cfg["empty_ratio"]
                ec.params["static_ratio"] = cfg["static_ratio"]
                ec.params["dynamic_ratio"] = cfg["dynamic_ratio"]
                ec.params["num_obstacles_static"] = cfg["num_obstacles_static"]
                ec.params["num_obstacles_dynamic"] = cfg["num_obstacles_dynamic"]
                evt.set_term_cfg(name, ec)
            except Exception:
                continue
    except Exception:
        pass

    try:
        evt = env.event_manager
        ec = evt.get_term_cfg("randomize_wall_positions")
        ec.params["min_walls"] = cfg["min_walls"]
        ec.params["max_walls"] = cfg["max_walls"]
        evt.set_term_cfg("randomize_wall_positions", ec)
    except Exception:
        pass

    env.cfg.episode_length_s = cfg["episode_length_s"]
    env._target_discount_factor = cfg["gamma"]

    # --- Stage-dependent reward weights ---
    rw = cfg.get("reward_weights")
    if rw is not None:
        try:
            rm = env.reward_manager
            for term_name, weight in rw.items():
                try:
                    tc = rm.get_term_cfg(term_name)
                    tc.weight = weight
                    rm.set_term_cfg(term_name, tc)
                except Exception:
                    pass
            weights_str = " ".join(f"{k}={v}" for k, v in rw.items())
            print(f"[Curriculum] Stage {stage} reward weights: {weights_str}", flush=True)
        except Exception:
            pass


__all__ = ["goal_obstacle_curriculum", "CURRICULUM_CONFIGS", "STAGES", "MAX_STAGE"]
