# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Charge 導航任務模組（SKRL 專用）

保留 3 個 task，所有實驗變體靠 --run_name + CLI 參數切換：
- NavRL: 當前 baseline (NavRL dense rewards)
- NavRL-Ground: 新版地面機器人 reward
- Curriculum: v9 純死亡機制 (舊版對照)
"""

import gymnasium as gym

from . import agents
from .cfg import (
    ChargeNavigationEnvCfgVLP16Curriculum,
    ChargeNavigationEnvCfgVLP16CurriculumNavRL,
)

##
# 註冊 Gymnasium 環境 (SKRL)
##

_skrl_agents = "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.agents"

# ============================================================================
# VLP-16 Curriculum: 20×20m + 8 階段課程 (v9 純死亡機制 reward)
# ============================================================================
gym.register(
    id="Isaac-Navigation-Charge-VLP16-Curriculum",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16_curriculum:ChargeNavigationEnvCfgVLP16Curriculum",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16.yaml",
    },
)

# ============================================================================
# VLP-16 Curriculum NavRL: NavRL-Style Dense Rewards（主力 baseline）
# 消融實驗用 CLI 參數切換: --v_gate_mode / --progress_gate_mode / --use_gap_reward / --use_safety_shield
# ============================================================================
gym.register(
    id="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16_curriculum:ChargeNavigationEnvCfgVLP16CurriculumNavRL",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16.yaml",
    },
)

# ============================================================================
# NavRL-Ground v1: 地面差速機器人版 NavRL 獎勵（新版 reward 設計）
# 也可用 --reward_mode navrl_ground_v1 從 NavRL task 切換
# ============================================================================
gym.register(
    id="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-Ground",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16_curriculum:ChargeNavigationEnvCfgVLP16CurriculumNavRLGround",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16.yaml",
    },
)
