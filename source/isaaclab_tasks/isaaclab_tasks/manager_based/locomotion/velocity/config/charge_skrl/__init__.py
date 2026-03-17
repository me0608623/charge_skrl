# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Charge 導航任務模組（SKRL 專用）

此模組負責將 Charge 機器人的導航環境註冊到 Gymnasium 環境註冊表中，
使用 SKRL 作為強化學習框架。

Phase 0 配置：
- 16x16m 房間，四面牆壁，混合障礙物
- 單 Goal：到達即終止 episode
- Episode 長度 20 秒
"""

import gymnasium as gym

from . import agents
from .cfg import (
    ChargeNavigationEnvCfgPhase0,
    ChargeNavigationEnvCfgPhase0_PLAY,
    ChargeNavigationEnvCfgPhase0NavRL,
    ChargeNavigationEnvCfgCompetitive,
    ChargeNavigationEnvCfgVLP16,
    ChargeNavigationEnvCfgVLP16_PLAY,
)

# 實驗性配置（依賴模組可能尚未完成）
try:
    from .cfg.charge_env_cfg_maze import ChargeNavigationEnvCfgMaze
    _MAZE_AVAILABLE = True
except ImportError:
    ChargeNavigationEnvCfgMaze = None  # type: ignore[assignment, misc]
    _MAZE_AVAILABLE = False

##
# 註冊 Gymnasium 環境 (SKRL)
##

# SKRL agent config 模組路徑
_skrl_agents = "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.agents"

# ============================================================================
# Phase 0: 空房間 + 混合障礙物（車輛動力學校準，10-Goal 動態重生）
# ============================================================================

# Phase 0: 標準 MLP 架構
gym.register(
    id="Isaac-Navigation-Charge-Phase0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0:ChargeNavigationEnvCfgPhase0",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_phase0.yaml",
    },
)

gym.register(
    id="Isaac-Navigation-Charge-Phase0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0:ChargeNavigationEnvCfgPhase0_PLAY",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_phase0.yaml",
    },
)

# Phase 0 NavRL: NavRL 風格獎勵函數
gym.register(
    id="Isaac-Navigation-Charge-Phase0-NavRL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0_navrl:ChargeNavigationEnvCfgPhase0NavRL",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_phase0.yaml",
    },
)

# Phase 0 NavRL CNN: NavRL 風格獎勵 + CNN 架構
gym.register(
    id="Isaac-Navigation-Charge-Phase0-NavRL-CNN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0_navrl:ChargeNavigationEnvCfgPhase0NavRL",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_navrl.yaml",
    },
)

# Phase 0 CNN: 多分支 CNN 特徵萃取器（LiDAR Conv1d + 排列不變障礙物 + State MLP）
gym.register(
    id="Isaac-Navigation-Charge-Phase0-CNN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0:ChargeNavigationEnvCfgPhase0",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_charge_cnn.yaml",
    },
)

# ============================================================================
# Competitive: 競爭式多 Agent 訓練（實驗性）
# ============================================================================
if ChargeNavigationEnvCfgCompetitive is not None:
    gym.register(
        id="Isaac-Navigation-Charge-Competitive",
        entry_point="isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.competition.competitive_env:CompetitiveNavigationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_competitive:ChargeNavigationEnvCfgCompetitive",
            "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_competitive.yaml",
        },
    )

# ============================================================================
# Maze: 迷宮導航（實驗性）
# ============================================================================
if _MAZE_AVAILABLE:
    gym.register(
        id="Isaac-Navigation-Charge-Maze",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_maze:ChargeNavigationEnvCfgMaze",
            "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_phase0.yaml",
        },
    )

# ============================================================================
# VLP-16 v2: 離散動作 + 79D 觀測 baseline
# ============================================================================
gym.register(
    id="Isaac-Navigation-Charge-VLP16",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16:ChargeNavigationEnvCfgVLP16",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16.yaml",
    },
)

gym.register(
    id="Isaac-Navigation-Charge-VLP16-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16:ChargeNavigationEnvCfgVLP16_PLAY",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16.yaml",
    },
)

# ============================================================================
# VLP-16 對照實驗 (A/B/C 用同一環境，僅 PPO 配置不同; D 用新環境)
# ============================================================================

# Exp A: Reward Scale ÷10
gym.register(
    id="Isaac-Navigation-Charge-VLP16-ExpA",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16:ChargeNavigationEnvCfgVLP16",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16_expA.yaml",
    },
)

# Exp B: gamma 0.99
gym.register(
    id="Isaac-Navigation-Charge-VLP16-ExpB",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16:ChargeNavigationEnvCfgVLP16",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16_expB.yaml",
    },
)

# Exp C: value_coef 1.0
gym.register(
    id="Isaac-Navigation-Charge-VLP16-ExpC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16:ChargeNavigationEnvCfgVLP16",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16_expC.yaml",
    },
)

# Exp D: 近障速度懲罰
gym.register(
    id="Isaac-Navigation-Charge-VLP16-ExpD",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16_expD:ChargeNavigationEnvCfgVLP16_ExpD",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16_expD.yaml",
    },
)
