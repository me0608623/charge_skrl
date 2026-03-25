# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
播放訓練好的 Charge VLP16 AC agent（支援 CADN + NavRL reward modes）。

預設 6 envs，可指定 --stage 1~8 逐一觀察各階段。
不帶 --stage 時自動依序跑所有 stage 各一輪（每 stage 跑 500 步後切換）。

使用方法:
    # 自動依序跑 8 個 stage（每 stage 500 步）
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac.py \
        --checkpoint logs/.../agent_152334.pt

    # 固定某一 stage
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac.py \
        --checkpoint logs/.../agent_152334.pt --stage 7

    # Headless 快速評估
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac.py \
        --checkpoint logs/.../agent_152334.pt --headless --max_steps 10000
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import glob
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from isaaclab.app import AppLauncher

# ============================================================================
# CLI Arguments
# ============================================================================
parser = argparse.ArgumentParser(description="Play trained Charge VLP16 AC agent.")
parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL",
                    help="Task name.")
parser.add_argument("--num_envs", type=int, default=6, help="Number of environments.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--stage", type=int, default=None,
                    help="Fixed curriculum stage (1-8). Default: auto-cycle all stages.")
parser.add_argument("--steps_per_stage", type=int, default=500,
                    help="Steps per stage in auto-cycle mode.")
parser.add_argument("--max_steps", type=int, default=5000, help="Max total steps.")
parser.add_argument("--camera", type=str, default="top", choices=["top", "follow", "side"],
                    help="Camera view.")
parser.add_argument("--reward_mode", type=str, default="navrl_ground_v4",
                    choices=["current", "navrl_ground_v1", "navrl_ground_v2", "navrl_ground_v3", "navrl_ground_v4"],
                    help="Reward mode (for matching training config).")
parser.add_argument("--curriculum_version", type=str, default="goal_first_v2",
                    choices=["baseline_v1", "goal_first_v1", "goal_first_v2"],
                    help="Curriculum version (for loading stage configs).")
parser.add_argument("--use_cadn", action="store_true", default=False,
                    help="Use CADN normalizer (must match training config).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import skrl
from packaging import version
from skrl.utils.runner.torch import Runner
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper

sys.path.insert(0, str(Path(__file__).parent))
import vlp16_models as _vlp16_models_module
import charge_models as _charge_models_module
_CUSTOM_MODEL_MODULES = {
    "charge_models": _charge_models_module,
    "vlp16_models": _vlp16_models_module,
}

import isaaclab_tasks  # noqa: F401

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    print(f"[ERROR] skrl >= {SKRL_VERSION} required, got {skrl.__version__}")
    exit()


# ============================================================================
# Patch Runner for custom model classes
# ============================================================================
def patch_runner_for_custom_models():
    original_generate_models = Runner._generate_models

    def patched_generate_models(self, env, cfg):
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("No 'models' defined in cfg")
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True

            if separate:
                for role in models_cfg:
                    model_class_name = models_cfg[role].get("class")
                    if not model_class_name:
                        raise ValueError(f"No 'class' in 'models:{role}'")
                    del models_cfg[role]["class"]

                    if "." in model_class_name:
                        parts = model_class_name.split(".")
                        module = _CUSTOM_MODEL_MODULES.get(parts[0])
                        if module is not None:
                            model_class = getattr(module, parts[1])
                        else:
                            raise ValueError(f"Module '{parts[0]}' not found")
                    else:
                        model_class = self._component(model_class_name)

                    models[agent_id][role] = model_class(
                        observation_space=observation_spaces[agent_id],
                        action_space=action_spaces[agent_id],
                        device=device,
                        **self._process_cfg(models_cfg[role]),
                    )
            else:
                raise ValueError("Shared models not supported")
        return models

    Runner._generate_models = patched_generate_models


patch_runner_for_custom_models()


# ============================================================================
# Checkpoint finder
# ============================================================================
def find_latest_checkpoint():
    log_base = "logs/skrl/Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-AC"
    if not os.path.exists(log_base):
        return None

    for run_dir in sorted(os.listdir(log_base), reverse=True):
        best = os.path.join(log_base, run_dir, "best_model", "best_agent.pt")
        if os.path.exists(best):
            return best
        ckpt_best = os.path.join(log_base, run_dir, "checkpoints", "best_agent.pt")
        if os.path.exists(ckpt_best):
            return ckpt_best
        ckpt_dir = os.path.join(log_base, run_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            pts = sorted(glob.glob(os.path.join(ckpt_dir, "agent_*.pt")))
            if pts:
                return pts[-1]
    return None


# ============================================================================
# Stage application (mirrors curriculum's _apply_stage for play mode)
# ============================================================================
def apply_stage_to_env(env, stage_cfg):
    """Apply a stage config to a running env (obstacles, walls, goals, episode length)."""
    try:
        cmd = env.unwrapped.command_manager.get_term("goal_command")
        cmd.cfg.num_goals = stage_cfg["num_goals"]
        cmd.cfg.ranges.distance = stage_cfg["goal_distance"]
    except Exception:
        pass

    try:
        evt = env.unwrapped.event_manager
        for name in ["randomize_obstacles", "randomize_obstacles_startup"]:
            try:
                ec = evt.get_term_cfg(name)
                ec.params["empty_ratio"] = stage_cfg["empty_ratio"]
                ec.params["static_ratio"] = stage_cfg["static_ratio"]
                ec.params["dynamic_ratio"] = stage_cfg["dynamic_ratio"]
                ec.params["num_obstacles_static"] = stage_cfg["num_obstacles_static"]
                ec.params["num_obstacles_dynamic"] = stage_cfg["num_obstacles_dynamic"]
                evt.set_term_cfg(name, ec)
            except Exception:
                continue
    except Exception:
        pass

    try:
        evt = env.unwrapped.event_manager
        ec = evt.get_term_cfg("randomize_wall_positions")
        ec.params["min_walls"] = stage_cfg["min_walls"]
        ec.params["max_walls"] = stage_cfg["max_walls"]
        evt.set_term_cfg("randomize_wall_positions", ec)
    except Exception:
        pass

    env.unwrapped.cfg.episode_length_s = stage_cfg["episode_length_s"]


# ============================================================================
# Main
# ============================================================================
def main():
    import math

    # Load stages
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
        _load_stages,
    )
    stages, max_stage, _ = _load_stages(args_cli.curriculum_version)

    print(f"\n{'='*70}")
    print(f"  Charge VLP16 AC — Play Mode")
    print(f"{'='*70}")
    print(f"  Task:       {args_cli.task}")
    print(f"  Reward:     {args_cli.reward_mode}")
    print(f"  Curriculum: {args_cli.curriculum_version} ({max_stage} stages)")
    print(f"  CADN:       {'ON' if args_cli.use_cadn else 'OFF'}")
    print(f"  Envs:       {args_cli.num_envs}")
    if args_cli.stage:
        print(f"  Stage:      {args_cli.stage} (fixed)")
    else:
        print(f"  Stage:      auto-cycle (1→{max_stage}, {args_cli.steps_per_stage} steps each)")
    print(f"  Checkpoint: {args_cli.checkpoint or 'Auto-detect'}")
    print(f"  Camera:     {args_cli.camera}")
    print(f"{'='*70}\n")

    # --- Load env config ---
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
        ChargeNavigationEnvCfgVLP16CurriculumNavRL,
    )
    env_cfg = ChargeNavigationEnvCfgVLP16CurriculumNavRL()
    env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Camera
    from isaaclab.envs import ViewerCfg
    if args_cli.camera == "top":
        env_cfg.viewer = ViewerCfg(
            eye=(0.0, 0.0, 30.0), lookat=(0.0, 0.0, 0.0),
            origin_type="env", env_index=0, resolution=(1920, 1080),
        )
    elif args_cli.camera == "follow":
        env_cfg.viewer = ViewerCfg(
            eye=(-3.0, 0.0, 3.0), lookat=(2.0, 0.0, 0.0),
            origin_type="asset_root", env_index=0, asset_name="robot",
            resolution=(1920, 1080),
        )
    elif args_cli.camera == "side":
        env_cfg.viewer = ViewerCfg(
            eye=(15.0, -15.0, 15.0), lookat=(0.0, 0.0, 0.0),
            origin_type="env", env_index=0, resolution=(1920, 1080),
        )

    # Disable curriculum (play mode uses manual stage control)
    env_cfg.curriculum = None

    # Set initial stage config
    init_stage = args_cli.stage or 1
    s = stages[init_stage]
    env_cfg.commands.goal_command.num_goals = s["num_goals"]
    env_cfg.commands.goal_command.ranges.distance = s["goal_distance"]
    env_cfg.commands.goal_command.ranges.angle = (-math.pi, math.pi)

    # --- Load agent config ---
    agent_yaml = Path(
        "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
        "locomotion/velocity/config/charge_skrl/agents/skrl_ppo_cfg_vlp16.yaml"
    )
    with open(agent_yaml) as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["agent"]["experiment"]["directory"] = "logs/skrl/play"
    agent_cfg["agent"]["experiment"]["experiment_name"] = "play"
    agent_cfg["trainer"]["timesteps"] = 0

    # --- Create env ---
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # --- Setup agent ---
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    runner = Runner(env, agent_cfg)

    # CADN: replace state_preprocessor before loading checkpoint
    if args_cli.use_cadn:
        from cadn import PerBranchCADN
        cadn = PerBranchCADN(device=env.device)
        runner.agent._state_preprocessor = cadn
        runner.agent.checkpoint_modules["state_preprocessor"] = cadn
        print("[INFO] CADN enabled")

    # Load checkpoint
    checkpoint_path = args_cli.checkpoint
    if checkpoint_path:
        checkpoint_path = retrieve_file_path(checkpoint_path)
    else:
        checkpoint_path = find_latest_checkpoint()

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            runner.agent.load(checkpoint_path)
            print(f"[OK] Loaded: {checkpoint_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[WARN] No checkpoint — running random policy!")

    runner.agent.set_running_mode("eval")

    # --- Apply initial stage ---
    apply_stage_to_env(env, stages[init_stage])
    print(f"[Stage {init_stage}] {s.get('name', '')} — "
          f"{s['num_goals']}G / {s['num_obstacles_static']}S+{s['num_obstacles_dynamic']}D / "
          f"walls {s['min_walls']}-{s['max_walls']}")

    # --- Play loop ---
    print(f"\n{'='*70}")
    print(f"  Running... (max {args_cli.max_steps} steps, Ctrl+C to stop)")
    print(f"{'='*70}\n")

    current_stage = init_stage
    stage_results = {}  # stage → {success, collision, timeout}
    stage_step_count = 0
    step = 0
    start_time = time.time()

    def _init_stage_counters(stg):
        if stg not in stage_results:
            stage_results[stg] = {"success": 0, "collision": 0, "timeout": 0}

    _init_stage_counters(current_stage)

    try:
        obs, _ = env.reset()
        for step in range(args_cli.max_steps):
            with torch.no_grad():
                actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
            obs, reward, terminated, truncated, infos = env.step(actions)
            stage_step_count += 1

            # Count episode outcomes
            try:
                tm = env.unwrapped.unwrapped.termination_manager
                for name in tm._term_names:
                    buf = tm.get_term(name)
                    if buf is not None:
                        count = int(buf.sum().item())
                        if count > 0:
                            if "goal_reached" in name:
                                stage_results[current_stage]["success"] += count
                            elif "collision" in name:
                                stage_results[current_stage]["collision"] += count
                            elif "time_out" in name:
                                stage_results[current_stage]["timeout"] += count
            except Exception:
                pass

            # Auto-cycle stages
            if args_cli.stage is None and stage_step_count >= args_cli.steps_per_stage:
                # Print stage summary
                r = stage_results[current_stage]
                total_ep = r["success"] + r["collision"] + r["timeout"]
                if total_ep > 0:
                    sr = r["success"] / total_ep * 100
                    cr = r["collision"] / total_ep * 100
                    to = r["timeout"] / total_ep * 100
                    s_cfg = stages[current_stage]
                    print(f"  Stage {current_stage} ({s_cfg.get('name', ''):20s}) | "
                          f"Ep={total_ep:>3d} SR={sr:5.1f}% CR={cr:5.1f}% TO={to:5.1f}%")

                # Next stage
                current_stage = current_stage % max_stage + 1
                _init_stage_counters(current_stage)
                stage_step_count = 0
                s_cfg = stages[current_stage]
                apply_stage_to_env(env, s_cfg)
                obs, _ = env.reset()

            # Periodic status for fixed-stage mode
            elif args_cli.stage is not None and step % 500 == 0 and step > 0:
                r = stage_results[current_stage]
                total_ep = r["success"] + r["collision"] + r["timeout"]
                if total_ep > 0:
                    sr = r["success"] / total_ep * 100
                    cr = r["collision"] / total_ep * 100
                    to = r["timeout"] / total_ep * 100
                    print(f"  Step {step:>5} | Stage {current_stage} | "
                          f"Ep={total_ep} SR={sr:.1f}% CR={cr:.1f}% TO={to:.1f}%")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    # --- Summary ---
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  PLAY SUMMARY — {args_cli.reward_mode} / {args_cli.curriculum_version}")
    print(f"{'='*70}")
    print(f"  Duration: {elapsed:.1f}s | Steps: {step + 1}")
    print(f"")
    print(f"  {'Stage':>5s}  {'Name':20s}  {'Ep':>4s}  {'SR':>6s}  {'CR':>6s}  {'TO':>6s}")
    print(f"  {'-'*65}")
    total_all = {"success": 0, "collision": 0, "timeout": 0}
    for stg in sorted(stage_results.keys()):
        r = stage_results[stg]
        total_ep = r["success"] + r["collision"] + r["timeout"]
        if total_ep == 0:
            continue
        sr = r["success"] / total_ep * 100
        cr = r["collision"] / total_ep * 100
        to = r["timeout"] / total_ep * 100
        name = stages[stg].get("name", "")
        print(f"  {stg:>5d}  {name:20s}  {total_ep:>4d}  {sr:5.1f}%  {cr:5.1f}%  {to:5.1f}%")
        for k in total_all:
            total_all[k] += r[k]

    grand_total = sum(total_all.values())
    if grand_total > 0:
        print(f"  {'-'*65}")
        print(f"  {'ALL':>5s}  {'':20s}  {grand_total:>4d}  "
              f"{total_all['success']/grand_total*100:5.1f}%  "
              f"{total_all['collision']/grand_total*100:5.1f}%  "
              f"{total_all['timeout']/grand_total*100:5.1f}%")
    print(f"{'='*70}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
