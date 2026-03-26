# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
播放訓練好的 Charge VLP16 Curriculum AC agent — 自動逐階段展示。

預設 6 envs，從 Stage 1 開始，每階段跑 200 steps 後自動升階到 Stage 8。

使用方法:
    # 預設：6 envs, stage 1→8, 每階 200 steps
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py

    # 指定 checkpoint
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py \
        --checkpoint logs/skrl/.../best_agent.pt

    # 自訂起始階段和每階步數
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py \
        --start_stage 3 --steps_per_stage 500

    # 只跑單一階段（不自動升階）
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py \
        --start_stage 5 --end_stage 5

    # 跟隨視角
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py --camera follow

    # 不使用課程學習，自訂障礙物與牆壁數量
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py \
        --no_curriculum --num_static 5 --num_dynamic 4 --num_walls 6 --max_steps 2000
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
parser = argparse.ArgumentParser(description="Play trained Charge VLP16 Curriculum AC agent.")
parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-VLP16-Curriculum",
                    help="Task name (used for gym.make).")
parser.add_argument("--num_envs", type=int, default=6, help="Number of environments.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--start_stage", type=int, default=1, help="Starting curriculum stage (1-8).")
parser.add_argument("--end_stage", type=int, default=8, help="Final curriculum stage (1-8).")
parser.add_argument("--steps_per_stage", type=int, default=200, help="Steps per stage before advancing.")
parser.add_argument("--curriculum_version", type=str, default="baseline_v1",
                    help="Curriculum version to use (baseline_v1, goal_first_v1).")
parser.add_argument("--no_curriculum", action="store_true", default=False,
                    help="Disable stage cycling; use fixed obstacle/wall counts from CLI.")
parser.add_argument("--num_static", type=int, default=3, help="Static obstacles per env (--no_curriculum mode).")
parser.add_argument("--num_dynamic", type=int, default=3, help="Dynamic obstacles per env (--no_curriculum mode).")
parser.add_argument("--num_walls", type=int, default=None, help="Number of internal walls (overrides curriculum stage walls if set).")
parser.add_argument("--max_steps", type=int, default=1600, help="Max steps in --no_curriculum mode.")
parser.add_argument("--camera", type=str, default="top", choices=["top", "follow", "side"],
                    help="Camera view.")
parser.add_argument("--use_cadn", action="store_true", default=False,
                    help="Use PerBranchCADN normalizer (required when checkpoint was trained with CADN).")

# AppLauncher args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- Play 模式渲染最佳化 ---
if not args_cli.headless:
    if args_cli.rendering_mode is None or args_cli.rendering_mode == "balanced":
        args_cli.rendering_mode = "performance"
    if not getattr(args_cli, "enable_cameras", False):
        args_cli.enable_cameras = True
    extra_kit = " ".join([
        "--/rtx/indirectDiffuse/enabled=true",
        "--/rtx/shadows/enabled=false",
        "--/app/asyncRendering=true",
        "--/app/asyncRenderingLowLatency=true",
    ])
    args_cli.kit_args = f"{args_cli.kit_args} {extra_kit}" if args_cli.kit_args else extra_kit

# Launch sim
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

# Custom model modules
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
    """自動找到最新的 best_agent.pt。"""
    task_name = args_cli.task or ""
    if "NavRL" in task_name:
        log_base = "logs/skrl/Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-AC"
    else:
        log_base = "logs/skrl/Isaac-Navigation-Charge-VLP16-Curriculum-AC"
    if not os.path.exists(log_base):
        print(f"[CHECKPOINT] Directory not found: {log_base}")
        return None

    for run_dir in sorted(os.listdir(log_base), reverse=True):
        # best_model/best_agent.pt
        best = os.path.join(log_base, run_dir, "best_model", "best_agent.pt")
        if os.path.exists(best):
            print(f"[CHECKPOINT] Found: {best}")
            return best
        # checkpoints/best_agent.pt
        ckpt_best = os.path.join(log_base, run_dir, "checkpoints", "best_agent.pt")
        if os.path.exists(ckpt_best):
            print(f"[CHECKPOINT] Found: {ckpt_best}")
            return ckpt_best
        # highest agent_NNNN.pt
        ckpt_dir = os.path.join(log_base, run_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            pts = sorted(glob.glob(os.path.join(ckpt_dir, "agent_*.pt")))
            if pts:
                print(f"[CHECKPOINT] Found: {pts[-1]}")
                return pts[-1]

    print("[CHECKPOINT] No checkpoint found")
    return None


# ============================================================================
# Main
# ============================================================================
def main():
    import math

    use_curriculum = not args_cli.no_curriculum

    # --- Load env config ---
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
        ChargeNavigationEnvCfgVLP16Curriculum,
        ChargeNavigationEnvCfgVLP16CurriculumNavRL,
    )
    if "NavRL" in args_cli.task:
        env_cfg = ChargeNavigationEnvCfgVLP16CurriculumNavRL()
    else:
        env_cfg = ChargeNavigationEnvCfgVLP16Curriculum()
    env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Disable training curriculum (we control stage transitions or use fixed config)
    env_cfg.curriculum = None

    # Play mode: goal/robot 離牆壁/邊界至少 1.0m
    env_cfg.commands.goal_command.wall_safe_margin = 1.0

    # Play 模式: render_interval 折衷（每 action 渲染 5 幀 ~25 FPS）
    if not args_cli.headless:
        env_cfg.sim.render_interval = 4

    if use_curriculum:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
            _apply_stage, _load_stages,
        )
        stages, max_stage, _ = _load_stages(args_cli.curriculum_version)
        start_stage = max(1, min(args_cli.start_stage, max_stage))
        end_stage = max(start_stage, min(args_cli.end_stage, max_stage))
        steps_per_stage = args_cli.steps_per_stage

        print(f"\n{'='*70}")
        print(f"  Charge VLP16 Curriculum — Play Mode")
        print(f"{'='*70}")
        print(f"  Envs:            {args_cli.num_envs}")
        print(f"  Curriculum:      {args_cli.curriculum_version}")
        print(f"  Stages:          {start_stage} → {end_stage}  ({end_stage - start_stage + 1} stages)")
        print(f"  Steps/stage:     {steps_per_stage}")
        print(f"  Checkpoint:      {args_cli.checkpoint or 'Auto-detect'}")
        print(f"  Camera:          {args_cli.camera}")
        print(f"{'='*70}\n")

        # Apply initial stage config BEFORE env creation
        s1 = stages[start_stage]
        for evt_attr in ["randomize_obstacles", "randomize_obstacles_startup"]:
            evt_term = getattr(env_cfg.events, evt_attr, None)
            if evt_term is not None:
                evt_term.params.update({
                    "empty_ratio": s1["empty_ratio"],
                    "static_ratio": s1["static_ratio"],
                    "dynamic_ratio": s1["dynamic_ratio"],
                    "num_obstacles_static": s1["num_obstacles_static"],
                    "num_obstacles_dynamic": s1["num_obstacles_dynamic"],
                })
        wall_evt = getattr(env_cfg.events, "randomize_wall_positions", None)
        if wall_evt is not None:
            if args_cli.num_walls is not None:
                wall_evt.params["min_walls"] = args_cli.num_walls
                wall_evt.params["max_walls"] = args_cli.num_walls
            else:
                wall_evt.params["min_walls"] = s1["min_walls"]
                wall_evt.params["max_walls"] = s1["max_walls"]

        env_cfg.commands.goal_command.num_goals = s1["num_goals"]
        env_cfg.commands.goal_command.num_obstacles = s1["num_obstacles_static"] + s1["num_obstacles_dynamic"]
        env_cfg.commands.goal_command.ranges.distance = s1["goal_distance"]
        env_cfg.commands.goal_command.ranges.angle = (-math.pi, math.pi)

    else:
        # --- Fixed mode: user-specified obstacles/walls ---
        n_s = args_cli.num_static
        n_d = args_cli.num_dynamic
        n_w = args_cli.num_walls if args_cli.num_walls is not None else 4
        has_obs = (n_s + n_d) > 0

        # 障礙物分配: 同時有 static + dynamic → mixed 模式
        if n_s > 0 and n_d > 0:
            obs_params = {
                "empty_ratio": 0.0, "static_ratio": 0.0, "dynamic_ratio": 0.0,
                "mixed_ratio": 1.0,
                "num_obstacles_static": n_s, "num_obstacles_dynamic": n_d,
            }
        elif n_d > 0:
            obs_params = {
                "empty_ratio": 0.0, "static_ratio": 0.0, "dynamic_ratio": 1.0,
                "num_obstacles_static": 0, "num_obstacles_dynamic": n_d,
            }
        elif n_s > 0:
            obs_params = {
                "empty_ratio": 0.0, "static_ratio": 1.0, "dynamic_ratio": 0.0,
                "num_obstacles_static": n_s, "num_obstacles_dynamic": 0,
            }
        else:
            obs_params = {
                "empty_ratio": 1.0, "static_ratio": 0.0, "dynamic_ratio": 0.0,
                "num_obstacles_static": 0, "num_obstacles_dynamic": 0,
            }

        max_obs = max(n_s + n_d, 10)
        obs_params["max_obstacles"] = max_obs
        for evt_attr in ["randomize_obstacles", "randomize_obstacles_startup"]:
            evt_term = getattr(env_cfg.events, evt_attr, None)
            if evt_term is not None:
                evt_term.params.update(obs_params)
        # 同步 move_dynamic_obstacles 的 max_obstacles
        move_evt = getattr(env_cfg.events, "move_dynamic_obstacles", None)
        if move_evt is not None:
            move_evt.params["max_obstacles"] = max_obs
        wall_evt = getattr(env_cfg.events, "randomize_wall_positions", None)
        if wall_evt is not None:
            wall_evt.params["min_walls"] = n_w
            wall_evt.params["max_walls"] = n_w

        env_cfg.commands.goal_command.num_goals = 1
        env_cfg.commands.goal_command.num_obstacles = n_s + n_d
        env_cfg.commands.goal_command.ranges.distance = (2.0, 14.0)
        env_cfg.commands.goal_command.ranges.angle = (-math.pi, math.pi)

        print(f"\n{'='*70}")
        print(f"  Charge VLP16 — Fixed Play Mode (no curriculum)")
        print(f"{'='*70}")
        print(f"  Envs:            {args_cli.num_envs}")
        print(f"  Static obs:      {n_s}")
        print(f"  Dynamic obs:     {n_d}")
        print(f"  Walls:           {n_w}")
        print(f"  Max steps:       {args_cli.max_steps}")
        print(f"  Checkpoint:      {args_cli.checkpoint or 'Auto-detect'}")
        print(f"  Camera:          {args_cli.camera}")
        print(f"{'='*70}\n")

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

    # --- Load agent config from YAML ---
    agent_yaml = Path(__file__).parent.parent.parent.parent / (
        "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
        "locomotion/velocity/config/charge_skrl/agents/skrl_ppo_cfg_vlp16.yaml"
    )
    if not agent_yaml.exists():
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
        print("[INFO] CADN normalizer enabled")

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

    # --- Unwrap to get the real ManagerBasedRLEnv for _apply_stage ---
    raw_env = env.unwrapped
    while hasattr(raw_env, "unwrapped") and raw_env is not raw_env.unwrapped:
        raw_env = raw_env.unwrapped

    # --- Play loop ---
    global_start = time.time()
    global_step = 0

    def _count_episodes(raw_env, counts):
        """Accumulate termination counts from current step."""
        try:
            tm = raw_env.termination_manager
            for name in tm._term_names:
                buf = tm.get_term(name)
                if buf is not None:
                    c = int(buf.sum().item())
                    if c > 0:
                        if "goal_reached" in name:
                            counts["success"] += c
                        elif "collision" in name:
                            counts["collision"] += c
                        elif "time_out" in name:
                            counts["timeout"] += c
        except Exception:
            pass

    if use_curriculum:
        # ── Curriculum mode: cycle through stages ──
        stage_results = []

        try:
            for current_stage in range(start_stage, end_stage + 1):
                s_cfg = stages[current_stage]
                s_name = s_cfg.get("name", "")
                s_goals = s_cfg["num_goals"]
                s_static = s_cfg["num_obstacles_static"]
                s_dynamic = s_cfg["num_obstacles_dynamic"]
                if args_cli.num_walls is not None:
                    s_walls = str(args_cli.num_walls)
                else:
                    s_walls = f"{s_cfg['min_walls']}-{s_cfg['max_walls']}"

                print(f"\n{'─'*70}")
                print(f"  Stage {current_stage}/{max_stage}: {s_name}")
                print(f"  {s_goals}G / {s_static}S+{s_dynamic}D / walls {s_walls} / "
                      f"γ={s_cfg['gamma']} / ep={s_cfg['episode_length_s']}s")
                print(f"  Running {steps_per_stage} steps...")
                print(f"{'─'*70}")

                _apply_stage(raw_env, current_stage)

                # Override walls after _apply_stage if CLI specified
                if args_cli.num_walls is not None:
                    try:
                        evt = raw_env.event_manager
                        ec = evt.get_term_cfg("randomize_wall_positions")
                        ec.params["min_walls"] = args_cli.num_walls
                        ec.params["max_walls"] = args_cli.num_walls
                        evt.set_term_cfg("randomize_wall_positions", ec)
                    except Exception:
                        pass

                obs, _ = env.reset()

                counts = {"success": 0, "collision": 0, "timeout": 0}
                stage_start = time.time()

                for _ in range(steps_per_stage):
                    with torch.no_grad():
                        actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
                    obs, reward, terminated, truncated, infos = env.step(actions)
                    global_step += 1
                    _count_episodes(raw_env, counts)

                stage_elapsed = time.time() - stage_start
                total_ep = sum(counts.values())
                sr = counts["success"] / total_ep * 100 if total_ep > 0 else 0
                cr = counts["collision"] / total_ep * 100 if total_ep > 0 else 0
                tr = counts["timeout"] / total_ep * 100 if total_ep > 0 else 0

                stage_results.append({
                    "stage": current_stage, "name": s_name,
                    "episodes": total_ep, "sr": sr, "cr": cr, "tr": tr,
                    "elapsed": stage_elapsed,
                })
                print(f"  Stage {current_stage} done: {stage_elapsed:.1f}s | "
                      f"Ep={total_ep} SR={sr:.0f}% CR={cr:.0f}% TO={tr:.0f}%")

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user")

        # Curriculum summary
        total_elapsed = time.time() - global_start
        print(f"\n{'='*70}")
        print(f"  PLAY SUMMARY — {args_cli.curriculum_version}")
        print(f"{'='*70}")
        print(f"  {'Stage':<8} {'Name':<20} {'Ep':>4} {'SR':>6} {'CR':>6} {'TO':>6} {'Time':>6}")
        print(f"  {'─'*56}")
        total_ep_all = 0
        total_sr_all = 0
        total_cr_all = 0
        for r in stage_results:
            print(f"  {r['stage']:<8} {r['name']:<20} {r['episodes']:>4} "
                  f"{r['sr']:>5.0f}% {r['cr']:>5.0f}% {r['tr']:>5.0f}% {r['elapsed']:>5.1f}s")
            total_ep_all += r["episodes"]
            total_sr_all += r["episodes"] * r["sr"] / 100
            total_cr_all += r["episodes"] * r["cr"] / 100
        print(f"  {'─'*56}")
        avg_sr = total_sr_all / total_ep_all * 100 if total_ep_all > 0 else 0
        avg_cr = total_cr_all / total_ep_all * 100 if total_ep_all > 0 else 0
        print(f"  {'Total':<8} {'':<20} {total_ep_all:>4} {avg_sr:>5.0f}% {avg_cr:>5.0f}%")
        print(f"\n  Duration: {total_elapsed:.1f}s | Steps: {global_step}")
        print(f"{'='*70}\n")

    else:
        # ── Fixed mode: single configuration, run max_steps ──
        counts = {"success": 0, "collision": 0, "timeout": 0}

        print(f"  Running {args_cli.max_steps} steps... (Ctrl+C to stop)\n")

        try:
            obs, _ = env.reset()
            for step in range(args_cli.max_steps):
                with torch.no_grad():
                    actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
                obs, reward, terminated, truncated, infos = env.step(actions)
                global_step += 1
                _count_episodes(raw_env, counts)

                if (step + 1) % 200 == 0:
                    elapsed = time.time() - global_start
                    total_ep = sum(counts.values())
                    if total_ep > 0:
                        sr = counts["success"] / total_ep * 100
                        cr = counts["collision"] / total_ep * 100
                        tr = counts["timeout"] / total_ep * 100
                        print(f"  Step {step+1:>5} | {elapsed:.0f}s | "
                              f"Ep={total_ep} SR={sr:.1f}% CR={cr:.1f}% TO={tr:.1f}%")

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user")

        # Fixed mode summary
        total_elapsed = time.time() - global_start
        total_ep = sum(counts.values())
        print(f"\n{'='*70}")
        print(f"  PLAY SUMMARY — Fixed Mode")
        print(f"{'='*70}")
        print(f"  Static obs:  {args_cli.num_static}")
        print(f"  Dynamic obs: {args_cli.num_dynamic}")
        print(f"  Walls:       {args_cli.num_walls}")
        print(f"  Duration:    {total_elapsed:.1f}s")
        print(f"  Steps:       {global_step}")
        print(f"  Episodes:    {total_ep}")
        if total_ep > 0:
            print(f"  Success:     {counts['success']} ({counts['success']/total_ep*100:.1f}%)")
            print(f"  Collision:   {counts['collision']} ({counts['collision']/total_ep*100:.1f}%)")
            print(f"  Timeout:     {counts['timeout']} ({counts['timeout']/total_ep*100:.1f}%)")
        print(f"{'='*70}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
