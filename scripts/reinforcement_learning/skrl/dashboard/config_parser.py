"""解析所有訓練配置檔案為結構化 dict — 不 import torch/Isaac Lab."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # IsaacLab/

CHARGE_SKRL = (
    PROJECT_ROOT
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "config"
    / "charge_skrl"
)

SKRL_SCRIPTS = PROJECT_ROOT / "scripts" / "reinforcement_learning" / "skrl"


# ═══════════════════════════════════════════════════════════════════════════
# 1. PPO Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════
def parse_ppo_config() -> dict:
    yaml_path = CHARGE_SKRL / "agents" / "skrl_ppo_cfg_vlp16.yaml"
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    agent = raw.get("agent", {})
    lr_scheduler_str = agent.get("learning_rate_scheduler", "LinearLR")
    lr_kwargs = agent.get("learning_rate_scheduler_kwargs", {}) or {}

    total_timesteps = raw.get("trainer", {}).get("timesteps", 234_375)
    rollouts = agent.get("rollouts", 128)
    epochs = agent.get("learning_epochs", 6)
    mini_batches = agent.get("mini_batches", 16)
    total_rollouts = total_timesteps // rollouts if rollouts else 0
    grad_steps_per_rollout = epochs * mini_batches
    total_grad_steps = total_rollouts * grad_steps_per_rollout

    # Extract scheduler class name from import string
    lr_class = "LinearLR"
    if isinstance(lr_scheduler_str, str) and "LinearLR" in lr_scheduler_str:
        lr_class = "LinearLR"

    return {
        "seed": raw.get("seed", 42),
        "models": raw.get("models", {}),
        "agent": {
            "learning_rate": agent.get("learning_rate", 1e-4),
            "lr_scheduler": lr_class,
            "lr_start_factor": lr_kwargs.get("start_factor", 1.0),
            "lr_end_factor": lr_kwargs.get("end_factor", 0.3),
            "lr_total_iters": lr_kwargs.get("total_iters", total_grad_steps),
            "discount_factor": agent.get("discount_factor", 0.990),
            "lambda_gae": agent.get("lambda", 0.95),
            "rollouts": rollouts,
            "learning_epochs": epochs,
            "mini_batches": mini_batches,
            "entropy_loss_scale": agent.get("entropy_loss_scale", 0.01),
            "value_loss_scale": agent.get("value_loss_scale", 1.0),
            "grad_norm_clip": agent.get("grad_norm_clip", 1.0),
            "ratio_clip": agent.get("ratio_clip", 0.2),
            "value_clip": agent.get("value_clip", 0.2),
            "clip_predicted_values": agent.get("clip_predicted_values", False),
            "kl_threshold": agent.get("kl_threshold", 0.0),
            "state_preprocessor": agent.get("state_preprocessor", "RunningStandardScaler"),
            "value_preprocessor": agent.get("value_preprocessor", "RunningStandardScaler"),
        },
        "trainer": {
            "timesteps": total_timesteps,
            "checkpoint_interval": agent.get("experiment", {}).get("checkpoint_interval",
                                  raw.get("trainer", {}).get("checkpoint_interval", 11718)),
        },
        "derived": {
            "total_rollouts": total_rollouts,
            "grad_steps_per_rollout": grad_steps_per_rollout,
            "total_grad_steps": total_grad_steps,
            "effective_batch_size": f"{rollouts} / {mini_batches} = {rollouts // mini_batches if mini_batches else '?'}",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Curriculum Stages
# ═══════════════════════════════════════════════════════════════════════════
def parse_curriculum_configs() -> dict:
    """Import CURRICULUM_CONFIGS directly — pure Python, no GPU."""
    curriculum_path = CHARGE_SKRL / "curriculum"
    sys.path.insert(0, str(curriculum_path))
    try:
        # The module guards Isaac Lab imports behind TYPE_CHECKING
        from goal_obstacle_curriculum import CURRICULUM_CONFIGS  # type: ignore
        result = {}
        for version_name, version_cfg in CURRICULUM_CONFIGS.items():
            stages = []
            for i, s in enumerate(version_cfg.get("stages", []), 1):
                stages.append({
                    "stage": i,
                    "name": s.get("name", ""),
                    "num_goals": s.get("num_goals", 0),
                    "num_static": s.get("num_obstacles_static", 0),
                    "num_dynamic": s.get("num_obstacles_dynamic", 0),
                    "min_walls": s.get("min_walls", 0),
                    "max_walls": s.get("max_walls", 0),
                    "gamma": s.get("gamma", 0.99),
                    "episode_s": s.get("episode_length_s", 60),
                    "empty_ratio": s.get("empty_ratio", 0),
                    "upgrade_sr": s.get("upgrade_sr", 0),
                    "upgrade_max_cr": s.get("upgrade_max_cr", 0.4),
                    "upgrade_max_to": s.get("upgrade_max_to", 0.3),
                    "min_stage_updates": s.get("min_stage_updates", 50),
                    "reward_weights": s.get("reward_weights"),
                })
            result[version_name] = {
                "upgrade_pass_required": version_cfg.get("upgrade_pass_required", 5),
                "stages": stages,
            }
        return result
    except Exception as e:
        return {"_error": str(e)}
    finally:
        sys.path.pop(0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Neural Network Architecture
# ═══════════════════════════════════════════════════════════════════════════
def parse_nn_architecture() -> dict:
    return {
        "observation_dim": 139,
        "obs_layout": [
            {"name": "ego", "dim": 4, "start": 0, "color": "#7aa2f7",
             "fields": ["norm_accel (1D)", "norm_velocity (1D)", "norm_angular_vel (1D)", "body_radius (1D)"]},
            {"name": "goal", "dim": 2, "start": 4, "color": "#9ece6a",
             "fields": ["x_goal_robot (1D)", "y_goal_robot (1D)"]},
            {"name": "LiDAR", "dim": 72, "start": 6, "color": "#ff9e64",
             "fields": ["72 bins × 5° resolution, VLP-16 min-pooled, normalized [0,1]"]},
            {"name": "obstacles", "dim": 60, "start": 78, "color": "#f7768e",
             "fields": ["Top-10 × 6D: [x, y, vx, vy, radius, mask] in body frame"]},
            {"name": "time", "dim": 1, "start": 138, "color": "#565f89",
             "fields": ["remaining_ratio [0,1]"]},
        ],
        "feature_extractor": {
            "output_dim": 128,
            "branches": [
                {
                    "name": "LiDAR Conv1d",
                    "input": "72D",
                    "output": "64D",
                    "layers": [
                        "Conv1d(1→32, k=5, s=1, pad=2) + ReLU",
                        "Conv1d(32→64, k=5, s=2, pad=2) + ReLU",
                        "Conv1d(64→64, k=3, s=2, pad=1) + ReLU",
                        "AdaptiveMaxPool1d(1)",
                        "Linear(64→64) + LayerNorm(64)",
                    ],
                },
                {
                    "name": "Obstacle MLP",
                    "input": "60D (10×6)",
                    "output": "32D",
                    "layers": [
                        "Linear(6→32) + ReLU  (per-object)",
                        "Linear(32→32) + ReLU",
                        "MaxPool over 10 objects",
                        "LayerNorm(32)",
                    ],
                },
                {
                    "name": "State MLP",
                    "input": "7D (ego4+goal2+time1)",
                    "output": "32D",
                    "layers": [
                        "Linear(7→32) + ReLU",
                        "Linear(32→32) + ReLU",
                        "LayerNorm(32)",
                    ],
                },
            ],
            "fusion": "Concatenate [64D, 32D, 32D] → 128D",
        },
        "policy_head": {
            "name": "VLP16DiscretePolicy (MultiCategoricalMixin)",
            "layers": ["Linear(128→128) + ReLU", "Linear(128→38)"],
            "output": "38 logits → MultiDiscrete([19, 19])",
        },
        "value_head": {
            "name": "VLP16Value (DeterministicMixin)",
            "layers": ["Linear(128→64) + ReLU", "Linear(64→32) + ReLU", "Linear(32→1)"],
            "output": "Scalar V(s)",
            "init": "orthogonal(gain=0.01), bias=-8.0",
        },
        "param_counts": {
            "LiDAR Conv1d branch": "~12,500",
            "Obstacle MLP branch": "~1,300",
            "State MLP branch": "~1,300",
            "Policy Head": "~17,000",
            "Value Head": "~3,500",
            "Total": "~35,600",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Reward Functions
# ═══════════════════════════════════════════════════════════════════════════
def parse_reward_config() -> list[dict]:
    return [
        {
            "name": "reaching_goal",
            "name_zh": "到達目標",
            "weight": 500,
            "sign": "+",
            "range": "[0, 1]",
            "formula": r"r = \mathbb{1}[d_{\mathrm{goal}} < 0.35\,\mathrm{m}]",
            "description": "Terminal sparse reward on goal reach",
            "params": {"threshold": "0.35 m (= body_radius)"},
        },
        {
            "name": "potential_progress",
            "name_zh": "勢函數進展",
            "weight": 12,
            "sign": "+",
            "range": "[-1, 1]",
            "formula": r"r = \mathrm{clamp}(d_{t-1} - d_t,\; -1,\; 1)",
            "description": "PBRS: reward for reducing distance to goal",
            "params": {"clip": "[-1.0, 1.0]"},
        },
        {
            "name": "velocity_to_goal",
            "name_zh": "目標速度（含 v_gate）",
            "weight": 10,
            "sign": "+",
            "range": "[-1, 1]",
            "formula": r"r = \mathrm{clamp}\!\left(\frac{\vec{v} \cdot \hat{g}}{v_{\max}},\; -1,\; 1\right) \times v_{\mathrm{gate}}(d_{\mathrm{safe}})",
            "description": "Directional velocity reward with safety gating",
            "params": {
                "v_max": "1.0 m/s",
                "d_attenuate": "1.0 m (baseline) / 0.6 m (softer)",
                "d_full": "2.5 m",
                "v_gate_floor": "0.0 (baseline) / 0.2 (floor)",
            },
            "gate_formula": r"v_{\mathrm{gate}} = \mathrm{clamp}\!\left(\frac{d_{\mathrm{safe}} - d_{\mathrm{att}}}{d_{\mathrm{full}} - d_{\mathrm{att}}},\; \mathrm{floor},\; 1\right)",
        },
        {
            "name": "safe_progress",
            "name_zh": "安全進展（含 progress_gate）",
            "weight": 4,
            "sign": "+",
            "range": "[-1, 1]",
            "formula": r"r = (d_{t-1} - d_t) \times \mathrm{gate}(d_{\mathrm{safe}})",
            "description": "Progress × piecewise-linear safety gate (negative in danger zone)",
            "params": {
                "d_danger": "0.8 m (baseline) / 0.55 m (delayed_negative)",
                "d_comfort": "2.0 m",
                "negative_scale": "0.5 (baseline) / 0.2 (weaken_negative)",
            },
            "gate_formula": r"\mathrm{gate} = \begin{cases} -\eta & d_{\mathrm{safe}} < d_{\mathrm{danger}} \\ \text{linear}(0,1) & d_{\mathrm{danger}} \le d_{\mathrm{safe}} \le d_{\mathrm{comfort}} \\ 1 & d_{\mathrm{safe}} > d_{\mathrm{comfort}} \end{cases}",
        },
        {
            "name": "safety_log_distance",
            "name_zh": "對數安全距離",
            "weight": 3,
            "sign": "+",
            "range": "[-6, 2]",
            "formula": r"r = \mathrm{clamp}(\ln d_{\mathrm{safe}},\; -6,\; 2) \times \mathrm{clamp}\!\left(\frac{\|\vec{v}\|}{v_{\mathrm{thr}}},\; 0.1,\; 1\right)",
            "description": "Log-distance safety signal coupled with speed",
            "params": {
                "bottom_k": "10 (closest rays)",
                "speed_threshold": "0.1 m/s",
                "min_speed_factor": "0.1",
            },
        },
        {
            "name": "collision_terminal",
            "name_zh": "碰撞懲罰",
            "weight": -100,
            "sign": "-",
            "range": "[0, 1]",
            "formula": r"r = \mathbb{1}[d_{\min} \le 0.45\,\mathrm{m}]",
            "description": "Terminal penalty on collision (LiDAR min ≤ threshold)",
            "params": {"threshold": "0.45 m (body 0.35 + buffer 0.10)"},
        },
        {
            "name": "acceleration_penalty",
            "name_zh": "加速度懲罰（含 deadzone）",
            "weight": -0.05,
            "sign": "-",
            "range": "[0, ∞)",
            "formula": r"r = \max(0,\; |a| - \delta)^2",
            "description": "Deadzone acceleration penalty for motor protection",
            "params": {"dead_zone": "0.1 m/s²"},
        },
        {
            "name": "angular_velocity_penalty",
            "name_zh": "角速度懲罰（情境感知）",
            "weight": -0.05,
            "sign": "-",
            "range": "[0, ∞)",
            "formula": r"r = \omega^2 \times (1 - \mathrm{prox\_factor})",
            "description": "Context-aware: reduced penalty when near obstacles (allows sharp turns)",
            "params": {
                "proximity_distance": "1.5 m",
                "max_reduction": "0.7",
            },
        },
        {
            "name": "time_penalty",
            "name_zh": "時間懲罰",
            "weight": -0.1,
            "sign": "-",
            "range": "[0, 1]",
            "formula": r"r = 1.0 \quad \text{(constant per step)}",
            "description": "Per-step time pressure",
            "params": {},
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 5. Action Space
# ═══════════════════════════════════════════════════════════════════════════
def parse_action_space() -> dict:
    return {
        "type": "MultiDiscrete([19, 19])",
        "total_actions": 361,
        "center_idx": 9,
        "axes": [
            {"name": "linear_accel", "bins": 19, "physical_range": "[-0.5, +0.5] m/s²",
             "mapping": "ratio = (idx - 9) / 9 ∈ [-1, 1]"},
            {"name": "angular_vel", "bins": 19, "physical_range": "[-0.785, +0.785] rad/s (±π/4)",
             "mapping": "ω = ratio × ω_max"},
        ],
        "physical_limits": {
            "v_max": "1.0 m/s",
            "a_max": "0.5 m/s²",
            "omega_max": "0.25π ≈ 0.785 rad/s",
            "dt": "0.2 s (decimation=20, sim_dt=0.01)",
            "allows_reverse": True,
        },
        "dynamic_bounds_formula": r"a \in \left[\max\!\left(-a_{\max},\; \frac{-v_{\max} - v}{dt}\right),\; \min\!\left(a_{\max},\; \frac{v_{\max} - v}{dt}\right)\right]",
        "safety_shield": [
            {"mode": "off", "description": "No intervention (baseline)"},
            {"mode": "soft", "description": "Linear speed reduction: d_safe < 1.2m → scale, d_safe < 0.55m → stop",
             "d_safe": 1.2, "d_danger": 0.55},
            {"mode": "hard", "description": "Force v=0 at d_safe < 0.55m, soft above",
             "d_safe": 1.2, "d_danger": 0.55},
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Observation Space
# ═══════════════════════════════════════════════════════════════════════════
def parse_observation_space() -> dict:
    return {
        "total_dim": 139,
        "lidar_pipeline": [
            {"step": "VLP-16 Raw", "detail": "16 channels × 360° = 5760 rays, max 20m"},
            {"step": "2D Projection", "detail": "Drop Z, compute XY Euclidean distance"},
            {"step": "Sim-to-Real Corruption", "detail": "Displacement N(0, 0.02²), holes 0.5%, distractors 0.2%"},
            {"step": "Reshape", "detail": "[5760] → [16, 72, 5] (channels, bins, rays/bin)"},
            {"step": "Min Pool", "detail": "Min over rays/bin → [16, 72], min over channels → [72]"},
            {"step": "Safe Margin", "detail": "d - body_radius (0.35m), clamp ≥ 0"},
            {"step": "Normalize", "detail": "d / 20.0 → [0, 1]"},
        ],
        "obstacle_features": [
            {"field": "px", "desc": "Position X (body frame)", "norm": "/ 8.0"},
            {"field": "py", "desc": "Position Y (body frame)", "norm": "/ 8.0"},
            {"field": "vx", "desc": "Relative velocity X (body frame)", "norm": "/ 1.5"},
            {"field": "vy", "desc": "Relative velocity Y (body frame)", "norm": "/ 1.5"},
            {"field": "r", "desc": "Obstacle radius", "norm": "raw (m)"},
            {"field": "m", "desc": "Valid mask", "norm": "{0, 1}"},
        ],
        "body_frame_rotation": r"\begin{bmatrix} x_b \\ y_b \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix}",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. Ablation Experiments
# ═══════════════════════════════════════════════════════════════════════════
def parse_ablation_families() -> list[dict]:
    return [
        {
            "id": 1,
            "name": "v_gate",
            "name_zh": "目標吸引力衰減",
            "cli_arg": "--v_gate_mode",
            "variants": [
                {"name": "baseline", "params": "d_attenuate=1.0, v_gate_floor=0.0",
                 "desc": "Full attenuation below d_attenuate"},
                {"name": "floor", "params": "d_attenuate=1.0, v_gate_floor=0.2",
                 "desc": "Retain 20% goal attraction near obstacles"},
                {"name": "softer", "params": "d_attenuate=0.6, v_gate_floor=0.0",
                 "desc": "Delay gate onset to 0.6m"},
            ],
        },
        {
            "id": 2,
            "name": "progress_gate",
            "name_zh": "危險區域門檻",
            "cli_arg": "--progress_gate_mode",
            "variants": [
                {"name": "baseline", "params": "d_danger=0.8, negative_scale=0.5",
                 "desc": "Standard danger zone with η=-0.5"},
                {"name": "delayed_negative", "params": "d_danger=0.55",
                 "desc": "Shrink danger zone (0.8→0.55m)"},
                {"name": "weaken_negative", "params": "negative_scale=0.2",
                 "desc": "Reduce negative zone strength (0.5→0.2)"},
            ],
        },
        {
            "id": 3,
            "name": "gap_reward",
            "name_zh": "繞行指引獎勵",
            "cli_arg": "--use_gap_reward --gap_reward_type --gap_reward_weight",
            "variants": [
                {"name": "off", "params": "(default, no flag)", "desc": "No gap-seeking reward"},
                {"name": "heading w=1.0", "params": "--gap_reward_type heading --gap_reward_weight 1.0",
                 "desc": "Align heading with largest passable arc"},
                {"name": "heading w=2.0", "params": "--gap_reward_type heading --gap_reward_weight 2.0",
                 "desc": "Stronger heading alignment"},
                {"name": "clearance w=1.0", "params": "--gap_reward_type clearance --gap_reward_weight 1.0",
                 "desc": "Reward expanding forward 60° clearance"},
                {"name": "clearance w=2.0", "params": "--gap_reward_type clearance --gap_reward_weight 2.0",
                 "desc": "Stronger clearance reward"},
            ],
        },
        {
            "id": 4,
            "name": "shield",
            "name_zh": "動作級安全盾牌",
            "cli_arg": "--use_safety_shield --shield_mode",
            "variants": [
                {"name": "off", "params": "(default, no flag)", "desc": "No action clipping"},
                {"name": "soft", "params": "--shield_mode soft",
                 "desc": "Linear speed reduction d<1.2m, stop d<0.55m"},
                {"name": "hard", "params": "--shield_mode hard",
                 "desc": "Force v=0 at d<0.55m, soft above"},
            ],
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 8. Diagnostic Metrics
# ═══════════════════════════════════════════════════════════════════════════
def parse_diagnostic_metrics() -> dict:
    return {
        "metrics": [
            {"key": "ablation/stuck_count", "name": "Stuck Count",
             "formula": "consecutive ≥5 steps: speed<0.02 AND Δdist<0.01", "range": "[0, ∞)"},
            {"key": "ablation/freeze_ratio", "name": "Freeze Ratio",
             "formula": "steps with speed<0.02 / total_steps", "range": "[0, 1]"},
            {"key": "ablation/oscillation_score", "name": "Oscillation Score",
             "formula": "v_toward sign-flip frequency (window=10)", "range": "[0, 1]"},
            {"key": "ablation/retreat_ratio", "name": "Retreat Ratio",
             "formula": "(d_safe<1.5 AND v_toward<0) / near_steps", "range": "[0, 1]"},
            {"key": "ablation/progress_near_obs", "name": "Progress Near Obs",
             "formula": "mean(d_prev - d_curr) when d_safe<1.5", "range": "[-0.5, 0.5]"},
            {"key": "ablation/v_toward_near_obs", "name": "Goal Vel Near Obs",
             "formula": "mean(v_toward) when d_safe<1.5", "range": "[-1, 1]"},
            {"key": "ablation/d_safe_mean", "name": "Safe Distance Mean",
             "formula": "mean of all d_safe values", "range": "[0, 20]"},
            {"key": "ablation/d_safe_min", "name": "Safe Distance Min",
             "formula": "min of d_safe per rollout", "range": "[0, 20]"},
            {"key": "ablation/danger_ratio", "name": "Danger Zone Ratio",
             "formula": "steps with d_safe<0.8 / total_steps", "range": "[0, 1]"},
            {"key": "ablation/speed_near_obs", "name": "Speed Near Obs",
             "formula": "mean(speed) when d_safe<1.5", "range": "[0, 1]"},
            {"key": "ablation/front_clearance", "name": "Front Clearance",
             "formula": "mean d_safe (proxy for forward heading)", "range": "[0, 20]"},
            {"key": "ablation/avg_ep_length", "name": "Avg Episode Length",
             "formula": "mean steps at termination", "range": "[1, 450]"},
            {"key": "ablation/collision_ep_len", "name": "Collision Ep Length",
             "formula": "mean steps when collision occurred", "range": "[1, 450]"},
            {"key": "ablation/shield_rate", "name": "Shield Rate",
             "formula": "steps where shield intervened / total", "range": "[0, 1]"},
            {"key": "ablation/action_override_rate", "name": "Action Override Rate",
             "formula": "actions actually modified by shield / total", "range": "[0, 1]"},
        ],
        "thresholds": {
            "STUCK_WINDOW": 5,
            "STUCK_SPEED_THRESH": 0.02,
            "STUCK_DIST_EPS": 0.01,
            "NEAR_OBS_D_SAFE": 1.5,
            "DANGER_D_SAFE": 0.8,
            "OSC_WINDOW": 10,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9. Environment Scene
# ═══════════════════════════════════════════════════════════════════════════
def parse_environment_scene() -> dict:
    return {
        "arena": {"width": 20, "height": 20, "wall_thickness": 1.0, "wall_height": 1.5},
        "robot": {
            "body_radius": 0.35,
            "collision_threshold": 0.45,
            "goal_reach_threshold": 0.35,
        },
        "internal_walls": {
            "max_slots": 8,
            "thickness": 1.0,
            "note": "Per-env randomized from WALL_SLOT_SPECS",
        },
        "obstacles": {
            "max_total": 20,
            "cuboid_count": 10,
            "cylinder_count": 10,
            "cuboid_sizes": "0.4-0.7m edge, 0.8-1.4m height",
            "cylinder_sizes": "0.2-0.35m radius, 0.8-1.5m height",
        },
        "simulation": {
            "sim_dt": 0.01,
            "decimation": 20,
            "control_dt": 0.2,
            "control_freq": "5 Hz",
            "physics_freq": "100 Hz",
        },
        "goal_command": {
            "distance_range": "2-13 m",
            "angle_range": "-π to +π",
            "wall_boundary": "9.5 m",
            "safe_margin_walls": "0.5 m",
            "safe_margin_obstacles": "1.0 m",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 10. CLI 配置解讀
# ═══════════════════════════════════════════════════════════════════════════
def parse_cli_config() -> dict:
    return {
        "task": "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL",
        "reward_mode": "navrl_ground_v2",
        "reward_mode_zh": "NavRL-Ground v2（論文對齊版）",
        "curriculum_version": "goal_first_v1",
        "curriculum_version_zh": "先導航再避障（8 階段）",
        "dynamic_safety_mode": "closing_risk",
        "dynamic_safety_mode_zh": "速度感知碰撞風險",
        "num_envs": 6144,
        "seed": 1,
        "run_name": "rw_groundv2_closingrisk_full_seed1",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 11. 獎勵模式對比表
# ═══════════════════════════════════════════════════════════════════════════
def parse_reward_modes() -> dict:
    terms = [
        "alive", "goal_velocity", "goal_progress", "static_safety",
        "dynamic_safety", "smoothness", "time_penalty", "reaching_goal",
        "collision_ground",
    ]
    modes = {
        "navrl_ground_v1": {
            "name_zh": "NavRL-Ground v1（軟閘版）",
            "desc_zh": "8 項獎勵，goal_velocity 與 goal_progress 帶 soft gate/scale",
            "weights": {
                "alive": 0, "goal_velocity": 10.0, "goal_progress": 12.0,
                "static_safety": 3.0, "dynamic_safety": 4.0,
                "smoothness": -0.05, "time_penalty": -0.1,
                "reaching_goal": 500, "collision_ground": -100,
            },
            "notes": {
                "goal_velocity": "use_soft_gate=True, beta=0.2",
                "goal_progress": "use_soft_scale=True, gamma=0.3",
                "dynamic_safety": "mode=log_distance, max_obs=10",
            },
        },
        "navrl_ground_v2": {
            "name_zh": "NavRL-Ground v2（論文對齊版）★ 當前使用",
            "desc_zh": "9 項獎勵，移除 gate/scale，新增 alive，降低 terminal 權重",
            "weights": {
                "alive": 0.2, "goal_velocity": 2.0, "goal_progress": 3.0,
                "static_safety": 2.0, "dynamic_safety": 2.0,
                "smoothness": -0.1, "time_penalty": 0,
                "reaching_goal": 100, "collision_ground": -50,
            },
            "notes": {
                "goal_velocity": "無 gate（100% 保留）, bottom_k=36",
                "goal_progress": "無 scale, progress_clip=0.25",
                "dynamic_safety": "mode=closing_risk, max_obs=10, σ=2.0",
                "alive": "每步存活獎勵（NavRL 原始設計）",
                "time_penalty": "已移除（alive + γ 折扣自然提供時間壓力）",
            },
        },
        "navrl_ground_v3": {
            "name_zh": "NavRL-Ground v3（20 障礙物版）",
            "desc_zh": "與 v2 相同結構，支援 20 障礙物，reaching_goal=500, alive=0",
            "weights": {
                "alive": 0, "goal_velocity": 2.0, "goal_progress": 3.0,
                "static_safety": 2.0, "dynamic_safety": 2.0,
                "smoothness": -0.1, "time_penalty": 0,
                "reaching_goal": 500, "collision_ground": -50,
            },
            "notes": {
                "reaching_goal": "↑ 500（v2 的 100 不足以超越累積 per-step）",
                "alive": "移除（與目標導向矛盾）",
                "dynamic_safety": "max_obs=20",
            },
        },
        "navrl_ground_v4": {
            "name_zh": "NavRL-Ground v4（最終調校版）",
            "desc_zh": "v3 微調：reaching_goal=500, alive=0（確認版）",
            "weights": {
                "alive": 0, "goal_velocity": 2.0, "goal_progress": 3.0,
                "static_safety": 2.0, "dynamic_safety": 2.0,
                "smoothness": -0.1, "time_penalty": 0,
                "reaching_goal": 500, "collision_ground": -50,
            },
            "notes": {},
        },
    }
    return {"terms": terms, "modes": modes}


# ═══════════════════════════════════════════════════════════════════════════
# 12. WandB 指標紀錄（含理想值、趨勢、壞信號）
# ═══════════════════════════════════════════════════════════════════════════
def parse_wandb_metrics() -> dict:
    return {
        "categories": [
            {
                "id": "perf",
                "name_zh": "核心效能",
                "icon": "🎯",
                "metrics": [
                    {"key": "perf/success_rate", "name_zh": "成功率",
                     "ideal": ">0.70", "good_trend": "穩定上升，隨課程階段提升",
                     "bad_signal": "長期停滯 <0.3，可能策略崩潰或獎勵設計問題",
                     "range": "[0, 1]", "status_thresholds": [0.7, 0.4]},
                    {"key": "perf/collision_rate", "name_zh": "碰撞率",
                     "ideal": "<0.25", "good_trend": "穩定下降至 <0.15",
                     "bad_signal": "持續 >0.4，避障機制失效或 dynamic_safety 權重不足",
                     "range": "[0, 1]", "status_thresholds": [0.25, 0.4]},
                    {"key": "perf/timeout_rate", "name_zh": "超時率",
                     "ideal": "<0.20", "good_trend": "隨速度學習而下降",
                     "bad_signal": ">0.5，agent 太保守或動力不足（goal_velocity 信號弱）",
                     "range": "[0, 1]", "status_thresholds": [0.2, 0.5]},
                    {"key": "perf/episode_length", "name_zh": "平均 Episode 長度",
                     "ideal": "接近 episode_max", "good_trend": "穩定增加（存活更久）",
                     "bad_signal": "突降 = 大量碰撞；過短 = 快速死亡",
                     "range": "[1, 450]", "status_thresholds": [200, 50]},
                ],
            },
            {
                "id": "reward",
                "name_zh": "獎勵分項",
                "icon": "💰",
                "metrics": [
                    {"key": "Reward / goal_velocity", "name_zh": "目標速度獎勵",
                     "ideal": ">0（正向前進）", "good_trend": "穩定正值",
                     "bad_signal": "持續負值 = 退縮行為或頻繁遠離目標",
                     "range": "[-2, 2]", "status_thresholds": [0, -0.5]},
                    {"key": "Reward / goal_progress", "name_zh": "目標進展獎勵",
                     "ideal": ">0", "good_trend": "穩定正值，表示持續接近目標",
                     "bad_signal": "負值 = 遠離目標，策略方向錯誤",
                     "range": "[-3, 3]", "status_thresholds": [0, -0.5]},
                    {"key": "Reward / static_safety", "name_zh": "靜態安全獎勵",
                     "ideal": ">0", "good_trend": "初期波動後穩定正值",
                     "bad_signal": "持續大負值 = 太靠近障礙物",
                     "range": "[-6, 2]", "status_thresholds": [0, -1]},
                    {"key": "Reward / dynamic_safety", "name_zh": "動態安全獎勵",
                     "ideal": ">0", "good_trend": "Phase C（Stage 5+）後出現並穩定",
                     "bad_signal": "負值 + collision↑ = closing_risk 捕捉到高風險但策略未學會應對",
                     "range": "[-6, 2]", "status_thresholds": [0, -1]},
                    {"key": "Reward / alive", "name_zh": "存活獎勵",
                     "ideal": "0.2/step", "good_trend": "穩定（與 episode_length 正相關）",
                     "bad_signal": "突降 = episode 過短（大量碰撞）",
                     "range": "[0, 0.2]", "status_thresholds": [0.15, 0.05]},
                    {"key": "Reward / smoothness", "name_zh": "平滑度懲罰",
                     "ideal": "接近 0", "good_trend": "從負值逐漸趨近 0（動作更平滑）",
                     "bad_signal": "持續 < -0.05 = 動作抖動嚴重",
                     "range": "[-0.5, 0]", "status_thresholds": [-0.02, -0.05]},
                    {"key": "Reward / reaching_goal", "name_zh": "到達目標獎勵",
                     "ideal": ">0", "good_trend": "隨 SR 上升而增加",
                     "bad_signal": "= 0 表示完全沒到終點",
                     "range": "[0, 100]", "status_thresholds": [10, 0]},
                    {"key": "Reward / collision_ground", "name_zh": "碰撞懲罰",
                     "ideal": "接近 0", "good_trend": "碰撞減少→趨近 0",
                     "bad_signal": "持續大負值 = 頻繁碰撞（-50/次）",
                     "range": "[-50, 0]", "status_thresholds": [-5, -20]},
                ],
            },
            {
                "id": "curriculum",
                "name_zh": "課程進度",
                "icon": "📈",
                "metrics": [
                    {"key": "Curriculum / stage", "name_zh": "目前課程階段",
                     "ideal": "持續遞增至 8", "good_trend": "階梯式上升（每階段穩定後升級）",
                     "bad_signal": "長期卡在 Stage 1-2 = 基本導航都學不會",
                     "range": "[1, 8]", "status_thresholds": [4, 2]},
                    {"key": "Curriculum / success_rate", "name_zh": "階段成功率",
                     "ideal": ">upgrade_sr", "good_trend": "每階段初始下降後快速回升",
                     "bad_signal": "長期低於 downgrade 閾值（0.15）= 崩潰",
                     "range": "[0, 1]", "status_thresholds": [0.6, 0.2]},
                    {"key": "Curriculum / sr_gap", "name_zh": "升級差距",
                     "ideal": ">0", "good_trend": "從負轉正 = 即將升級",
                     "bad_signal": "持續大負值 = 離升級很遠",
                     "range": "[-1, 1]", "status_thresholds": [0, -0.3]},
                    {"key": "Curriculum / collision_rate", "name_zh": "階段碰撞率",
                     "ideal": "<0.35", "good_trend": "穩定在 upgrade_max_cr 以下",
                     "bad_signal": ">0.5 可能觸發降級",
                     "range": "[0, 1]", "status_thresholds": [0.35, 0.6]},
                    {"key": "Curriculum / gamma", "name_zh": "折扣因子 γ",
                     "ideal": "隨階段升高", "good_trend": "0.990→0.998 逐階提升",
                     "bad_signal": "降回低值 = 被降級",
                     "range": "[0.99, 0.998]", "status_thresholds": [0.995, 0.991]},
                ],
            },
            {
                "id": "behavior",
                "name_zh": "行為診斷",
                "icon": "🤖",
                "metrics": [
                    {"key": "behavior/freeze_ratio", "name_zh": "凍結比例",
                     "ideal": "<0.05", "good_trend": "持續降低",
                     "bad_signal": ">0.15 = 學到「不動」策略（凍結死亡）",
                     "range": "[0, 1]", "status_thresholds": [0.05, 0.15]},
                    {"key": "behavior/oscillation", "name_zh": "擺盪分數",
                     "ideal": "<0.30", "good_trend": "降低 = 動作更穩定",
                     "bad_signal": ">0.5 = 在障礙物前來回擺盪，無法決定方向",
                     "range": "[0, 1]", "status_thresholds": [0.3, 0.5]},
                    {"key": "behavior/retreat_ratio", "name_zh": "退縮比例",
                     "ideal": "<0.10", "good_trend": "低且穩定",
                     "bad_signal": ">0.3 = 恐懼退縮，近障礙物時不敢前進",
                     "range": "[0, 1]", "status_thresholds": [0.1, 0.3]},
                    {"key": "behavior/danger_zone_ratio", "name_zh": "危險區域停留比例",
                     "ideal": "<0.10", "good_trend": "降低 = 學會保持安全距離",
                     "bad_signal": ">0.2 = 頻繁進入 d_safe<0.8m 危險區",
                     "range": "[0, 1]", "status_thresholds": [0.1, 0.2]},
                    {"key": "behavior/obstacle_distance_avg", "name_zh": "平均安全距離",
                     "ideal": ">0.5m", "good_trend": "穩定在 0.5m 以上",
                     "bad_signal": "<0.3m = 安全裕度不足，碰撞風險高",
                     "range": "[0, 20]", "status_thresholds": [0.5, 0.3]},
                    {"key": "behavior/stuck_events", "name_zh": "卡住次數",
                     "ideal": "0", "good_trend": "維持 0",
                     "bad_signal": ">5/rollout = 策略陷入局部最小值",
                     "range": "[0, ∞)", "status_thresholds": [1, 5]},
                    {"key": "behavior/progress_near_obstacle", "name_zh": "近障礙物時的進展",
                     "ideal": ">0", "good_trend": "正值 = 即使在障礙物旁也能前進",
                     "bad_signal": "持續負值 = 近障礙物時會遠離目標",
                     "range": "[-0.5, 0.5]", "status_thresholds": [0, -0.1]},
                ],
            },
            {
                "id": "training",
                "name_zh": "訓練健康",
                "icon": "🔧",
                "metrics": [
                    {"key": "Loss / Policy loss", "name_zh": "策略損失",
                     "ideal": "先升後穩定下降", "good_trend": "平穩收斂",
                     "bad_signal": "爆炸（突增 10x）= 梯度問題，檢查 grad_norm_clip",
                     "range": "[0, ∞)", "status_thresholds": None},
                    {"key": "Loss / Value loss", "name_zh": "價值損失",
                     "ideal": "穩定下降", "good_trend": "持續降低 = Critic 正在學習",
                     "bad_signal": "不降反升 = Critic 學不到（檢查 clip_predicted_values）",
                     "range": "[0, ∞)", "status_thresholds": None},
                    {"key": "train/module_entropy", "name_zh": "模組熵（梯度平衡）",
                     "ideal": "≈0（Actor/Critic 梯度平衡）",
                     "good_trend": "穩定在 [-1, 1] 區間",
                     "bad_signal": "<-2 = 策略崩潰（Actor 梯度消失）; >2 = Critic 落後",
                     "range": "[-3, 3]", "status_thresholds": None},
                    {"key": "train/actor_grad_norm", "name_zh": "Actor 梯度範數",
                     "ideal": "1e-4 ~ 1.0", "good_trend": "非零且穩定",
                     "bad_signal": "→0 = 策略死亡（entropy 過低或 LR 衰減過快）",
                     "range": "[0, ∞)", "status_thresholds": None},
                    {"key": "train/critic_grad_norm", "name_zh": "Critic 梯度範數",
                     "ideal": "1e-4 ~ 1.0", "good_trend": "非零且穩定",
                     "bad_signal": "→0 = Critic 死亡（value_loss_scale 太低或被 clip 殺死）",
                     "range": "[0, ∞)", "status_thresholds": None},
                    {"key": "Learning / learning_rate", "name_zh": "學習率",
                     "ideal": "1e-4→3e-5 線性衰減", "good_trend": "平滑線性下降曲線",
                     "bad_signal": "突變或歸零 = Scheduler 配置異常",
                     "range": "[3e-5, 1e-4]", "status_thresholds": None},
                ],
            },
            {
                "id": "robot",
                "name_zh": "機器人狀態",
                "icon": "🚗",
                "metrics": [
                    {"key": "robot/speed_mean", "name_zh": "平均線速度",
                     "ideal": "0.3-0.8 m/s", "good_trend": "穩定在中間區間",
                     "bad_signal": "<0.05 = 凍結; >0.95 = 失控衝撞",
                     "range": "[0, 1]", "status_thresholds": [0.3, 0.05]},
                    {"key": "robot/progress_per_step", "name_zh": "每步目標進展",
                     "ideal": ">0", "good_trend": "正值穩定",
                     "bad_signal": "持續負值 = 策略方向錯誤",
                     "range": "[-0.5, 0.5]", "status_thresholds": [0, -0.05]},
                    {"key": "robot/min_obstacle_dist_mean", "name_zh": "平均最近障礙物距離",
                     "ideal": ">0.5m", "good_trend": "穩定在安全距離",
                     "bad_signal": "<0.3m = 頻繁擦邊，碰撞風險高",
                     "range": "[0, 20]", "status_thresholds": [0.5, 0.3]},
                    {"key": "robot/angular_vel_mean", "name_zh": "平均角速度",
                     "ideal": "0.1-0.3 rad/s", "good_trend": "穩定（不過度旋轉）",
                     "bad_signal": ">0.5 = 原地旋轉; ≈0 = 不會轉彎",
                     "range": "[0, 0.785]", "status_thresholds": [0.3, 0.5]},
                ],
            },
            {
                "id": "action",
                "name_zh": "動作管線",
                "icon": "🎮",
                "metrics": [
                    {"key": "action/saturation_rate", "name_zh": "動作飽和率",
                     "ideal": "<0.05", "good_trend": "低（動作空間未被用盡）",
                     "bad_signal": ">0.2 = 動作空間不足，策略被迫走極端",
                     "range": "[0, 1]", "status_thresholds": [0.05, 0.2]},
                    {"key": "action/applied_std", "name_zh": "應用動作標準差",
                     "ideal": "0.3-1.0", "good_trend": "穩定（保持探索能力）",
                     "bad_signal": "→0 = 策略過度收斂（premature convergence）",
                     "range": "[0, ∞)", "status_thresholds": [0.3, 0.1]},
                    {"key": "action/raw_mean", "name_zh": "原始動作均值",
                     "ideal": "接近 0", "good_trend": "穩定（不偏向某方向）",
                     "bad_signal": "大偏移 = 策略偏差（bias in policy head）",
                     "range": "[-∞, ∞]", "status_thresholds": None},
                ],
            },
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 13. CADN (Curriculum-Aware Dual-Rate Normalizer)
# ═══════════════════════════════════════════════════════════════════════════
def parse_cadn_config() -> dict:
    """Parse CADN architecture, branch parameters, and diagnostic metrics."""
    return {
        "full_name": "Curriculum-Aware Dual-Rate Normalizer",
        "abbreviation": "CADN",
        "obs_dim": 139,
        "cli_flag": "--use_cadn",
        "replaces": "skrl.resources.preprocessors.torch.RunningStandardScaler",
        "source_file": "scripts/reinforcement_learning/skrl/cadn_preprocessor.py",
        "motivation": {
            "problem": "RunningStandardScaler 使用全域單速率 EMA（或 Welford），"
                       "當課程切換導致觀測分布突變時，統計量追蹤過慢（需數千步才穩定），"
                       "造成正規化後的值偏移，策略在新階段初期表現崩潰。",
            "root_causes": [
                {
                    "id": "RC1",
                    "title": "單速率追蹤延遲",
                    "desc": "全域 α 固定，課程跳階後 μ/σ 需 ~1/α 步才收斂到新分布，"
                            "期間正規化輸出偏離 N(0,1)，信噪比劣化。",
                },
                {
                    "id": "RC2",
                    "title": "跨語義混合統計",
                    "desc": "LiDAR (0~10m) 與 obstacle 6D (含 vx/vy ∈ [-2,2]) 混在同一 scaler，"
                            "不同維度的分布漂移速度與幅度不同，單一 α 無法兼顧。",
                },
                {
                    "id": "RC3",
                    "title": "Zero-padding 污染",
                    "desc": "Obstacle slots 不足 10 個時以零填充，使 scaler 的 μ 偏向 0、σ 偏小，"
                            "有值 slot 被過度放大，NaN 風險上升。",
                },
            ],
        },
        "design": {
            "dual_rate_ema": {
                "title": "雙速率 EMA (Dual-Rate EMA)",
                "fast_desc": "Fast channel (α_f): 快速追蹤近期分布，對課程切換反應靈敏",
                "slow_desc": "Slow channel (α_s): 維持長期穩定基線，避免噪音干擾",
                "blend_desc": "Adaptive blending: 根據 drift 指標自動混合 fast/slow",
                "formula_update": r"\mu_f \leftarrow (1-\alpha_f)\,\mu_f + \alpha_f\,\bar{x}_{\text{batch}}",
                "formula_drift": r"d = \frac{\|\mu_f - \mu_s\|}{\sqrt{D}\;\|\sqrt{\sigma^2_s}\|}",
                "formula_blend": r"w = \text{clamp}\!\left(\frac{d - \tau_{\text{lo}}}{\tau_{\text{hi}} - \tau_{\text{lo}}},\; 0,\; 1\right)",
                "formula_output": r"\mu^* = w\,\mu_f + (1-w)\,\mu_s, \quad \sigma^{*2} = w\,\sigma^2_f + (1-w)\,\sigma^2_s",
                "formula_normalize": r"\hat{x} = \text{clamp}\!\left(\frac{x - \mu^*}{\sqrt{\sigma^{*2}} + \epsilon},\; -5,\; 5\right)",
            },
            "per_branch": {
                "title": "Per-Branch 分支正規化",
                "desc": "將 139D 觀測拆為 3 語義分支，各自獨立追蹤統計量",
            },
        },
        "branches": [
            {
                "name": "State",
                "dims": "ego(4) + goal(2) + time(1) = 7D",
                "size": 7,
                "slice": "[0:4] + [4:6] + [138:139]",
                "alpha_fast": 0.008,
                "alpha_slow": 0.0003,
                "tau_lo": 0.25,
                "tau_hi": 0.55,
                "epsilon": 1e-8,
                "half_life_fast": "~87 steps",
                "half_life_slow": "~2310 steps",
                "rationale": "ego/goal 在課程切換時變化最小，用較保守的 α_f",
            },
            {
                "name": "LiDAR",
                "dims": "72 rays (5° resolution, 360°)",
                "size": 72,
                "slice": "[6:78]",
                "alpha_fast": 0.005,
                "alpha_slow": 0.0002,
                "tau_lo": 0.30,
                "tau_hi": 0.60,
                "epsilon": 1e-8,
                "half_life_fast": "~139 steps",
                "half_life_slow": "~3466 steps",
                "rationale": "LiDAR 分布在多障礙階段會顯著左移（更多近距讀數），"
                             "但 72D 高維需更穩定的基線，故 α 最小",
            },
            {
                "name": "Obstacle",
                "dims": "top-10 × 6D [x,y,vx,vy,r,m] = 60D",
                "size": 60,
                "slice": "[78:138]",
                "alpha_fast": 0.01,
                "alpha_slow": 0.0005,
                "tau_lo": 0.20,
                "tau_hi": 0.50,
                "epsilon": 1e-6,
                "half_life_fast": "~69 steps",
                "half_life_slow": "~1386 steps",
                "rationale": "障礙物數量隨課程劇變（0→7），分布變化最大，需最快 α_f；"
                             "ε=1e-6 避免零填充 slot 導致除以極小值",
            },
        ],
        "wandb_metrics": [
            {
                "key": "cadn/drift_state",
                "name_zh": "State 分支漂移量",
                "desc": "fast/slow 均值差的正規化距離",
                "ideal": "<0.2",
                "good_signal": "穩定階段 <0.1，課程跳階瞬間 spike 後快速回落",
                "bad_signal": "持續 >0.5 = 分布未收斂，α_slow 太慢或課程跳太快",
                "range": "[0, ∞)",
            },
            {
                "key": "cadn/drift_lidar",
                "name_zh": "LiDAR 分支漂移量",
                "desc": "LiDAR 72D 的 fast/slow drift",
                "ideal": "<0.3",
                "good_signal": "Phase 3+ 穩定在 0.1~0.2（障礙物增加使 LiDAR 分布左移）",
                "bad_signal": ">0.6 持續不降 = LiDAR 正規化失效，檢查 ray 範圍設定",
                "range": "[0, ∞)",
            },
            {
                "key": "cadn/drift_obstacle",
                "name_zh": "Obstacle 分支漂移量",
                "desc": "Obstacle 60D 的 fast/slow drift",
                "ideal": "<0.3",
                "good_signal": "課程跳階 spike 後 50~100 steps 回穩",
                "bad_signal": ">0.5 持續 = 零填充比例劇變，考慮增加 ε",
                "range": "[0, ∞)",
            },
            {
                "key": "cadn/blend_weight_state",
                "name_zh": "State blend weight",
                "desc": "w=0: 用 slow (穩定), w=1: 用 fast (追蹤)",
                "ideal": "穩定期 ≈0, 切換期 spike",
                "good_signal": "課程跳階時 spike 至 0.5~1.0，數百步後回到 ≈0",
                "bad_signal": "長期 >0.5 = 永遠用 fast channel，基線沒建立",
                "range": "[0, 1]",
            },
            {
                "key": "cadn/blend_weight_lidar",
                "name_zh": "LiDAR blend weight",
                "desc": "LiDAR 分支的 fast/slow 混合權重",
                "ideal": "穩定期 ≈0, 切換期 spike",
                "good_signal": "與 drift_lidar 同步，spike → recovery",
                "bad_signal": "不隨 drift 變化 = tau 閾值設定不當",
                "range": "[0, 1]",
            },
            {
                "key": "cadn/blend_weight_obstacle",
                "name_zh": "Obstacle blend weight",
                "desc": "Obstacle 分支的 fast/slow 混合權重",
                "ideal": "穩定期 ≈0, 切換期 spike",
                "good_signal": "課程增加障礙物時 spike 最明顯（因 zero-padding 比例變化）",
                "bad_signal": "Phase 1 (0 障礙) 就 >0 = ε 設定有誤",
                "range": "[0, 1]",
            },
        ],
        "comparison": {
            "title": "RunningStandardScaler vs CADN",
            "rows": [
                {
                    "aspect": "追蹤速度",
                    "rss": "單一全域 α（固定或 Welford 遞增）",
                    "cadn": "雙速率 α_f / α_s + 自適應混合",
                },
                {
                    "aspect": "分支感知",
                    "rss": "139D 統一統計",
                    "cadn": "3 分支獨立（State 7D / LiDAR 72D / Obs 60D）",
                },
                {
                    "aspect": "課程適應",
                    "rss": "被動等統計量漂移收斂",
                    "cadn": "主動偵測 drift → 切換 fast channel",
                },
                {
                    "aspect": "Zero-padding",
                    "rss": "μ/σ 被零值拉偏",
                    "cadn": "Obs 分支 ε=1e-6 + 獨立統計，不污染其他分支",
                },
                {
                    "aspect": "Clip 保護",
                    "rss": "無 (或全域 clip)",
                    "cadn": "Per-branch clip ±5.0",
                },
                {
                    "aspect": "診斷",
                    "rss": "無（黑盒）",
                    "cadn": "6 項 WandB 指標（drift × 3 + blend × 3）",
                },
            ],
        },
        "experiment_design": {
            "hypothesis": "CADN 在課程切換時能更快穩定觀測正規化，"
                          "減少策略在新階段的初期崩潰（SR drop），"
                          "提升整體訓練效率與最終表現。",
            "control": "RunningStandardScaler (SKRL 預設)",
            "treatment": "CADN (--use_cadn)",
            "metrics_to_compare": [
                "perf/success_rate — 課程跳階後 SR 恢復速度",
                "perf/collision_rate — 新障礙階段的碰撞率",
                "train/policy_loss — 正規化穩定性影響 loss 波動",
                "cadn/drift_* — 量化分布漂移幅度與恢復時間",
            ],
            "expected_outcome": [
                "課程跳階後 SR 恢復時間縮短 30~50%",
                "Phase 3+ 碰撞率降低（LiDAR 正規化更準確）",
                "policy_loss 在課程切換時 spike 幅度更小",
            ],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Master loader
# ═══════════════════════════════════════════════════════════════════════════
def load_all_configs() -> dict:
    return {
        "ppo": parse_ppo_config(),
        "curriculum": parse_curriculum_configs(),
        "nn": parse_nn_architecture(),
        "rewards": parse_reward_config(),
        "actions": parse_action_space(),
        "observations": parse_observation_space(),
        "ablation": parse_ablation_families(),
        "diagnostics": parse_diagnostic_metrics(),
        "environment": parse_environment_scene(),
        "cli": parse_cli_config(),
        "reward_modes": parse_reward_modes(),
        "wandb_metrics": parse_wandb_metrics(),
        "cadn": parse_cadn_config(),
    }
