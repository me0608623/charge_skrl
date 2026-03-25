"""獎勵模組 — VLP16 訓練使用的獎勵函數

保留的模組：
- utils: 工具函數（診斷、檢查）
- goal_rewards: 目標相關獎勵
- safety_rewards: 安全相關獎勵（避障、碰撞）
- potential_based_rewards: Potential-Based Reward Shaping（Phase 0）
"""

from .utils import (
    _print_diagnostics,
    _check_reward_term,
)

from .goal_rewards import (
    reaching_goal,
    heading_to_goal_distance_weighted,
    progress_to_goal,
    approaching_goal_bonus,
    alignment_reward,
)

from .safety_rewards import (
    collision_occurred,
    collision_contact_occurred,
    progressive_collision_penalty,
)

from .potential_based_rewards import (
    potential_progress_reward,
    near_obstacle_penalty,
    smooth_collision_penalty,
    collision_terminal_penalty,
    per_step_time_penalty,
    acceleration_squared_penalty,
    discrete_acceleration_squared_penalty,
    angular_velocity_squared_penalty,
    velocity_too_low_penalty,
    proximity_brake_penalty,
    risk_speed_penalty,
)

from .smoothness_rewards import (
    deadzone_acceleration_penalty,
    context_aware_angular_velocity_penalty,
)

from .navrl_rewards import (
    velocity_to_goal_reward,
    safety_log_distance_reward,
    safe_progress_reward,
)

from .gap_rewards import (
    heading_to_gap_reward,
    forward_clearance_improvement_reward,
)

from .navrl_ground_rewards import (
    goal_velocity_reward,
    goal_progress_reward,
    static_safety_reward,
    dynamic_safety_reward,
    control_smoothness_penalty,
    alive_reward,
)

__all__ = [
    # 工具函數
    "_print_diagnostics",
    "_check_reward_term",
    # 目標獎勵
    "reaching_goal",
    "heading_to_goal_distance_weighted",
    "progress_to_goal",
    "approaching_goal_bonus",
    "alignment_reward",
    # 安全獎勵
    "collision_occurred",
    "collision_contact_occurred",
    "progressive_collision_penalty",
    # Potential-Based Reward Shaping
    "potential_progress_reward",
    "near_obstacle_penalty",
    "smooth_collision_penalty",
    "collision_terminal_penalty",
    "per_step_time_penalty",
    "acceleration_squared_penalty",
    "discrete_acceleration_squared_penalty",
    "angular_velocity_squared_penalty",
    "velocity_too_low_penalty",
    "proximity_brake_penalty",
    "risk_speed_penalty",
    # Smoothness rewards
    "deadzone_acceleration_penalty",
    "context_aware_angular_velocity_penalty",
    # NavRL-style dense rewards
    "velocity_to_goal_reward",
    "safety_log_distance_reward",
    "safe_progress_reward",
    # Gap-seeking rewards
    "heading_to_gap_reward",
    "forward_clearance_improvement_reward",
    # NavRL-Ground v1 rewards
    "goal_velocity_reward",
    "goal_progress_reward",
    "static_safety_reward",
    "dynamic_safety_reward",
    "control_smoothness_penalty",
    "alive_reward",
]
