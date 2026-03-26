# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
VLP-16 v3 環境配置（SKRL 版本）— 突破 50% 成功率瓶頸

核心架構：
- Velodyne VLP-16 16 線 LiDAR（16ch x 360 horizontal, max 20m）
- v3 動作：MultiDiscrete([19, 19]) — 兩組獨立 Categorical，解決維度詛咒
  v_max=1.0 m/s, a_max=0.5 m/s², ω_max=0.25π rad/s
- 觀測設計 (139D):
  * Policy: 139D = ego(4) + goal(2) + static(72) + obs(60) + time(1)
  * Critic: 139D（與 Policy 相同，暫不加特權資訊）

場景：16x16m 迷宮，6 面內部牆壁，10 個障礙物
Episode: 45 秒，decimation=20（env dt = 0.2s）

v3 變更：
- collision_terminal: -80 → -200（碰撞期望值大幅降低）
- near_obstacle_penalty: 線性 → 指數型排斥力 -5.0（明確斥力梯度）
- COLLISION_THRESHOLD: 0.7m → 0.45m（釋放窄道可行駛空間）
- 動作空間：Discrete(361) → MultiDiscrete([19, 19])
- γ: 0.98 → 0.995（有效視野 50→200 步）
"""

import math

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import MultiMeshRayCasterCfg, ContactSensorCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .charge_cfg import CHARGE_CFG
from ..multi_goal_command import MultiGoalCommandCfg
from ..goal_command import GoalCommandCfg

# 動作類
from ..mdp.actions import DiscreteDifferentialDriveActionCfg

# v2 觀測函數 (139D = ego 4 + goal 2 + static 72 + obs 60 + time 1)
from ..mdp.observations import (
    normalized_linear_acceleration,
    normalized_linear_velocity,
    base_angular_velocity_z,
    robot_radius_obs,
    goal_position_in_robot_frame,
    time_remaining_ratio,
    dynamic_obstacles_state,
)
from ..mdp.observations.obs_functions import (
    lidar_vlp16_to_2d_bins,
    topk_obstacles_6d,
)

# 獎勵函數
from ..mdp.rewards import (
    reaching_goal,
    collision_occurred,
    collision_contact_occurred,
    obstacle_proximity_termination,
)
from ..mdp.rewards.potential_based_rewards import (
    potential_progress_reward,
    near_obstacle_penalty,
    exponential_obstacle_penalty,
    collision_terminal_penalty,
    per_step_time_penalty,
    discrete_acceleration_squared_penalty,
    angular_velocity_squared_penalty,
    velocity_too_low_penalty,
    proximity_brake_penalty,
    risk_speed_penalty,
)

# 終止條件
from ..mdp.terminations import (
    goal_reached,
    robot_tipped_over,
    physics_explosion,
    wall_collision_termination,
)

# 事件
from ..mdp.events import (
    randomize_obstacles_by_difficulty,
    move_obstacles_vectorized,
)
from ..mdp.events.reset import reset_root_state_random_safe

from isaaclab.envs.mdp.terminations import time_out

# 域隨機化
from ..domain_randomization import apply_domain_randomization

# 核心狀態管理
from ..mdp.events.state import set_obstacle_metadata

# ============================================================================
# 環境常數
# ============================================================================
ROBOT_BODY_RADIUS = 0.35  # 實際機器人半徑 [m]
GOAL_REACH_THRESHOLD = ROBOT_BODY_RADIUS
COLLISION_BUFFER = 0.10  # 安全裕量：縮減以釋放窄道可行駛空間 [m]
COLLISION_THRESHOLD = round(ROBOT_BODY_RADIUS + COLLISION_BUFFER, 2)  # 0.45m
MAX_OBSTACLES = 100  # 場景最大障礙物 entity 數量（實際使用由 event params 控制）


# ============================================================================
# VLP-16 場景配置（迷宮：4 面外牆 + 10 面內部牆 + 10 個障礙物）
# ============================================================================
@configclass
class MySceneCfgVLP16(InteractiveSceneCfg):
    """VLP-16 場景 - 16x16m 迷宮

    LiDAR: VLP-16 16-channel, 360 horizontal (1 deg res), max 20m
    場景: 四面外牆 + 10 面內部牆壁 + 10 個障礙物
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot = CHARGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # VLP-16 LiDAR: 16 channels, 360 horizontal, 1.0 deg res, max 20m
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.6)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        max_distance=20.0,
        debug_vis=False,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/ground", track_mesh_transforms=False),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Wall_.*", track_mesh_transforms=False),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Obstacle_.*", track_mesh_transforms=True),
        ],
        update_period=0.2,
    )

    # 接觸感測器
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacle_.*"],
        debug_vis=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0),
    )

    def __post_init__(self):
        InteractiveSceneCfg.__post_init__(self)

        room_size = 8.0
        wall_thickness = 0.2
        wall_height = 1.5
        wall_length = room_size * 2 + wall_thickness
        wall_color = (0.5, 0.5, 0.5)

        wall_rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        wall_collision_props = sim_utils.CollisionPropertiesCfg()
        wall_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=wall_color, metallic=0.1)

        # 四面外牆
        self.wall_north = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_North",
            spawn=sim_utils.CuboidCfg(size=(wall_length, wall_thickness, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, room_size, wall_height / 2)),
        )
        self.wall_south = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_South",
            spawn=sim_utils.CuboidCfg(size=(wall_length, wall_thickness, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -room_size, wall_height / 2)),
        )
        self.wall_east = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_East",
            spawn=sim_utils.CuboidCfg(size=(wall_thickness, wall_length, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(room_size, 0.0, wall_height / 2)),
        )
        self.wall_west = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_West",
            spawn=sim_utils.CuboidCfg(size=(wall_thickness, wall_length, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-room_size, 0.0, wall_height / 2)),
        )

        # 6 面內部牆壁（迷宮結構，匹配 wall_layout.py MAZE_WALLS）
        internal_walls = [
            # (size, pos) — 均勻分佈在 4 象限，形成走廊
            ((3.5, 0.2, 1.5), (-6.25, 4.0, 0.75)),    # wall_internal_0: Top-left horizontal
            ((0.2, 3.0, 1.5), (3.0, 6.5, 0.75)),      # wall_internal_1: Top-right vertical
            ((3.0, 0.2, 1.5), (4.0, 0.0, 0.75)),      # wall_internal_2: Right-middle horizontal
            ((0.2, 4.0, 1.5), (-3.0, -1.0, 0.75)),    # wall_internal_3: Center-left vertical
            ((3.0, 0.2, 1.5), (0.5, -5.0, 0.75)),     # wall_internal_4: Bottom-center horizontal
            ((0.2, 3.0, 1.5), (6.5, -5.5, 0.75)),     # wall_internal_5: Bottom-right vertical
        ]

        for i, (size, pos) in enumerate(internal_walls):
            setattr(self, f"wall_internal_{i}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Wall_Internal_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=wall_rigid_props,
                    collision_props=wall_collision_props,
                    visual_material=wall_visual,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ))

        # 100 個障礙物（循環 10 種外觀，初始隱藏在 Z = -10.0）
        # 實際使用數量由 event params 的 num_obstacles_static/dynamic 控制
        # 觀測只取 Top-K=10 最近的（obs 維度不變 60D）
        HIDDEN_Z = -10.0
        _obstacle_templates = [
            {"type": "cuboid", "size": (0.5, 0.5, 1.2), "color": (0.8, 0.2, 0.2)},
            {"type": "cylinder", "radius": 0.3, "height": 1.0, "color": (0.8, 0.8, 0.2)},
            {"type": "cuboid", "size": (0.7, 0.7, 1.4), "color": (0.2, 0.4, 0.8)},
            {"type": "cylinder", "radius": 0.25, "height": 0.8, "color": (0.2, 0.8, 0.2)},
            {"type": "cuboid", "size": (0.6, 0.6, 1.0), "color": (0.8, 0.2, 0.8)},
            {"type": "cylinder", "radius": 0.35, "height": 1.2, "color": (0.8, 0.5, 0.2)},
            {"type": "cuboid", "size": (0.4, 0.4, 0.9), "color": (0.2, 0.8, 0.8)},
            {"type": "cylinder", "radius": 0.2, "height": 1.5, "color": (0.5, 0.5, 0.5)},
            {"type": "cuboid", "size": (0.55, 0.55, 1.1), "color": (0.9, 0.9, 0.9)},
            {"type": "cylinder", "radius": 0.28, "height": 1.1, "color": (0.3, 0.3, 0.3)},
        ]

        obstacle_sizes: list[float] = []
        for i in range(MAX_OBSTACLES):
            cfg = _obstacle_templates[i % len(_obstacle_templates)]
            if cfg["type"] == "cuboid":
                spawn_cfg = sim_utils.CuboidCfg(
                    size=cfg["size"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg["color"], metallic=0.2),
                )
                size_scalar = max(cfg["size"][0], cfg["size"][1])
            else:
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=cfg["radius"], height=cfg["height"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg["color"], metallic=0.2),
                )
                size_scalar = cfg["radius"] * 2.0

            setattr(self, f"obstacle_{i}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, HIDDEN_Z)),
            ))
            obstacle_sizes.append(size_scalar)

        set_obstacle_metadata(MAX_OBSTACLES, obstacle_sizes)


# ============================================================================
# VLP-16 命令配置（MultiGoalCommand，5 個目標）
# ============================================================================
@configclass
class CommandsCfgVLP16:
    """VLP-16 命令：MultiGoalCommand，5 個目標，距離 3-8m"""
    goal_command = MultiGoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        ranges=GoalCommandCfg.Ranges(
            distance=(3.0, 8.0),
            angle=(-math.pi, math.pi),
        ),
        wall_boundary=7.5,
        wall_safe_margin=1.0,
        obstacle_safe_distance=1.0,
        num_obstacles=0,
        num_goals=5,
    )


# ============================================================================
# VLP-16 離散動作配置
# ============================================================================
@configclass
class ActionsCfgVLP16:
    """VLP-16 離散動作：Discrete(361) = 19×19 中心對稱 + 動態加速度邊界

    num_bins = 19 → 19×19 = 361 flat discrete actions
    v_max = 1.0 m/s, a_max = 0.5 m/s², ω_max = 0.25π rad/s
    dt = 0.2s (decimation=20, sim.dt=0.01)
    """
    diff_drive = DiscreteDifferentialDriveActionCfg(
        asset_name="robot",
        debug_vis=True,
        num_bins=19,
        max_linear_velocity=1.0,
        max_linear_accel=0.5,
        max_angular_vel=0.25 * math.pi,
    )


# ============================================================================
# VLP-16 觀測配置 v2（139D = ego 4 + goal 2 + static 72 + obs 60 + time 1）
# ============================================================================
@configclass
class ObservationsCfgVLP16:
    """v2 觀測配置 — 139D Policy / 139D Critic (暫不加特權資訊)

    觀測向量佈局：
      [0]      ego: normalized_linear_acceleration  — ā_t          1D
      [1]      ego: normalized_linear_velocity      — v̄_t          1D
      [2]      ego: normalized_angular_velocity     — ω̄_t          1D
      [3]      ego: robot_radius                    — r_a          1D
      [4:6]    goal: waypoint in robot frame        — (x, y)       2D
      [6:78]   static: LiDAR 72 bins                — d_1..d_72    72D
      [78:138] obs: Top-10 obstacles × 6D           — o_1..o_10    60D
      [138]    time: remaining ratio                — T_rem         1D
      ─────────────────────────────────────────────────────────────────
      Total:                                                      139D

    obs 每物件 6D: o_i = [x_i^robot, y_i^robot, vx_i^robot, vy_i^robot, r_i, m_i]
      - (x, y, vx, vy) 為 robot body frame 下的位置與速度
      - r_i 為障礙物半徑
      - m_i 為有效位元 (1=可見, 0=填充/遮擋)

    設計原則：
      - 單幀 observation（不做 frame stacking）
      - goal = current local waypoint（robot frame）
      - static = LiDAR 72 bins（幾何距離）
      - obs = Top-10 物件級障礙物觀測（速度+半徑+遮擋）
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy 觀測組 (139D)

        ego (4D) + goal (2D) + static (72D) + obs (60D) + time (1D) = 139D
        """
        # --- ego state (4D) ---
        # ā_t: 歸一化前進加速度 — 1D
        linear_acceleration = ObsTerm(
            func=normalized_linear_acceleration,
            params={"asset_cfg": SceneEntityCfg("robot"), "max_acceleration": 1.0},
        )
        # v̄_t: 歸一化前進速度 — 1D
        linear_velocity = ObsTerm(
            func=normalized_linear_velocity,
            params={"asset_cfg": SceneEntityCfg("robot"), "max_linear_velocity": 1.0},
        )
        # ω̄_t: 歸一化角速度 — 1D
        angular_velocity = ObsTerm(
            func=base_angular_velocity_z,
            params={"asset_cfg": SceneEntityCfg("robot"), "max_angular_velocity": 1.5},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # r_a: 機器人半徑 — 1D
        radius = ObsTerm(
            func=robot_radius_obs,
            params={"radius": ROBOT_BODY_RADIUS},
        )

        # --- goal state (2D) ---
        # current waypoint in robot frame — (x_g, y_g)
        goal_position = ObsTerm(
            func=goal_position_in_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # --- static state (72D) ---
        # VLP-16 LiDAR 72 bins (單幀，不做 history stacking)
        lidar_static = ObsTerm(
            func=lidar_vlp16_to_2d_bins,
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),
                "num_channels": 16,
                "num_horizontal": 360,
                "num_bins": 72,
                "r_max": 20.0,
                "r_robot": ROBOT_BODY_RADIUS,
                "displacement_std": 0.02,
                "hole_rate": 0.005,
                "distractor_rate": 0.002,
                "distractor_range": (0.2, 2.0),
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(0.0, 1.0),
        )

        # --- obs state (60D) ---
        # Top-10 obstacles × 6D = 60D (body-frame, LOS 遮擋)
        obstacle_obs = ObsTerm(
            func=topk_obstacles_6d,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "top_k": 10,
                "max_obstacles": MAX_OBSTACLES,
                "max_distance": 8.0,
                "v_max": 1.5,
                "wall_occlusion": True,
                "speed_threshold": 0.01,
            },
        )

        # --- time state (1D) ---
        time_remaining = ObsTerm(
            func=time_remaining_ratio,
        )

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 觀測組 — 與 Policy 相同 (139D)

        暫不加入特權資訊。保持對稱 Critic，先建立 baseline。
        """
        # --- ego state (4D) ---
        linear_acceleration = ObsTerm(
            func=normalized_linear_acceleration,
            params={"asset_cfg": SceneEntityCfg("robot"), "max_acceleration": 1.0},
        )
        linear_velocity = ObsTerm(
            func=normalized_linear_velocity,
            params={"asset_cfg": SceneEntityCfg("robot"), "max_linear_velocity": 1.0},
        )
        angular_velocity = ObsTerm(
            func=base_angular_velocity_z,
            params={"asset_cfg": SceneEntityCfg("robot"), "max_angular_velocity": 1.5},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        radius = ObsTerm(
            func=robot_radius_obs,
            params={"radius": ROBOT_BODY_RADIUS},
        )

        # --- goal state (2D) ---
        goal_position = ObsTerm(
            func=goal_position_in_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # --- static state (72D) ---
        lidar_static = ObsTerm(
            func=lidar_vlp16_to_2d_bins,
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),
                "num_channels": 16,
                "num_horizontal": 360,
                "num_bins": 72,
                "r_max": 20.0,
                "r_robot": ROBOT_BODY_RADIUS,
                "displacement_std": 0.02,
                "hole_rate": 0.005,
                "distractor_rate": 0.002,
                "distractor_range": (0.2, 2.0),
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(0.0, 1.0),
        )

        # --- obs state (60D) ---
        # Top-10 obstacles × 6D = 60D (body-frame, LOS 遮擋)
        obstacle_obs = ObsTerm(
            func=topk_obstacles_6d,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "top_k": 10,
                "max_obstacles": MAX_OBSTACLES,
                "max_distance": 8.0,
                "v_max": 1.5,
                "wall_occlusion": True,
                "speed_threshold": 0.01,
            },
        )

        # --- time state (1D) ---
        time_remaining = ObsTerm(
            func=time_remaining_ratio,
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================================
# VLP-16 獎勵配置（匹配 env.yaml 精確值）
# ============================================================================
@configclass
class RewardsCfgVLP16:
    """VLP-16 獎勵配置 — 安全導航 v3（強化避障）

    v3 變更（突破 50% 成功率瓶頸）：
    - collision_terminal: -80 → -200（碰撞期望值大幅降低，迫使繞路）
    - near_obstacle_penalty: 線性 → 指數型排斥力（靠近牆壁時感受明確斥力梯度）
    - COLLISION_THRESHOLD: 0.7m → 0.45m（釋放窄道可行駛空間）

    ┌───────────────────────────────┬──────────┬────────────────────────┐
    │ Term                          │ Weight   │ 類別                   │
    ├───────────────────────────────┼──────────┼────────────────────────┤
    │ reaching_goal                 │ +250     │ 成功                   │
    │ potential_progress            │ +60      │ 前進                   │
    │ exponential_obstacle_penalty  │ -5.0     │ 安全（指數型排斥力）   │
    │ collision_terminal            │ -200     │ 安全（碰撞終止懲罰）   │
    │ time_penalty                  │ -0.3     │ 防呆                   │
    │ velocity_too_low              │ -0.3     │ 防呆                   │
    │ acceleration_penalty          │ -0.15    │ 平滑                   │
    │ angular_vel_penalty           │ -0.15    │ 平滑                   │
    └───────────────────────────────┴──────────┴────────────────────────┘
    """

    # 成功：到達目標
    reaching_goal = RewTerm(
        func=reaching_goal,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": GOAL_REACH_THRESHOLD, "body_radius": ROBOT_BODY_RADIUS},
        weight=250.0,
    )

    # 前進：PBRS 距離縮減
    potential_progress = RewTerm(
        func=potential_progress_reward,
        params={"robot_cfg": SceneEntityCfg("robot")},
        weight=60.0,
    )

    # 安全：指數型排斥力（d_min < 1.0m 時指數增加，靠近時極大懲罰）
    near_obstacle_penalty = RewTerm(
        func=exponential_obstacle_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "safe_distance": 1.0, "steepness": 6.0},
        weight=-5.0,
    )

    # 安全：碰撞終止懲罰（LiDAR）— v3 加重至 -200
    collision_terminal = RewTerm(
        func=collision_terminal_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
        weight=-200.0,
    )

    # 防呆：時間懲罰
    time_penalty = RewTerm(
        func=per_step_time_penalty,
        params={},
        weight=-0.3,
    )

    # 防呆：速度過低懲罰
    velocity_too_low = RewTerm(
        func=velocity_too_low_penalty,
        params={"robot_cfg": SceneEntityCfg("robot")},
        weight=-0.3,
    )

    # 平滑：加速度懲罰
    acceleration_penalty = RewTerm(
        func=discrete_acceleration_squared_penalty,
        params={},
        weight=-0.15,
    )

    # 平滑：角速度懲罰
    angular_velocity_penalty = RewTerm(
        func=angular_velocity_squared_penalty,
        params={"robot_cfg": SceneEntityCfg("robot")},
        weight=-0.15,
    )


# ============================================================================
# VLP-16 終止條件（含 Fix 2: physics_explosion）
# ============================================================================
@configclass
class TerminationsCfgVLP16:
    """VLP-16 終止條件

    - time_out: Episode 超時（45s）
    - goal_reached: 到達目標
    - robot_tipped_over: 翻倒
    - collision: LiDAR 碰撞（1.0m）
    - physics_explosion: Fix 2 — 物理引擎爆炸檢測（速度/位置異常）
    """
    time_out = DoneTerm(func=time_out, time_out=True)
    goal_reached = DoneTerm(
        func=goal_reached,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": GOAL_REACH_THRESHOLD, "body_radius": ROBOT_BODY_RADIUS},
    )
    robot_tipped_over = DoneTerm(
        func=robot_tipped_over,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 障礙物碰撞偵測 — 直接位置距離計算（不依賴 LiDAR 射線或 PhysX 接觸力）
    # kinematic obstacles + LiDAR 1.6m 高度導致 LiDAR/contact 偵測都失效
    collision = DoneTerm(
        func=obstacle_proximity_termination,
        params={"threshold": COLLISION_THRESHOLD, "max_obstacles": MAX_OBSTACLES},
    )
    # AABB 牆壁碰撞偵測
    wall_collision = DoneTerm(
        func=wall_collision_termination,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": COLLISION_THRESHOLD},
    )
    # Fix 2: 物理引擎爆炸檢測
    physics_explosion = DoneTerm(
        func=physics_explosion,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ============================================================================
# VLP-16 事件配置
# ============================================================================
@configclass
class EventCfgVLP16:
    """VLP-16 事件：隨機安全重置 + 域隨機化 + 混合障礙物 + 動態移動"""

    reset_base = EventTerm(
        func=reset_root_state_random_safe,
        mode="reset",
        params={
            "pose_range": {
                "x": (-5.0, 5.0),
                "y": (-5.0, 5.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_obstacles = None  # 由 randomize_obstacles 處理

    domain_randomization = EventTerm(
        func=apply_domain_randomization,
        mode="reset",
        params={"enable_physics": True, "enable_sensor_noise": True, "enable_external_force": True},
    )

    randomize_obstacles_startup = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="startup",
        params={
            "empty_ratio": 0.5, "static_ratio": 0.3, "dynamic_ratio": 0.2,
            "num_obstacles_static": 5, "num_obstacles_dynamic": 8,
            "max_obstacles": 10, "speed_range": 1.2, "min_speed": 0.3,
            "min_robot_distance": 1.5, "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.0, "max_spawn_attempts": 50,
            "boundary": 7.5, "active_obstacle_ratio": 0.25, "debug": False,
        },
    )

    randomize_obstacles = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="reset",
        params={
            "empty_ratio": 0.5, "static_ratio": 0.3, "dynamic_ratio": 0.2,
            "num_obstacles_static": 5, "num_obstacles_dynamic": 8,
            "max_obstacles": 10, "speed_range": 1.2, "min_speed": 0.3,
            "min_robot_distance": 1.5, "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.0, "max_spawn_attempts": 50,
            "boundary": 7.5, "active_obstacle_ratio": 0.25, "debug": False,
        },
    )

    move_dynamic_obstacles = EventTerm(
        func=move_obstacles_vectorized,
        mode="interval",
        interval_range_s=(0.2, 0.2),
        params={
            "move_dt": 0.2, "speed_min": 0.3, "speed_max": 1.2,
            "goal_reach_threshold": 0.5, "speed_resample_steps": 10,
            "area_limit": 6.0, "max_obstacles": 10, "bound_limit": 7.0,
        },
    )


# ============================================================================
# VLP-16 完整環境配置
# ============================================================================
@configclass
class ChargeNavigationEnvCfgVLP16(ManagerBasedRLEnvCfg):
    """VLP-16 離散動作環境（匹配 2026-03-07_15-18-14 訓練 run）

    - decimation=20, sim.dt=0.01 -> env dt = 0.2s
    - episode_length_s = 45.0
    - 1024 envs, env_spacing = 18.0
    - v2 觀測：Policy 139D / Critic 139D
    - Fix 2: physics_explosion 終止
    - Fix 3: velocity_too_low_penalty 獎勵
    """

    scene: MySceneCfgVLP16 = MySceneCfgVLP16(num_envs=1024, env_spacing=18.0)
    observations: ObservationsCfgVLP16 = ObservationsCfgVLP16()
    actions: ActionsCfgVLP16 = ActionsCfgVLP16()
    rewards: RewardsCfgVLP16 = RewardsCfgVLP16()
    terminations: TerminationsCfgVLP16 = TerminationsCfgVLP16()
    events: EventCfgVLP16 = EventCfgVLP16()
    commands: CommandsCfgVLP16 = CommandsCfgVLP16()

    def __post_init__(self):
        self.decimation = 20
        self.episode_length_s = 45.0
        self.is_finite_horizon = False
        self.sim.dt = 0.01
        self.sim.render_interval = 20
        self.viewer.eye = (7.5, 7.5, 7.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.seed = 42


# ============================================================================
# Phase 2: 避障微調配置
# ============================================================================

@configclass
class RewardsCfgVLP16Phase2(RewardsCfgVLP16):
    """Phase 2 獎勵配置 — 強化避障，保留導航能力

    變更（相對 Phase 1）：
    - progress_reward: +60 → +45（降低前進壓力，避免衝向障礙物）
    - near_obstacle_penalty: 移除指數型排斥力（由 proximity_brake 取代）
    - proximity_brake_penalty: -4.0（線性制動，1.5m→0.45m）
    - risk_speed_penalty: -5.0（靠近障礙物時懲罰高速）
    - collision_terminal: -200（保持 Phase 1 值，新 dense penalty 已提供安全梯度）

    量級驗算（dt=0.2s, progress ≈ +0.9/step）：
    - d=1.5m+: brake=0, risk=0（empty env 不受影響）
    - d=1.0m: brake=-0.381(42%), risk=-0.167(19%)  → 合計 -0.548(61%)
    - d=0.5m: brake=-0.762(85%), risk=-0.333(37%)  → 合計 -1.095(122%)
    """

    # 前進：降低權重，減少衝向障礙物的壓力
    potential_progress = RewTerm(
        func=potential_progress_reward,
        params={"robot_cfg": SceneEntityCfg("robot")},
        weight=45.0,
    )

    # 移除指數型排斥力（由 proximity_brake 取代）
    near_obstacle_penalty = None  # type: ignore[assignment]

    # 新增：近距離制動懲罰（線性 1.5m→0.45m）
    # d=1.0m: -0.381/step (42% of progress), d=0.5m: -0.762/step (85%)
    proximity_brake = RewTerm(
        func=proximity_brake_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "safe_distance": 1.5, "collision_threshold": COLLISION_THRESHOLD},
        weight=-4.0,
    )

    # 新增：風險速度懲罰（靠近障礙物時高速行駛懲罰）
    # d=1.0m, v=0.5: -0.167/step (19%), d=0.5m: -0.333/step (37%)
    risk_speed = RewTerm(
        func=risk_speed_penalty,
        params={"robot_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("lidar"), "risk_distance": 1.5},
        weight=-5.0,
    )

    # 安全：碰撞終止懲罰（保持 Phase 1 值 -200，新 dense penalty 已提供安全梯度）
    collision_terminal = RewTerm(
        func=collision_terminal_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
        weight=-200.0,
    )


@configclass
class EventCfgVLP16Phase2(EventCfgVLP16):
    """Phase 2 事件配置 — 增加障礙物比例（抗遺忘混合策略）

    環境比例變更：
    - empty: 0.50 → 0.35（保留較多空環境防止遺忘）
    - static: 0.30 → 0.35（增加靜態障礙物）
    - dynamic: 0.20 → 0.30（增加動態障礙物）
    """

    randomize_obstacles_startup = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="startup",
        params={
            "empty_ratio": 0.35, "static_ratio": 0.35, "dynamic_ratio": 0.30,
            "num_obstacles_static": 5, "num_obstacles_dynamic": 8,
            "max_obstacles": 10, "speed_range": 1.2, "min_speed": 0.3,
            "min_robot_distance": 1.5, "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.0, "max_spawn_attempts": 50,
            "boundary": 7.5, "active_obstacle_ratio": 0.25, "debug": False,
        },
    )

    randomize_obstacles = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="reset",
        params={
            "empty_ratio": 0.35, "static_ratio": 0.35, "dynamic_ratio": 0.30,
            "num_obstacles_static": 5, "num_obstacles_dynamic": 8,
            "max_obstacles": 10, "speed_range": 1.2, "min_speed": 0.3,
            "min_robot_distance": 1.5, "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.0, "max_spawn_attempts": 50,
            "boundary": 7.5, "active_obstacle_ratio": 0.25, "debug": False,
        },
    )


@configclass
class ChargeNavigationEnvCfgVLP16Phase2(ChargeNavigationEnvCfgVLP16):
    """Phase 2 環境配置 — 避障微調

    繼承 Phase 1 所有設定（scene, obs, actions, terminations, commands），
    僅覆蓋 rewards 和 events。
    """

    rewards: RewardsCfgVLP16Phase2 = RewardsCfgVLP16Phase2()
    events: EventCfgVLP16Phase2 = EventCfgVLP16Phase2()
