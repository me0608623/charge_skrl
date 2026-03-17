# ==============================================================================
# 實驗 D：新增近障速度懲罰 — 直接打 collision 瓶頸
# ==============================================================================
# 基於 VLP16 v3 配置，新增 near_obstacle_speed_penalty
# 不修改原有 reward 結構，只多加一項行為 shaping
# ==============================================================================

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .charge_env_cfg_vlp16 import (
    ChargeNavigationEnvCfgVLP16,
    ChargeNavigationEnvCfgVLP16_PLAY,
    RewardsCfgVLP16,
)
from ..mdp.rewards.potential_based_rewards import near_obstacle_speed_penalty


@configclass
class RewardsCfgVLP16_ExpD(RewardsCfgVLP16):
    """VLP16 v3 + 近障速度懲罰

    新增項:
        near_obstacle_speed: weight=-2.0
        當 LiDAR min < 3.0m 時，速度越快懲罰越大
        鼓勵 agent 在窄道/近障時減速通過
    """

    near_obstacle_speed = RewTerm(
        func=near_obstacle_speed_penalty,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "robot_cfg": SceneEntityCfg("robot"),
            "danger_distance": 1.5,
            "safe_distance": 3.0,
        },
        weight=-2.0,
    )


@configclass
class ChargeNavigationEnvCfgVLP16_ExpD(ChargeNavigationEnvCfgVLP16):
    """VLP16 + 近障速度懲罰實驗"""
    rewards: RewardsCfgVLP16_ExpD = RewardsCfgVLP16_ExpD()


@configclass
class ChargeNavigationEnvCfgVLP16_ExpD_PLAY(ChargeNavigationEnvCfgVLP16_ExpD):
    """Play 版本"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 18.0
