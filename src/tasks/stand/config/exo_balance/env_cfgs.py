"""exo_balance 无绳站立环境配置。"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots import EXO_BALANCE_ACTION_SCALE, get_exo_balance_robot_cfg
from src.tasks.stand import mdp
from src.tasks.stand.stand_env_cfg import make_stand_env_cfg


def exo_balance_stand_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """创建外骨骼平地无绳站立任务配置。"""
    cfg = make_stand_env_cfg()

    # 把通用站立任务里的 robot 换成外骨骼资产。
    cfg.scene.entities = {"robot": get_exo_balance_robot_cfg()}

    foot_geom_names = ("left_foot_collision", "right_foot_collision")

    # CPU 冒烟训练先不启用 ContactSensorCfg，因为 mujoco_warp 的 CPU sensor kernel
    # 在当前环境里会编译失败。后续上 CUDA 训练时再恢复脚底接触观测和脚滑奖励。
    cfg.scene.sensors = cfg.scene.sensors or ()

    # 动作尺度按关节类别设置，先保守，能站住后再放大。
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, mdp.VelocityLimitedJointPositionActionCfg)
    joint_pos_action.scale = EXO_BALANCE_ACTION_SCALE

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

    cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
    if play:
        # 可视化时不要频繁终止和随机扰动，便于观察策略行为。
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("foot_friction", None)
        cfg.events.pop("base_com", None)

    return cfg


def exo_balance_stand_push_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """创建外骨骼抗扰动站立任务配置。"""
    cfg = exo_balance_stand_flat_env_cfg(play=play)

    if play:
        return cfg

    # 每隔一段时间给 base_link 一个短促随机外力，训练机器人被推后自己拉回平衡。
    cfg.events["random_body_impulse"] = EventTermCfg(
        func=mdp.apply_body_impulse,
        mode="step",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            "force_range": (-120.0, 120.0),
            "torque_range": (-12.0, 12.0),
            "duration_s": (0.08, 0.18),
            "cooldown_s": (0.8, 2.0),
            "body_point_offset": (0.0, 0.0, 0.18),
        },
    )

    # 抗扰动站立不只要求“不倒”，还要求被推后别一路漂走。
    cfg.rewards["base_xy_position_l2"] = RewardTermCfg(
        func=mdp.base_xy_position_l2,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["body_orientation_l2"].weight = -3.0
    cfg.rewards["body_ang_vel"].weight = -0.25
    cfg.rewards["joint_deviation_l2"].weight = -0.25

    return cfg
