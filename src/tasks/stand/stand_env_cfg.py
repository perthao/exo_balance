"""无绳站立任务通用环境配置。

这个文件对应宇树的 `velocity_env_cfg.py`，只是把目标从“速度跟踪”缩小成
“先别倒、双脚踩住、动作平滑”。站稳以后再新增 velocity 任务。
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.stand.mdp as mdp


def make_stand_env_cfg() -> ManagerBasedRlEnvCfg:
    """创建站立任务基础配置，机器人专用信息在 config/exo_balance 里补。"""
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.0, n_max=1.0),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    critic_terms = {
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.3, n_max=0.3),
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=1,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": mdp.VelocityLimitedJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.10,
            velocity_limit=15.7079632679,
            # 站立 XML 的 stand keyframe 里关节目标就是 0。
            # 这里不用 MuJoCo 默认关节角做 offset，避免 CPU 训练时 float/double 混用。
            use_default_offset=False,
        )
    }

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.02, 0.02),
                    "y": (-0.02, 0.02),
                    # reset_root_state_uniform 这里写的是绝对高度，不是相对偏移。
                    # 之前写 0.0 会把机器人每次重置到地面以下，导致 base_too_low 立刻终止。
                    "z": (0.993, 0.993),
                    "roll": (-0.03, 0.03),
                    "pitch": (-0.03, 0.03),
                    "yaw": (-0.05, 0.05),
                },
                "velocity_range": {},
            },
        ),
        # 第一阶段先使用 XML 的 stand keyframe 重置关节。
        # mjlab 当前 CPU 路径下 reset_joints_by_offset 会产生 double/float 不匹配，
        # 等站立闭环跑通后，再补一个 float32 的小扰动重置函数。
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),
                "operation": "abs",
                "ranges": (0.6, 1.4),
                "shared_random": True,
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),
                "operation": "add",
                "ranges": {
                    0: (-0.02, 0.02),
                    1: (-0.02, 0.02),
                    2: (-0.02, 0.02),
                },
            },
        ),
    }

    rewards = {
        "alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
        "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-80.0),
        "body_orientation_l2": RewardTermCfg(
            func=mdp.body_orientation_l2,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
        ),
        "base_height_l2": RewardTermCfg(
            func=mdp.base_height_l2,
            weight=-8.0,
            params={"target_height": 0.993},
        ),
        "base_linear_velocity_l2": RewardTermCfg(
            func=mdp.base_linear_velocity_l2,
            weight=-0.5,
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.15,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
        ),
        "joint_deviation_l2": RewardTermCfg(
            func=mdp.joint_deviation_l2,
            weight=-0.4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        ),
        "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(60.0)},
        ),
        "base_too_low": TerminationTermCfg(
            func=mdp.base_height_below,
            params={"min_height": 0.50},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            sensors=(),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        metrics={"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=2.0,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            njmax=500,
            nconmax=64,
            contact_sensor_maxmatch=64,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=12.0,
    )
