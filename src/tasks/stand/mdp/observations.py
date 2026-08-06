"""站立任务专用观测函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def base_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """读取基座坐标系下的角速度，等价于站立任务需要的 IMU 角速度观测。"""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b.float()


def base_lin_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """读取基座坐标系下的线速度，给 critic 使用。"""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b.float()


def projected_gravity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """读取重力在基座坐标系下的投影，并转成 float32。"""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b.float()


def joint_pos_rel(env: ManagerBasedRlEnv, biased: bool = False, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """读取相对默认站姿的关节角，并转成 float32。"""
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    return (joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[:, asset_cfg.joint_ids]).float()


def joint_vel_rel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """读取相对默认速度的关节速度，并转成 float32。"""
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return (asset.data.joint_vel[:, asset_cfg.joint_ids] - default_joint_vel[:, asset_cfg.joint_ids]).float()


def last_action(env: ManagerBasedRlEnv, action_name: str | None = None) -> torch.Tensor:
    """读取上一帧动作，并转成 float32。"""
    if action_name is None:
        return env.action_manager.action.float()
    return env.action_manager.get_term(action_name).raw_action.float()


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """返回左右脚是否接触地面，接触为 1，不接触为 0。"""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return (sensor.data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """返回左右脚接触力，并做 log 压缩，避免大冲击力把观测数值拉爆。"""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.force is not None
    forces = sensor.data.force.flatten(start_dim=1)
    return (torch.sign(forces) * torch.log1p(torch.abs(forces))).float()
