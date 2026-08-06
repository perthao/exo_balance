"""站立自平衡任务奖励函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def body_orientation_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """惩罚身体倾斜；值越小表示身体越接近直立。"""
    asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
        projected_gravity_b = quat_apply_inverse(body_quat_w, asset.data.gravity_vec_w)
        return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1).float()
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1).float()


def body_angular_velocity_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """惩罚机身 roll/pitch 方向角速度，减少左右晃和前后倒。"""
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    return torch.sum(torch.square(ang_vel[:, :2]), dim=1).float()


def base_height_l2(env: ManagerBasedRlEnv, target_height: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """惩罚机身高度偏离目标高度，帮助机器人别蹲塌也别跳起。"""
    asset: Entity = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2]
    return torch.square(height - target_height).float()


def base_xy_position_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """惩罚机身在水平面上漂远，抗扰动站立时用来鼓励被推后回到原地附近。"""
    asset: Entity = env.scene[asset_cfg.name]
    env_origins = env.scene.env_origins[:, :2]
    xy_error = asset.data.root_link_pos_w[:, :2] - env_origins
    return torch.sum(torch.square(xy_error), dim=1).float()


def base_linear_velocity_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """惩罚机身线速度，站立任务里希望整体尽量不飘。"""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1).float()


def foot_contact_bonus(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """奖励双脚接触地面；两只脚都踩住时奖励最大。"""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    contact = (sensor.data.found > 0).float()
    return torch.mean(contact, dim=1).float()


def foot_slip_l2(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """惩罚脚已经接触地面时还在水平滑动。"""
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    contact = (sensor.data.found > 0).float()
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    slip = torch.sum(torch.square(foot_vel_xy), dim=-1) * contact
    return torch.sum(slip, dim=1).float()


def joint_deviation_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """惩罚关节偏离默认站姿，站立早期先让机器人学会保持初始姿态。"""
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.mean(torch.square(joint_pos - default_pos), dim=1).float()
