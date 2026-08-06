"""站立任务终止条件。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def base_height_below(env: ManagerBasedRlEnv, min_height: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """机身低于最小高度时终止，避免摔倒后继续无效采样。"""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_height


def illegal_contact(env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 10.0) -> torch.Tensor:
    """非脚部碰到地面时终止，逼策略只用脚站住。"""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)
        return (force_mag > force_threshold).any(dim=-1).any(dim=-1)
    assert data.found is not None
    return torch.any(data.found, dim=-1)
