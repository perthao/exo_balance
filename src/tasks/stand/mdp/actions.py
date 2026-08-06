"""站立任务专用动作项。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class VelocityLimitedJointPositionActionCfg(JointPositionActionCfg):
    """带电机速度限幅的位置控制动作配置。"""

    velocity_limit: float = 15.7079632679
    """电机最大速度，单位 rad/s。150 RPM = 15.7079632679 rad/s。"""

    def build(self, env: ManagerBasedRlEnv) -> "VelocityLimitedJointPositionAction":
        """创建带速度限幅的位置控制动作项。"""
        return VelocityLimitedJointPositionAction(self, env)


class VelocityLimitedJointPositionAction(JointPositionAction):
    """把网络输出的位置目标限制成硬件允许的最大目标变化速度。"""

    cfg: VelocityLimitedJointPositionActionCfg

    def __init__(self, cfg: VelocityLimitedJointPositionActionCfg, env: ManagerBasedRlEnv):
        """初始化动作项，并计算每个控制周期允许变化的最大角度。"""
        super().__init__(cfg=cfg, env=env)
        self._max_delta = float(cfg.velocity_limit) * float(env.step_dt)

    def process_actions(self, actions: torch.Tensor) -> None:
        """先按 scale/offset 得到目标角，再按速度上限裁剪相邻两步变化量。"""
        previous_target = self._processed_actions.clone()
        super().process_actions(actions)
        delta = torch.clamp(self._processed_actions - previous_target, -self._max_delta, self._max_delta)
        self._processed_actions = previous_target + delta

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """重置动作历史，避免新 episode 沿用上一轮摔倒前的目标角。"""
        super().reset(env_ids=env_ids)
        self._processed_actions[env_ids] = 0.0
