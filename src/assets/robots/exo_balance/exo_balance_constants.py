"""exo_balance 机器人资产、执行器和初始姿态配置。

这个文件对应宇树的 `g1_constants.py`。XML 负责描述机器人长什么样，
这里负责告诉训练代码：用哪个 XML、哪些关节可控、默认站姿是什么、动作尺度多大。
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

from src import SRC_PATH

# ---- MJCF 路径 --------------------------------------------------------------
EXO_BALANCE_XML: Path = (
    SRC_PATH / "assets" / "robots" / "exo_balance" / "xmls" / "exo_balance_standing.xml"
)

if not EXO_BALANCE_XML.exists():
    raise FileNotFoundError(
        f"找不到训练 XML: {EXO_BALANCE_XML}\n"
        "请先运行: python3 scripts/make_standing_scene.py"
    )


def get_assets(meshdir: str) -> dict[str, bytes]:
    """把 XML 旁边的 STL 资源打包给 mjlab，避免并行环境找不到 mesh。"""
    assets: dict[str, bytes] = {}
    update_assets(assets, EXO_BALANCE_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    """读取 standing XML，并把 mesh 资源挂到 MuJoCo spec 上。"""
    spec = mujoco.MjSpec.from_file(str(EXO_BALANCE_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


# ---- 初始站姿 ---------------------------------------------------------------
# 现在先使用零关节角。后续如果无绳站立一开始就摔，需要优先调这里。
STAND_KEYFRAME = EntityCfg.InitialStateCfg(
    # 使用 XML 里已经写好的 stand keyframe，避免 mjlab 再额外生成第二个 keyframe。
    joint_pos=None,
)


# ---- 执行器 -----------------------------------------------------------------
# standing XML 里已有 12 个 <position> 执行器，这里只包住 XML 执行器。
# 这样训练动作会写到同一批 actuator，不会重复创建 actuator。
EXO_BALANCE_XML_POSITION_ACTUATORS = XmlPositionActuatorCfg(
    target_names_expr=(".*_joint",),
)

EXO_BALANCE_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(EXO_BALANCE_XML_POSITION_ACTUATORS,),
    soft_joint_pos_limit_factor=0.9,
)


def get_exo_balance_robot_cfg() -> EntityCfg:
    """返回一份新的外骨骼机器人配置，避免多个任务共用同一个可变对象。"""
    return EntityCfg(
        init_state=STAND_KEYFRAME,
        spec_fn=get_spec,
        articulation=EXO_BALANCE_ARTICULATION,
    )


# 动作尺度参考宇树写法，但先对外骨骼保守一点。
# 网络输出范围一般是 [-1, 1]，乘这个 scale 后成为关节目标角偏移。
EXO_BALANCE_ACTION_SCALE: dict[str, float] = {
    ".*_hip_abduction.*": 0.12,
    ".*_hip_yaw.*": 0.10,
    ".*_hip_pitch.*": 0.18,
    ".*_knee.*": 0.18,
    ".*_ankle_abduction.*": 0.08,
    ".*_ankle_pitch.*": 0.12,
}


if __name__ == "__main__":
    # 手动调试用：直接启动 MuJoCo viewer 看 standing XML。
    import mujoco.viewer as viewer

    viewer.launch(get_spec().compile())
