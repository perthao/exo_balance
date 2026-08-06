#!/usr/bin/env python3
"""生成 exo_balance 的无绳站立训练场景。

这个文件不会覆盖原始机器人 XML。它从
`exo_balance.xml` 复制机器人结构，然后新增自由根、地面、足底简化碰撞、
足端 site、腰部电池和 12 个位置执行器，作为强化学习站立任务的模型入口。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
XML_DIR = ROOT / "src" / "assets" / "robots" / "exo_balance" / "xmls"
ROBOT_XML = XML_DIR / "exo_balance.xml"

OUT_XML = XML_DIR / "scene_exo_balance_standing.xml"
OUT_ROBOT_XML = XML_DIR / "exo_balance_standing.xml"

# ---- 从 URDF/当前模型确认后的本体参数 -------------------------------------
# base_link 的惯量来自原始 URDF；只保留对角惯量，方便 MuJoCo 调试。
BASE_INERTIAL = {
    "pos": "0.078916219 0.0093931604 0.010346157",
    "mass": "1.4",
    "diaginertia": "0.012599903 0.0047904399 0.012666433",
}

# CAD 自动导出的旧关节名 -> 训练更容易看懂的新关节名。
# body 名暂时不改，方便继续追踪 mesh 和惯量挂在哪个原始 link 上。
JOINT_NAME_MAP = {
    "link_006_joint": "left_hip_abduction_joint",
    "link_007_joint": "left_hip_yaw_joint",
    "link_008_joint": "left_hip_pitch_joint",
    "link_009_joint": "left_knee_joint",
    "link_012_joint": "left_ankle_abduction_joint",
    "link_013_joint": "left_ankle_pitch_joint",
    "link_002_joint": "right_hip_abduction_joint",
    "link_003_joint": "right_hip_yaw_joint",
    "link_004_joint": "right_hip_pitch_joint",
    "link_005_joint": "right_knee_joint",
    "link_014_joint": "right_ankle_abduction_joint",
    "link_015_joint": "right_ankle_pitch_joint",
}

# 当前质量策略：保留 URDF 原始各部位惯量，只额外添加腰部电池。
# 原始模型约 33.5 kg，加 5 kg 电池后约 38.5 kg，控制在 40 kg 以内。
TARGET_TOTAL_MASS = 38.5
EXTRA_MASS_BODIES = [
    # 父 body，硬件名字，局部位置，盒子半尺寸，质量 kg。
    ("base_link", "waist_battery", "0 0 0.05", "0.14 0.07 0.04", 5.00),
]

# 关节被动参数和位置执行器刚度。先保持保守，避免站立训练一开始疯狂抖动。
JOINT_DAMPING = "2.0"
JOINT_ARMATURE = "0.01"
JOINT_FRICTIONLOSS = "0.03"
POSITION_KP = "45"

# ---- 站立训练常用参数 -------------------------------------------------------
# 这个高度让默认零关节角时，视觉脚底和足底简化碰撞盒都接近地面。
BASE_START_POS = "0 0 0.993"

# 左右脚在当前 CAD 树里的末端 body。真正训练时，contact sensor 会盯这两个 geom。
IMU_SITE_POS = "0 0 0"
LEFT_FOOT_BODY = "link_013"
RIGHT_FOOT_BODY = "link_015"
LEFT_FOOT_SITE_POS = "0 0 0"
RIGHT_FOOT_SITE_POS = "0 0 0"
FOOT_COLLISION_POS = "0 0 0.113"
FOOT_COLLISION_SIZE = "0.11 0.045 0.025"
FOOT_FRICTION = "1.0 0.01 0.001"

# 电机硬件限制。URDF 里的 velocity 单位是 rad/s；150 RPM = 15.707963 rad/s。
MOTOR_FORCE_RANGE = "-200 200"
MOTOR_VELOCITY_LIMIT_RPM = 150.0
MOTOR_VELOCITY_LIMIT_RAD_S = MOTOR_VELOCITY_LIMIT_RPM * 2.0 * 3.141592653589793 / 60.0

# 每个关节默认给 0 作为站立初始姿态。后续如果零姿态不是自然站姿，就改这里。
STANDING_KEY_JOINTS = {
    "left_hip_abduction_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_hip_pitch_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_abduction_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "right_hip_abduction_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_abduction_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
}


def box_diaginertia(mass: float, half_size_text: str) -> str:
    """根据 MuJoCo 盒子半尺寸计算实心盒子的对角惯量。"""
    sx, sy, sz = (float(v) for v in half_size_text.split())
    ixx = mass * (sy * sy + sz * sz) / 3.0
    iyy = mass * (sx * sx + sz * sz) / 3.0
    izz = mass * (sx * sx + sy * sy) / 3.0
    return f"{ixx:.8g} {iyy:.8g} {izz:.8g}"


def total_inertial_mass(root: ET.Element) -> float:
    """统计 XML 里所有 inertial 的质量，用来确认总质量。"""
    total = 0.0
    for inertial in root.findall(".//inertial"):
        mass = inertial.get("mass")
        if mass:
            total += float(mass)
    return total


def add_asset(scene: ET.Element, robot_root: ET.Element) -> None:
    """添加地面材质，并复制原机器人 XML 里的 mesh 资源。"""
    asset = ET.SubElement(scene, "asset")
    # 添加天空纹理，打开 viewer 时背景不会是纯黑。
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.35 0.45 0.55",
            "rgb2": "0.02 0.03 0.04",
            "width": "512",
            "height": "3072",
        },
    )
    # 添加棋盘地面纹理，用来观察脚底是否贴地。
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "ground_checker",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.24 0.25 0.25",
            "rgb2": "0.18 0.19 0.19",
            "width": "512",
            "height": "512",
        },
    )
    # 添加地面材质，后面的 floor geom 会引用这个材质。
    ET.SubElement(
        asset,
        "material",
        {
            "name": "ground_mat",
            "texture": "ground_checker",
            "texrepeat": "8 8",
            "texuniform": "true",
            "reflectance": "0.15",
        },
    )
    robot_asset = robot_root.find("asset")
    if robot_asset is not None:
        # 复制原始 STL mesh 声明，否则新场景找不到机器人外观。
        for child in list(robot_asset):
            asset.append(deepcopy(child))


def indent(elem: ET.Element, level: int = 0) -> None:
    """给 ElementTree 输出加缩进，方便人工检查 XML。"""
    space = "  "
    i = "\n" + level * space
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + space
        for child in elem:
            indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def disable_mesh_collisions(base_body: ET.Element) -> None:
    """把 CAD mesh 改成视觉用，训练接地只依赖后面新增的足底盒子。"""
    for geom in base_body.findall(".//geom"):
        if geom.get("mesh") is None:
            continue
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("group", "1")
        geom.set("density", "0")


def add_ground(worldbody: ET.Element) -> None:
    """添加平地、光源和相机，训练和检查 XML 都会用到。"""
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "sun",
            "pos": "0 -1.5 3",
            "dir": "0 0 -1",
            "directional": "true",
            "diffuse": "0.8 0.8 0.8",
            "specular": "0.2 0.2 0.2",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": "0 0 0",
            "size": "5 5 0.05",
            "material": "ground_mat",
            "condim": "3",
            "friction": FOOT_FRICTION,
        },
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": "2.0 -2.5 1.2",
            "xyaxes": "0.78 0.62 0 -0.24 0.30 0.92",
        },
    )


def add_foot_training_parts(base_body: ET.Element) -> None:
    """给左右脚添加训练用 site 和足底碰撞盒，不再添加足部配重。"""
    for body_name, side_name, site_pos in (
        (LEFT_FOOT_BODY, "left", LEFT_FOOT_SITE_POS),
        (RIGHT_FOOT_BODY, "right", RIGHT_FOOT_SITE_POS),
    ):
        body = base_body.find(f".//body[@name='{body_name}']")
        if body is None:
            print(f"[WARN] 找不到足端 body: {body_name}")
            continue

        # site 给观测和奖励使用，名字保持和宇树类似：left_foot/right_foot。
        ET.SubElement(
            body,
            "site",
            {"name": f"{side_name}_foot", "pos": site_pos, "size": "0.025", "rgba": "0 1 0 1"},
        )

        # 训练时只让这个盒子和地面接触，避免复杂 STL 碰撞导致抖动和穿模。
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{side_name}_foot_collision",
                "type": "box",
                "pos": FOOT_COLLISION_POS,
                "size": FOOT_COLLISION_SIZE,
                "condim": "3",
                "priority": "1",
                "friction": FOOT_FRICTION,
                "rgba": "0.1 0.9 0.2 0.25",
            },
        )



def add_extra_masses(base_body: ET.Element) -> None:
    """添加明确硬件含义的额外质量，当前只有腰部 5 kg 电池。"""
    for parent_name, hardware_name, pos, size, mass in EXTRA_MASS_BODIES:
        parent = base_body if parent_name == "base_link" else base_body.find(f".//body[@name='{parent_name}']")
        if parent is None:
            print(f"[WARN] 找不到额外质量父 body: {parent_name}")
            continue
        extra_body = ET.SubElement(parent, "body", {"name": hardware_name, "pos": pos})
        ET.SubElement(
            extra_body,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": f"{mass:.2f}",
                "diaginertia": box_diaginertia(mass, size),
            },
        )
        ET.SubElement(
            extra_body,
            "geom",
            {
                "name": f"{hardware_name}_geom",
                "type": "box",
                "size": size,
                "rgba": "0.1 0.35 0.9 0.35",
                "contype": "0",
                "conaffinity": "0",
            },
        )


def add_position_actuators(scene: ET.Element, joint_names: list[str]) -> None:
    """添加 12 个 XML 位置执行器，训练层会用 XmlPositionActuatorCfg 包住它们。"""
    actuator = ET.SubElement(scene, "actuator")
    for name in joint_names:
        actuator_base = name[:-6] if name.endswith("_joint") else name
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{actuator_base}_actuator",
                "joint": name,
                "kp": POSITION_KP,
                "ctrllimited": "true",
                "ctrlrange": "-1.57 1.57",
                "forcelimited": "true",
                "forcerange": MOTOR_FORCE_RANGE,
            },
        )


def add_imu_sensors(scene: ET.Element) -> None:
    """添加训练观测需要的 IMU 传感器，名字对齐宇树任务配置。"""
    sensor = ET.SubElement(scene, "sensor")
    ET.SubElement(sensor, "gyro", {"name": "imu_ang_vel", "site": "imu"})
    ET.SubElement(sensor, "velocimeter", {"name": "imu_lin_vel", "site": "imu"})
    ET.SubElement(sensor, "accelerometer", {"name": "imu_lin_acc", "site": "imu"})


def add_standing_keyframe(scene: ET.Element, joint_names: list[str]) -> None:
    """添加站立初始 keyframe，让 MuJoCo 和训练重置都能引用同一套默认姿态。"""
    key = ET.SubElement(scene, "keyframe")
    qpos_values = [*BASE_START_POS.split(), "1", "0", "0", "0"]
    qpos_values.extend(str(STANDING_KEY_JOINTS.get(name, 0.0)) for name in joint_names)
    ctrl_values = [str(STANDING_KEY_JOINTS.get(name, 0.0)) for name in joint_names]
    ET.SubElement(
        key,
        "key",
        {
            "name": "stand",
            "qpos": " ".join(qpos_values),
            "ctrl": " ".join(ctrl_values),
        },
    )


def add_robot_to_worldbody(scene: ET.Element, robot_root: ET.Element, worldbody: ET.Element) -> None:
    """把原始机器人复制进 worldbody，并补训练需要的关节、足底、电池和执行器。"""
    base_body = ET.SubElement(worldbody, "body", {"name": "base_link", "pos": BASE_START_POS})
    ET.SubElement(base_body, "freejoint", {"name": "floating_base_joint"})
    ET.SubElement(base_body, "inertial", BASE_INERTIAL)
    # IMU site 放在外骨骼基座上，站立任务用它读取角速度和线速度。
    ET.SubElement(base_body, "site", {"name": "imu", "pos": IMU_SITE_POS, "size": "0.01"})

    original_worldbody = robot_root.find("worldbody")
    if original_worldbody is None:
        raise RuntimeError(f"No <worldbody> in {ROBOT_XML}")
    for child in list(original_worldbody):
        base_body.append(deepcopy(child))

    joint_names: list[str] = []
    for joint in base_body.findall(".//joint"):
        old_name = joint.get("name")
        if not old_name:
            continue
        name = JOINT_NAME_MAP.get(old_name, old_name)
        joint.set("name", name)
        joint_names.append(name)
        joint.set("damping", JOINT_DAMPING)
        joint.set("armature", JOINT_ARMATURE)
        joint.set("frictionloss", JOINT_FRICTIONLOSS)
        joint.set("limited", "true")
        # joint 的 actuatorfrcrange 会再次限制实际施加到关节上的力矩。
        # 如果这里只保留原始 XML 的 -10 10，即使 actuator 写 200 也发不出来。
        joint.set("actuatorfrcrange", MOTOR_FORCE_RANGE)
        # MuJoCo joint 没有直接的速度限幅字段；这个值在训练动作层执行限幅。
        joint.set("user", f"{MOTOR_VELOCITY_LIMIT_RAD_S:.8f}")

    disable_mesh_collisions(base_body)
    add_foot_training_parts(base_body)
    add_extra_masses(base_body)
    add_position_actuators(scene, joint_names)
    # 训练阶段先不生成 XML 内置 IMU sensor。CPU 上 mujoco_warp 编译 sensor kernel
    # 容易出错；站立任务直接从 EntityData 读取基座速度，物理含义相同。
    add_standing_keyframe(scene, joint_names)


def add_common_header(scene: ET.Element) -> None:
    """添加 MuJoCo 编译和仿真基础参数。"""
    ET.SubElement(scene, "compiler", {"angle": "radian", "meshdir": "assets", "autolimits": "true"})
    ET.SubElement(scene, "size", {"nuser_jnt": "1"})
    ET.SubElement(
        scene,
        "option",
        {"timestep": "0.005", "gravity": "0 0 -9.81", "integrator": "RK4"},
    )
    ET.SubElement(scene, "statistic", {"center": "0 0 0.5", "extent": "1.6"})


def make_robot_asset() -> ET.Element:
    """生成训练用 robot-only XML，不自带地面，地面由 mjlab 环境创建。"""
    robot_root = ET.parse(ROBOT_XML).getroot()
    robot_xml = ET.Element("mujoco", {"model": "exo_balance_standing"})
    add_common_header(robot_xml)
    add_asset(robot_xml, robot_root)
    worldbody = ET.SubElement(robot_xml, "worldbody")
    add_robot_to_worldbody(robot_xml, robot_root, worldbody)
    return robot_xml


def make_scene() -> ET.Element:
    """生成可单独打开查看的无绳站立场景，包含地面、灯光和相机。"""
    robot_root = ET.parse(ROBOT_XML).getroot()

    scene = ET.Element("mujoco", {"model": "scene_exo_balance_standing"})
    add_common_header(scene)

    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3"})
    ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

    add_asset(scene, robot_root)

    worldbody = ET.SubElement(scene, "worldbody")
    add_ground(worldbody)
    add_robot_to_worldbody(scene, robot_root, worldbody)
    return scene


def write_pretty_xml(root: ET.Element, path: Path) -> None:
    """把生成好的 standing XML 写到磁盘。"""
    indent(root)
    raw = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    pretty = "\n".join(line for line in pretty.splitlines() if line.strip()) + "\n"
    path.write_text(pretty, encoding="utf-8")


if __name__ == "__main__":
    robot_asset = make_robot_asset()
    write_pretty_xml(robot_asset, OUT_ROBOT_XML)
    scene = make_scene()
    write_pretty_xml(scene, OUT_XML)
    print(OUT_ROBOT_XML.relative_to(ROOT))
    print(OUT_XML.relative_to(ROOT))
    print(f"total_mass_kg={total_inertial_mass(scene):.3f} target_kg={TARGET_TOTAL_MASS:.3f}")
