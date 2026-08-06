# XML 微调参数

当前只保留无绳站立训练版本。不要直接改生成出来的 XML，改完生成脚本后重新运行：

```bash
./scripts/run_standing_scene.sh
```

生成文件：

```text
exo_balance.xml              原始 MuJoCo 本体，不直接训练
exo_balance_standing.xml     训练用 robot-only XML
scene_exo_balance_standing.xml  可单独打开查看的带地面场景
```

主要修改文件：

```text
scripts/make_standing_scene.py
```

## 常用参数

位置：`scripts/make_standing_scene.py` 顶部。

```python
BASE_START_POS = "0 0 0.993"
FOOT_COLLISION_POS = "0 0 0.113"
FOOT_COLLISION_SIZE = "0.11 0.045 0.025"
FOOT_FRICTION = "1.0 0.01 0.001"
MOTOR_FORCE_RANGE = "-200 200"
MOTOR_VELOCITY_LIMIT_RPM = 150.0
TARGET_TOTAL_MASS = 38.5
EXTRA_MASS_BODIES = [
    ("base_link", "waist_battery", "0 0 0.05", "0.14 0.07 0.04", 5.00),
]
```

## 怎么调

```text
视觉脚底穿进地面：增大 FOOT_COLLISION_POS 的 z，或增大 BASE_START_POS 的 z
视觉脚底离地太高：减小 FOOT_COLLISION_POS 的 z，或减小 BASE_START_POS 的 z
脚底接触面积太小：增大 FOOT_COLLISION_SIZE 的 x/y
接触太滑：增大 FOOT_FRICTION 第一个数
电机最大力矩：改 MOTOR_FORCE_RANGE，现在是 -200 200 Nm
电机最大速度：改 MOTOR_VELOCITY_LIMIT_RPM，现在是 150 RPM = 15.708 rad/s
腰部电池重量：改 EXTRA_MASS_BODIES 里最后一个数，现在是 5.00 kg
```

## 当前质量

```text
base_link       1.4 kg
left leg        16.0 kg
right leg       16.1 kg
waist_battery   5.0 kg
total           38.5 kg
```

说明：当前版本不再添加足部配重，也不再添加全身透明配重。MuJoCo 现在直接使用 XML/URDF 的 `inertial mass` 和 `diaginertia`，不是按 mesh 材料密度自动算质量。如果以后要精细体现电机材料更重，需要回到对应 link 的 `mass` 和 `inertia` 改。

## 关节和执行器

关节命名在 `JOINT_NAME_MAP` 里改。训练中使用的 12 个关节是：

```text
left_hip_abduction_joint
left_hip_yaw_joint
left_hip_pitch_joint
left_knee_joint
left_ankle_abduction_joint
left_ankle_pitch_joint
right_hip_abduction_joint
right_hip_yaw_joint
right_hip_pitch_joint
right_knee_joint
right_ankle_abduction_joint
right_ankle_pitch_joint
```

注意：MuJoCo 里有两层力矩限制。脚本会同时设置 joint 的 `actuatorfrcrange` 和 actuator 的 `forcerange`。速度限制不在 XML 位置执行器里直接生效，而是在训练动作类 `VelocityLimitedJointPositionActionCfg` 里按 `15.708 rad/s` 限幅。
