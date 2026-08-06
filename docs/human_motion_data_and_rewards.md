# 标准人体运动数据和模仿奖励设计

目标：让外骨骼机器人从固定初始位姿开始，学习接近标准人体运动的姿态、速度、步态节奏和抗扰动恢复能力。

## 推荐下载顺序

### 1. AddBiomechanics

下载页：

```text
https://addbiomechanics.org/assets/download_data.html
https://addbiomechanics.org/download_data.html
```

命令行下载：

```bash
wget -O AddBiomechanicsDataset.zip http://archive.simtk.org/addbiomechanics/addbiomechanics.zip
```

用途：最适合外骨骼，因为它不只有人体姿态，还有 OpenSim 模型、关节角、关节速度、关节力矩、地面反力。后续可以拿来设计“像人一样站、走、受力”的奖励。

论文/引用方向：AddBiomechanics: Capturing the Physics of Human Motion at Scale。

### 2. AMASS

官网：

```text
https://amass.is.tue.mpg.de/
```

用途：大规模人体动作模仿。它把很多 mocap 数据统一成 SMPL/SMPL-H 参数，适合做“标准人体运动轨迹库”。需要注册账号并同意许可。

论文：Mahmood et al., AMASS: Archive of Motion Capture as Surface Shapes, ICCV 2019。

### 3. CMU Motion Capture Database

官网：

```text
https://mocap.cs.cmu.edu/
```

用途：免费 mocap 数据，包含 walking、running、turning 等。格式比较原始，常见是 ASF/AMC，也有第三方 BVH 转换。适合作为第一批小规模步态参考。

### 4. KIT Whole-Body Human Motion Database

官网：

```text
https://motion-database.humanoids.kit.edu/
```

用途：全身动作数据库，和机器人动作迁移关系更近。需要免费注册下载。它的数据里有 MMM 表示，适合参考“人体动作转机器人动作”的论文路线。

论文：Mandery et al., Unifying Representations and Large-Scale Whole-Body Motion Databases for Studying Human Motion, IEEE T-RO 2016。

### 5. OpenCap

官网：

```text
https://www.opencap.ai/
https://www.opencap.ai/get-started
https://github.com/opencap-org/opencap-core
```

用途：自己用手机采集人体动作，然后导出运动学/动力学结果。后面如果要采集你自己的外骨骼穿戴者步态，它会很有用。

论文：OpenCap: Human movement dynamics from smartphone videos, PLoS Computational Biology 2023。

### 6. Human3.6M

官网/论文入口：

```text
http://vision.imar.ro/human3.6m/
https://pubmed.ncbi.nlm.nih.gov/26353306/
```

用途：3D 人体姿态估计基准数据。它适合做人类姿态识别/视觉，不是首选的外骨骼动力学训练数据。

## 推荐先下载哪个

第一优先级：AddBiomechanics。

原因：外骨骼需要关节角、速度、力矩、足底接触和地面反力，AddBiomechanics 比纯视觉姿态数据更接近强化学习需要的状态和奖励。

第二优先级：AMASS。

原因：动作多，格式统一，适合做人形动作模仿。但它不是直接的外骨骼关节空间，需要做 retarget。

第三优先级：CMU/KIT。

原因：补充走路、转身、站起、上下台阶等动作。

## 数据到机器人动作的流程

```text
人体数据下载
-> 统一成每帧人体关节角/根部姿态/足端位置
-> 选择动作片段，例如静站、原地重心转移、慢走
-> 重采样到控制频率，例如 50 Hz
-> retarget 到外骨骼 12 个关节
-> 保存为 motion npz
-> 训练时每局从第 0 帧初始位姿开始
-> 奖励函数逐帧跟踪参考动作
```

## 初始位姿要求

当前环境已经改成每局固定回到初始位姿：

```text
x = 0
y = 0
z = 0.993
roll = 0
pitch = 0
yaw = 0
关节角 = XML stand keyframe
```

位置：

```text
src/tasks/stand/stand_env_cfg.py
```

## 奖励函数设计

先不要一步到位训练跑步。建议分四阶段：

### 阶段 1：标准站姿

目标：机器人回到初始人体站姿，不倒、不漂、不抖。

```text
r_alive                  活着奖励
r_base_height            pelvis/base 高度接近人体站姿
r_orientation            身体竖直
r_joint_posture          关节接近人体标准站姿
r_joint_velocity         关节速度小
r_action_smooth          控制动作平滑
r_base_xy                不水平漂移
```

### 阶段 2：人体静态平衡动作

目标：学习人体小幅重心转移，例如左右重心移动、踝策略、髋策略。

```text
r_joint_ref              当前关节角接近参考人体关节角
r_joint_vel_ref          当前关节速度接近参考人体关节速度
r_base_ref               base/pelvis 位置和姿态接近参考
r_foot_contact           该踩地的脚踩地
r_cop_stability          压力中心留在足底支撑区域内
```

### 阶段 3：慢速步态

目标：跟踪人体慢走周期。

```text
r_phase_joint_ref        按步态相位跟踪关节角
r_foot_pos_ref           足端轨迹接近人体脚轨迹
r_foot_clearance         摆动脚有离地高度
r_stance_contact         支撑脚保持接触
r_no_foot_slip           支撑脚不滑
r_base_velocity_ref      base 速度接近人体行走速度
```

### 阶段 4：扰动恢复

目标：被推后回到人体式平衡策略。

```text
r_recover_to_ref         扰动后回到当前参考相位
r_base_xy_return         base 不越漂越远
r_orientation_recover    身体姿态快速恢复
r_effort                 控制力矩不过大
```

## 推荐总奖励形式

```text
reward =
  1.0  * alive
  2.0  * exp(-joint_error / sigma_joint)
  1.0  * exp(-joint_vel_error / sigma_vel)
  1.5  * exp(-base_pose_error / sigma_base)
  1.0  * exp(-foot_pos_error / sigma_foot)
  0.8  * contact_match
 -0.05 * action_rate
 -0.000001 * joint_acc
 -0.0005 * torque_square
 -80.0 * termination
```

说明：`exp(-error/sigma)` 比直接 `-error` 更适合动作模仿，因为接近参考动作时奖励更明显，偏差太大时不会无限爆炸。

## 下一步要做的代码结构

建议新增：

```text
src/tasks/motion/
src/tasks/motion/motion_env_cfg.py
src/tasks/motion/mdp/rewards.py
src/tasks/motion/mdp/commands.py
src/tasks/motion/config/exo_balance/env_cfgs.py
src/tasks/motion/config/exo_balance/rl_cfg.py
data/motions/
```

`data/motions/` 不进 git，只保存下载和转换后的动作文件。

## 先做哪个动作

第一条 motion 不建议直接用走路。建议顺序：

```text
1. 标准站立 5-10 秒
2. 左右重心转移
3. 原地小幅抬脚
4. 慢速步态周期
```

外骨骼模型和人体腿部不完全一样，必须先 retarget，不能直接把人体关节角硬塞进机器人。
