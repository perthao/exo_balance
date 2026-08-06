# exo_balance

外骨骼机器人 MuJoCo 仿真与强化学习训练项目。

当前确定版本：`38.5 kg` 无绳站立模型 + 抗扰动 PPO 训练。早期悬挂查看、步态 viewer、70 kg 配重版本已经删除，不作为当前路线使用。

## 快速运行

```bash
cd ~/research/exo_balance
./scripts/run_standing_scene.sh
./scripts/run_push_train.sh
```

## 目录说明

```text
scripts/      一键生成 XML、训练、回放入口
src/assets/   机器人 MuJoCo XML 和 STL 训练资产
src/tasks/    抗扰动站立强化学习任务配置和 MDP 函数
urdf/         原始 URDF 导出资料，训练不直接从这里加载
```

## 主要文档

```text
scripts/README.md                                      运行、训练、测试、回放命令
src/assets/robots/exo_balance/xmls/README.md           XML、质量、电机、足底接触微调
src/tasks/stand/README.md                              强化学习任务代码结构
docs/human_motion_data_and_rewards.md                  人体运动数据下载、论文和模仿奖励设计
```

## 当前训练入口

```text
ExoBalance-Stand-Push-Flat
```

默认训练脚本：

```bash
./scripts/run_push_train.sh
```

默认 `256` 个并行环境、`6000` 轮，按当前机器速度预计约 `50-60` 分钟。

## 版本管理说明

训练日志、W&B 离线记录、checkpoint、Python 缓存、原始 zip 和重复 STL 导出副本不会进入 git。训练需要的 XML 和 STL 资产保留在：

```text
src/assets/robots/exo_balance/xmls/
src/assets/robots/exo_balance/xmls/assets/
```
