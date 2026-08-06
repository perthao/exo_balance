# stand 任务代码说明

这里放无绳站立强化学习任务代码，最终训练入口是：

```text
ExoBalance-Stand-Push-Flat
```

运行、测试、回放命令统一看：

```text
scripts/README.md
```

## 文件作用

```text
stand_env_cfg.py                         通用站立环境骨架
config/exo_balance/__init__.py           注册任务名
config/exo_balance/env_cfgs.py           外骨骼环境参数、速度扰动、奖励权重
config/exo_balance/rl_cfg.py             PPO 网络和训练轮次
mdp/actions.py                           关节位置动作和 150 RPM 速度限幅
mdp/observations.py                      观测量
mdp/rewards.py                           奖励函数
mdp/terminations.py                      摔倒/过低终止条件
```

## 当前训练策略

当前采用抗扰动站立训练：训练过程中参考宇树 G1 的做法，随机给机器人根部速度一个扰动，让策略学会被扰动后回到平衡位置。

主要参数位置：

```text
速度扰动大小：config/exo_balance/env_cfgs.py 里的 push_robot
奖励权重：config/exo_balance/env_cfgs.py 里的 rewards
PPO 网络：config/exo_balance/rl_cfg.py
默认轮次：config/exo_balance/rl_cfg.py，当前 6000
```

旧的普通站立任务函数还保留给抗扰动任务复用，但当前实际训练和文档只推荐 `ExoBalance-Stand-Push-Flat`。
