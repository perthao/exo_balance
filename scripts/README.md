# scripts 使用说明

当前最终路线：`38.5 kg` 无绳模型 + 抗扰动站立训练。早期悬挂查看、步态 viewer、普通站立长训脚本已经删除，不再作为当前版本使用。

所有命令都从项目根目录运行：

```bash
cd ~/research/exo_balance
```

## 1. 生成并检查 XML

```bash
./scripts/run_standing_scene.sh
```

这个脚本会自动激活 `unitree_rl_mjlab`，然后执行：

```text
重新生成 exo_balance_standing.xml
重新生成 scene_exo_balance_standing.xml
检查 MuJoCo 能否加载
输出 nq/nv/nu/body 数量和总质量
```

有图形界面时可以打开 MuJoCo viewer：

```bash
./scripts/run_standing_scene.sh --view
```

## 2. 抗扰动训练

```bash
./scripts/run_push_train.sh
```

默认参数：

```text
任务名：ExoBalance-Stand-Push-Flat
并行环境：256
训练轮次：6000
预计时间：约 50-60 分钟
日志目录：logs/rsl_rl/exo_balance_stand_push/日期时间/
```

如果想临时改训练轮次或并行环境，直接在脚本后面追加参数：

```bash
./scripts/run_push_train.sh --agent.max-iterations=3000
./scripts/run_push_train.sh --env.scene.num-envs=512 --agent.max-iterations=6000
```

如果已经有训练结果，想从最新模型继续训练：

```bash
./scripts/run_resume_train.sh
```

它会自动从 `logs/rsl_rl/exo_balance_stand_push/` 里找最新 run 和最新 `model_*.pt`，不用手动输入日期和模型号。

训练时重点看这些指标：

```text
Mean episode length 越接近 600 越好
Episode_Termination/fell_over 越接近 0 越好
Episode_Termination/base_too_low 越接近 0 越好
Episode_Reward/base_xy_position_l2 绝对值越小，说明被推后漂移越少
Episode_Metrics/mean_action_acc 越小，动作越平稳
```

## 3. 看训练结果

最简单的回放方式：

```bash
./scripts/run_latest_play.sh
```

它会自动找到最新 checkpoint，例如：

```text
logs/rsl_rl/exo_balance_stand_push/最新日期时间/model_最新数字.pt
```

如果要手动指定某个 checkpoint，也可以直接用 `play.py`：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitree_rl_mjlab
export MPLCONFIGDIR=/tmp/matplotlib
export PYTHONPATH="$PWD:$PWD/../robot_rl/unitree_rl_mjlab:$PYTHONPATH"

python3 scripts/play.py ExoBalance-Stand-Push-Flat \
  --checkpoint-file logs/rsl_rl/exo_balance_stand_push/日期时间/model_5999.pt \
  --device cuda:0
```

回放默认会打开测试扰动，量级和训练一致，参考宇树 G1 velocity 任务：

```text
x/y 速度：-0.5 到 0.5 m/s
z 速度：-0.4 到 0.4 m/s
roll/pitch 角速度：-0.52 到 0.52 rad/s
yaw 角速度：-0.78 到 0.78 rad/s
间隔：5.0 到 6.0 s
```

想稍微更猛一点：

```bash
./scripts/run_latest_play.sh --push-scale 1.2
```

想只看安静站立，不加随机扰动：

```bash
./scripts/run_latest_play.sh --test-push False
```

如果关闭 MuJoCo 窗口后终端暂时没有返回输入行，按一次 `Ctrl+C`。现在 `play.py` 会捕获它并关闭 viewer/env。

注意：`./scripts/run_standing_scene.sh --view` 只看静态 XML，不加载策略，也没有随机扰动。

如果没有图形界面，可以先只做 CPU 零动作检查：

```bash
python3 scripts/play.py ExoBalance-Stand-Push-Flat --agent zero --no-terminations True --device cpu
```

## 4. 当前保留脚本

```text
make_standing_scene.py   生成当前无绳训练 XML
run_standing_scene.sh    一键生成并检查 XML
train.py                 训练入口，命名对照宇树项目
play.py                  回放/测试入口，命名对照宇树项目
run_push_train.sh        一键抗扰动训练
run_resume_train.sh      从最新 checkpoint 继续训练
run_latest_play.sh       一键回放最新 checkpoint
```

## 5. 微调位置

XML、质量、电机、足底接触参数看：

```text
src/assets/robots/exo_balance/xmls/README.md
```

奖励、扰动力、PPO 参数看：

```text
src/tasks/stand/config/exo_balance/env_cfgs.py
src/tasks/stand/config/exo_balance/rl_cfg.py
```

注意：旧 70 kg/配重版本已经删除。旧 checkpoint 不建议继续用于当前 38.5 kg 模型，当前版本请重新训练。
