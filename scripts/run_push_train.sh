#!/usr/bin/env bash
set -euo pipefail

# 进入项目根目录，保证所有相对路径都从 exo_balance 开始。
cd "$(dirname "$0")/.."

# 激活已经配置好的 unitree_rl_mjlab 环境。
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_rl_mjlab

# Matplotlib 写到 /tmp，避免没有权限或缓存目录混乱。
export MPLCONFIGDIR=/tmp/matplotlib

# 让 Python 同时找到 exo_balance 自己的 src 和参考的 unitree_rl_mjlab。
export PYTHONPATH="$PWD:$PWD/../robot_rl/unitree_rl_mjlab:${PYTHONPATH:-}"

# 抗扰动站立训练默认用 256 个并行环境、6000 轮。
# 按之前这台机器的速度，大约 50-60 分钟完成；后面追加参数可以覆盖默认值。
python3 scripts/train.py ExoBalance-Stand-Push-Flat --env.scene.num-envs=256 --agent.max-iterations=6000 "$@"
