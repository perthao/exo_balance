#!/usr/bin/env bash
set -euo pipefail

# 进入项目根目录，保证相对路径稳定。
cd "$(dirname "$0")/.."

# 自动激活训练环境。
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_rl_mjlab

# 统一缓存和 Python 路径。
export MPLCONFIGDIR=/tmp/matplotlib
export PYTHONPATH="$PWD:$PWD/../robot_rl/unitree_rl_mjlab:${PYTHONPATH:-}"

# 从最新抗扰动 checkpoint 继续训练。load-run/load-checkpoint 用正则，mjlab 会取最新。
python3 scripts/train.py ExoBalance-Stand-Push-Flat \
  --env.scene.num-envs=256 \
  --agent.resume True \
  --agent.load-run ".*" \
  --agent.load-checkpoint "model_.*.pt" \
  "$@"
