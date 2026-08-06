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

# 从抗扰动训练日志里找最新写入的 model_*.pt。
LATEST_CHECKPOINT="$(${PYTHON:-python3} - <<'PY'
from pathlib import Path

root = Path("logs/rsl_rl/exo_balance_stand_push")
models = sorted(root.glob("*/model_*.pt"), key=lambda p: (p.stat().st_mtime, p.name))
if not models:
    raise SystemExit("找不到 checkpoint：请先运行 ./scripts/run_push_train.sh")
print(models[-1])
PY
)"

echo "[INFO] 使用最新 checkpoint: ${LATEST_CHECKPOINT}"

# 默认打开宇树式速度扰动；额外参数会原样传给 play.py，例如 --push-scale 1.2。
python3 scripts/play.py ExoBalance-Stand-Push-Flat \
  --checkpoint-file "${LATEST_CHECKPOINT}" \
  --device cuda:0 \
  "$@"
