#!/usr/bin/env bash
# 生成并检查无绳站立训练 XML。

set -euo pipefail

# 所有路径都从脚本位置推导，避免依赖当前终端在哪个目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
XML_PATH="${PROJECT_ROOT}/src/assets/robots/exo_balance/xmls/scene_exo_balance_standing.xml"
ROBOT_XML_PATH="${PROJECT_ROOT}/src/assets/robots/exo_balance/xmls/exo_balance_standing.xml"
PYTHON_BIN="${PYTHON:-python3}"

cd "${PROJECT_ROOT}"

# 自动进入已经配置好的训练环境，保证能 import mujoco。
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_rl_mjlab

# 统一缓存和 Python 路径，和训练脚本保持一致。
export MPLCONFIGDIR=/tmp/matplotlib
export PYTHONPATH="$PWD:$PWD/../robot_rl/unitree_rl_mjlab:${PYTHONPATH:-}"

# 先重新生成 standing XML，保证参数修改后立即生效。
"${PYTHON_BIN}" scripts/make_standing_scene.py

# 用 MuJoCo 加载一次 XML，确认模型维度、执行器数量和总质量。
XML_PATH="${XML_PATH}" ROBOT_XML_PATH="${ROBOT_XML_PATH}" PROJECT_ROOT="${PROJECT_ROOT}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

try:
    import mujoco
except ModuleNotFoundError as exc:
    raise SystemExit("找不到 mujoco。请先 conda activate unitree_rl_mjlab 后再运行本脚本。") from exc

for key in ("ROBOT_XML_PATH", "XML_PATH"):
    xml_path = Path(os.environ[key])
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    print(f"xml={xml_path.relative_to(Path(os.environ['PROJECT_ROOT']))}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom}")
    print(f"total_mass_kg={sum(model.body_mass):.3f}")

project_root = Path(os.environ["PROJECT_ROOT"])
PY

if [[ "${1:-}" == "--view" ]]; then
  # 打开 MuJoCo 原生 viewer。服务器无图形界面时不要加 --view。
  "${PYTHON_BIN}" -m mujoco.viewer --mjcf="${XML_PATH}"
fi
