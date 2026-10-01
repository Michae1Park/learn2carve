#!/usr/bin/env bash
# One-time setup. Everything lands inside this repo:
#   third_party/IsaacLab   Isaac Lab source, pinned to ISAACLAB_COMMIT
#   .venv-sim              Python 3.12 env: Isaac Sim 6.1 + Isaac Lab + skrl (~30 GB)
# Installing Isaac Sim accepts the NVIDIA Omniverse EULA.
set -euo pipefail
cd "$(dirname "$0")/.."

ISAACLAB_COMMIT=51b1f61ac5dbf9b464d83fc0fab86e58c46cfd7a   # main, 2026-09-30 (Isaac Lab 3.0, Isaac Sim 6.1)

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

if [ ! -d third_party/IsaacLab/.git ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
fi
git -C third_party/IsaacLab fetch -q origin "$ISAACLAB_COMMIT"
git -C third_party/IsaacLab checkout -q "$ISAACLAB_COMMIT"

export OMNI_KIT_ACCEPT_EULA=YES
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-sim"
uv sync --project third_party/IsaacLab --extra isaacsim --extra skrl --extra test

.venv-sim/bin/python -c "import isaaclab, isaacsim, skrl; print('isaaclab', isaaclab.__version__ if hasattr(isaaclab, '__version__') else 'ok', '| skrl', skrl.__version__)"
