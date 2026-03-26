#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Charge-SKRL Training Dashboard — One-click Launcher
# ═══════════════════════════════════════════════════════════════
# Usage:  ./launch_dashboard.sh
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD="$SCRIPT_DIR/scripts/reinforcement_learning/skrl/dashboard/launcher.py"

# Dashboard 不需要 Isaac Sim，直接用 conda python
PYTHON="/home/aa/miniconda3/envs/env_isaaclab/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Please activate conda env: conda activate env_isaaclab"
    exit 1
fi

exec "$PYTHON" "$DASHBOARD" "$@"
