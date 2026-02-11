#!/usr/bin/env bash
# Create a separate conda env for DPAD (TensorFlow 2.15) inside the xCEBRA folder.
# Usage: ./setup_dpad_env.sh   OR   bash setup_dpad_env.sh
# Run from the xCEBRA project root.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ENV_DIR="${ENV_DIR:-./env_dpad}"
echo "Creating conda env at: $ENV_DIR (Python 3.11)"
conda create --prefix "$ENV_DIR" python=3.11 -y --solver classic
echo "Activate with: conda activate $ENV_DIR"
echo "Then run: pip install -r requirements-dpad.txt"
echo ""
echo "To install now, run:"
echo "  conda activate $ENV_DIR && pip install -r requirements-dpad.txt"
