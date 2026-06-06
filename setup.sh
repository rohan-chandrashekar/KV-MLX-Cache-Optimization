#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Python env ready (MLX + mlx-lm)."
echo "Confirm a small 4-bit model fits this machine, then build Phase 0:"
echo "  python scripts/baseline.py   # created in Phase 0"
