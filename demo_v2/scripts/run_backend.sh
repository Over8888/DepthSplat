#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/demo_v2
HOST="${DEPTHSPLAT_V3_HOST:-0.0.0.0}"
PORT="${DEPTHSPLAT_V3_PORT:-8012}"
python -m uvicorn app.main:app --host "${HOST}" --port "${PORT}"
