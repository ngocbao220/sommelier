#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/sommelier/bin/python}"

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/sommelier_mplconfig}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/sommelier_numba_cache}"

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" refactor/run_pipeline.py "$@"

