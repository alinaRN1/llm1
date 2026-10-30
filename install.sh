#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
CC=gcc CXX=g++ CMAKE_BUILD_PARALLEL_LEVEL=4 .venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m ipykernel install --prefix "$PWD/.venv" --name llm1 --display-name 'ЛЛ1 (.venv)'
MPLCONFIGDIR=/tmp/llm1-mpl .venv/bin/python -m pytest -q tests
