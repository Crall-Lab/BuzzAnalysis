#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-BuzzAnalysis}"
CONDA_BIN="${CONDA_BIN:-/home/august/miniconda3/bin/conda}"
ENV_PY="/home/august/miniconda3/envs/${ENV_NAME}/bin/python"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Could not find conda at: $CONDA_BIN" >&2
  echo "Set CONDA_BIN=/path/to/conda and rerun this script." >&2
  exit 1
fi

if [[ ! -x "$ENV_PY" ]]; then
  echo "Could not find Python for conda env '${ENV_NAME}' at: $ENV_PY" >&2
  echo "Create the env first or pass the env name as the first argument." >&2
  exit 1
fi

echo "Installing GUI review dependencies into conda env: ${ENV_NAME}"
"$CONDA_BIN" install -n "$ENV_NAME" -c conda-forge -y \
  pyside6 \
  pyqtgraph \
  pillow \
  opencv \
  pandas \
  numpy \
  scipy \
  shapely \
  labelme \
  pytest

echo "Verifying imports with: $ENV_PY"
"$ENV_PY" - <<'PY'
import cv2
import numpy
import pandas
import PIL
import PySide6
import labelme
import pyqtgraph
import pytest
import scipy
import shapely

print("GUI/analysis environment is ready.")
print(f"PySide6: {PySide6.__version__}")
print(f"labelme: {labelme.__version__}")
print(f"pyqtgraph: {pyqtgraph.__version__}")
print(f"opencv: {cv2.__version__}")
print(f"pandas: {pandas.__version__}")
print(f"numpy: {numpy.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"shapely: {shapely.__version__}")
print(f"pytest: {pytest.__version__}")
PY
