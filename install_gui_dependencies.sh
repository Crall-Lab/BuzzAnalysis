#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-BuzzAnalysis}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "Conda was not found. Set CONDA_BIN=/path/to/conda and rerun." >&2
  exit 1
fi

# Let conda resolve named environments; their prefix need not be under ~/miniconda3.
"$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)'
echo "Installing GUI and analysis dependencies into: ${ENV_NAME}"
"$CONDA_BIN" install -n "$ENV_NAME" -c conda-forge -y \
  pyside6 pyqtgraph pillow opencv pandas numpy scipy 'shapely>=2' \
  labelme pytest pyarrow tqdm matplotlib
"$CONDA_BIN" run -n "$ENV_NAME" python -c \
  'import cv2, numpy, pandas, PIL, PySide6, labelme, pyqtgraph, pytest, scipy, shapely, pyarrow, tqdm, matplotlib; print("GUI/analysis imports passed.")'
