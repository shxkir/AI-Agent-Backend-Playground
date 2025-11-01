#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_SERVICE_DIR="${PROJECT_ROOT}/python_ai"
VENV_PATH="${PY_SERVICE_DIR}/.venv"
PYTHON_BIN="python3"

cd "${PY_SERVICE_DIR}"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Creating virtual environment at ${VENV_PATH}..."
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

pip install --upgrade \
  fastapi \
  uvicorn \
  langchain \
  langgraph \
  chromadb \
  pydantic \
  openai \
  anthropic

echo "Starting FastAPI AI service..."
uvicorn app:app --port 8001 --reload
