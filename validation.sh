#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
VENV_DIR='/home/bolun/Projects/venv311'

cd "${SCRIPT_DIR}" || exit

# Activate the virtual environment
if [ -f ${VENV_DIR}/bin/activate ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "Virtual environment not found at ${VENV_DIR}."
    exit 1
fi

source "${VENV_DIR}/bin/activate"

PYTHONPATH="${SCRIPT_DIR}" python "${SCRIPT_DIR}/Quark/Factor/validation.py"

deactivate