#!/bin/bash

set -e

if [ $# -lt 1 ]; then
    echo "Usage: ./python_run <script.py> [args...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

source "$PROJECT_DIR/venv/bin/activate"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

SCRIPT="$1"
shift

python "$SCRIPT_DIR/$SCRIPT" "$@"
