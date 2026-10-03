#!/usr/bin/env bash
# Start the facegate service in the foreground (Ctrl-C to stop).
set -euo pipefail
cd "$(dirname "$0")"
exec ./.venv/bin/python run.py "$@"
