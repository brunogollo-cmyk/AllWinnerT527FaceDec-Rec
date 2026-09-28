#!/usr/bin/env bash
# Measure real capture / detect / embed timings on this machine.
set -euo pipefail
cd "$(dirname "$0")"
exec ./.venv/bin/python bench.py "$@"
