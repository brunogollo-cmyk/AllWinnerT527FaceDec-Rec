#!/usr/bin/env bash
# Build (or rebuild) the face database from faces/<person_name>/*.jpg
set -euo pipefail
cd "$(dirname "$0")"
exec ./.venv/bin/python scripts/enroll_cli.py "$@"
