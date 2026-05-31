#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

pip install -e . -q

PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"

echo "Starting UI server at http://${HOST}:${PORT}"
exec uvicorn deepreview.api.app:app --host "$HOST" --port "$PORT" --reload
