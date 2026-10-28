#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-}"
shift || true
case "$MODE" in
  supervised) SCRIPT="train_supervised.py" ;;
  weakly) SCRIPT="train_weakly.py" ;;
  infer) SCRIPT="demo.py" ;;
  evaluate) SCRIPT="evaluate.py" ;;
  *) echo "Usage: bash scripts/run.sh <supervised|weakly|infer|evaluate> [script arguments]" >&2
     exit 2 ;;
esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
ENTRY="$ROOT/scripts/$SCRIPT"
[[ -x "$PYTHON" ]] || { echo "Environment missing. Run bash scripts/setup.sh first." >&2; exit 1; }
[[ -f "$ENTRY" ]] || { echo "Missing $ENTRY." >&2; exit 1; }
exec "$PYTHON" "$ENTRY" "$@"

