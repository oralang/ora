#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

exec python3 "$PROJECT_ROOT/scripts/test_ora_features.py" --output "" --emit mlir --max-error-lines 8 "$@"
