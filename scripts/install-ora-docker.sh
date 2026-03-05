#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/ora-docker"
TARGET="/usr/local/bin/ora"

if [[ ! -x "$LAUNCHER" ]]; then
  chmod +x "$LAUNCHER"
fi

if [[ ! -w "$(dirname "$TARGET")" ]]; then
  if command -v sudo >/dev/null 2>&1; then
    sudo ln -sf "$LAUNCHER" "$TARGET"
  else
    echo "error: need write access to $(dirname "$TARGET") or install sudo" >&2
    exit 1
  fi
else
  ln -sf "$LAUNCHER" "$TARGET"
fi

echo "installed Docker launcher at $TARGET"
echo "default image: oralang/ora:latest (override with ORA_IMAGE)"
