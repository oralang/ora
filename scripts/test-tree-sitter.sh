#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRAMMAR_DIR="$ROOT_DIR/tree-sitter-ora"
CANARY_LIST="$GRAMMAR_DIR/canary-files.txt"
WRAPPER="$GRAMMAR_DIR/scripts/tree-sitter-cli.js"

if [[ ! -d "$GRAMMAR_DIR" ]]; then
  echo "error: missing $GRAMMAR_DIR"
  exit 1
fi

if [[ -f "$WRAPPER" ]]; then
  TS_BIN="node $WRAPPER"
elif command -v tree-sitter >/dev/null 2>&1; then
  TS_BIN="tree-sitter"
elif [[ -x "$GRAMMAR_DIR/node_modules/.bin/tree-sitter" ]]; then
  TS_BIN="$GRAMMAR_DIR/node_modules/.bin/tree-sitter"
else
  echo "error: tree-sitter CLI not found"
  echo "hint: install it with one of:"
  echo "  npm install -g tree-sitter-cli"
  echo "  (cd $GRAMMAR_DIR && npm install)"
  exit 1
fi

echo "Using tree-sitter CLI: $TS_BIN"

pushd "$GRAMMAR_DIR" >/dev/null
bash -lc "$TS_BIN generate"
bash -lc "$TS_BIN test"
popd >/dev/null

if [[ -f "$CANARY_LIST" ]]; then
  echo "Running canary parse sweep"
  while IFS= read -r rel_path; do
    [[ -z "$rel_path" ]] && continue
    [[ "$rel_path" =~ ^# ]] && continue
    file="$ROOT_DIR/$rel_path"

    if [[ ! -f "$file" ]]; then
      echo "error: canary file not found: $rel_path"
      exit 1
    fi

    if ! output="$(cd "$GRAMMAR_DIR" && bash -lc "$TS_BIN parse \"$file\"" 2>&1)"; then
      echo "error: parser command failed for $rel_path"
      printf "%s\n" "$output"
      exit 1
    fi
    if printf "%s\n" "$output" | rg -q "ERROR"; then
      echo "error: parse produced ERROR node for $rel_path"
      printf "%s\n" "$output"
      exit 1
    fi
  done < "$CANARY_LIST"
fi

echo "tree-sitter checks passed"
