#!/usr/bin/env sh
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/harness" <<'SH'
#!/usr/bin/env sh
printf 'sample::__bytecode_bytes\t10\n'
printf 'sample::__deploy_gas\t110\n'
printf 'sample::c0:run\t210\n'
SH
chmod +x "$TMP/harness"

cat > "$TMP/baseline.tsv" <<'EOF'
sample::__bytecode_bytes	10
sample::__deploy_gas	100
sample::c0:run	200
EOF

ORA_METRICS_HARNESS="$TMP/harness" \
ORA_METRICS_BASELINE="$TMP/baseline.tsv" \
python3 "$ROOT/scripts/metrics-check.py" --check-size --report-dir "$TMP/report" >"$TMP/size.out"

ORA_METRICS_HARNESS="$TMP/harness" \
ORA_METRICS_BASELINE="$TMP/baseline.tsv" \
python3 "$ROOT/scripts/metrics-check.py" --check >"$TMP/strict.out" 2>"$TMP/strict.err" && {
    echo "check-metrics-report: strict check accepted gas drift" >&2
    exit 1
}

test -s "$TMP/report/current.tsv"
test -s "$TMP/report/compare.tsv"
test -s "$TMP/report/changed.tsv"
test -s "$TMP/report/summary.json"
grep -q '"gated_drift": false' "$TMP/report/summary.json"
grep -q 'deploy_gas.*local' "$TMP/report/compare.tsv"
grep -q 'runtime_gas.*local' "$TMP/report/compare.tsv"

cat > "$TMP/harness" <<'SH'
#!/usr/bin/env sh
printf 'sample::__bytecode_bytes\t11\n'
printf 'sample::__deploy_gas\t100\n'
printf 'sample::c0:run\t200\n'
SH
chmod +x "$TMP/harness"

ORA_METRICS_HARNESS="$TMP/harness" \
ORA_METRICS_BASELINE="$TMP/baseline.tsv" \
python3 "$ROOT/scripts/metrics-check.py" --check-size >"$TMP/size-drift.out" 2>"$TMP/size-drift.err" && {
    echo "check-metrics-report: size check accepted bytecode drift" >&2
    exit 1
}
grep -q 'bytecode-size drift' "$TMP/size-drift.err"

echo "check-metrics-report: ok"
