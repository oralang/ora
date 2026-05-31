#!/usr/bin/env sh
set -eu

fail() {
  echo "check-sir-shift-operand-order: $*" >&2
  exit 1
}

[ -d src/mlir/ora/lowering ] || fail "missing src/mlir/ora/lowering directory"

matches="$(
  grep -R -n -E "create<sir::(ShlOp|ShrOp|SarOp)>" src/mlir/ora/lowering --include="*.cpp" --include="*.h" \
    | awk '
      function trim(s) {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", s)
        return s
      }
      {
        # SIR shift ops are declared as (shift, x). In builder calls this is
        # create<sir::*Op>(loc, result_type, shift, value). Flag calls whose
        # first value operand does not look like an explicit shift/count.
        line = $0
        n = split(line, parts, ",")
        if (n < 3) {
          print line
          next
        }
        shift = trim(parts[3])
        if (shift ~ /(^|[^A-Za-z0-9_])(shift|byteIndex|one|zero|c[0-9][A-Za-z0-9_]*|c224_ls)([^A-Za-z0-9_]|$)/)
          next
        print line
      }
    ' \
    || true
)"

if [ -n "$matches" ]; then
  echo "$matches" >&2
  fail "sir::ShlOp/ShrOp/SarOp builder calls must pass the shift/count operand before the value operand"
fi

echo "check-sir-shift-operand-order: ok"
