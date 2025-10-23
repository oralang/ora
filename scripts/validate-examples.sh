#!/usr/bin/env bash
# Validate all Ora example files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ORA_BIN="$PROJECT_ROOT/zig-out/bin/ora"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ ! -f "$ORA_BIN" ]; then
    echo -e "${RED}Error: Ora compiler not found at $ORA_BIN${NC}"
    echo "Please run 'zig build' first"
    exit 1
fi

echo "=================================="
echo "Validating Ora Examples"
echo "=================================="
echo ""

total=0
passed=0
failed=0

# Find all .ora files
while IFS= read -r -d '' file; do
    total=$((total + 1))
    rel_path="${file#$PROJECT_ROOT/}"
    
    # Run parser
    if "$ORA_BIN" parse "$file" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $rel_path"
        passed=$((passed + 1))
    else
        echo -e "${RED}✗${NC} $rel_path"
        failed=$((failed + 1))
        # Show error details
        "$ORA_BIN" parse "$file" 2>&1 | grep -E "(error|Error)" | head -3 | sed 's/^/  /'
    fi
done < <(find "$PROJECT_ROOT/ora-example" -name "*.ora" -print0 | sort -z)

echo ""
echo "=================================="
echo "Summary:"
echo "  Total:  $total"
echo -e "  ${GREEN}Passed: $passed${NC}"
echo -e "  ${RED}Failed: $failed${NC}"
echo "=================================="

if [ $failed -gt 0 ]; then
    exit 1
fi

