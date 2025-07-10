#!/bin/bash

# Test all .ora example files with the Ora compiler
# This script tests each compilation phase (lex, parse, analyze, compile) for every .ora file

set -e

echo "Testing all .ora example files..."
echo "================================"

# Build the compiler first
echo "Building compiler..."
zig build

# Counters
total_files=0
passed_files=0
failed_files=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test phases to run
phases=("lex" "parse" "analyze" "compile")

# Function to test a single file
test_file() {
    local file="$1"
    local filename=$(basename "$file")
    
    echo -e "${YELLOW}Testing:${NC} $file"
    
    # Test each phase
    for phase in "${phases[@]}"; do
        if ! ./zig-out/bin/ora "$phase" "$file" >/dev/null 2>&1; then
            echo -e "${RED}FAILED:${NC} $file (phase: $phase)"
            # Show error output for debugging
            echo "Error output:"
            ./zig-out/bin/ora "$phase" "$file" 2>&1 || true
            echo ""
            return 1
        fi
    done
    
    echo -e "${GREEN}PASSED:${NC} $file"
    return 0
}

# Find and test all .ora files
while IFS= read -r -d '' file; do
    total_files=$((total_files + 1))
    
    if test_file "$file"; then
        passed_files=$((passed_files + 1))
    else
        failed_files=$((failed_files + 1))
    fi
    
    echo ""
done < <(find examples -name "*.ora" -type f -print0)

# Summary
echo "================================"
echo "Example testing complete:"
echo -e "${GREEN}Passed:${NC} $passed_files/$total_files files"

if [ $failed_files -gt 0 ]; then
    echo -e "${RED}Failed:${NC} $failed_files files"
    exit 1
else
    echo -e "${GREEN}All examples passed!${NC}"
    exit 0
fi 