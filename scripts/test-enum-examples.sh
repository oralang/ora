#!/bin/bash

# Test script for enum examples
# This script compiles and tests all enum-related examples

set -e

echo "🧪 Testing Enum Type Implementation"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
    fi
}

# Function to run a test
run_test() {
    local test_file="$1"
    local test_name="$2"
    
    echo -e "${YELLOW}Running $test_name...${NC}"
    
    # Compile the test file
    if zig build test-file -Dfile="$test_file" > /dev/null 2>&1; then
        print_status 0 "$test_name compilation passed"
        return 0
    else
        print_status 1 "$test_name compilation failed"
        return 1
    fi
}

# Initialize test counters
total_tests=0
passed_tests=0

# Basic enum tests
echo -e "\n${YELLOW}=== Basic Enum Tests ===${NC}"

# Test 1: Basic enum functionality
total_tests=$((total_tests + 1))
if run_test "examples/core/enum_basic_test.ora" "Basic enum functionality"; then
    passed_tests=$((passed_tests + 1))
fi

# Test 2: Advanced enum features
total_tests=$((total_tests + 1))
if run_test "examples/advanced/enum_advanced_test.ora" "Advanced enum features"; then
    passed_tests=$((passed_tests + 1))
fi

# Test 3: Comprehensive enum test
total_tests=$((total_tests + 1))
if run_test "examples/advanced/enum_comprehensive_test.ora" "Comprehensive enum test"; then
    passed_tests=$((passed_tests + 1))
fi

# Additional enum feature tests
echo -e "\n${YELLOW}=== Enum Feature Tests ===${NC}"

# Test lexical analysis
echo -e "${YELLOW}Testing lexical analysis...${NC}"
total_tests=$((total_tests + 1))
if echo "enum TestEnum { A, B, C }" | zig build lex > /dev/null 2>&1; then
    print_status 0 "Enum lexical analysis"
    passed_tests=$((passed_tests + 1))
else
    print_status 1 "Enum lexical analysis"
fi

# Test parsing
echo -e "${YELLOW}Testing parsing...${NC}"
total_tests=$((total_tests + 1))
if echo "contract Test { enum Status { Active, Inactive } }" | zig build parse > /dev/null 2>&1; then
    print_status 0 "Enum parsing"
    passed_tests=$((passed_tests + 1))
else
    print_status 1 "Enum parsing"
fi

# Test semantic analysis
echo -e "${YELLOW}Testing semantic analysis...${NC}"
total_tests=$((total_tests + 1))
if zig build analyze -Dfile="examples/core/enum_basic_test.ora" > /dev/null 2>&1; then
    print_status 0 "Enum semantic analysis"
    passed_tests=$((passed_tests + 1))
else
    print_status 1 "Enum semantic analysis"
fi

# Test HIR generation
echo -e "${YELLOW}Testing HIR generation...${NC}"
total_tests=$((total_tests + 1))
if zig build hir -Dfile="examples/core/enum_basic_test.ora" > /dev/null 2>&1; then
    print_status 0 "Enum HIR generation"
    passed_tests=$((passed_tests + 1))
else
    print_status 1 "Enum HIR generation"
fi

# Test Yul generation
echo -e "${YELLOW}Testing Yul generation...${NC}"
total_tests=$((total_tests + 1))
if zig build yul -Dfile="examples/core/enum_basic_test.ora" > /dev/null 2>&1; then
    print_status 0 "Enum Yul generation"
    passed_tests=$((passed_tests + 1))
else
    print_status 1 "Enum Yul generation"
fi

# Test complete compilation
echo -e "${YELLOW}Testing complete compilation...${NC}"
total_tests=$((total_tests + 1))
if zig build compile -Dfile="examples/core/enum_basic_test.ora" > /dev/null 2>&1; then
    print_status 0 "Complete enum compilation"
    passed_tests=$((passed_tests + 1))
else
    print_status 1 "Complete enum compilation"
fi

# Performance benchmark
echo -e "\n${YELLOW}=== Performance Benchmark ===${NC}"
start_time=$(date +%s%3N)
zig build compile -Dfile="examples/advanced/enum_comprehensive_test.ora" > /dev/null 2>&1
end_time=$(date +%s%3N)
compile_time=$((end_time - start_time))

echo -e "${GREEN}Compilation time: ${compile_time}ms${NC}"

# Print summary
echo -e "\n${YELLOW}=== Test Summary ===${NC}"
echo "Total tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $((total_tests - passed_tests))"

if [ $passed_tests -eq $total_tests ]; then
    echo -e "${GREEN}All enum tests passed! 🎉${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please check the implementation.${NC}"
    exit 1
fi 