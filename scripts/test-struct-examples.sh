#!/bin/bash

# Comprehensive test script for Ora struct implementations
# Tests all struct-related examples across multiple compilation phases
# Suitable for local development and CI/CD environments

set -e

echo "🏗️  Testing Ora Struct Implementation"
echo "===================================="

# Build the compiler first
echo "📦 Building compiler..."
if ! zig build; then
    echo "❌ Build failed!"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Counters
total_tests=0
passed_tests=0
failed_tests=0
total_files=0
passed_files=0
failed_files=0

# Test phases to run
phases=("lex" "parse" "analyze" "hir" "yul" "compile")

# Struct-specific example files
struct_examples=(
    "examples/struct_assignment_test.ora"
    "examples/struct_comprehensive_test.ora"
    "examples/struct_memory_model_test.ora"
    "examples/struct_memory_optimization_test.ora"
    "examples/memory_layout_demo.ora"
    "examples/struct_yul_test.ora"
    "examples/struct_advanced_features_test.ora"
)

# Function to test a single file with a specific phase
test_file_phase() {
    local file="$1"
    local phase="$2"
    local filename=$(basename "$file")
    
    total_tests=$((total_tests + 1))
    
    echo -n "  🔍 Testing $phase phase... "
    
    # Create output directory for test results
    mkdir -p test-results
    
    # Run the test and capture output
    if ./zig-out/bin/ora "$phase" "$file" > "test-results/${filename}_${phase}.out" 2>&1; then
        echo -e "${GREEN}✅ PASSED${NC}"
        passed_tests=$((passed_tests + 1))
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}"
        failed_tests=$((failed_tests + 1))
        # Show error output for debugging
        echo -e "${RED}Error output:${NC}"
        cat "test-results/${filename}_${phase}.out"
        echo ""
        return 1
    fi
}

# Function to test a single file across all phases
test_file() {
    local file="$1"
    local filename=$(basename "$file")
    
    total_files=$((total_files + 1))
    
    echo -e "${YELLOW}📄 Testing:${NC} $file"
    
    local file_passed=true
    
    # Test each phase
    for phase in "${phases[@]}"; do
        if ! test_file_phase "$file" "$phase"; then
            file_passed=false
        fi
    done
    
    if $file_passed; then
        echo -e "${GREEN}✅ ALL PHASES PASSED:${NC} $file"
        passed_files=$((passed_files + 1))
    else
        echo -e "${RED}❌ SOME PHASES FAILED:${NC} $file"
        failed_files=$((failed_files + 1))
    fi
    
    echo ""
    return 0
}

# Function to test specific struct features
test_struct_features() {
    echo -e "${BLUE}🧪 Testing Struct Features${NC}"
    echo "========================="
    
    # Test memory layout optimization
    echo -e "${PURPLE}Memory Layout Optimization:${NC}"
    test_file "examples/memory_layout_demo.ora"
    test_file "examples/struct_memory_optimization_test.ora"
    
    # Test HIR integration
    echo -e "${PURPLE}HIR Integration:${NC}"
    test_file "examples/struct_comprehensive_test.ora"
    test_file "examples/struct_memory_model_test.ora"
    
    # Test Yul code generation
    echo -e "${PURPLE}Yul Code Generation:${NC}"
    test_file "examples/struct_yul_test.ora"
    test_file "examples/struct_assignment_test.ora"
    
    # Test advanced features
    echo -e "${PURPLE}Advanced Features:${NC}"
    test_file "examples/struct_advanced_features_test.ora"
}

# Function to check if examples exist
check_examples() {
    echo -e "${BLUE}📋 Checking example files...${NC}"
    
    local missing_files=0
    
    for file in "${struct_examples[@]}"; do
        if [[ -f "$file" ]]; then
            echo -e "${GREEN}✅ Found:${NC} $file"
        else
            echo -e "${RED}❌ Missing:${NC} $file"
            missing_files=$((missing_files + 1))
        fi
    done
    
    if [ $missing_files -gt 0 ]; then
        echo -e "${RED}❌ Missing $missing_files example files!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All example files found!${NC}"
    echo ""
}

# Function to run performance benchmarks
run_benchmarks() {
    echo -e "${BLUE}⚡ Running Performance Benchmarks${NC}"
    echo "================================"
    
    local benchmark_file="examples/struct_memory_optimization_test.ora"
    
    if [[ -f "$benchmark_file" ]]; then
        echo "📊 Benchmarking struct memory optimization..."
        
        # Time the compilation phases
        echo -n "  ⏱️  Parse time: "
        time ./zig-out/bin/ora parse "$benchmark_file" >/dev/null 2>&1
        
        echo -n "  ⏱️  HIR time: "
        time ./zig-out/bin/ora hir "$benchmark_file" >/dev/null 2>&1
        
        echo -n "  ⏱️  Yul time: "
        time ./zig-out/bin/ora yul "$benchmark_file" >/dev/null 2>&1
        
        echo ""
    fi
}

# Function to generate test report
generate_report() {
    echo -e "${BLUE}📊 Test Report${NC}"
    echo "=============="
    
    echo "Test Results:"
    echo "  Total tests: $total_tests"
    echo -e "  Passed: ${GREEN}$passed_tests${NC}"
    echo -e "  Failed: ${RED}$failed_tests${NC}"
    echo ""
    
    echo "File Results:"
    echo "  Total files: $total_files"
    echo -e "  Passed: ${GREEN}$passed_files${NC}"
    echo -e "  Failed: ${RED}$failed_files${NC}"
    echo ""
    
    if [ $failed_tests -gt 0 ] || [ $failed_files -gt 0 ]; then
        echo -e "${RED}❌ Some tests failed. Check test-results/ directory for details.${NC}"
        return 1
    else
        echo -e "${GREEN}✅ All tests passed!${NC}"
        return 0
    fi
}

# Main execution
main() {
    # Check if we're in the right directory
    if [[ ! -f "build.zig" ]]; then
        echo -e "${RED}❌ Please run this script from the project root directory${NC}"
        exit 1
    fi
    
    # Check if examples exist
    check_examples
    
    # Run the struct feature tests
    test_struct_features
    
    # Run performance benchmarks if requested
    if [[ "$1" == "--benchmark" ]]; then
        run_benchmarks
    fi
    
    # Generate report
    if generate_report; then
        echo -e "${GREEN}🎉 All struct tests completed successfully!${NC}"
        exit 0
    else
        echo -e "${RED}💥 Some tests failed. See output above for details.${NC}"
        exit 1
    fi
}

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --benchmark    Run performance benchmarks"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run all struct tests"
    echo "  $0 --benchmark        # Run tests with benchmarks"
    echo ""
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --benchmark)
        main --benchmark
        ;;
    "")
        main
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        show_help
        exit 1
        ;;
esac 