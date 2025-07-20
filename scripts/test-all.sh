#!/bin/bash

# Unified Test Runner for Ora Language Compiler
# Combines all test scripts into a single, comprehensive test suite

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test counters
total_tests=0
passed_tests=0
failed_tests=0
skipped_tests=0

# Configuration
VERBOSE=false
QUICK_MODE=false
CATEGORIES=""
PHASES=("lex" "parse" "analyze")

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "PASS") echo -e "${GREEN}✅ $message${NC}" ;;
        "FAIL") echo -e "${RED}❌ $message${NC}" ;;
        "SKIP") echo -e "${YELLOW}⏭️  $message${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  $message${NC}" ;;
        "WARN") echo -e "${YELLOW}⚠️  $message${NC}" ;;
    esac
}

# Function to print section headers
print_section() {
    echo ""
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}$(echo "$1" | sed 's/./=/g')${NC}"
}

# Function to print subsection headers
print_subsection() {
    echo ""
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}$(echo "$1" | sed 's/./-/g')${NC}"
}

# Function to show usage
show_help() {
    echo "Usage: $0 [OPTIONS] [CATEGORIES]"
    echo ""
    echo "Options:"
    echo "  -h, --help       Show this help message"
    echo "  -v, --verbose    Show verbose output"
    echo "  -q, --quick      Quick mode (analyze phase only)"
    echo "  --phases PHASES  Specify phases to test (comma-separated: lex,parse,analyze)"
    echo ""
    echo "Categories:"
    echo "  all              Run all tests (default)"
    echo "  struct           Test struct examples only"
    echo "  enum             Test enum examples only"  
    echo "  core             Test core examples only"
    echo "  advanced         Test advanced examples only"
    echo "  examples         Test general examples only"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run all tests"
    echo "  $0 struct             # Test only struct examples"
    echo "  $0 -q core advanced   # Quick test of core and advanced examples"
    echo "  $0 --phases parse,analyze  # Test only parse and analyze phases"
}

# Function to build the compiler
build_compiler() {
    print_section "Building Ora Compiler"
    
    if $VERBOSE; then
        zig build
    else
        if ! zig build > /dev/null 2>&1; then
            print_status "FAIL" "Compiler build failed"
            exit 1
        fi
    fi
    
    print_status "PASS" "Compiler built successfully"
}

# Function to test a single file with specified phases
test_file() {
    local file="$1"
    local phases=("${@:2}")
    local filename=$(basename "$file")
    
    if [[ ${#phases[@]} -eq 0 ]]; then
        phases=("${PHASES[@]}")
    fi
    
    total_tests=$((total_tests + 1))
    
    if $VERBOSE; then
        echo "  Testing: $filename"
    fi
    
    for phase in "${phases[@]}"; do
        if ! ./zig-out/bin/ora "$phase" "$file" > /dev/null 2>&1; then
            print_status "FAIL" "$filename (phase: $phase)"
            if $VERBOSE; then
                echo "    Error output:"
                ./zig-out/bin/ora "$phase" "$file" 2>&1 | sed 's/^/    /'
            fi
            failed_tests=$((failed_tests + 1))
            return 1
        fi
    done
    
    print_status "PASS" "$filename"
    passed_tests=$((passed_tests + 1))
    return 0
}

# Function to test struct examples
test_struct_examples() {
    print_subsection "Testing Core Struct Examples"
    
    local found_files=false
    for file in examples/core/struct_*.ora; do
        if [ -f "$file" ]; then
            found_files=true
            test_file "$file" "${PHASES[@]}"
        fi
    done
    
    if ! $found_files; then
        print_status "SKIP" "No core struct examples found"
        skipped_tests=$((skipped_tests + 1))
    fi
    
    print_subsection "Testing Advanced Struct Examples"
    
    found_files=false
    for file in examples/advanced/struct_*.ora; do
        if [ -f "$file" ]; then
            found_files=true
            test_file "$file" "${PHASES[@]}"
        fi
    done
    
    if ! $found_files; then
        print_status "SKIP" "No advanced struct examples found"
        skipped_tests=$((skipped_tests + 1))
    fi
}

# Function to test enum examples
test_enum_examples() {
    print_subsection "Testing Core Enum Examples"
    
    local found_files=false
    for file in examples/core/enum_*.ora; do
        if [ -f "$file" ]; then
            found_files=true
            test_file "$file" "${PHASES[@]}"
        fi
    done
    
    if ! $found_files; then
        print_status "SKIP" "No core enum examples found"
        skipped_tests=$((skipped_tests + 1))
    fi
    
    print_subsection "Testing Advanced Enum Examples"
    
    found_files=false
    for file in examples/advanced/enum_*.ora; do
        if [ -f "$file" ]; then
            found_files=true
            test_file "$file" "${PHASES[@]}"
        fi
    done
    
    if ! $found_files; then
        print_status "SKIP" "No advanced enum examples found"
        skipped_tests=$((skipped_tests + 1))
    fi
}

# Function to test core examples
test_core_examples() {
    print_subsection "Testing Core Examples"
    
    local found_files=false
    for file in examples/core/*.ora; do
        if [ -f "$file" ]; then
            found_files=true
            test_file "$file" "${PHASES[@]}"
        fi
    done
    
    if ! $found_files; then
        print_status "SKIP" "No core examples found"
        skipped_tests=$((skipped_tests + 1))
    fi
}

# Function to test advanced examples
test_advanced_examples() {
    print_subsection "Testing Advanced Examples"
    
    local found_files=false
    for file in examples/advanced/*.ora; do
        if [ -f "$file" ]; then
            found_files=true
            test_file "$file" "${PHASES[@]}"
        fi
    done
    
    if ! $found_files; then
        print_status "SKIP" "No advanced examples found"
        skipped_tests=$((skipped_tests + 1))
    fi
}

# Function to test all examples (recursive)
test_all_examples() {
    print_subsection "Testing All Examples"
    
    local found_files=false
    while IFS= read -r -d '' file; do
        found_files=true
        test_file "$file" "${PHASES[@]}"
    done < <(find examples -name "*.ora" -type f -print0 | sort -z)
    
    if ! $found_files; then
        print_status "SKIP" "No example files found"
        skipped_tests=$((skipped_tests + 1))
    fi
}

# Function to print final summary
print_summary() {
    print_section "Test Summary"
    
    echo "Total tests: $total_tests"
    
    if [ $passed_tests -gt 0 ]; then
        print_status "PASS" "$passed_tests tests passed"
    fi
    
    if [ $failed_tests -gt 0 ]; then
        print_status "FAIL" "$failed_tests tests failed"
    fi
    
    if [ $skipped_tests -gt 0 ]; then
        print_status "SKIP" "$skipped_tests tests skipped"
    fi
    
    echo ""
    
    if [ $failed_tests -gt 0 ]; then
        print_status "FAIL" "Some tests failed"
        exit 1
    else
        print_status "PASS" "All tests passed!"
        exit 0
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quick)
            QUICK_MODE=true
            PHASES=("analyze")
            shift
            ;;
        --phases)
            if [ -n "$2" ]; then
                IFS=',' read -ra PHASES <<< "$2"
                shift 2
            else
                echo "Error: --phases requires a comma-separated list of phases"
                exit 1
            fi
            ;;
        -*)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
        *)
            CATEGORIES="$CATEGORIES $1"
            shift
            ;;
    esac
done

# Set default category if none specified
if [ -z "$CATEGORIES" ]; then
    CATEGORIES="all"
fi

# Main execution
main() {
    print_section "Ora Language Compiler - Unified Test Suite"
    
    if $QUICK_MODE; then
        print_status "INFO" "Running in quick mode (analyze phase only)"
    fi
    
    if $VERBOSE; then
        print_status "INFO" "Verbose mode enabled"
    fi
    
    print_status "INFO" "Testing phases: ${PHASES[*]}"
    print_status "INFO" "Testing categories: $CATEGORIES"
    
    # Build the compiler first
    build_compiler
    
    # Run tests based on categories
    print_section "Running Tests"
    
    for category in $CATEGORIES; do
        case $category in
            all)
                test_all_examples
                ;;
            struct)
                test_struct_examples
                ;;
            enum)
                test_enum_examples
                ;;
            core)
                test_core_examples
                ;;
            advanced)
                test_advanced_examples
                ;;
            examples)
                test_all_examples
                ;;
            *)
                print_status "WARN" "Unknown category: $category"
                ;;
        esac
    done
    
    # Print final summary
    print_summary
}

# Run main function
main 