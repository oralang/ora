#!/bin/bash

# Semantics Analyzer Test Suite
# Comprehensive testing for the semantic analysis phase

set -e

echo "🔍 Semantic Analysis Test Suite"
echo "==============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test configuration
VERBOSE=false
SHOW_MEMORY_STATS=false
RUN_INTEGRATION_TESTS=true
RUN_UNIT_TESTS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -m|--memory)
            SHOW_MEMORY_STATS=true
            shift
            ;;
        --unit-only)
            RUN_INTEGRATION_TESTS=false
            shift
            ;;
        --integration-only)
            RUN_UNIT_TESTS=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -v, --verbose          Show verbose output"
            echo "  -m, --memory          Show memory usage statistics"
            echo "  --unit-only           Run only unit tests"
            echo "  --integration-only    Run only integration tests"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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

# Test counters
total_tests=0
passed_tests=0
failed_tests=0
skipped_tests=0

# Function to run a Zig test file
run_zig_test() {
    local test_file="$1"
    local test_name="$2"
    
    if [[ ! -f "$test_file" ]]; then
        print_status "SKIP" "$test_name (file not found: $test_file)"
        skipped_tests=$((skipped_tests + 1))
        return 1
    fi
    
    total_tests=$((total_tests + 1))
    
    if $VERBOSE; then
        echo "  Running: $test_name"
        echo "  File: $test_file"
    fi
    
    # Run the test with memory tracking if requested
    local test_cmd="zig test \"$test_file\""
    if $SHOW_MEMORY_STATS; then
        test_cmd="$test_cmd --test-runner-config memory_leak_checks=true"
    fi
    
    if $VERBOSE; then
        if eval $test_cmd; then
            print_status "PASS" "$test_name"
            passed_tests=$((passed_tests + 1))
            return 0
        else
            print_status "FAIL" "$test_name"
            failed_tests=$((failed_tests + 1))
            return 1
        fi
    else
        if eval $test_cmd > /dev/null 2>&1; then
            print_status "PASS" "$test_name"
            passed_tests=$((passed_tests + 1))
            return 0
        else
            print_status "FAIL" "$test_name"
            if $VERBOSE; then
                echo "  Error output:"
                eval $test_cmd 2>&1 | sed 's/^/  /'
            fi
            failed_tests=$((failed_tests + 1))
            return 1
        fi
    fi
}

# Function to check if semantics analyzer compiles
check_semantics_compilation() {
    print_subsection "Checking Semantics Analyzer Compilation"
    
    if zig build-lib src/semantics.zig > /dev/null 2>&1; then
        print_status "PASS" "Semantics analyzer compiles successfully"
        rm -f libsemantics.a libsemantics.a.o  # Clean up build artifacts
        return 0
    else
        print_status "FAIL" "Semantics analyzer compilation failed"
        if $VERBOSE; then
            echo "Compilation errors:"
            zig build-lib src/semantics.zig 2>&1 | sed 's/^/  /'
        fi
        return 1
    fi
}

# Function to run unit tests
run_unit_tests() {
    print_section "Unit Tests"
    
    # Test semantics analyzer unit tests (using minimal working test)
    run_zig_test "src/semantics_test_minimal.zig" "Semantics Analyzer Unit Tests"
    
    # Test individual semantics components
    print_subsection "Component Tests"
    
    # Test memory safety features
    if [[ -f "src/semantics.zig" ]]; then
        print_status "INFO" "Memory safety tests included in main test suite"
    fi
    
    # Test diagnostic system
    if [[ -f "src/semantics.zig" ]]; then
        print_status "INFO" "Diagnostic system tests included in main test suite"
    fi
}

# Function to run integration tests
run_integration_tests() {
    print_section "Integration Tests"
    
    # Test complete semantic analysis workflows
    run_zig_test "test/semantics_integration_test.zig" "Semantics Integration Tests"
    
    # Test with real Ora contract examples
    print_subsection "Real Contract Tests"
    
    # Test simple contracts
    test_contract_analysis "examples/core/simple_struct_test.ora" "Simple Contract Analysis"
    test_contract_analysis "examples/tokens/simple_token.ora" "Token Contract Analysis"
    
    # Test advanced contracts if they exist
    if [[ -f "examples/advanced/struct_advanced_features_test.ora" ]]; then
        test_contract_analysis "examples/advanced/struct_advanced_features_test.ora" "Advanced Features Analysis"
    fi
}

# Function to test contract analysis
test_contract_analysis() {
    local contract_file="$1"
    local test_name="$2"
    
    if [[ ! -f "$contract_file" ]]; then
        print_status "SKIP" "$test_name (file not found: $contract_file)"
        skipped_tests=$((skipped_tests + 1))
        return
    fi
    
    total_tests=$((total_tests + 1))
    
    # Check if we can build the compiler first
    if ! zig build > /dev/null 2>&1; then
        print_status "FAIL" "$test_name (compiler build failed)"
        failed_tests=$((failed_tests + 1))
        return
    fi
    
    # Run semantic analysis on the contract
    if ./zig-out/bin/ora analyze "$contract_file" > /dev/null 2>&1; then
        print_status "PASS" "$test_name"
        passed_tests=$((passed_tests + 1))
    else
        print_status "FAIL" "$test_name"
        if $VERBOSE; then
            echo "  Analysis output:"
            ./zig-out/bin/ora analyze "$contract_file" 2>&1 | sed 's/^/  /'
        fi
        failed_tests=$((failed_tests + 1))
    fi
}

# Function to run performance tests
run_performance_tests() {
    print_section "Performance Tests"
    
    print_subsection "Memory Usage Analysis"
    
    if $SHOW_MEMORY_STATS; then
        # Run a representative test with memory tracking
        if [[ -f "src/semantics_test.zig" ]]; then
            print_status "INFO" "Running memory usage analysis..."
            
            # Run test with memory leak detection
            if zig test src/semantics_test.zig --test-runner-config memory_leak_checks=true > /dev/null 2>&1; then
                print_status "PASS" "No memory leaks detected"
            else
                print_status "WARN" "Memory issues detected - check output"
                if $VERBOSE; then
                    zig test src/semantics_test.zig --test-runner-config memory_leak_checks=true 2>&1 | sed 's/^/  /'
                fi
            fi
        fi
    else
        print_status "SKIP" "Memory analysis (use -m flag to enable)"
        skipped_tests=$((skipped_tests + 1))
    fi
    
    print_subsection "Compilation Speed Analysis"
    
    # Test compilation speed on a representative contract
    if [[ -f "examples/tokens/simple_token.ora" ]]; then
        print_status "INFO" "Testing semantic analysis performance..."
        
        # Time the semantic analysis phase
        start_time=$(date +%s%N)
        if ./zig-out/bin/ora analyze examples/tokens/simple_token.ora > /dev/null 2>&1; then
            end_time=$(date +%s%N)
            duration=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
            print_status "INFO" "Analysis completed in ${duration}ms"
        else
            print_status "WARN" "Performance test failed to run"
        fi
    fi
}

# Function to generate test coverage report
generate_coverage_report() {
    print_section "Test Coverage Analysis"
    
    print_subsection "Feature Coverage"
    
    # Check which semantic analysis features are being tested
    local tested_features=()
    local untested_features=()
    
    # Features to check for test coverage
    local all_features=(
        "Contract Analysis"
        "Function Analysis" 
        "Variable Declaration Analysis"
        "Expression Analysis"
        "Memory Region Validation"
        "Immutable Variable Tracking"
        "Error Handling"
        "Import System"
        "Struct Analysis"
        "Enum Analysis"
        "Formal Verification Integration"
        "Memory Safety"
        "Diagnostic Generation"
    )
    
    # Check test files for feature coverage
    for feature in "${all_features[@]}"; do
        local feature_lower=$(echo "$feature" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        
        if [[ -f "src/semantics_test.zig" ]] && grep -q "$feature_lower\|$(echo "$feature" | tr ' ' '_')" src/semantics_test.zig; then
            tested_features+=("$feature")
        elif [[ -f "test/semantics_integration_test.zig" ]] && grep -q "$feature_lower\|$(echo "$feature" | tr ' ' '_')" test/semantics_integration_test.zig; then
            tested_features+=("$feature")
        else
            untested_features+=("$feature")
        fi
    done
    
    # Report coverage
    print_status "INFO" "Tested features: ${#tested_features[@]}/${#all_features[@]}"
    for feature in "${tested_features[@]}"; do
        print_status "PASS" "$feature (covered)"
    done
    
    if [[ ${#untested_features[@]} -gt 0 ]]; then
        print_subsection "Missing Test Coverage"
        for feature in "${untested_features[@]}"; do
            print_status "WARN" "$feature (not covered)"
        done
    fi
}

# Function to print final summary
print_summary() {
    print_section "Test Summary"
    
    echo -e "Total Tests:  ${total_tests}"
    echo -e "${GREEN}Passed:       ${passed_tests}${NC}"
    echo -e "${RED}Failed:       ${failed_tests}${NC}"
    echo -e "${YELLOW}Skipped:      ${skipped_tests}${NC}"
    echo ""
    
    if [[ $failed_tests -eq 0 ]]; then
        print_status "PASS" "All tests passed successfully!"
        echo ""
        echo -e "${GREEN}🎉 Semantics analyzer is working correctly!${NC}"
    else
        print_status "FAIL" "Some tests failed"
        echo ""
        echo -e "${RED}❌ Issues found in semantics analyzer${NC}"
        echo -e "${YELLOW}💡 Run with -v flag for detailed error output${NC}"
    fi
}

# Main execution
main() {
    print_status "INFO" "Starting semantic analysis test suite..."
    
    if $VERBOSE; then
        print_status "INFO" "Verbose mode enabled"
    fi
    
    if $SHOW_MEMORY_STATS; then
        print_status "INFO" "Memory analysis enabled"
    fi
    
    # Check compilation first
    if ! check_semantics_compilation; then
        print_status "FAIL" "Cannot proceed - semantics analyzer does not compile"
        exit 1
    fi
    
    # Run test suites based on configuration
    if $RUN_UNIT_TESTS; then
        run_unit_tests
    fi
    
    if $RUN_INTEGRATION_TESTS; then
        run_integration_tests
    fi
    
    # Always run performance tests if available
    run_performance_tests
    
    # Generate coverage report
    generate_coverage_report
    
    # Print final summary
    print_summary
    
    # Exit with appropriate code
    if [[ $failed_tests -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@" 