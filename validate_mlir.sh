#!/bin/bash

# MLIR Validation Script for Ora Compiler
# This script validates generated MLIR using mlir-opt
# Updated: October 21, 2025 - Uses modern CLI flags
#
# Usage:
#   ./validate_mlir.sh                  # Validate all files in ora-example/
#   ./validate_mlir.sh file.ora         # Validate specific file
#   ./validate_mlir.sh -v file.ora      # Verbose mode
#   ./validate_mlir.sh --save-all       # Save all MLIR outputs
#   ./validate_mlir.sh --help           # Show help

set -e

# Options
VERBOSE=0
SAVE_ALL=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[VALIDATE]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    if [ $VERBOSE -eq 1 ]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

show_help() {
    echo "MLIR Validation Script for Ora Compiler"
    echo ""
    echo "Usage:"
    echo "  ./validate_mlir.sh [options] [files...]"
    echo ""
    echo "Options:"
    echo "  -v, --verbose      Enable verbose output"
    echo "  --save-all         Save all generated MLIR files (success and failure)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./validate_mlir.sh                     # Validate all files in ora-example/"
    echo "  ./validate_mlir.sh contract.ora        # Validate specific file"
    echo "  ./validate_mlir.sh -v contract.ora     # Verbose validation"
    echo "  ./validate_mlir.sh --save-all          # Save all MLIR outputs"
    echo ""
    exit 0
}

# Check if mlir-opt is available
MLIR_OPT=""
if [ -f "vendor/mlir/bin/mlir-opt" ]; then
    MLIR_OPT="vendor/mlir/bin/mlir-opt"
elif [ -f "vendor/llvm-project/build-mlir/bin/mlir-opt" ]; then
    MLIR_OPT="vendor/llvm-project/build-mlir/bin/mlir-opt"
else
    print_error "mlir-opt not found. Please ensure MLIR is built."
    exit 1
fi

print_info "Using mlir-opt: $MLIR_OPT"

# Function to validate a single file
validate_file() {
    local input_file="$1"
    local basename=$(basename "$input_file" .ora)
    local temp_dir=$(mktemp -d)
    local mlir_file="$temp_dir/${basename}.mlir"
    
    print_status "Validating $input_file..."
    
    # Generate MLIR using modern --emit-mlir flag
    # Output goes to stdout, so we capture it directly
    if ! zig-out/bin/ora --emit-mlir "$input_file" > "$mlir_file" 2>&1; then
        print_error "Failed to generate MLIR for $input_file"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Check if file was created and has content
    if [ ! -s "$mlir_file" ]; then
        print_error "Generated MLIR file is empty for $input_file"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Validate with mlir-opt
    if $MLIR_OPT "$mlir_file" --allow-unregistered-dialect >/dev/null 2>&1; then
        print_status "✅ $input_file - MLIR is valid"
        
        # Save successful MLIR if requested
        if [ $SAVE_ALL -eq 1 ]; then
            local success_dir="validated_mlir"
            mkdir -p "$success_dir"
            cp "$mlir_file" "$success_dir/${basename}.mlir"
            print_info "MLIR saved to: $success_dir/${basename}.mlir"
        fi
        
        rm -rf "$temp_dir"
        return 0
    else
        print_error "❌ $input_file - MLIR validation failed:"
        
        if [ $VERBOSE -eq 1 ]; then
            $MLIR_OPT "$mlir_file" --allow-unregistered-dialect 2>&1 | sed 's/^/    /'
        else
            $MLIR_OPT "$mlir_file" --allow-unregistered-dialect 2>&1 | head -10 | sed 's/^/    /'
            echo "    ... (use -v for full output)"
        fi
        
        # Save failed MLIR for debugging
        local failed_dir="failed_mlir"
        mkdir -p "$failed_dir"
        cp "$mlir_file" "$failed_dir/${basename}_failed.mlir"
        print_warning "Failed MLIR saved to: $failed_dir/${basename}_failed.mlir"
        
        rm -rf "$temp_dir"
        return 1
    fi
}

# Parse command line arguments
FILES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        --save-all)
            SAVE_ALL=1
            shift
            ;;
        -*)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# Main validation logic
if [ ${#FILES[@]} -eq 0 ]; then
    print_status "Validating all .ora files in ora-example/"
    failed_files=()
    total_files=0
    passed_files=0
    
    for file in ora-example/*.ora; do
        if [ -f "$file" ]; then
            total_files=$((total_files + 1))
            if validate_file "$file"; then
                passed_files=$((passed_files + 1))
            else
                failed_files+=("$file")
            fi
        fi
    done
    
    echo ""
    print_status "========================================="
    print_status "Validation Summary:"
    print_status "  Total files:  $total_files"
    print_status "  Passed:       $passed_files"
    print_status "  Failed:       ${#failed_files[@]}"
    print_status "========================================="
    
    if [ ${#failed_files[@]} -eq 0 ]; then
        print_status "🎉 All files passed validation!"
        exit 0
    else
        print_error "❌ ${#failed_files[@]} file(s) failed validation:"
        for file in "${failed_files[@]}"; do
            echo "    - $file"
        done
        exit 1
    fi
else
    # Validate specific files
    failed_files=()
    total_files=0
    passed_files=0
    
    for file in "${FILES[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "File not found: $file"
            failed_files+=("$file")
            total_files=$((total_files + 1))
            continue
        fi
        
        total_files=$((total_files + 1))
        if validate_file "$file"; then
            passed_files=$((passed_files + 1))
        else
            failed_files+=("$file")
        fi
    done
    
    echo ""
    print_status "========================================="
    print_status "Validation Summary:"
    print_status "  Total files:  $total_files"
    print_status "  Passed:       $passed_files"
    print_status "  Failed:       ${#failed_files[@]}"
    print_status "========================================="
    
    if [ ${#failed_files[@]} -eq 0 ]; then
        print_status "🎉 All specified files passed validation!"
        exit 0
    else
        print_error "❌ ${#failed_files[@]} file(s) failed validation"
        exit 1
    fi
fi
