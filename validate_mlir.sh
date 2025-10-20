#!/bin/bash

# MLIR Validation Script for Ora Compiler
# This script validates generated MLIR using mlir-opt

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

print_status "Using mlir-opt: $MLIR_OPT"

# Function to validate a single file
validate_file() {
    local input_file="$1"
    local temp_file="/tmp/$(basename "$input_file").mlir"
    
    print_status "Validating $input_file..."
    
    # Generate MLIR and extract clean output
    if ! zig-out/bin/ora mlir "$input_file" > "$temp_file" 2>/dev/null; then
        print_error "Failed to generate MLIR for $input_file"
        return 1
    fi
    
    # Extract just the MLIR part (skip debug output)
    local clean_file="/tmp/$(basename "$input_file")_clean.mlir"
    if grep -q "=== MLIR Output ===" "$temp_file"; then
        sed -n '/=== MLIR Output ===/,$p' "$temp_file" | tail -n +2 > "$clean_file"
    else
        # Filter out debug messages and keep only MLIR
        grep -v "^info:" "$temp_file" | grep -v "^ERROR:" | grep -v "^WARNING:" | grep -v "^Error " > "$clean_file"
    fi
    
    # Validate with mlir-opt
    if $MLIR_OPT "$clean_file" --allow-unregistered-dialect >/dev/null 2>&1; then
        print_status "✅ $input_file - MLIR is valid"
        return 0
    else
        print_error "❌ $input_file - MLIR validation failed:"
        $MLIR_OPT "$clean_file" --allow-unregistered-dialect 2>&1 | sed 's/^/    /'
        return 1
    fi
}

# Main validation logic
if [ $# -eq 0 ]; then
    print_status "Validating all .ora files in ora-example/"
    failed_files=()
    
    for file in ora-example/*.ora; do
        if [ -f "$file" ]; then
            if ! validate_file "$file"; then
                failed_files+=("$file")
            fi
        fi
    done
    
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
    for file in "$@"; do
        if [ ! -f "$file" ]; then
            print_error "File not found: $file"
            failed_files+=("$file")
            continue
        fi
        
        if ! validate_file "$file"; then
            failed_files+=("$file")
        fi
    done
    
    if [ ${#failed_files[@]} -eq 0 ]; then
        print_status "🎉 All specified files passed validation!"
        exit 0
    else
        print_error "❌ ${#failed_files[@]} file(s) failed validation"
        exit 1
    fi
fi
