#!/bin/bash

# Simple test script for all .ora examples
# Similar to the user's original suggestion

# Build first
zig build

echo "Testing all .ora example files..."

# Test all examples  
for file in examples/*.ora; do
    if [[ -f "$file" ]]; then
        echo "Testing $file"
        if ! zig build run -- compile "$file" >/dev/null 2>&1; then
            echo "FAILED: $file"
        else
            echo "PASSED: $file"
        fi
    fi
done

echo "Done testing examples." 