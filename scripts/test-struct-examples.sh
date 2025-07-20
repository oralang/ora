#!/bin/bash

set -e

echo "Testing Struct Examples"
echo "======================"

echo ""
echo "Testing Core Struct Examples:"
echo "------------------------------"

for file in examples/core/struct_*.ora; do
    if [ -f "$file" ]; then
        echo "Testing $file..."
        ./zig-out/bin/ora analyze "$file" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "  ✅ PASS"
        else
            echo "  ❌ FAIL"
            exit 1
        fi
    fi
done

echo ""
echo "Testing Advanced Struct Examples:"
echo "----------------------------------"

for file in examples/advanced/struct_*.ora; do
    if [ -f "$file" ]; then
        echo "Testing $file..."
        ./zig-out/bin/ora analyze "$file" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "  ✅ PASS"
        else
            echo "  ❌ FAIL"
            exit 1
        fi
    fi
done

echo ""
echo "🎉 All struct examples passed!" 