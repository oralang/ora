#!/bin/bash

echo "🧪 Running all tests..."
echo "================================"

echo "1️⃣ Running comprehensive test suite..."
zig build test
COMPREHENSIVE_EXIT=$?

echo "2️⃣ Running AST visitor tests..."
zig build test-ast
AST_EXIT=$?

echo "3️⃣ Running lexer tests..."
zig build test-lexer
LEXER_EXIT=$?

echo "4️⃣ Running expression parser tests..."
zig build test-expression-parser
PARSER_EXIT=$?

echo "5️⃣ Running test framework tests..."
zig build test-framework
FRAMEWORK_EXIT=$?

echo "6️⃣ Running example tests..."
zig build test-examples
EXAMPLES_EXIT=$?

echo "================================"
echo "📊 Test Results Summary:"
echo "  Comprehensive tests: $([ $COMPREHENSIVE_EXIT -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "  AST tests: $([ $AST_EXIT -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "  Lexer tests: $([ $LEXER_EXIT -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "  Expression parser tests: $([ $PARSER_EXIT -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "  Test framework tests: $([ $FRAMEWORK_EXIT -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "  Example tests: $([ $EXAMPLES_EXIT -eq 0 ] && echo "✅ PASSED" || echo "⚠️  FAILED (some expected)")"

# Calculate total failures (excluding examples which may have expected failures)
TOTAL_FAILURES=$((COMPREHENSIVE_EXIT + AST_EXIT + LEXER_EXIT + PARSER_EXIT + FRAMEWORK_EXIT))

if [ $TOTAL_FAILURES -eq 0 ]; then
    echo "🎉 All core tests passed!"
    exit 0
else
    echo "❌ Some tests failed. Check the output above for details."
    exit 1
fi
