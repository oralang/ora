#!/bin/bash
# Script to run the lexer examples

# Build the lexer usage example
echo "Building lexer_usage.zig..."
zig build-exe lexer_usage.zig -I ../../src/ -lc

# Run the lexer usage example
echo -e "\nRunning lexer usage example:"
./lexer_usage

# Process the Ora example files using the main compiler
echo -e "\nProcessing string_literals.ora:"
../../zig-out/bin/ora tokenize string_literals.ora

echo -e "\nProcessing number_literals.ora:"
../../zig-out/bin/ora tokenize number_literals.ora

echo -e "\nProcessing error_recovery.ora with recovery enabled:"
../../zig-out/bin/ora tokenize error_recovery.ora --recover

echo -e "\nProcessing performance_optimization.ora:"
../../zig-out/bin/ora tokenize performance_optimization.ora

# Clean up
echo -e "\nCleaning up..."
rm -f lexer_usage

echo -e "\nDone!"