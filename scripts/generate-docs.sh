#!/bin/bash

# Ora Documentation Generation Script
# This script generates Zig documentation and integrates it into the Docusaurus website

set -e

echo "🔨 Generating Zig documentation..."

# Generate Zig docs
zig build docs

echo "📁 Processing documentation for website..."

# Create target directory if it doesn't exist
mkdir -p website/static/api-docs

# Copy generated docs to website static folder
cp -r zig-out/docs/* website/static/api-docs/

# Extract sources.tar if it exists for better accessibility
if [ -f "website/static/api-docs/sources.tar" ]; then
    echo "📦 Extracting sources.tar for better web accessibility..."
    cd website/static/api-docs
    tar -xf sources.tar
    # Create a simple redirect index for the extracted content
    if [ -d "src" ]; then
        echo '<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Ora API Documentation</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; margin: 40px; }
        h1 { color: #2563eb; }
        .links { margin-top: 20px; }
        .links a { display: block; margin: 10px 0; color: #1d4ed8; text-decoration: none; }
        .links a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Ora API Documentation</h1>
    <div class="links">
        <a href="src/root.zig.html">Root Module</a>
        <a href="src/ast.zig.html">AST Module</a>
        <a href="src/typer.zig.html">Type System</a>
        <a href="src/ir.zig.html">IR (HIR) Module</a>
        <a href="src/codegen_yul.zig.html">Yul Code Generation</a>
        <a href="src/parser.zig.html">Parser Module</a>
        <a href="src/lexer.zig.html">Lexer Module</a>
    </div>
</body>
</html>' > docs-index.html
    fi
    cd ../../..
fi

echo "✅ API documentation generated and processed for website/static/api-docs/"
echo "📖 Documentation will be available at /api-docs/ when the website is built"

# Optional: Start development server if requested
if [ "$1" = "--serve" ]; then
    echo "🚀 Starting development server..."
    cd website
    pnpm start
fi 