#!/bin/bash

# Ora Documentation Generation Script
# This script generates Zig documentation and integrates it into the Docusaurus website

set -e

echo "🔨 Generating Zig documentation..."

# Generate Zig docs
zig build docs

echo "📁 Copying documentation to website..."

# Create target directory if it doesn't exist
mkdir -p website/static/api-docs

# Copy generated docs to website static folder
cp -r zig-out/docs/* website/static/api-docs/

echo "✅ API documentation generated and copied to website/static/api-docs/"
echo "📖 Documentation will be available at /api-docs/ when the website is built"

# Optional: Start development server if requested
if [ "$1" = "--serve" ]; then
    echo "🚀 Starting development server..."
    cd website
    pnpm start
fi 