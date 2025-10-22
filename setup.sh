#!/usr/bin/env bash
# Ora Compiler Setup Script
# Automates installation of all development dependencies

set -e  # Exit on error

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}🚀 Ora Compiler Setup${NC}"
echo "This script will install all dependencies required to build and develop the Ora compiler."
echo ""

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM=Linux;;
    Darwin*)    PLATFORM=Mac;;
    CYGWIN*|MINGW*|MSYS*) PLATFORM=Windows;;
    *)          PLATFORM="UNKNOWN:${OS}"
esac

echo -e "${BOLD}📍 Detected platform: ${PLATFORM}${NC}"
echo ""

# Check Zig version
echo -e "${BOLD}1️⃣  Checking Zig installation...${NC}"
if command -v zig &> /dev/null; then
    ZIG_VERSION=$(zig version)
    echo -e "${GREEN}✅ Zig ${ZIG_VERSION} found${NC}"
    
    # Check if version is at least 0.15.0
    if [[ "${ZIG_VERSION}" < "0.15.0" ]]; then
        echo -e "${YELLOW}⚠️  Zig 0.15.1+ is recommended. Current version: ${ZIG_VERSION}${NC}"
        echo "   Download from: https://ziglang.org/download/"
    fi
else
    echo -e "${RED}❌ Zig not found${NC}"
    echo "   Please install Zig 0.15.1+ from: https://ziglang.org/download/"
    echo "   Or use:"
    if [[ "$PLATFORM" == "Mac" ]]; then
        echo "   brew install zig"
    elif [[ "$PLATFORM" == "Linux" ]]; then
        echo "   snap install zig --classic --beta"
    fi
    exit 1
fi
echo ""

# Install system dependencies
echo -e "${BOLD}2️⃣  Installing system dependencies...${NC}"

if [[ "$PLATFORM" == "Mac" ]]; then
    echo "Installing dependencies via Homebrew..."
    
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}❌ Homebrew not found. Please install from: https://brew.sh${NC}"
        exit 1
    fi
    
    echo "Updating Homebrew..."
    brew update
    
    echo "Installing CMake, Boost, and OpenSSL..."
    brew install cmake boost openssl || echo "⚠️  Some packages may already be installed"
    
    echo -e "${GREEN}✅ macOS dependencies installed${NC}"

elif [[ "$PLATFORM" == "Linux" ]]; then
    echo "Installing dependencies via apt (Ubuntu/Debian)..."
    
    if ! command -v apt-get &> /dev/null; then
        echo -e "${YELLOW}⚠️  apt-get not found. Please install dependencies manually:${NC}"
        echo "   - build-essential"
        echo "   - cmake"
        echo "   - clang"
        echo "   - libc++-dev libc++abi-dev"
        echo "   - libboost-all-dev"
        echo "   - libssl-dev"
        echo "   - pkg-config"
    else
        echo "Updating package list..."
        sudo apt-get update -qq
        
        echo "Installing build tools and libraries..."
        sudo apt-get install -y \
            build-essential \
            cmake \
            clang \
            libc++-dev \
            libc++abi-dev \
            libboost-all-dev \
            libssl-dev \
            pkg-config \
            git
        
        echo -e "${GREEN}✅ Linux dependencies installed${NC}"
    fi

elif [[ "$PLATFORM" == "Windows" ]]; then
    echo -e "${YELLOW}⚠️  Windows setup requires manual installation:${NC}"
    echo "   1. Install CMake: https://cmake.org/download/"
    echo "   2. Install Visual Studio Build Tools 2022"
    echo "   3. Install vcpkg and run:"
    echo "      vcpkg install boost:x64-windows openssl:x64-windows"
    echo ""
    echo "   Or use Chocolatey:"
    echo "      choco install cmake openssl boost-msvc-14.3"
fi
echo ""

# Clone submodules
echo -e "${BOLD}3️⃣  Fetching Git submodules...${NC}"
if [ -d ".git" ]; then
    echo "Initializing submodules (vendor/solidity)..."
    git submodule update --init --depth=1 vendor/solidity
    echo -e "${GREEN}✅ Submodules initialized${NC}"
else
    echo -e "${YELLOW}⚠️  Not a git repository. Skipping submodule initialization.${NC}"
    echo "   If you cloned without submodules, run:"
    echo "   git submodule update --init --depth=1 vendor/solidity"
fi
echo ""

# Build MLIR libraries (if vendor/llvm-project exists)
echo -e "${BOLD}4️⃣  Checking MLIR libraries...${NC}"
if [ -d "vendor/mlir/lib" ] && [ -f "vendor/mlir/lib/libMLIRSupport.a" ]; then
    echo -e "${GREEN}✅ MLIR libraries already built${NC}"
else
    if [ -d "vendor/llvm-project" ]; then
        echo -e "${YELLOW}⚠️  MLIR libraries not found. These will be built automatically on first compile.${NC}"
        echo "   This may take 10-30 minutes on first run."
    else
        echo -e "${YELLOW}⚠️  vendor/llvm-project not found.${NC}"
        echo "   MLIR libraries will need to be downloaded or built separately."
    fi
fi
echo ""

# Build the compiler
echo -e "${BOLD}5️⃣  Building Ora compiler...${NC}"
echo "Running: zig build"
if zig build; then
    echo -e "${GREEN}✅ Compiler built successfully${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    echo "   Check the error messages above for details."
    exit 1
fi
echo ""

# Run tests
echo -e "${BOLD}6️⃣  Running tests...${NC}"
echo "Running: zig build test"
if zig build test; then
    echo -e "${GREEN}✅ All tests passed${NC}"
else
    echo -e "${YELLOW}⚠️  Some tests failed${NC}"
    echo "   This might be expected during development."
fi
echo ""

# Summary
echo -e "${BOLD}🎉 Setup Complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run the compiler:"
echo "     ${BOLD}./zig-out/bin/ora compile examples/hello.ora${NC}"
echo ""
echo "  2. Run tests:"
echo "     ${BOLD}zig build test${NC}"
echo ""
echo "  3. See available commands:"
echo "     ${BOLD}./zig-out/bin/ora --help${NC}"
echo ""
echo "For more information, see CONTRIBUTING.md"
echo ""

