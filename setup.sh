#!/usr/bin/env bash
# Ora bootstrap script.
# Installs/checks local dependencies, syncs submodules, and builds Ora.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

PROFILE="full"          # minimal | full | dev
RUN_TESTS=0
SKIP_DEPS=0
SKIP_SUBMODULES=0
SKIP_BUILD=0
NO_LLVM_BOOTSTRAP=0

MIN_ZIG_VERSION="0.15.0"
LLVM_REPO_URL="https://github.com/llvm/llvm-project.git"
LLVM_COMMIT="ee8c14be14deabace692ab51f5d5d432b0a83d58"

# colors (enabled only on tty)
if [[ -t 1 ]]; then
  BOLD='\033[1m'
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  NC='\033[0m'
else
  BOLD=''
  RED=''
  GREEN=''
  YELLOW=''
  BLUE=''
  NC=''
fi

log_info() { printf "%b[info]%b %s\n" "$BLUE" "$NC" "$*"; }
log_ok()   { printf "%b[ok]%b   %s\n" "$GREEN" "$NC" "$*"; }
log_warn() { printf "%b[warn]%b %s\n" "$YELLOW" "$NC" "$*"; }
log_err()  { printf "%b[err]%b  %s\n" "$RED" "$NC" "$*" >&2; }

die() {
  log_err "$*"
  exit 1
}

usage() {
  cat <<USAGE
Ora Setup

Usage:
  ./setup.sh [options]

Options:
  --profile <minimal|full|dev>   Setup profile (default: full)
  --run-tests                    Run tests after build
  --skip-deps                    Do not install system packages
  --skip-submodules              Do not sync/update git submodules
  --skip-build                   Do not build Ora
  --no-llvm-bootstrap            Do not auto-clone vendor/llvm-project when missing
  -h, --help                     Show this help

Profiles:
  minimal  Build fast with -Dskip-mlir=true when MLIR artifacts already exist.
  full     Build complete toolchain (default).
  dev      Full profile plus developer-oriented tool checks.
USAGE
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile)
        [[ $# -ge 2 ]] || die "--profile requires a value"
        PROFILE="$2"
        shift 2
        ;;
      --run-tests)
        RUN_TESTS=1
        shift
        ;;
      --skip-deps)
        SKIP_DEPS=1
        shift
        ;;
      --skip-submodules)
        SKIP_SUBMODULES=1
        shift
        ;;
      --skip-build)
        SKIP_BUILD=1
        shift
        ;;
      --no-llvm-bootstrap)
        NO_LLVM_BOOTSTRAP=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown argument: $1"
        ;;
    esac
  done

  case "$PROFILE" in
    minimal|full|dev) ;;
    *) die "invalid profile '$PROFILE' (expected: minimal|full|dev)" ;;
  esac
}

version_ge() {
  # Returns 0 when $1 >= $2
  local IFS=.
  local -a lhs rhs
  local i
  lhs=($1)
  rhs=($2)

  for ((i=${#lhs[@]}; i<3; i++)); do lhs[i]=0; done
  for ((i=${#rhs[@]}; i<3; i++)); do rhs[i]=0; done

  for i in 0 1 2; do
    if ((10#${lhs[i]} > 10#${rhs[i]})); then
      return 0
    fi
    if ((10#${lhs[i]} < 10#${rhs[i]})); then
      return 1
    fi
  done
  return 0
}

platform_detect() {
  case "$(uname -s)" in
    Darwin) PLATFORM="macos" ;;
    Linux) PLATFORM="linux" ;;
    *) PLATFORM="unsupported" ;;
  esac

  if [[ "$PLATFORM" == "unsupported" ]]; then
    die "unsupported platform: $(uname -s). Use Linux or macOS."
  fi

  log_info "platform: $PLATFORM"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

check_zig() {
  require_cmd zig
  local zig_ver
  zig_ver="$(zig version)"
  log_info "zig version: $zig_ver"

  if ! version_ge "$zig_ver" "$MIN_ZIG_VERSION"; then
    die "Zig $MIN_ZIG_VERSION+ required (found $zig_ver)"
  fi

  log_ok "Zig version is compatible"
}

has_mlir_artifacts() {
  local lib_dir="$ROOT_DIR/vendor/mlir/lib"
  [[ -d "$lib_dir" ]] || return 1
  compgen -G "$lib_dir/libMLIR-C.*" >/dev/null || return 1
  compgen -G "$lib_dir/libMLIROraDialectC.*" >/dev/null || return 1
  compgen -G "$lib_dir/libMLIRSIRDialect.*" >/dev/null || return 1
  return 0
}

install_deps_macos() {
  require_cmd brew

  local packages=(git cmake z3)
  local dev_packages=(python ninja)

  if [[ "$PROFILE" == "dev" ]]; then
    packages+=("${dev_packages[@]}")
  fi

  log_info "installing packages via Homebrew: ${packages[*]}"
  brew install "${packages[@]}"
  log_ok "macOS dependencies are installed"
}

install_deps_linux() {
  require_cmd apt-get

  local -a packages=(
    build-essential
    git
    cmake
    clang
    libc++-dev
    libc++abi-dev
    pkg-config
    z3
    libz3-dev
  )

  if [[ "$PROFILE" == "dev" ]]; then
    packages+=(python3 ninja-build cargo)
  fi

  local SUDO=""
  if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      SUDO="sudo"
    else
      die "apt-get requires root privileges (install sudo or run as root)"
    fi
  fi

  log_info "updating apt package index"
  $SUDO apt-get update -qq

  log_info "installing packages via apt: ${packages[*]}"
  $SUDO apt-get install -y "${packages[@]}"
  log_ok "Linux dependencies are installed"
}

install_system_dependencies() {
  if [[ "$SKIP_DEPS" -eq 1 ]]; then
    log_warn "skipping dependency installation (--skip-deps)"
    return
  fi

  case "$PLATFORM" in
    macos) install_deps_macos ;;
    linux) install_deps_linux ;;
    *) die "unsupported platform: $PLATFORM" ;;
  esac
}

sync_submodules() {
  if [[ "$SKIP_SUBMODULES" -eq 1 ]]; then
    log_warn "skipping submodule sync (--skip-submodules)"
    return
  fi

  if [[ ! -d "$ROOT_DIR/.git" ]]; then
    log_warn "not a git checkout; skipping submodule sync"
    return
  fi

  require_cmd git
  log_info "syncing submodules"
  git -C "$ROOT_DIR" submodule sync --recursive
  git -C "$ROOT_DIR" submodule update --init --recursive --depth=1
  log_ok "submodules synced"
}

ensure_llvm_source() {
  if [[ "$NO_LLVM_BOOTSTRAP" -eq 1 ]]; then
    if [[ -d "$ROOT_DIR/vendor/llvm-project/llvm" ]]; then
      log_ok "llvm-project source is present"
      return
    fi
    die "vendor/llvm-project is missing and --no-llvm-bootstrap was specified"
  fi

  require_cmd git
  log_info "ensuring pinned llvm-project commit: $LLVM_COMMIT"

  if [[ ! -d "$ROOT_DIR/vendor/llvm-project/.git" ]]; then
    rm -rf "$ROOT_DIR/vendor/llvm-project"
    git init "$ROOT_DIR/vendor/llvm-project"
    git -C "$ROOT_DIR/vendor/llvm-project" remote add origin "$LLVM_REPO_URL"
  fi

  if ! git -C "$ROOT_DIR/vendor/llvm-project" remote get-url origin >/dev/null 2>&1; then
    git -C "$ROOT_DIR/vendor/llvm-project" remote add origin "$LLVM_REPO_URL"
  fi

  git -C "$ROOT_DIR/vendor/llvm-project" fetch --depth=1 origin "$LLVM_COMMIT"
  git -C "$ROOT_DIR/vendor/llvm-project" checkout --detach FETCH_HEAD
  log_ok "llvm-project pinned at $(git -C "$ROOT_DIR/vendor/llvm-project" rev-parse HEAD)"
}

build_ora() {
  if [[ "$SKIP_BUILD" -eq 1 ]]; then
    log_warn "skipping build (--skip-build)"
    return
  fi

  local -a cmd
  if [[ "$PROFILE" == "minimal" ]] && has_mlir_artifacts; then
    cmd=(zig build -Dskip-mlir=true)
    log_info "using fast build with existing MLIR artifacts"
  else
    cmd=(zig build)
    if [[ "$PROFILE" == "minimal" ]]; then
      log_warn "MLIR artifacts not found; running full build instead"
    fi
  fi

  log_info "running: ${cmd[*]}"
  (cd "$ROOT_DIR" && "${cmd[@]}")
  log_ok "build completed"
}

run_tests() {
  if [[ "$RUN_TESTS" -eq 0 ]]; then
    return
  fi

  local -a cmd
  if [[ "$PROFILE" == "minimal" ]] && has_mlir_artifacts; then
    cmd=(zig build test -Dskip-mlir=true)
  else
    cmd=(zig build test)
  fi

  log_info "running tests: ${cmd[*]}"
  (cd "$ROOT_DIR" && "${cmd[@]}")
  log_ok "tests completed"
}

print_optional_tool_status() {
  if [[ "$PROFILE" != "dev" ]]; then
    return
  fi

  local tools=(python3 cargo cast anvil)
  local tool
  log_info "developer tool status"
  for tool in "${tools[@]}"; do
    if command -v "$tool" >/dev/null 2>&1; then
      log_ok "$tool is available"
    else
      log_warn "$tool not found"
    fi
  done
}

main() {
  parse_args "$@"

  printf "%bOra Setup%b\n" "$BOLD" "$NC"
  log_info "profile: $PROFILE"

  platform_detect
  check_zig
  install_system_dependencies
  sync_submodules

  if [[ "$PROFILE" != "minimal" ]]; then
    ensure_llvm_source
  fi

  build_ora
  run_tests
  print_optional_tool_status

  printf "\n"
  log_ok "setup finished"
  printf "%s\n" "next: ./zig-out/bin/ora --help"
}

main "$@"
