#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<USAGE
Usage: $0 <image-repo> <version-tag> [extra buildx args...]

Examples:
  $0 oralang/ora v0.1.0
  $0 oralang/ora nightly-20260303 --no-cache
USAGE
  exit 1
fi

IMAGE_REPO="$1"
VERSION_TAG="$2"
shift 2

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t "${IMAGE_REPO}:latest" \
  -t "${IMAGE_REPO}:${VERSION_TAG}" \
  --push \
  "$@" \
  .
