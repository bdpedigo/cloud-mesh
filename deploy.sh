#!/usr/bin/env bash
# deploy.sh — Full deployment pipeline in one command.
#
# Override the config file via CONFIG_PATH env var (default: config.toml).
#
# Usage:
#   bash deploy.sh                   # check prereqs, then deploy to GKE
#   bash deploy.sh --build           # build + push image first, then deploy
#   bash deploy.sh --local           # deploy to local kind cluster
#   bash deploy.sh --build --local   # build (no push) then deploy locally
#   bash deploy.sh --skip-check      # skip prerequisite checks
#   CONFIG_PATH=config-bdp.toml bash deploy.sh --build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-config.toml}"

BUILD=false
LOCAL=false
SKIP_CHECK=false

for arg in "$@"; do
    case "$arg" in
        --build)       BUILD=true ;;
        --local)       LOCAL=true ;;
        --skip-check)  SKIP_CHECK=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [[ "$SKIP_CHECK" == false ]]; then
    echo "=== Checking prerequisites ==="
    bash "$SCRIPT_DIR/check.sh" || { echo ""; echo "Fix the issues above, then re-run."; exit 1; }
    echo ""
fi

if [[ "$BUILD" == true ]]; then
    echo "=== Building Docker image ==="
    if [[ "$LOCAL" == true ]]; then
        bash "$SCRIPT_DIR/build.sh" --no-push
    else
        bash "$SCRIPT_DIR/build.sh"
    fi
    echo ""
fi

echo "=== Deploying ==="
if [[ "$LOCAL" == true ]]; then
    bash "$SCRIPT_DIR/make_cluster.sh" --local
else
    bash "$SCRIPT_DIR/make_cluster.sh"
fi
