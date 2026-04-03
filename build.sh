#!/usr/bin/env bash
# build.sh — Build (and optionally push) the Docker image.
#
# Reads docker_image from config.toml — no hardcoded image names needed.
# Override the config file via CONFIG_PATH env var (default: config.toml).
#
# Usage:
#   bash build.sh            # build and push to registry
#   bash build.sh --no-push  # build only (e.g. for local/kind testing)
#   CONFIG_PATH=config-bdp.toml bash build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-config.toml}"

NO_PUSH=false
for arg in "$@"; do
    case "$arg" in
        --no-push) NO_PUSH=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

DOCKER_IMAGE="$(python3 - <<'EOF'
import tomllib, os
with open(os.environ["CONFIG_PATH"], "rb") as f:
    cfg = tomllib.load(f)
print(cfg["job"]["docker_image"])
EOF
)"

echo "Config:        $CONFIG_PATH"
echo "Building image: $DOCKER_IMAGE"
docker buildx build --platform linux/amd64 -t "$DOCKER_IMAGE" "$SCRIPT_DIR"

if [[ "$NO_PUSH" == true ]]; then
    echo "Skipping push (--no-push)."
else
    echo "Pushing image: $DOCKER_IMAGE"
    docker push "$DOCKER_IMAGE"
fi
