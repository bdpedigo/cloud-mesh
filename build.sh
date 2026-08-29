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

# Tag with git SHA so each build is uniquely identifiable.
# Also updates config.toml so make_cluster.sh deploys the exact same image.
GIT_SHA="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo "nogit")"
DOCKER_IMAGE_SHA="${DOCKER_IMAGE%:*}:${GIT_SHA}"

echo "Git SHA tag:    $DOCKER_IMAGE_SHA"
docker buildx build --platform linux/amd64 \
    -t "$DOCKER_IMAGE" \
    -t "$DOCKER_IMAGE_SHA" \
    "$SCRIPT_DIR"

if [[ "$NO_PUSH" == true ]]; then
    echo "Skipping push (--no-push)."
else
    echo "Pushing image: $DOCKER_IMAGE and $DOCKER_IMAGE_SHA"
    docker push "$DOCKER_IMAGE"
    docker push "$DOCKER_IMAGE_SHA"
fi

# Write the SHA-tagged image back to config.toml so make_cluster.sh uses it.
python3 - <<PYEOF
import re, os
path = os.environ["CONFIG_PATH"]
with open(path) as f:
    content = f.read()
new = re.sub(
    r'(docker_image\s*=\s*")[^"]+(")',
    lambda m: m.group(1) + "$DOCKER_IMAGE_SHA" + m.group(2),
    content
)
with open(path, "w") as f:
    f.write(new)
print(f"Updated {path}: docker_image = $DOCKER_IMAGE_SHA")
PYEOF
