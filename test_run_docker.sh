#!/usr/bin/env bash
# run_docker.sh — Run the worker image locally for testing.
#
# Reads docker_image from config.toml. Mounts your CloudVolume secrets and
# exits after one task (CLOUD_MESH_MAX_RUNS=1).
# Override the config file via CONFIG_PATH env var (default: config.toml).
#
# Usage:
#   bash test_run_docker.sh
#   CONFIG_PATH=config-bdp.toml bash test_run_docker.sh
set -euo pipefail

export CONFIG_PATH="${CONFIG_PATH:-config.toml}"

DOCKER_IMAGE="$(python3 - <<'EOF'
import tomllib, os
with open(os.environ["CONFIG_PATH"], "rb") as f:
    cfg = tomllib.load(f)
print(cfg["job"]["docker_image"])
EOF
)"

echo "Running image: $DOCKER_IMAGE"
docker run --rm --platform linux/amd64 \
    -v ~/.cloudvolume/secrets:/root/.cloudvolume/secrets \
    -e CLOUD_MESH_MAX_RUNS=1 \
    "$DOCKER_IMAGE"
