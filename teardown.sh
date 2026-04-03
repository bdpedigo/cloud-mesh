#!/usr/bin/env bash
# teardown.sh — Tear down cloud-mesh deployments.
#
# Override the config file via CONFIG_PATH env var (default: config.toml).
#
# Usage:
#   bash teardown.sh           # delete K8s deployment, then prompt to delete GKE cluster
#   bash teardown.sh --local   # delete K8s deployment and local kind cluster
#   CONFIG_PATH=config-bdp.toml bash teardown.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-config.toml}"

LOCAL=false
for arg in "$@"; do
    case "$arg" in
        --local) LOCAL=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Read cluster config
eval "$(python3 - <<'EOF'
import tomllib, os
with open(os.environ["CONFIG_PATH"], "rb") as f:
    cfg = tomllib.load(f)
c = cfg["cluster"]
print(f'CLUSTER_NAME="{c["cluster_name"]}"')
print(f'ZONE="{c["zone"]}"')
print(f'PROJECT="{c["project"]}"')
EOF
)"

# Delete the Kubernetes deployment (non-destructive to the cluster itself)
if kubectl get deployment cloud-mesh &>/dev/null; then
    echo "Deleting deployment 'cloud-mesh'..."
    kubectl delete deployment cloud-mesh
else
    echo "No deployment 'cloud-mesh' found, skipping."
fi

if [[ "$LOCAL" == true ]]; then
    echo "Deleting kind cluster 'cloud-mesh'..."
    kind delete cluster --name cloud-mesh
else
    echo ""
    echo "GKE cluster: $CLUSTER_NAME  (zone: $ZONE, project: $PROJECT)"
    read -r -p "Delete the GKE cluster? This cannot be undone. [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        gcloud container clusters delete "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT"
        echo "Cluster deleted."
    else
        echo "Cluster kept. To delete it later:"
        echo "  gcloud container clusters delete $CLUSTER_NAME --zone $ZONE --project $PROJECT"
    fi
fi
