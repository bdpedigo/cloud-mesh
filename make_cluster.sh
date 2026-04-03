#!/usr/bin/env bash
# make_cluster.sh — Create a cluster and deploy the cloud-mesh workers.
#
# Modes:
#   bash make_cluster.sh           — create a GKE cluster (production)
#   bash make_cluster.sh --local   — create a local kind cluster (for testing)
#
# Prerequisites (both modes):
#   - kubectl available
#   - envsubst: brew install gettext (macOS) or apt install gettext-base (Linux)
#   - CloudVolume secrets in ~/.cloudvolume/secrets/
#   - Docker image already built
#
# Additional prerequisites for GKE mode:
#   - gcloud CLI authenticated: gcloud auth login && gcloud auth application-default login
#   - kubectl via gcloud: gcloud components install kubectl
#
# Additional prerequisites for --local mode:
#   - kind: brew install kind  (https://kind.sigs.k8s.io)
#
# All configurable values live in config.toml (or CONFIG_PATH).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-config.toml}"

LOCAL=false
if [[ "${1:-}" == "--local" ]]; then
    LOCAL=true
fi

# ── Read config into shell variables ──────────────────────────────────────────
# Uses Python's stdlib tomllib (Python 3.11+) — no extra dependencies needed.
eval "$(python3 - <<'EOF'
import tomllib, os
with open(os.environ["CONFIG_PATH"], "rb") as f:
    cfg = tomllib.load(f)
c, j = cfg["cluster"], cfg["job"]
pairs = {
    "PROJECT":       c["project"],
    "ZONE":          c["zone"],
    "CLUSTER_NAME":  c["cluster_name"],
    "MACHINE_TYPE":  c["machine_type"],
    "NUM_NODES":     str(c["num_nodes"]),
    "DISK_SIZE":     str(c["disk_size_gb"]),
    "NETWORK":       c["network"],
    "SUBNETWORK":    c["subnetwork"],
    "DOCKER_IMAGE":  j["docker_image"],
    "NUM_REPLICAS":  str(j["num_replicas"]),
    "QUEUE_URL":     j["queue_url"],
    "OUTPUT_BUCKET": j["output_bucket"],
    "LEASE_SECONDS": str(j["lease_seconds"]),
    "MAX_RUNS":      str(j["max_runs"]),
    "N_JOBS":        str(j["n_jobs"]),
    "RECOMPUTE":     str(j["recompute"]).lower(),
    "LOGGING_LEVEL": j["logging_level"],
}
r = cfg["resources"]
pairs.update({
    "CPU_REQUEST":               r["cpu_request"],
    "MEMORY_REQUEST":            r["memory_request"],
    "EPHEMERAL_STORAGE_REQUEST": r["ephemeral_storage_request"],
    "MEMORY_LIMIT":              r["memory_limit"],
    "EPHEMERAL_STORAGE_LIMIT":   r["ephemeral_storage_limit"],
})
for k, v in pairs.items():
    print(f'export {k}="{v}"')
EOF
)"

echo "Docker image: $DOCKER_IMAGE  replicas: $NUM_REPLICAS"
echo "Queue:        $QUEUE_URL"
echo "Output:       $OUTPUT_BUCKET"
echo ""

if [[ "$LOCAL" == true ]]; then
    # ── Local: kind cluster ───────────────────────────────────────────────────
    if kind get clusters 2>/dev/null | grep -q "^cloud-mesh$"; then
        echo "kind cluster 'cloud-mesh' already exists, reusing."
    else
        echo "Creating kind cluster 'cloud-mesh'..."
        kind create cluster --name cloud-mesh
    fi

    kubectl config use-context kind-cloud-mesh

    echo "Loading image '$DOCKER_IMAGE' into kind cluster..."
    kind load docker-image "$DOCKER_IMAGE" --name cloud-mesh
else
    # ── GKE cluster ───────────────────────────────────────────────────────────
    echo "Project: $PROJECT  Cluster: $CLUSTER_NAME ($MACHINE_TYPE × $NUM_NODES)  Zone: $ZONE"
    echo ""

    gcloud config set project "$PROJECT"

    gcloud container --project "$PROJECT" clusters create "$CLUSTER_NAME" \
        --zone "$ZONE" \
        --no-enable-basic-auth \
        --release-channel "stable" \
        --machine-type "$MACHINE_TYPE" \
        --image-type "COS_CONTAINERD" \
        --disk-type "pd-standard" \
        --disk-size "$DISK_SIZE" \
        --metadata disable-legacy-endpoints=true \
        --scopes "https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring" \
        --preemptible \
        --num-nodes "$NUM_NODES" \
        --logging=SYSTEM,WORKLOAD \
        --monitoring=SYSTEM \
        --enable-ip-alias \
        --network "$NETWORK" \
        --subnetwork "$SUBNETWORK" \
        --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
        --enable-autoupgrade \
        --enable-autorepair \
        --max-unavailable-upgrade 0 \
        --max-pods-per-node "256" \
        --node-locations "$ZONE" \
        --enable-shielded-nodes \
        --shielded-secure-boot \
        --shielded-integrity-monitoring

    gcloud container clusters get-credentials --zone "$ZONE" "$CLUSTER_NAME"
fi

# ── Push CloudVolume secrets into the cluster ─────────────────────────────────
# Picks up every file in ~/.cloudvolume/secrets/ automatically.
# Each file becomes a key in the secret, mounted at /root/.cloudvolume/secrets/<filename>.
SECRETS_DIR="$HOME/.cloudvolume/secrets"
SECRET_ARGS=()
for f in "$SECRETS_DIR"/*; do
    [[ -f "$f" ]] && SECRET_ARGS+=("--from-file=$f")
done

if kubectl get secret secrets &>/dev/null; then
    echo "Secret 'secrets' already exists, replacing..."
    kubectl delete secret secrets
fi
kubectl create secret generic secrets "${SECRET_ARGS[@]}"

# ── Deploy ────────────────────────────────────────────────────────────────────
if [[ "$LOCAL" == true ]]; then
    # Use the locally-loaded image; don't try to pull from a registry.
    envsubst < "$SCRIPT_DIR/kube-task.yml.tpl" \
        | sed 's/imagePullPolicy: Always/imagePullPolicy: Never/' \
        | kubectl apply -f -
else
    envsubst < "$SCRIPT_DIR/kube-task.yml.tpl" | kubectl apply -f -
fi

echo ""
echo "Cluster ready. Useful commands:"
echo "  kubectl get pods"
echo "  kubectl logs -f <pod-name>"
echo "  kubectl describe nodes"
echo "  kubectl delete deployment cloud-mesh          # tear down workers"
if [[ "$LOCAL" == true ]]; then
    echo "  kind delete cluster --name cloud-mesh         # delete local cluster"
else
    echo "  gcloud container clusters delete $CLUSTER_NAME --zone $ZONE  # delete cluster"
fi