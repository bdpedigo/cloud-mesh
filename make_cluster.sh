#!/usr/bin/env bash
# make_cluster.sh — Create a GKE cluster and deploy the cloud-mesh workers.
#
# Prerequisites:
#   - gcloud CLI authenticated: gcloud auth login && gcloud auth application-default login
#   - kubectl available: gcloud components install kubectl
#   - envsubst available: brew install gettext (macOS) or apt install gettext-base (Linux)
#   - CloudVolume secrets in ~/.cloudvolume/secrets/
#
# Usage:
#   bash make_cluster.sh
#
# All configurable values live in config.toml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Read config.toml into shell variables ─────────────────────────────────────
# Uses Python's stdlib tomllib (Python 3.11+) — no extra dependencies needed.
eval "$(python3 - <<'EOF'
import tomllib, os, sys
with open("config.toml", "rb") as f:
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
for k, v in pairs.items():
    print(f'export {k}="{v}"')
EOF
)"

echo "Project:      $PROJECT"
echo "Cluster:      $CLUSTER_NAME ($MACHINE_TYPE × $NUM_NODES nodes)"
echo "Zone:         $ZONE"
echo "Docker image: $DOCKER_IMAGE  replicas: $NUM_REPLICAS"
echo "Queue:        $QUEUE_URL"
echo "Output:       $OUTPUT_BUCKET"
echo ""

# ── Create the cluster ────────────────────────────────────────────────────────
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

# Fetch credentials so kubectl works
gcloud container clusters get-credentials --zone "$ZONE" "$CLUSTER_NAME"

# ── Push CloudVolume secrets into the cluster ─────────────────────────────────
# Picks up every file in ~/.cloudvolume/secrets/ automatically.
# Each file becomes a key in the secret, mounted at /root/.cloudvolume/secrets/<filename>.
SECRETS_DIR="$HOME/.cloudvolume/secrets"
SECRET_ARGS=()
for f in "$SECRETS_DIR"/*; do
    [[ -f "$f" ]] && SECRET_ARGS+=("--from-file=$f")
done
kubectl create secret generic secrets "${SECRET_ARGS[@]}"

# ── Deploy ────────────────────────────────────────────────────────────────────
envsubst < "$SCRIPT_DIR/kube-task.yml.tpl" | kubectl apply -f -

echo ""
echo "Cluster ready. Useful commands:"
echo "  kubectl get pods"
echo "  kubectl logs -f <pod-name>"
echo "  kubectl describe nodes"
echo "  kubectl delete deployment cloud-mesh   # tear down workers"
echo "  gcloud container clusters delete $CLUSTER_NAME --zone $ZONE  # delete cluster"