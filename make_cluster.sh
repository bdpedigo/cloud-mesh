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
eval "$(python3 - <<'EOF'
import tomllib, os
with open(os.environ["CONFIG_PATH"], "rb") as f:
    cfg = tomllib.load(f)
c, j, p = cfg["cluster"], cfg["job"], cfg["pubsub"]
pairs = {
    "PROJECT":       c["project"],
    "ZONE":          c["zone"],
    "CLUSTER_NAME":  c["cluster_name"],
    "MACHINE_TYPE":  c["machine_type"],
    "NUM_NODES":     str(c["num_nodes"]),
    "DISK_SIZE":     str(c["disk_size_gb"]),
    "NETWORK":       c["network"],
    "SUBNETWORK":    c["subnetwork"],
    # Optional: pin nodes to a specific service account. Empty -> GKE uses
    # the project's default compute service account.
    "NODE_SERVICE_ACCOUNT": c.get("node_service_account", ""),
    "DOCKER_IMAGE":  j["docker_image"],
    "NUM_REPLICAS":  str(j["num_replicas"]),
    "OUTPUT_BUCKET": j["output_bucket"],
    "N_JOBS":        str(j["n_jobs"]),
    "RECOMPUTE":     str(j["recompute"]).lower(),
    "LOGGING_LEVEL": j["logging_level"],
    "SLACK_SECRET_NAME": j["slack_secret_name"],
    "PUBSUB_TOPIC":        p["topic"],
    "PUBSUB_SUBSCRIPTION": p["subscription"],
    "PUBSUB_DLT":          p["dead_letter_topic"],
    "PUBSUB_DLS":          p["dead_letter_subscription"],
    "PUBSUB_MAX_DELIVERIES": str(p["max_delivery_attempts"]),
}
r = cfg["resources"]
pairs.update({
    "CPU_REQUEST":               r["cpu_request"],
    "CPU_LIMIT":                 r["cpu_limit"],
    "MEMORY_REQUEST":            r["memory_request"],
    "EPHEMERAL_STORAGE_REQUEST": r["ephemeral_storage_request"],
    "MEMORY_LIMIT":              r["memory_limit"],
    "EPHEMERAL_STORAGE_LIMIT":   r["ephemeral_storage_limit"],
})
m = cfg.get("monitor", {})
pairs.update({
    "MONITOR_SCHEDULE":  m.get("schedule",  "*/10 * * * *"),
    "MONITOR_DATASTACK": m.get("datastack", ""),
})
for k, v in pairs.items():
    print(f'export {k}="{v}"')
EOF
)"

echo "Docker image: $DOCKER_IMAGE  replicas: $NUM_REPLICAS"
echo "Subscription: $PUBSUB_SUBSCRIPTION"
echo "Dead letter:  $PUBSUB_DLT"
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

    if gcloud container clusters describe "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT" &>/dev/null; then
        echo "Cluster '$CLUSTER_NAME' already exists, skipping creation."
    else
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
            --workload-pool="${PROJECT}.svc.id.goog" \
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
            --shielded-integrity-monitoring \
            ${NODE_SERVICE_ACCOUNT:+--service-account="$NODE_SERVICE_ACCOUNT"}
    fi

    gcloud container clusters get-credentials --zone "$ZONE" "$CLUSTER_NAME"

    # Enforce node count from config.toml (idempotent; no-op if already correct).
    CURRENT_NODES=$(gcloud container clusters describe "$CLUSTER_NAME" \
        --zone "$ZONE" --project "$PROJECT" \
        --format='value(currentNodeCount)')
    if [[ "$CURRENT_NODES" != "$NUM_NODES" ]]; then
        echo "Resizing default-pool from $CURRENT_NODES to $NUM_NODES nodes..."
        gcloud container clusters resize "$CLUSTER_NAME" \
            --node-pool default-pool \
            --num-nodes "$NUM_NODES" \
            --zone "$ZONE" \
            --project "$PROJECT" \
            --quiet
    else
        echo "Cluster already at $NUM_NODES nodes."
    fi

    # Create namespace and Kubernetes service account for Workload Identity
    kubectl create namespace workers || true
    kubectl create serviceaccount ksa-worker -n workers || true

    # Annotate the KSA with the GSA email
    kubectl annotate serviceaccount ksa-worker -n workers \
        iam.gke.io/gcp-service-account=gke-worker-sa@${PROJECT}.iam.gserviceaccount.com --overwrite

    # Bind KSA -> GSA via Workload Identity
    gcloud iam service-accounts add-iam-policy-binding gke-worker-sa@${PROJECT}.iam.gserviceaccount.com \
        --role roles/iam.workloadIdentityUser \
        --member "serviceAccount:${PROJECT}.svc.id.goog[workers/ksa-worker]" \
        --project="${PROJECT}"

    # Grant the worker GSA read access to Cloud Monitoring metrics so the
    # monitor CronJob can query Pub/Sub queue depth. Idempotent.
    gcloud projects add-iam-policy-binding "${PROJECT}" \
        --member "serviceAccount:gke-worker-sa@${PROJECT}.iam.gserviceaccount.com" \
        --role roles/monitoring.viewer \
        --condition=None >/dev/null

    # Grant the worker GSA full object access on the output bucket so the
    # monitor can overwrite _monitor_state.json (create alone is not enough;
    # rewrite requires storage.objects.delete). Idempotent.
    BUCKET_NAME_ONLY="${OUTPUT_BUCKET#gs://}"
    BUCKET_NAME_ONLY="${BUCKET_NAME_ONLY%%/*}"
    gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME_ONLY}" \
        --member "serviceAccount:gke-worker-sa@${PROJECT}.iam.gserviceaccount.com" \
        --role roles/storage.objectUser \
        --condition=None >/dev/null

    # Allow the monitor CronJob (running as ksa-worker) to list pods in the
    # workers namespace so it can report how many workers are alive. Idempotent.
    kubectl apply -f - <<'EOF'
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: workers
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ksa-worker-pod-reader
  namespace: workers
subjects:
- kind: ServiceAccount
  name: ksa-worker
  namespace: workers
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
EOF
fi

# ── Pub/Sub: create dead letter topic and wire it up ─────────────────────────
DLT_NAME="${PUBSUB_DLT##*/}"   # strip "projects/.../topics/" prefix
DLS_NAME="${PUBSUB_DLS##*/}"   # strip "projects/.../subscriptions/" prefix
TOPIC_NAME="${PUBSUB_TOPIC##*/}"
SUB_NAME="${PUBSUB_SUBSCRIPTION##*/}"

# Create dead letter topic if it doesn't exist
if gcloud pubsub topics describe "$PUBSUB_DLT" --project="$PROJECT" &>/dev/null; then
    echo "Dead letter topic '$DLT_NAME' already exists, skipping."
else
    echo "Creating dead letter topic '$DLT_NAME'..."
    gcloud pubsub topics create "$DLT_NAME" --project="$PROJECT"
fi

# Create dead letter subscription if it doesn't exist
if gcloud pubsub subscriptions describe "$PUBSUB_DLS" --project="$PROJECT" &>/dev/null; then
    echo "Dead letter subscription '$DLS_NAME' already exists, skipping."
else
    echo "Creating dead letter subscription '$DLS_NAME'..."
    gcloud pubsub subscriptions create "$DLS_NAME" \
        --topic="$DLT_NAME" \
        --project="$PROJECT"
fi

# Grant Pub/Sub the rights it needs to forward dead-lettered messages
PUBSUB_SA="service-$(gcloud projects describe $PROJECT --format='value(projectNumber)')@gcp-sa-pubsub.iam.gserviceaccount.com"

gcloud pubsub topics add-iam-policy-binding "$DLT_NAME" \
    --member="serviceAccount:${PUBSUB_SA}" \
    --role="roles/pubsub.publisher" \
    --project="$PROJECT"

gcloud pubsub subscriptions add-iam-policy-binding "$SUB_NAME" \
    --member="serviceAccount:${PUBSUB_SA}" \
    --role="roles/pubsub.subscriber" \
    --project="$PROJECT"

# Update the main subscription with dead letter policy
echo "Configuring dead letter policy on '$SUB_NAME' (max $PUBSUB_MAX_DELIVERIES attempts)..."

gcloud pubsub subscriptions update "$SUB_NAME" \
    --dead-letter-topic="$DLT_NAME" \
    --max-delivery-attempts="$PUBSUB_MAX_DELIVERIES" \
    --project="$PROJECT"

echo "Dead letter policy configured."

# ── Push CloudVolume secrets into the cluster ─────────────────────────────────
SECRETS_DIR="$HOME/.cloudvolume/secrets"
SECRET_ARGS=()
for f in "$SECRETS_DIR"/*; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f")
    if [[ "$base" == "google-secret.json" || "$base" == "google-secret-sa.json" ]]; then
        echo "Skipping $base (use Workload Identity instead)"
        continue
    fi
    SECRET_ARGS+=("--from-file=$f")
done

if kubectl get secret secrets &>/dev/null; then
    echo "Secret 'secrets' already exists, replacing..."
    kubectl delete secret secrets
fi
kubectl create secret generic secrets "${SECRET_ARGS[@]}"

# ── Push CAVE auth token into the cluster ────────────────────────────────────
CAVE_SECRET_FILE="$HOME/.cloudvolume/secrets/cave-secret.json"
CAVE_SECRET_FILE_NATIVE=$(python3 -c "import os; print(os.path.expanduser(r'~/.cloudvolume/secrets/cave-secret.json'))")
if [[ -f "$CAVE_SECRET_FILE" ]]; then
    CAVE_AUTH_TOKEN=$(python3 -c "import json; print(json.load(open(r'$CAVE_SECRET_FILE_NATIVE'))['token'])")
else
    CAVE_AUTH_TOKEN="${CAVE_AUTH_TOKEN:-}"
fi

if [[ -z "$CAVE_AUTH_TOKEN" ]]; then
    echo "ERROR: CAVE auth token not found in $CAVE_SECRET_FILE or \$CAVE_AUTH_TOKEN"
    exit 1
fi

if kubectl get secret cave-secret -n workers &>/dev/null; then
    echo "Secret 'cave-secret' already exists, replacing..."
    kubectl delete secret cave-secret -n workers
fi
kubectl create secret generic cave-secret -n workers \
    --from-literal=CAVE_AUTH_TOKEN="$CAVE_AUTH_TOKEN"
echo "CAVE auth token stored as Kubernetes secret."

# ── Push Slack webhook into the cluster ──────────────────────────────────────
echo "Fetching Slack webhook from GCP Secret Manager ('$SLACK_SECRET_NAME')..."
SLACK_WEBHOOK_URL=$(gcloud secrets versions access latest \
    --secret="$SLACK_SECRET_NAME" \
    --project="$PROJECT" 2>/dev/null || true)

if [[ -z "$SLACK_WEBHOOK_URL" ]]; then
    echo "WARNING: Could not fetch '$SLACK_SECRET_NAME' from Secret Manager; skipping Slack secret."
else
    if kubectl get secret slack-secret -n workers &>/dev/null; then
        echo "Secret 'slack-secret' already exists, replacing..."
        kubectl delete secret slack-secret -n workers
    fi
    kubectl create secret generic slack-secret -n workers \
        --from-literal=SLACK_WEBHOOK_URL="$SLACK_WEBHOOK_URL"
    echo "Slack webhook stored as Kubernetes secret."
fi

# ── Deploy ────────────────────────────────────────────────────────────────────
if [[ "$LOCAL" == true ]]; then
    envsubst < "$SCRIPT_DIR/kube-task.yml.tpl" \
        | sed 's/imagePullPolicy: Always/imagePullPolicy: Never/' \
        | kubectl apply -f -
    envsubst < "$SCRIPT_DIR/monitor-cron.yml.tpl" \
        | sed 's/imagePullPolicy: Always/imagePullPolicy: Never/' \
        | kubectl apply -f -
else
    envsubst < "$SCRIPT_DIR/kube-task.yml.tpl" | kubectl apply -f -
    envsubst < "$SCRIPT_DIR/monitor-cron.yml.tpl" | kubectl apply -f -
fi

echo "Monitor CronJob: schedule '$MONITOR_SCHEDULE' datastack '$MONITOR_DATASTACK'"

echo ""
echo "Cluster ready. Useful commands:"
echo "  kubectl get pods -n workers"
echo "  kubectl logs -f -n workers <pod-name>"
echo "  kubectl describe nodes"
echo "  kubectl delete deployment cloud-mesh-worker -n workers   # tear down workers"
echo "  gcloud pubsub subscriptions pull $PUBSUB_DLS --limit=10  # inspect dead letters"
if [[ "$LOCAL" == true ]]; then
    echo "  kind delete cluster --name cloud-mesh                    # delete local cluster"
else
    echo "  gcloud container clusters delete $CLUSTER_NAME --zone $ZONE  # delete cluster"
fi
