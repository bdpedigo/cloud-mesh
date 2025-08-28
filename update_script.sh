#!/bin/bash
# Usage: ./update-script.sh myscript
# This updates the ConfigMap with the given script and restarts the pods.

set -euo pipefail

SCRIPT_NAME="$1"
DEPLOYMENT_NAME="cloud-mesh"
NAMESPACE="default" # change if not running in default namespace

# Delete old configmap (ignore error if not found)
kubectl delete configmap script --ignore-not-found -n "$NAMESPACE"

# Create new configmap with the chosen script
kubectl create configmap script \
  --from-file=run_script.py=./runners/"${SCRIPT_NAME}".py \
  -n "$NAMESPACE"

echo "✅ Updated configmap 'script' with ./runners/${SCRIPT_NAME}.py"

# Restart pods so they pick up the new ConfigMap
kubectl rollout restart deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE"

echo "🔄 Restarted deployment '$DEPLOYMENT_NAME'"

echo "⏳ Waiting for pods to be ready..."
kubectl rollout status deployment cloud-mesh

echo "⏳ Waiting for pods to appear..."
kubectl wait --for=condition=Ready pod -l run=cloud-mesh --timeout=120s

POD=$(kubectl get pods -l run=cloud-mesh -o jsonpath='{.items[0].metadata.name}')

echo "📜 Tailing logs from pod $POD..."
kubectl logs -f "$POD"