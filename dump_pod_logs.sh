#!/usr/bin/env bash
set -euo pipefail

NS="${1:-all}"
OUTROOT="pod-logs-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$OUTROOT"

pods=()
if [[ "$NS" == "all" ]]; then
  mapfile -t pods < <(kubectl get pods --all-namespaces -o jsonpath='{range .items[*]}{.metadata.namespace}"/"{.metadata.name}{"\n"}{end}')
else
  mapfile -t pods < <(kubectl get pods -n "$NS" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')
fi

for entry in "${pods[@]}"; do
  if [[ "$NS" == "all" ]]; then
    ns="${entry%%/*}"; pod="${entry#*/}"
  else
    ns="$NS"; pod="$entry"
  fi
  outdir="$OUTROOT/$ns/$pod"
  mkdir -p "$outdir"
  # containers (including init containers)
  mapfile -t containers < <(kubectl get pod -n "$ns" "$pod" -o jsonpath='{range .spec.initContainers[*]}{.name}{"\n"}{end}' 2>/dev/null || true)
  mapfile -t tmp < <(kubectl get pod -n "$ns" "$pod" -o jsonpath='{range .spec.containers[*]}{.name}{"\n"}{end}')
  containers+=("${tmp[@]}")

  for c in "${containers[@]}"; do
    # current stdout/stderr
    kubectl logs -n "$ns" "$pod" -c "$c" > "$outdir/${pod}__${c}.log" 2>&1 || echo "error-getting-current" > "$outdir/${pod}__${c}.log"
    # previous stdout/stderr (if any)
    if kubectl logs -n "$ns" "$pod" -c "$c" --previous > "$outdir/${pod}__${c}.previous.log" 2>&1; then
      :
    else
      echo "no-previous" > "$outdir/${pod}__${c}.previous.log"
    fi

    # Ensure current user can read/write/delete the files we just created:
    # Make files readable/writable by owner, and ensure owner owns them.
    # Use the current user's UID/GID if available.
    if command -v id >/dev/null 2>&1; then
      UID_NOW=$(id -u)
      GID_NOW=$(id -g)
      chown "${UID_NOW}:${GID_NOW}" "$outdir/${pod}__${c}.log" "$outdir/${pod}__${c}.previous.log" 2>/dev/null || true
    fi
    chmod u+rw "$outdir/${pod}__${c}.log" "$outdir/${pod}__${c}.previous.log" 2>/dev/null || true
  done
done

# also ensure the whole output tree is owned and writable by current user
if command -v id >/dev/null 2>&1; then
  UID_NOW=$(id -u)
  GID_NOW=$(id -g)
  chown -R "${UID_NOW}:${GID_NOW}" "$OUTROOT" 2>/dev/null || true
fi
chmod -R u+rwX "$OUTROOT" 2>/dev/null || true

tar -czf "${OUTROOT}.tar.gz" -C "$(dirname "$OUTROOT")" "$(basename "$OUTROOT")"
echo "Saved logs: ${OUTROOT}.tar.gz"