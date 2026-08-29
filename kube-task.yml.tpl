# Kubernetes Deployment template.
# Do NOT apply this file directly — make_cluster.sh runs envsubst to produce
# kube-task.yml and applies that. See make_cluster.sh for the full workflow.
#
# Variables substituted at deploy time (all come from config.toml via make_cluster.sh):
#   DOCKER_IMAGE, NUM_REPLICAS, QUEUE_URL, OUTPUT_BUCKET,
#   LEASE_SECONDS, MAX_RUNS, N_JOBS, RECOMPUTE, LOGGING_LEVEL,
#   CPU_REQUEST, CPU_LIMIT, MEMORY_REQUEST, EPHEMERAL_STORAGE_REQUEST,
#   MEMORY_LIMIT, EPHEMERAL_STORAGE_LIMIT, SLACK_SECRET_NAME
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cloud-mesh-worker
  namespace: workers
spec:
  replicas: ${NUM_REPLICAS}
  selector:
    matchLabels:
      app: cloud-mesh-worker
  template:
    metadata:
      labels:
        app: cloud-mesh-worker
    spec:
      serviceAccountName: ksa-worker
      containers:
      - name: worker
        image: ${DOCKER_IMAGE}
        imagePullPolicy: Always
        command: ["uv", "run", "worker_pubsub.py"]
        env:
        - name: GCP_PROJECT
          value: "${PROJECT}"
        - name: PUBSUB_SUBSCRIPTION
          value: "hks-todo-sub"
        - name: CLOUD_MESH_OUTPUT_BUCKET
          value: "${OUTPUT_BUCKET}"
        - name: CONFIG_PATH
          value: "/app/config.toml"
        - name: CLOUD_MESH_RECOMPUTE
          value: "${RECOMPUTE}"
        - name: CAVE_AUTH_TOKEN
          valueFrom:
            secretKeyRef:
              name: cave-secret
              key: CAVE_AUTH_TOKEN
        - name: SLACK_WEBHOOK_URL
          valueFrom:
            secretKeyRef:
              name: slack-secret
              key: SLACK_WEBHOOK_URL
        resources:
          requests:
            cpu: "${CPU_REQUEST}"
            memory: "${MEMORY_REQUEST}"
            ephemeral-storage: "${EPHEMERAL_STORAGE_REQUEST}"
          limits:
            cpu: "${CPU_LIMIT}"
            memory: "${MEMORY_LIMIT}"
            ephemeral-storage: "${EPHEMERAL_STORAGE_LIMIT}"
