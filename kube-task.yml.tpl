# Kubernetes Deployment template.
# Do NOT apply this file directly — make_cluster.sh runs envsubst to produce
# kube-task.yml and applies that. See make_cluster.sh for the full workflow.
#
# Variables substituted at deploy time (all come from config.toml via make_cluster.sh):
#   DOCKER_IMAGE, NUM_REPLICAS, QUEUE_URL, OUTPUT_BUCKET,
#   LEASE_SECONDS, MAX_RUNS, N_JOBS, RECOMPUTE, LOGGING_LEVEL
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    run: cloud-mesh
  name: cloud-mesh
spec:
  replicas: ${NUM_REPLICAS}
  selector:
    matchLabels:
      run: cloud-mesh
  strategy:
    rollingUpdate:
      maxSurge: 100%
      maxUnavailable: 100%
    type: RollingUpdate
  template:
    metadata:
      labels:
        run: cloud-mesh
    spec:
      containers:
        - image: ${DOCKER_IMAGE}
          name: cloud-mesh
          imagePullPolicy: Always
          command: ["/bin/sh"]
          args: ["-c", "while true; do uv run worker.py; done"]
          env:
            - name: GOOGLE_APPLICATION_CREDENTIALS
              value: "/root/.cloudvolume/secrets/google-secret.json"
            - name: CLOUD_MESH_QUEUE_URL
              value: "${QUEUE_URL}"
            - name: CLOUD_MESH_OUTPUT_BUCKET
              value: "${OUTPUT_BUCKET}"
            - name: CLOUD_MESH_LEASE_SECONDS
              value: "${LEASE_SECONDS}"
            - name: CLOUD_MESH_MAX_RUNS
              value: "${MAX_RUNS}"
            - name: CLOUD_MESH_N_JOBS
              value: "${N_JOBS}"
            - name: CLOUD_MESH_RECOMPUTE
              value: "${RECOMPUTE}"
            - name: CLOUD_MESH_LOGGING_LEVEL
              value: "${LOGGING_LEVEL}"
            # Thread count guards — keep at 1 to avoid over-subscription
            - name: OPENBLAS_NUM_THREADS
              value: "1"
            - name: MKL_NUM_THREADS
              value: "1"
            - name: NUMEXPR_NUM_THREADS
              value: "1"
            - name: OMP_NUM_THREADS
              value: "1"
          resources:
            requests:
              memory: "4Gi"
              cpu: "900m"
              ephemeral-storage: "4Gi"
            limits:
              memory: "32Gi"
              ephemeral-storage: "16Gi"
          volumeMounts:
            - name: secrets-volume
              mountPath: /root/.cloudvolume/secrets
      dnsPolicy: Default
      volumes:
        - name: secrets-volume
          secret:
            secretName: secrets
