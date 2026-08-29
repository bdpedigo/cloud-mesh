# Kubernetes CronJob template for the HKS pipeline monitor.
# Do NOT apply this file directly - make_cluster.sh runs envsubst to produce
# monitor-cron.yml and applies that.
#
# Variables substituted at deploy time (from config.toml via make_cluster.sh):
#   DOCKER_IMAGE, PROJECT, OUTPUT_BUCKET, MONITOR_SCHEDULE, MONITOR_DATASTACK
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cloud-mesh-monitor
  namespace: workers
spec:
  schedule: "${MONITOR_SCHEDULE}"
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 1
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      backoffLimit: 0
      ttlSecondsAfterFinished: 3600
      template:
        metadata:
          labels:
            app: cloud-mesh-monitor
        spec:
          serviceAccountName: ksa-worker
          restartPolicy: Never
          containers:
          - name: monitor
            image: ${DOCKER_IMAGE}
            imagePullPolicy: Always
            command: ["uv", "run", "monitor.py"]
            env:
            - name: GCP_PROJECT
              value: "${PROJECT}"
            - name: CLOUD_MESH_OUTPUT_BUCKET
              value: "${OUTPUT_BUCKET}"
            - name: CLOUD_MESH_DATASTACK
              value: "${MONITOR_DATASTACK}"
            - name: CONFIG_PATH
              value: "/app/config.toml"
            - name: SLACK_WEBHOOK_URL
              valueFrom:
                secretKeyRef:
                  name: slack-secret
                  key: SLACK_WEBHOOK_URL
            resources:
              requests:
                cpu: "100m"
                memory: "256Mi"
              limits:
                cpu: "500m"
                memory: "512Mi"
