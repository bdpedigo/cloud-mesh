# cloud-mesh

Batch extraction of condensed HKS morphology features from neuron meshes, orchestrated with GKE (Google Kubernetes Engine) and a cloud task queue.

Each Kubernetes pod runs `worker.py` in a loop, pulling `(root_id, datastack)` tasks from a queue, fetching the mesh via CloudVolume, running `meshmash.condensed_hks_pipeline`, and writing the result to a GCS bucket.

Note that you may want to do other things here with the mesh in-hand, e.g. map synapses
to the vertices of the mesh so that you don't have to look them up again later. The `worker.py` script is meant to be a template.

We also stress that this is just one possible deployment of this pipeline. The core logic in `worker.py` is portable to any environment where you can run Python and access the meshes, and write the results to a bucket. The GKE + task queue setup is just one way to orchestrate the work at scale, and can be swapped out for something else if desired.

```
local machine
  enqueue.py ──► task queue (GCS or SQS)
                      ▲
              GKE pods (worker.py) ──► GCS output bucket
```

---

## Prerequisites

| Requirement | Notes | Link |
|---|---|---|
| Google Cloud account | billing-enabled project required | [console.cloud.google.com](https://console.cloud.google.com) |
| `gcloud` CLI | install and run `gcloud auth login` + `gcloud auth application-default login` | [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install) |
| `kubectl` | `gcloud components install kubectl` | [kubernetes.io/docs/tasks/tools](https://kubernetes.io/docs/tasks/tools/) |
| `envsubst` | macOS: `brew install gettext`; Linux: `apt install gettext-base` | — |
| Docker | for building and pushing the worker image | [docs.docker.com/get-docker](https://docs.docker.com/get-docker/) |
| Python ≥ 3.12 | for running `enqueue.py` locally | [python.org](https://www.python.org/downloads/) |
| `uv` | fast Python package manager (used in the container and recommended locally) | [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| CloudVolume secrets | JSON credential files in `~/.cloudvolume/secrets/` — at minimum `cave-secret.json` and `google-secret.json` | [github.com/seung-lab/cloud-volume](https://github.com/seung-lab/cloud-volume#credentials) |

---

## Quick start

### 1 — Configure

Edit `config.toml` to set your GCP project, machine type, Docker image name, task queue URL, and output bucket.  
Edit `hks_parameters.toml` if you want to adjust pipeline hyperparameters.

### 2 — Build and push the Docker image

```bash
# Build for linux/amd64 (required for GKE)
docker buildx build --platform linux/amd64 -t your-user/cloud-mesh:v1 .

# Push to Docker Hub (or your registry of choice)
docker push your-user/cloud-mesh:v1
```

Update `docker_image` in `config.toml` to match.

### 3 — Install local dependencies (for enqueue.py)

```bash
uv sync
```

### 4 — Enqueue tasks

Populate the queue from a CSV before or while the cluster is running:

```bash
# From a CSV with a column of root IDs
uv run enqueue.py --csv path/to/roots.csv --col pt_root_id \
    --datastack minnie65_phase3_v1 --version 1412

# Or pass root IDs directly
uv run enqueue.py --ids 864691135494786192 864691135851320007 \
    --datastack minnie65_phase3_v1 --version 1412
```

### 5 — Create the cluster and deploy

```bash
bash make_cluster.sh
```

This will:
1. Create the GKE cluster using settings from `config.toml`
2. Push CloudVolume secrets into the cluster
3. Deploy the worker pods via `envsubst < kube-task.yml.tpl | kubectl apply -f -`

---

## Configuration reference

### `config.toml`

Controls cluster settings, Docker image name, task queue URL, and output bucket. See the file itself for all keys and their defaults.

### `hks_parameters.toml`

Controls `meshmash.condensed_hks_pipeline` kwargs, CloudVolume mesh-get options, and output dtype. See the file itself for all keys and their defaults.

---

## Output

Features are written to:
```
{output_bucket}/{datastack}/features/{root_id}.npz
```

Each `.npz` contains the condensed HKS feature matrix and per-vertex labels as produced by `meshmash.save_condensed_features`.

---

## Monitoring

```bash
# List running pods
kubectl get pods

# Stream logs from a pod
kubectl logs -f <pod-name>

# Check node resource usage
kubectl describe nodes

# Tear down the deployment (keeps the cluster)
kubectl delete deployment cloud-mesh

# Delete the cluster entirely
gcloud container clusters delete <cluster-name> --zone <zone>
```

---

## Docker build notes (macOS)

```bash
# If you get a buildx platform error, you may need to switch builder:
docker buildx use desktop-linux

# Build and run locally for testing (mounts your secrets):
docker run --rm --platform linux/amd64 \
    -v ~/.cloudvolume/secrets:/root/.cloudvolume/secrets \
    -e CLOUD_MESH_QUEUE_URL=gs://my-bucket/test-queue \
    your-user/cloud-mesh:v1
```