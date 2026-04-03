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
| `uv` | fast Python package manager — manages Python and dependencies locally and in the container | [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| CloudVolume secrets | JSON credential files in `~/.cloudvolume/secrets/` — at minimum `cave-secret.json` and `google-secret.json` | [github.com/seung-lab/cloud-volume](https://github.com/seung-lab/cloud-volume#credentials) |

---

## Quick start

### 1 — Configure

Edit `config.toml` to set your GCP project, machine type, Docker image name, task queue URL, and output bucket.  
Edit `hks_parameters.toml` if you want to adjust pipeline hyperparameters.

Then verify all prerequisites are in order:

```bash
bash check.sh
```

### 2 — Build and push the Docker image

```bash
bash build.sh
```

This reads `docker_image` from `config.toml` and builds for `linux/amd64` (required for GKE), then pushes to your registry.

### 3 — Install local dependencies (for enqueue.py)

```bash
uv sync
```

### 4 — Enqueue tasks

Populate the queue from a CSV before or while the cluster is running:

```bash
# From a CSV with a column of root IDs
uv run enqueue.py --csv path/to/roots.csv --col pt_root_id \
    --datastack minnie65_public

# Or pass root IDs directly
uv run enqueue.py --ids 864691135494786192 864691135851320007 \
    --datastack minnie65_public
```

### 5 — Create the cluster and deploy

```bash
bash make_cluster.sh
```

This will:
1. Create the GKE cluster using settings from `config.toml`
2. Push CloudVolume secrets into the cluster
3. Deploy the worker pods via `envsubst < kube-task.yml.tpl | kubectl apply -f -`

> **Tip:** Steps 2 and 5 can be combined into a single command:
> ```bash
> bash deploy.sh --build
> ```

---

## Incremental development walkthrough

When setting up the system from scratch — or debugging a problem — it helps to validate each layer independently before moving on to the next. The stages below let you do that.

### Stage 1 — Local Python (no queue)

**Tests:** Python environment, CloudVolume credentials, `meshmash` pipeline, GCS write.

**Prerequisites:** `uv sync` and CloudVolume secrets in `~/.cloudvolume/secrets/`.
Set `output_bucket` in `config.toml` to a writable GCS path.

```bash
uv run run_single.py --root-id 864691135436446706 --datastack minnie65_public
```

**Verify:** The script logs `saved features for ... → gs://...` and the `.npz` file appears in your output bucket.

---

### Stage 2 — Local Python + real task queue

**Tests:** Task queue integration, worker polling loop.

**Prerequisites:** Stage 1 passing. Set `queue_url` in `config.toml` to a queue path (e.g. `gs://my-bucket/queues/cloud-mesh`). If you do multiple rounds of debugging here, make sure that
you always enqueue prior to trying to run the worker, since the worker will wait for a while if there are no tasks to claim.

```bash
# Enqueue one task
uv run enqueue.py --ids 864691135436446706 --datastack minnie65_public

# Pull and process it (exits after one task)
CLOUD_MESH_MAX_RUNS=1 uv run worker.py
```

**Verify:** Worker logs show the task being claimed and processed, then exits cleanly.

---

### Stage 3 — Docker (local)

**Tests:** Container build, dependency installation, secrets volume mount, env var wiring.

**Prerequisites:** Stage 2 passing. Docker running locally.

```bash
# Build for linux/amd64 without pushing (will be run directly)
bash build.sh --no-push

# Enqueue a task (from your local Python env)
uv run enqueue.py --ids 864691135436446706 --datastack minnie65_public

# Run the worker inside the container
# config.toml is baked into the image, so queue/bucket settings are already present.
# Only the secrets (never baked in) and a MAX_RUNS override are needed here.
bash test_run_docker.sh
```

**Verify:** Container exits after processing one task; output `.npz` appears in GCS (or shows that it was already computed).

---

### Stage 4 — Local Kubernetes (kind)

**Tests:** Kubernetes manifest, secret injection, pod restart loop, resource requests.

**Prerequisites:** Stage 3 passing. Install [kind](https://kind.sigs.k8s.io/docs/user/quick-start/): `brew install kind`.

```bash
# Enqueue some tasks first
uv run enqueue.py --ids 864691135436446706 --datastack minnie65_public

# Stand up a local cluster, load the image, and deploy
bash make_cluster.sh --local
```

```bash
# Monitor
kubectl get pods
kubectl logs -f <pod-name>

# Tear down when done
bash teardown.sh --local
```

**Verify:** Pods show `Running` status; logs show tasks being processed.

> **Note:** `--local` mode uses `imagePullPolicy: Never` so Kubernetes uses
> the image loaded via `kind load docker-image` rather than pulling from a registry.
> This means you need to re-run `bash make_cluster.sh --local` after every image rebuild.

---

### Stage 5 — GKE (production)

When all four local stages pass, you're ready for the real cluster. Follow the [Quick start](#quick-start).

```bash
bash build.sh
bash make_cluster.sh
```

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

# Tear down deployment and (optionally) delete the cluster
bash teardown.sh
```

---

## Docker build notes (macOS)

```bash
# If you get a buildx platform error, you may need to switch builder:
docker buildx use desktop-linux

# Build and run locally for testing (mounts your secrets):
bash build.sh --no-push
bash test_run_docker.sh
```