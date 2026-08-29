# REF: https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# Copy application code and config
COPY worker_pubsub.py run_hks.py PSTaskQueue.py monitor.py config.toml hks_parameters.toml ./

# Install the project itself
RUN uv sync --frozen


ENV PATH="/app/.venv/bin:$PATH"
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# Default: run the Pub/Sub worker. Swap in worker.py to use the original
# task-queue path instead.
CMD ["uv", "run", "worker_pubsub.py"]
