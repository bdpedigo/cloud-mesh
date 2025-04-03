# REF: based on https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
FROM python:3.12-slim-bookworm

# Install git
RUN apt update
# RUN apt install -y git
RUN apt-get update && apt-get install -y build-essential

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.4.3 /uv /bin/uv

# Install the project with intermediate layers
# ADD .dockerignore .

# First, install the dependencies
WORKDIR /app
COPY ./cloud-mesh/uv.lock /app/cloud-mesh/uv.lock
COPY ./cloud-mesh/pyproject.toml /app/cloud-mesh/pyproject.toml
COPY ./cloud-mesh/README.md /app/cloud-mesh/README.md
COPY ./cloud-mesh/runners /app/cloud-mesh/runners
COPY ./cloud-mesh/models /app/cloud-mesh/models
WORKDIR /app/cloud-mesh

# RUN echo "meshrep" > /app/.uvignore
# ADD uv.lock /app/uv.lock
# ADD pyproject.toml /app/pyproject.toml
# RUN --mount=type=cache,target=/root/.cache/uv \
    # uv sync --frozen --no-install-project

# Then, install the rest of the project
ADD cloudigo /app/cloudigo
ADD meshmash /app/meshmash
RUN uv sync
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --frozen

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENV RUN_JOBS='True'
ENV TEST_RUN='False'

# CMD ["uv", "run", "runners/features_latest.py"]
CMD ["uv", "run", "runners/approach_comparisons.py"]