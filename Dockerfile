# REF: based on https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
FROM python:3.12-slim-bookworm

# Install git
RUN apt update
RUN apt install -y git
RUN apt-get update && apt-get install -y build-essential

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.4.3 /uv /bin/uv

# Install the project with intermediate layers
# ADD .dockerignore .

# First, install the dependencies
WORKDIR /app
COPY ./cloud-mesh /app/cloud-mesh
WORKDIR /app/cloud-mesh

# RUN echo "meshrep" > /app/.uvignore
# ADD uv.lock /app/uv.lock
# ADD pyproject.toml /app/pyproject.toml
# RUN --mount=type=cache,target=/root/.cache/uv \
    # uv sync --frozen --no-install-project

# Then, install the rest of the project
ADD . /app
RUN uv sync
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --frozen

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENV RUN_JOBS='True'
ENV TEST_RUN='False'

CMD ["uv", "run", "runners/predict_synapse_compartments_2024-11-07.py"]