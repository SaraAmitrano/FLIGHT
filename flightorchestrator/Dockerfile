# syntax=docker.io/docker/dockerfile:1.7-labs

# An example of using standalone Python builds with multistage images.

# First, build the application in the `/app` directory
FROM ghcr.io/astral-sh/uv:trixie-slim AS builder

ARG UID=1000
ARG GID=1000

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Configure the Python directory so it is consistent
ENV UV_PYTHON_INSTALL_DIR=/opt/python

# Only use the managed Python version
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python before the project for caching
RUN uv python install 3.12

WORKDIR /app

RUN mkdir -p /app/logs /app/storage

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,from=common,source=.,target=../common \
    uv sync --locked --no-install-project --no-dev --no-install-workspace \
    && uv pip uninstall --break-system-packages --python "$(uv python find --managed-python --system)" pip setuptools wheel

COPY src /app/main
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
COPY --from=common . /common
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-workspace

# Then, use a final image without uv
FROM debian:trixie-slim

ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# DEBUG
# ENV PYTHONMALLOC=debug
# ENV PYTHONASYNCIODEBUG=1

# Create a non-root user
RUN groupadd --gid ${GID} appuser && \
    useradd -ms /bin/bash --uid ${UID} --gid ${GID} appuser;

# Copy the Python version
COPY --from=builder --chown=python:python /opt/python /opt/python

# Copy the application from the builder
COPY --from=builder --chown=appuser:appuser /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

USER appuser

# Copy application code and static files
WORKDIR /app/main