# syntax=docker.io/docker/dockerfile:1.7-labs

# Stage 0: Common Layer
# ------------------------------------------------------------------------------
FROM ubuntu:22.04 AS commonlayer

ARG UID=1000
ARG GID=1000

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Stage 1: Builder Stage
# ------------------------------------------------------------------------------
FROM commonlayer AS builder

ARG UID=1000
ARG GID=1000

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_ROOT_USER_ACTION=ignore

# DEBUG
# ENV PYTHONMALLOC=debug
# ENV PYTHONASYNCIODEBUG=1

# Set working directory
WORKDIR /app

# Install essential dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV PATH=/opt/conda/bin:$PATH
RUN arch=$(uname -m) && \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-0-Linux" && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="${MINICONDA_URL}-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="${MINICONDA_URL}-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget ${MINICONDA_URL} -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo ". /opt/conda/etc/profile.d/mamba.sh" >> ~/.bashrc && \
    echo "conda activate appenv" >> ~/.bashrc && \
    conda install -n base --override-channels --yes -c conda-forge mamba 'python_abi=*=*cp*' && \
    mamba update --all --yes && \
    mamba clean --all --yes

# Copy environment file and create the Conda environment
COPY ./environment.yaml ./
RUN mamba env create -f environment.yaml -n appenv --yes && \
    mamba clean --all --yes

# Activate the environment
ENV PATH=/opt/conda/envs/appenv/bin:$PATH

# ------------------------------------------------------------------------------
# Stage 2: Final Stage
# ------------------------------------------------------------------------------
FROM commonlayer AS final

ARG UID=1000
ARG GID=1000

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH=/opt/conda/envs/appenv/bin:$PATH

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${GID} appuser && \
    useradd -ms /bin/bash --uid ${UID} --gid ${GID} appuser

# Create directories and set permissions
RUN mkdir -p /app/logs /app/storage /app/tmp && \
    chown -R appuser:appuser /app

USER appuser

# Copy Miniconda and environment from the builder stage
COPY --from=builder /opt/conda /opt/conda

# Copy application code and static files
WORKDIR /app/main
