# syntax=docker.io/docker/dockerfile:1.7-labs

FROM nvcr.io/nvidia/pytorch:24.12-py3-igpu

ARG UID=1000
ARG GID=1000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

# Set the working directory in the container
WORKDIR /app

COPY ./requirements.txt ./
COPY ./python_patches ./python_patches

RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt && \
    cd /usr/local/lib/python3.12/dist-packages && \
    find /app -type f -name '*.patch' -print0 | xargs -0 -n 1 patch -p1 && \
    cd /app && \
    rm -r python_patches && \
    pip uninstall -y pynvml  # to avoid conflicts with nvidia-ml-py3

WORKDIR /app/main
