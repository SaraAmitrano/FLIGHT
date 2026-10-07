#!/bin/bash

# This script is used to create a release for the FLIGHT project.
# It takes the path to the FLIGHT release directory as an argument.
if [ "$#" -lt 1 ]; then
    echo "Usage: bash $0 <path_to_release_directory> [uid] [gid]"
    echo ""
    echo "Arguments:"
    echo "  <path_to_release_directory>  Path to the directory where the release will be created (required)."
    echo "  [uid]                        User ID (optional, default: 1000)."
    echo "  [gid]                        Group ID (optional, default: 1000)."
    exit 1
fi
FLIGHT_VERSION=1.3.0

FLIGHT_RELEASE_DIR=$1
if [ ! -d "$FLIGHT_RELEASE_DIR" ]; then
    echo "Error: Directory $FLIGHT_RELEASE_DIR does not exist."
    exit 1
fi

FLIGHT_UID=${2:-1000}
FLIGHT_GID=${3:-1000}

# Create the unique .env file
(cat envs/.env.base_rel ; echo "";) > "$FLIGHT_RELEASE_DIR/.env"

# from the .env file, remove the line containing MYORG=

sed -i '/^MYORG=/d' "$FLIGHT_RELEASE_DIR/.env"
sed -i '/^MYAPP=/d' "$FLIGHT_RELEASE_DIR/.env"
sed -i 's/UID=1000/UID='$FLIGHT_UID'/g' "$FLIGHT_RELEASE_DIR/.env"
sed -i 's/GID=1000/GID='$FLIGHT_GID'/g' "$FLIGHT_RELEASE_DIR/.env"

# Copy necessary files to the release directory

# Makefiles
cp Makefile.release "$FLIGHT_RELEASE_DIR/Makefile"

# Data
cp -r data "$FLIGHT_RELEASE_DIR/data"

# Configs
cp -r configs "$FLIGHT_RELEASE_DIR/"

# Examples
cp -r examples "$FLIGHT_RELEASE_DIR/"

# Source code, but remove ruff.toml files
cp -r flightapi flightorchestrator flightflserver flightflclient common "$FLIGHT_RELEASE_DIR/" && rm -f "$FLIGHT_RELEASE_DIR/*/ruff.toml"

# Scripts
mkdir -p "$FLIGHT_RELEASE_DIR/scripts"
cp -r scripts "$FLIGHT_RELEASE_DIR/" && mv "$FLIGHT_RELEASE_DIR/scripts/build_compose-release.py" "$FLIGHT_RELEASE_DIR/scripts/build_compose.py"
cp scripts/build_compose-release.py "$FLIGHT_RELEASE_DIR/scripts/build_compose.py"

# Modify the docker-compose-template.yml file
sed -i 's/\${MYORG}/modal-unina/g' "$FLIGHT_RELEASE_DIR/scripts/docker/docker-compose-template.yml"
sed -i 's/\${MYAPP}/flight/g' "$FLIGHT_RELEASE_DIR/scripts/docker/docker-compose-template.yml"
sed -i 's/\${APP_DEV}//g' "$FLIGHT_RELEASE_DIR/scripts/docker/docker-compose-template.yml"
sed -i 's/\${APP_VERSION:-latest}/'$FLIGHT_VERSION'/g' "$FLIGHT_RELEASE_DIR/scripts/docker/docker-compose-template.yml"

mkdir -p "$FLIGHT_RELEASE_DIR/fldata"
mkdir -p "$FLIGHT_RELEASE_DIR/docker"
mkdir -p "$FLIGHT_RELEASE_DIR/docker-centralized"

# echo "DO NOT FORGET TO SET the permissions of the data and datasplit folders to 1000:1000"
