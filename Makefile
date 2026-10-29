.DEFAULT_GOAL := help

# Load some shared variables from envs/.env.base
ORG := $(shell grep '^MYORG=' envs/.env.base | cut -d= -f2 | tr -d '"')
APP := $(shell grep '^MYAPP=' envs/.env.base | cut -d= -f2 | tr -d '"')

MOCKUP_APP_VERSION := $(shell grep '^APP_VERSION=' envs/.env.mockup | cut -d= -f2 | tr -d '"')
NORMAL_APP_VERSION := $(shell grep '^APP_VERSION=' envs/.env.normal | cut -d= -f2 | tr -d '"')

CUR_SYS := 
CUR_MODE := prod

CUR_BUILD := $(CUR_SYS)$(CUR_MODE)

# The known modes your project supports
VALID_MODES := mockup-dev mockup-prod dev prod

# Map each mode to a set of environment files
ENVFILES_mockup-dev   = --env-file envs/.env.base --env-file envs/.env.mockup --env-file envs/.env.dev
ENVFILES_mockup-prod  = --env-file envs/.env.base --env-file envs/.env.mockup --env-file envs/.env.prod
ENVFILES_dev          = --env-file envs/.env.base --env-file envs/.env.normal --env-file envs/.env.dev
ENVFILES_prod         = --env-file envs/.env.base --env-file envs/.env.normal --env-file envs/.env.prod

ifdef C
CENTRALIZED := 1
endif

ifdef CENTRALIZED
COMPOSE_FILE_OPTIONS = -f docker-centralized/docker-compose-centralized.yml
EDGE_COMPOSE_FILE_OPTIONS = 
CENTRALIZED_FLAG = --centralized
else
COMPOSE_FILE_OPTIONS = -f docker/docker-compose.yml
EDGE_COMPOSE_FILE_OPTIONS = -f docker/docker-compose-edge.yml
CENTRALIZED_FLAG = 
endif

# Map each mode to an image tag
TAG_mockup-dev   = $(MOCKUP_APP_VERSION)-dev
TAG_mockup-prod  = $(MOCKUP_APP_VERSION)
TAG_dev          = $(NORMAL_APP_VERSION)-dev
TAG_prod         = $(NORMAL_APP_VERSION)

PHONY_TARGETS := help lock compose-% build-% forcebuild-% showcomposecmd-% run-% start-% stop-% logs-% clean-% restart-% rmlogs-%

ifndef CENTRALIZED
PHONY_TARGETS += buildedge-% forcebuildedge-% showcomposecmdedge-% runedge-% startedge-% stopedge-% logsedge-% cleanedge-% restartedge-% rmlogsedge-%
endif

.PHONY: $(PHONY_TARGETS)

compose-%:
	@echo "Configuring the environment for $(ORG)/$(APP):$(TAG_$*) ..."
	@python3 scripts/build_compose.py $(CENTRALIZED_FLAG) $(ENVFILES_$*)
	@echo "Done"

build-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Building $(ORG)/$(APP):$(TAG_$*) ..."
	@docker compose $(COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) build
	@echo "Done"

forcebuild-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Force-rebuilding $(ORG)/$(APP):$(TAG_$*) ..."
	@docker compose $(COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) build --no-cache
	@echo "Done"

showcomposecmd-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo docker compose $(COMPOSE_FILE_OPTIONS) $(ENVFILES_$*)

run-% start-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Starting $(ORG)/$(APP):$(TAG_$*) ..."
	@docker compose $(COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) up -d
	@echo "Done"

stop-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Stopping $(ORG)/$(APP):$(TAG_$*) ..."
	@docker compose $(COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) down -t 3
	@echo "Done"

ps-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Listing containers for $(ORG)/$(APP):$(TAG_$*) ..."
	@docker compose $(COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) ps
	@echo "Done"

logs-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Showing logs for $(ORG)/$(APP):$(TAG_$*) ..."
	@docker compose $(COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) logs -f -t
	@echo "Done"

clean-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@docker compose $(COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) down -t 3 --rmi all -v --remove-orphans
	@echo "Done"

restart-%: clean-% build-% run-%

rmlogs-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Removing all log files ..."
	@rm -f logs-$(CENTRALIZED_SUFFIX)/*/*
	@echo "Done"

ifndef CENTRALIZED
buildedge-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	 echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	 exit 1; \
	fi
	@echo "Building $(ORG)/$(APP):$(TAG_$*) ..."
	@docker compose $(EDGE_COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) build
	@echo "Done"

forcebuildedge-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Force-rebuilding $(ORG)/$(APP):$(TAG_$*) ..."
	@docker compose $(EDGE_COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) build --no-cache
	@echo "Done"

showcomposecmdedge-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo docker compose $(EDGE_COMPOSE_FILE_OPTIONS) $(ENVFILES_$*)

runedge-% startedge-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Starting $(ORG)/$(APP):$(TAG_$*) ..."
	@docker compose $(EDGE_COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) up -d
	@echo "Done"

stopedge-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Stopping $(ORG)/$(APP):$(TAG_$*) ..."
	@docker compose $(EDGE_COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) down -t 3
	@echo "Done"

psedge-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Listing containers for $(ORG)/$(APP):$(TAG_$*) ..."
	@docker compose $(EDGE_COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) ps
	@echo "Done"

logsedge-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Showing logs for $(ORG)/$(APP):$(TAG_$*) ..."
	@docker compose $(EDGE_COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) logs -f -t
	@echo "Done"

cleanedge-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@docker compose $(EDGE_COMPOSE_FILE_OPTIONS) $(ENVFILES_$*) down -t 3 --rmi all -v --remove-orphans
	@echo "Done"
endif

restartedge-%: cleanedge-% buildedge-% runedge-%

rmlogsedge-%:
	@if [ -z "$(findstring $*, $(VALID_MODES))" ]; then \
	  echo "Error: invalid environment '$*'. Valid modes: $(VALID_MODES)."; \
	  exit 1; \
	fi
	@echo "Removing all flclient logs ..."
	@rm -f logs-$(CENTRALIZED_SUFFIX)/flclient*/*
	@echo "Done"

###############################################################################
# Shortcut Targets
###############################################################################

compose: compose-$(CUR_BUILD)
build: build-$(CUR_BUILD)
forcebuild: forcebuild-$(CUR_BUILD)
showcomposecmd: showcomposecmd-$(CUR_BUILD)
run: run-$(CUR_BUILD)
start: start-$(CUR_BUILD)
stop: stop-$(CUR_BUILD)
ps: ps-$(CUR_BUILD)
logs: logs-$(CUR_BUILD)
clean: clean-$(CUR_BUILD)
restart: restart-$(CUR_BUILD)
rmlogs: rmlogs-$(CUR_BUILD)

ifndef CENTRALIZED
buildedge: buildedge-$(CUR_BUILD)
forcebuildedge: forcebuildedge-$(CUR_BUILD)
showcomposecmdedge: showcomposecmdedge-$(CUR_BUILD)
runedge: runedge-$(CUR_BUILD)
startedge: startedge-$(CUR_BUILD)
stopedge: stopedge-$(CUR_BUILD)
logsedge: logsedge-$(CUR_BUILD)
cleanedge: cleanedge-$(CUR_BUILD)
restartedge: restartedge-$(CUR_BUILD)
rmlogsedge: rmlogsedge-$(CUR_BUILD)
endif

###############################################################################
# Help
###############################################################################
# Conda environments locked by `make lock` (directories containing environment.yaml)
CONDA_LOCK_DIRS := flightflserver flightflclient
CONDA_LOCK_IMAGE := condaforge/miniforge3:25.3.0-3
CONDA_LOCK_VERSION := 4.0.3

lock:
	@for dir in $(CONDA_LOCK_DIRS); do \
	  echo "Locking $$dir/environment.yaml ..."; \
	  docker run --rm --user $$(id -u):$$(id -g) -e HOME=/tmp -e CONDA_PKGS_DIRS=/tmp/pkgs -v "$(CURDIR)/$$dir":/work -w /work $(CONDA_LOCK_IMAGE) \
	    bash -c "mamba create -p /tmp/lock --yes --quiet -c conda-forge conda-lock=$(CONDA_LOCK_VERSION) >/dev/null && \
	             /tmp/lock/bin/conda-lock lock -f environment.yaml -p linux-64 --kind explicit --filename-template 'conda-{platform}.lock'" \
	    || exit 1; \
	done
	@echo "Done"

help:
ifdef CENTRALIZED
	@echo "Mode: centralized"
else
	@echo "Mode: decentralized"
endif
	@echo "Usage: make <target>"
	@echo ""
	@echo "Environments: $(VALID_MODES)"
	@echo ""
ifdef CENTRALIZED
	@echo "Pattern Targets (append '-<env>'):"
	@echo "  compose-<env>, build-<env>, forcebuild-<env>, showcomposecmd-<env>, run-<env>, start-<env>, stop-<env>, logs-<env>, clean-<env>, restart-<env>, rmlogs-<env>"
else
	@echo "Pattern Targets (append '-<env>') on the host device:"
	@echo "  compose-<env>, build-<env>, forcebuild-<env>, showcomposecmd-<env>, run-<env>, start-<env>, stop-<env>, logs-<env>, clean-<env>, restart-<env>, rmlogs-<env>"
	@echo "Pattern Targets (append '-<env>') on the edge device:"
	@echo "  buildedge-<env>, forcebuildedge-<env>, showcomposecmdedge-<env>, runedge-<env>, startedge-<env>, stopedge-<env>, logsedge-<env>, cleanedge-<env>, restartedge-<env>, rmlogsedge-<env>"
endif
	@echo ""
	@echo "Examples:"
	@echo "  make build-mockup-dev"
	@echo "  make run-prod"
	@echo "  make tar-mockup-prod"
	@echo ""
	@echo "Dependency lock (regenerates conda-linux-64.lock from each environment.yaml):"
	@echo "  lock"
	@echo ""
ifdef CENTRALIZED
	@echo "Shortcut Targets (map to '-$(CUR_BUILD)'):"
	@echo "  compose, build, forcebuild, showcomposecmd, run, start, stop, logs, clean, restart, rmlogs"
else
	@echo "Shortcut Targets on the host device (map to '-$(CUR_BUILD)'):"
	@echo "  compose, build, forcebuild, showcomposecmd, run, start, stop, logs, clean, restart, rmlogs"
	@echo "Shortcut Targets on the edge device (map to '-$(CUR_BUILD)'):"
	@echo "  buildedge, forcebuildedge, showcomposecmdedge, runedge, startedge, stopedge, logsedge, cleanedge, restartedge, rmlogsedge"
endif
	@echo ""