TAG ?= $(USER)
BASE_IMAGE ?= nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04
WORKSPACE_DOCKERFILE ?= docker/workspace.Dockerfile
WORKSPACE_IMAGE ?= liepose/workspace
WORKDIR ?= /workspace
DATADIR ?= /workspace/dataset
SHM_SIZE ?= 16G
CUDA_VISIBLE_DEVICES ?= 0

EXTRA_ARGS ?=
EXTRA_ARGS := $(EXTRA_ARGS) --network host
EXTRA_ARGS := $(EXTRA_ARGS) -e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
EXTRA_ARGS := $(EXTRA_ARGS) -e WORKDIR=$(WORKDIR) -e DATADIR=$(DATADIR)
EXTRA_ARGS := $(if $(SHM_SIZE), $(EXTRA_ARGS) --shm-size $(SHM_SIZE), $(EXTRA_ARGS))
EXTRA_ARGS := $(if $(DISPLAY), $(EXTRA_ARGS) -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix, $(EXTRA_ARGS))
EXTRA_ARGS := $(if $(NTFY_ACCESS_KEY), $(EXTRA_ARGS) -e NTFY_ACCESS_KEY=$(NTFY_ACCESS_KEY), $(EXTRA_ARGS))
COMMAND ?= /bin/bash

.PHONY: download
download:
	cd dataset/; bash ./download_voc.sh
	cd dataset/bop_datasets/; bash ./download_bop.sh

.PHONY: build
build:
	docker build . -f $(WORKSPACE_DOCKERFILE) -t $(WORKSPACE_IMAGE):$(TAG) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg WORKDIR=$(WORKDIR)

.PHONY: run
run:
	if [ -n "${DISPLAY}" ]; then xhost +; fi
	docker run -it --gpus all --rm \
		-v $(realpath ./):$(WORKDIR) \
		-w $(WORKDIR) \
		$(EXTRA_ARGS) \
		$(WORKSPACE_IMAGE):$(TAG) \
		$(COMMAND)

.PHONY: dev
dev:
	@pip install pre-commit hatch
	-pre-commit install --hook-type pre-commit

.PHONY: lint
lint:
	pre-commit run
