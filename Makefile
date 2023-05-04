#!/usr/bin/env make

# customization points via makefile key-value arguments
#
# interpreter in images: 3.{8,10} currently available
# DOCKER_BASE_PYTHON=3.10
# test script executed with `docker_test`: mpi, notebooks_dir, oldest, vanilla, mpi, numpy_git, pip_installed
# PYMOR_TEST_SCRIPT=vanilla
# version pinned mirror to be used: stable or oldest
# PYPI_MIRROR=stable
# wheel check OS: debian_buster centos_8 debian_testing
# PYMOR_TEST_OS=debian_buster
# hypothesis profiles: dev, debug, ci, ci-pr, ci-large
# PYMOR_HYPOTHESIS_PROFILE=dev
# extra options to be passed to pytest
# PYMOR_PYTEST_EXTRA="--lf"

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
DOCKER_RUN=docker run -v $(THIS_DIR):/pymor --env-file  $(THIS_DIR)/.env
CI_COMMIT_REF_NAME?=$(shell git rev-parse --abbrev-ref HEAD)
COMPOSE_CMD=$(shell ((docker compose version 1> /dev/null) && echo "docker compose") || which docker-compose)
DOCKER_COMPOSE=CI_COMMIT_SHA=$(shell git log -1 --pretty=format:"%H") \
	CI_COMMIT_REF_NAME=$(CI_COMMIT_REF_NAME) \
	NB_USER=$(NB_USER) $(COMPOSE_SUDO) $(COMPOSE_CMD) -f .binder/docker-compose.yml -p pymor
NB_DIR=docs/source
NB_USER:=${USER}
ifeq ($(PYMOR_SUDO), 1)
	COMPOSE_SUDO:=sudo -E
else
	COMPOSE_SUDO:=
endif

PANDOC_MAJOR=$(shell ( which pandoc && pandoc --version | head  -n1 | cut -d ' ' -f 2 | cut -d '.' -f 1)) || echo "pandoc missing")
ifeq ($(PANDOC_MAJOR),1)
	PANDOC_FORMAT=-f markdown_github
endif
# this loads $(ENV_FILE) as both makefile variables and into shell env
ENV_FILE?=.env
include $(ENV_FILE)
export $(shell sed 's/=.*//' $(ENV_FILE))

.PHONY: docker README.html pylint test docs conda_update

all:
	./dependencies.py

# I want HTML (to preview the formatting :))
README.html: README.md
	pandoc $(PANDOC_FORMAT) -t html $< > $@

README: README.html

flake8:
	flake8 ./src

test:
	python setup.py pytest

jupyter:
	jupyter notebook --notebook-dir=$(NB_DIR) --NotebookApp.disable_check_xsrf=True

tutorials: NB_DIR="docs/_build/html"
tutorials: docs jupyter

full-test:
	@echo
	@echo "Ensuring that pytest-cov is installed ..."
	@echo "--------------------------------------------------------------------------------"
	@echo
	pip install pytest-cov
	@echo
	@echo "--------------------------------------------------------------------------------"
	@echo
	py.test --cov --cov-config=setup.cfg --cov-report=html --cov-report=xml src/pymortests

docs:
	PYTHONPATH=${PWD}/src/:${PYTHONPATH} make -C docs html

template:
	./dependencies.py
	./.ci/gitlab/template.ci.py

conda_update:
	./dependencies.py
	./.ci/create_conda_env.py $(THIS_DIR)/requirements*txt

# docker targets
docker_template:
	@$(DOCKER_RUN) pymor/ci_sanity:$(CI_IMAGE_TAG) /pymor/dependencies.py \
	  || $(DOCKER_RUN) pymor/ci_sanity:latest /pymor/dependencies.py
	@$(DOCKER_RUN) pymor/ci_sanity:$(CI_IMAGE_TAG) /pymor/.ci/gitlab/template.ci.py ${GITLAB_TOKEN} \
	  || $(DOCKER_RUN) pymor/ci_sanity:latest /pymor/.ci/gitlab/template.ci.py ${GITLAB_TOKEN}
	@echo Files changed:
	@git diff --name-only

docker_image:
	$(DOCKER_COMPOSE) build
	@if [ "$(COMPOSE_CMD)" = "docker-compose" ]; then echo \
		"in case of inexplicable errors try using compose as plugin: https://docs.docker.com/compose/install/"; fi

docker_docs: docker_image
	NB_DIR=notebooks $(DOCKER_COMPOSE) run jupyter ./.ci/gitlab/test_docs.bash

docker_run: docker_image
	$(DOCKER_COMPOSE) up -d pypi_mirror
	$(DOCKER_COMPOSE) run --service-ports pytest bash
	$(DOCKER_COMPOSE) down --remove-orphans -v

docker_exec: docker_image
	$(DOCKER_COMPOSE) run --service-ports pytest bash -l -c "${DOCKER_CMD}"

docker_tutorials: NB_DIR=docs/_build/html
docker_tutorials: docker_docs docker_jupyter

docker_test: docker_image
	PYMOR_TEST_SCRIPT=$(PYMOR_TEST_SCRIPT) $(DOCKER_COMPOSE) up pytest

docker_test_oldest: docker_image
	PYMOR_TEST_SCRIPT=oldest PYPI_MIRROR=oldest DOCKER_BASE_PYTHON=3.8 $(DOCKER_COMPOSE) up pytest

docker_run_oldest: DOCKER_BASE_PYTHON=3.8
docker_run_oldest: PYMOR_TEST_SCRIPT=oldest
docker_run_oldest: PYPI_MIRROR=oldest
docker_run_oldest: docker_image
	$(DOCKER_COMPOSE) up -d pypi_mirror
	$(DOCKER_COMPOSE) run pytest bash
	$(DOCKER_COMPOSE) down --remove-orphans -v

docker_jupyter: docker_image
	NB_DIR=$(NB_DIR) $(DOCKER_COMPOSE) up jupyter

docker_wheel_check: docker_image
	DOCKER_BASE_PYTHON=$(DOCKER_BASE_PYTHON) PYMOR_TEST_OS=$(PYMOR_TEST_OS) $(DOCKER_COMPOSE) run --service-ports \
		wheel_check bash
	$(DOCKER_COMPOSE) down --remove-orphans -v

docker_install_check: docker_image
	DOCKER_BASE_PYTHON=$(DOCKER_BASE_PYTHON) PYMOR_TEST_OS=$(PYMOR_TEST_OS) $(DOCKER_COMPOSE) run --service-ports install_check \
      /pymor/.ci/gitlab/install_checks/$(PYMOR_TEST_OS)/check.bash
	$(DOCKER_COMPOSE) down --remove-orphans -v

ci_base_image:
	podman build -t pymor/ci-base:3.10 -f $(THIS_DIR)/.ci/gitlab/Dockerfile.ci-base.3_10 $(THIS_DIR)

ci_fenics_base_image:
	podman build -t pymor/ci-fenics-base:3.11 -f $(THIS_DIR)/.ci/gitlab/Dockerfile.ci-fenics-base.3_11 $(THIS_DIR)

ci_requirements:
	podman run --rm -it -v=$(THIS_DIR):/src pymor/ci-base:3.10 \
		pip-compile --resolver backtracking \
			--extra docs-additional \
			--extra tests \
			--extra ci \
			--extra ann \
			--extra slycot \
			--extra pymess \
			--extra ipyparallel \
			--extra mpi \
			--extra gui \
			--extra jupyter \
			--extra vtk \
			--extra gmsh \
			--extra dune \
			--extra ngsolve \
			--extra scikit-fem \
			--extra-index-url https://download.pytorch.org/whl/cpu \
			-o requirements-ci.txt

ci_fenics_requirements:
	podman run --rm -it -v=$(THIS_DIR):/src pymor/ci-fenics-base:3.11 \
		. /venv/bin/activate; \
		pip-compile --resolver backtracking \
			--extra docs_additional \
			--extra tests \
			--extra ci \
			--extra ann \
			--extra ipyparallel \
			--extra mpi \
			--extra-index-url https://download.pytorch.org/whl/cpu \
			-o requirements-ci-fenics.txt

ci_conda_requirements:
	conda-lock --micromamba -c conda-forge --filter-extras -f pyproject.toml \
		--extras tests \
		--extras ci \
		--extras slycot \
		--extras ipyparallel \
		--extras mpi \
		--extras gui \
		--extras jupyter \
		--extras vtk \
		--extras gmsh 
	# pymess, dune, ngsolve, scikit-fem (no recent version) not available as conda-forge packages
	# pytorch not available for win64
	# docs-additional not needed
	conda-lock render \
		--extras tests \
		--extras ci \
		--extras slycot \
		--extras ipyparallel \
		--extras mpi \
		--extras gui \
		--extras jupyter \
		--extras vtk \
		--extras gmsh 

ci_image:
	podman build -t pymor/ci:3.10_$(shell sha256sum $(THIS_DIR)/requirements-ci.txt | cut -d " " -f 1) \
		-f $(THIS_DIR)/.ci/gitlab/Dockerfile.ci.3_10 $(THIS_DIR)

ci_fenics_image:
	podman build -t pymor/ci-fenics:3.11_$(shell sha256sum $(THIS_DIR)/requirements-ci-fenics.txt | cut -d " " -f 1) \
		-f $(THIS_DIR)/.ci/gitlab/Dockerfile.ci-fenics.3_11 $(THIS_DIR)

ci_image_push:
	podman push pymor/ci:3.10_$(shell sha256sum $(THIS_DIR)/requirements-ci.txt | cut -d " " -f 1) \
		zivgitlab.wwu.io/pymor/pymor/ci:3.10_$(shell sha256sum $(THIS_DIR)/requirements-ci.txt | cut -d " " -f 1)

ci_fenics_image_push:
	podman push pymor/ci-fenics:3.11_$(shell sha256sum $(THIS_DIR)/requirements-ci-fenics.txt | cut -d " " -f 1) \
		zivgitlab.wwu.io/pymor/pymor/ci-fenics:3.11_$(shell sha256sum $(THIS_DIR)/requirements-ci-fenics.txt | cut -d " " -f 1)
