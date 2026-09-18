VENV ?= /mnt/downloads/frigate-xdna-venvs/dev
PY := $(VENV)/bin/python
SRC := $(CURDIR)/src
UPSTREAM := $(CURDIR)/tests/upstream

.PHONY: test-unit test-contract test build-native image test-image-offline test-hardware test-frigate-e2e

test-unit:
	PYTHONPATH=$(SRC) $(PY) -m unittest discover -s tests/unit -t . -v

test-contract:
	PYTHONPATH=$(SRC):$(UPSTREAM) $(PY) -m unittest discover -s tests/contract -t . -v

test: test-unit test-contract

build-native:
	@echo "not yet implemented (Task 04: native IPC worker)" >&2; exit 3

image:
	@echo "not yet implemented (Task 03/05: appliance image)" >&2; exit 3

test-image-offline:
	@echo "not yet implemented (Task 03: compiler appliance gate)" >&2; exit 3

test-hardware:
	@echo "refusing: hardware tests are explicit opt-in with device ownership (Task 04+)" >&2; exit 8

test-frigate-e2e:
	@echo "not yet implemented (Task 05: full Frigate gate)" >&2; exit 3
