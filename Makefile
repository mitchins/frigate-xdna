VENV ?= /mnt/downloads/frigate-xdna-venvs/dev
PY ?= $(VENV)/bin/python
SRC := $(CURDIR)/src
UPSTREAM := $(CURDIR)/tests/upstream

.PHONY: test-unit test-contract test-integration test coverage build-native image test-image-offline test-hardware test-frigate-e2e

test-unit:
	PYTHONPATH=$(SRC) $(PY) -m unittest discover -s tests/unit -t . -v

test-contract:
	PYTHONPATH=$(SRC):$(UPSTREAM) $(PY) -m unittest discover -s tests/contract -t . -v

test-integration:
	PYTHONPATH=$(SRC):$(CURDIR) $(PY) -m unittest discover -s tests/integration -t . -v

test: test-unit test-contract test-integration

# Hardware-free coverage across all suites (CI gate). `coverage report`
# enforces the fail_under floor from pyproject.toml [tool.coverage.report].
coverage:
	rm -f .coverage coverage.xml
	PYTHONPATH=$(SRC) $(PY) -m coverage run -m unittest discover -s tests/unit -t .
	PYTHONPATH=$(SRC):$(UPSTREAM) $(PY) -m coverage run --append -m unittest discover -s tests/contract -t .
	PYTHONPATH=$(SRC):$(CURDIR) $(PY) -m coverage run --append -m unittest discover -s tests/integration -t .
	$(PY) -m coverage xml
	$(PY) -m coverage report

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
