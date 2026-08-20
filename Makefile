PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: install verify lint test analysis paper ci

install:
	$(PYTHON) -m pip install -e ".[test]"

verify:
	$(PYTHON) tools/verify_inputs.py

lint:
	$(PYTHON) -m ruff check src scripts tests tools
	$(PYTHON) -m ruff format --check src scripts tests tools

test: verify
	$(PYTHON) -m pytest

analysis: verify
	PYTHONPATH=src $(PYTHON) scripts/run_analysis.py

paper: analysis
	sh paper/build.sh

ci: lint test analysis
