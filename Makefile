.PHONY: environment-check scgpt-smoke baselines-smoke evaluate-smoke test week1-audit

PYTHON ?= python
PYTHONPATH := src

environment-check:
	mkdir -p results/compute
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/check_environment.py --config config/model.yaml --json-out results/compute/week1_environment_check.json || test -f results/compute/week1_environment_check.json
	$(PYTHON) -m pip freeze > results/compute/week1_pip_freeze.txt

scgpt-smoke:
	mkdir -p results/compute
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/benchmark_scgpt.py --config config/model.yaml --output results/compute/week1_scgpt_benchmark.json || test -f results/compute/week1_scgpt_benchmark.json

baselines-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/test_baselines.py

evaluate-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/test_evaluation.py

test:
	ruff format --check .
	ruff check .
	mypy src tests scripts
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q src tests scripts
	git diff --check

week1-audit: environment-check scgpt-smoke baselines-smoke evaluate-smoke
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/week1_audit.py