.PHONY: environment-check scgpt-smoke scgpt-shared-gpu baselines-smoke evaluate-smoke gpu-plan chat-report synthetic-smoke test week1-audit week2-adit

PYTHON ?= python
PYTHONPATH := src

environment-check:
	mkdir -p results/compute
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/check_environment.py --config config/model.yaml --json-out results/compute/week1_environment_check.json || test -f results/compute/week1_environment_check.json
	$(PYTHON) -m pip freeze > results/compute/week1_pip_freeze.txt

scgpt-smoke:
	mkdir -p results/compute
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/benchmark_scgpt.py --config config/model.yaml --output results/compute/week1_scgpt_benchmark.json || test -f results/compute/week1_scgpt_benchmark.json

scgpt-shared-gpu:
	mkdir -p results/compute
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/benchmark_scgpt.py --config config/model_shared_gpu.yaml --output results/compute/shared_gpu_scgpt_benchmark.json

baselines-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/test_baselines.py

evaluate-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/test_evaluation.py

gpu-plan:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/plan_gpu.py --token-length 2048 --cells 10000 --output results/compute/week3_gpu_plan.json

chat-report:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/write_chat_report.py

synthetic-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/synthetic_smoke.py

test:
	ruff format --check .
	ruff check .
	mypy src tests scripts
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q src tests scripts
	git diff --check

week1-audit: environment-check scgpt-smoke baselines-smoke evaluate-smoke
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/week1_audit.py

week2-adit:
	mkdir -p results/week2_adit
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_adit_week2.py --config config/week2_adit.yaml --output results/week2_adit/report.json
