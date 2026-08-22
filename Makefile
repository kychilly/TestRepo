.PHONY: environment-check scgpt-smoke scgpt-shared-gpu baselines-smoke evaluate-smoke gpu-plan chat-report synthetic-smoke test week1-audit week2-adit stage34-fixture grn-sanity-current audit-current readiness execution-readiness imported-dataset-audit archive-audit mutation-join combine-dataset week3-adit week3-candidates week3-validator-input week3-verdicts week3-experiments repo-safety a100-preflight a100-run adit-week-audit

PYTHON ?= python
PYTHONPATH := src:.

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
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/plan_gpu.py --token-length 1200 --cells 10000 --output results/compute/week3_gpu_plan.json

chat-report:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/write_chat_report.py

synthetic-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/synthetic_smoke.py

test:
	ruff format --check .
	ruff check .
	mypy src tests scripts
	for source_file in ./*.py; do mypy "$$source_file"; done
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q src tests scripts *.py
	git diff --check

week1-audit: environment-check scgpt-smoke baselines-smoke evaluate-smoke
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/week1_audit.py

week2-adit:
	mkdir -p results/week2_adit
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_adit_week2.py --config config/week2_adit.yaml --output results/week2_adit/report.json

stage34-fixture:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_stage34_validation.py --records examples/validator_gate_input.jsonl --candidates results/contracts/tp53/scgpt_candidate_output.jsonl --gold-outcomes examples/validator_gold.synthetic.jsonl --config config/stage34.yaml --seed 17 --output reports/stage34/fixture_feasibility.json

grn-sanity-current:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_grn_sanity_check.py --config config/week2_adit.yaml --train-prior "data/import_20260820/TP53 Dataset(preprocessed)/prior/grn_pilot_train_prior.csv" --held-out "data/import_20260820/TP53 Dataset(preprocessed)/prior/grn_pilot_adit_holdout_check.csv" --output reports/jeffrey_grn_run/grn_sanity_current.json

audit-current: stage34-fixture grn-sanity-current week2-adit
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

readiness:
	PYTHONPATH=src:. $(PYTHON) scripts/readiness_audit.py --pilot "data/import_20260820/TP53 Dataset(preprocessed)/processed/neftel_analysis_cohort.h5ad" --split splits/combined_full_cohort_neftel_patient_splits.json --mutations "data/import_20260820/TP53 Dataset(preprocessed)/pilot/patient_gene_mutation_join.csv" --grn-train "data/import_20260820/TP53 Dataset(preprocessed)/prior/grn_pilot_train_prior.csv" --grn-holdout "data/import_20260820/TP53 Dataset(preprocessed)/prior/grn_pilot_adit_holdout_check.csv" --output reports/readiness/current.json

execution-readiness:
	PYTHONPATH=src:. $(PYTHON) scripts/audit_execution_readiness.py --output reports/readiness/execution_readiness.json

imported-dataset-audit:
	PYTHONPATH=src:. $(PYTHON) scripts/audit_imported_datasets.py --output reports/readiness/imported_dataset_audit.json || test -f reports/readiness/imported_dataset_audit.json

archive-audit:
	PYTHONPATH=src:. $(PYTHON) scripts/audit_dataset_archives.py --output reports/readiness/dataset_archives.json || test -f reports/readiness/dataset_archives.json

mutation-join:
	PYTHONPATH=src:. $(PYTHON) scripts/build_tcga_pilot_mutation_join.py

combine-dataset:
	PYTHONPATH=src:. $(PYTHON) scripts/merge_neftel_states_into_full_cohort.py
	PYTHONPATH=src:. $(PYTHON) scripts/build_combined_dataset_archive.py

week3-adit:
	PYTHONPATH=src:. $(PYTHON) scripts/run_adit_week3.py --model-config config/model.yaml --adit-config config/week2_adit.yaml --output reports/week3_adit/report.json

week3-experiments:
	PYTHONPATH=src:. $(PYTHON) scripts/run_week3_experiments.py --config config/week3_adit.yaml --output reports/week3_adit/experiments

week3-candidates:
	PYTHONPATH=src:. $(PYTHON) scripts/build_internal_candidate_universe.py --adata "data/import_20260820/TP53 Dataset(preprocessed)/processed/neftel_analysis_cohort.h5ad" --output data/pilot/internal_candidate_universe.jsonl

week3-validator-input:
	PYTHONPATH=src:. $(PYTHON) scripts/build_week3_validator_outcomes.py

week3-verdicts:
	PYTHONPATH=src:. $(PYTHON) scripts/build_cross_cohort_verdicts.py

repo-safety:
	$(PYTHON) scripts/install_repo_safety.py

adit-week-audit:
	PYTHONPATH=src:. $(PYTHON) scripts/audit_adit_weeks.py --output reports/adit_week_audit/current.json

a100-preflight:
	test -n "$(A100_CONFIG)"
	test -n "$(GBM_A100_SCRATCH)"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/a100_preflight.py --config "$(A100_CONFIG)" --scratch "$(GBM_A100_SCRATCH)" --output "$(GBM_A100_SCRATCH)/preflight.json"

a100-run:
	test -n "$(A100_CONFIG)"
	test -n "$(GBM_A100_SCRATCH)"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_a100_week3.py --config "$(A100_CONFIG)" --scratch "$(GBM_A100_SCRATCH)"
