.PHONY: install test lint results demo clean

install:        ## install Python dependencies
	pip install -r requirements.txt

test:           ## run the smoke test suite
	pytest tests/ -q

lint:           ## fast error-only lint (syntax + undefined names)
	ruff check --select E9,F63,F7,F82 src scripts tests

results:        ## regenerate results/ tables from committed raw logs
	python scripts/build_results_table.py

# End-to-end demo: baselines + LLM-NAS on Jannis (OpenML 41168), then the table.
# Requires OPENAI_API_KEY for the LLM step.
demo:
	python scripts/run_baselines.py --openml_id 41168 --out_dir outputs/jannis_baselines
	python scripts/run_llm_nas.py  --openml_id 41168 --out_dir outputs/jannis_llm --budget 40
	python scripts/build_results_table.py

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf outputs/ catboost_info/
