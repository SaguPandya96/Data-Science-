.PHONY: install test run lab clean

install:
	pip install -r requirements.txt

test:
	pytest -q

run:
	python scripts/run_experiment.py --config config.yaml

lab:
	jupyter lab notebooks/01_evasion_gap.ipynb

clean:
	rm -rf results/*.csv results/*.json results/*.png
