.PHONY: install download-data analyze figures test lint

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

download-data:
	python scripts/download_data.py

analyze:
	python scripts/run_analysis.py

figures:
	python scripts/build_figures.py

test:
	python -m pytest

lint:
	python -m ruff check .
