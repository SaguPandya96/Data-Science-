.PHONY: install download-data analyze test lint

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

download-data:
	python scripts/download_data.py

analyze:
	python scripts/run_analysis.py

test:
	python -m pytest

lint:
	python -m ruff check .
