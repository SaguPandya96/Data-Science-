# EvalForge developer tasks.
# Every target runs fully offline with the deterministic mock provider.

PYTHON ?= python
PKG    := src/evalforge

.PHONY: help install install-dev lint format typecheck test test-unit test-integration \
        test-regression cov check generate validate demo dashboard annotate clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package (runtime only)
	$(PYTHON) -m pip install -e .

install-dev:  ## Install with dev, dashboard and provider extras
	$(PYTHON) -m pip install -e ".[dev,dashboard]"

lint:  ## Ruff lint + format check
	ruff check .
	ruff format --check .

format:  ## Apply Ruff formatting and autofixes
	ruff check --fix .
	ruff format .

typecheck:  ## Static type checking
	mypy src

test:  ## Full test suite
	pytest -q

test-unit:  ## Unit tests only
	pytest -q tests/unit

test-integration:  ## Integration tests only
	pytest -q tests/integration

test-regression:  ## Golden-fixture regression tests only
	pytest -q tests/regression

cov:  ## Test suite with coverage report
	pytest -q --cov=evalforge --cov-report=term-missing --cov-report=xml

check: lint typecheck test  ## Everything CI runs

generate:  ## Generate the standard 150-scenario suite
	evalforge generate --count 150 --seed 42
	evalforge validate-scenarios

demo:  ## Full offline demonstration (baseline + candidate + reports)
	evalforge demo

dashboard:  ## Launch the Streamlit dashboard
	evalforge dashboard

annotate:  ## Launch the human annotation interface
	evalforge annotate

clean:  ## Remove caches and regenerable artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
