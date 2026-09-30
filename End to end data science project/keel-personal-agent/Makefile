# Keel developer tasks. Everything except `chat` and `eval-live` runs offline.

PYTHON ?= python

.PHONY: install lint format typecheck test check eval demo app clean

install:  ## Install with dev and app extras
	$(PYTHON) -m pip install -e ".[dev,app]"

lint:  ## Ruff lint and format check
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:  ## Apply Ruff fixes and formatting
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

typecheck:  ## Static type checking
	$(PYTHON) -m mypy src

test:  ## Test suite (offline, scripted model)
	$(PYTHON) -m pytest

check: lint typecheck test  ## Everything CI runs

eval:  ## Memory benchmark; writes reports/metrics/
	$(PYTHON) scripts/run_benchmark.py

demo:  ## Seed a demo profile into ~/.keel/demo.db
	keel --db ~/.keel/demo.db demo

app:  ## Streamlit app on the demo profile
	KEEL_DB=~/.keel/demo.db streamlit run app/app.py

clean:  ## Remove caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
