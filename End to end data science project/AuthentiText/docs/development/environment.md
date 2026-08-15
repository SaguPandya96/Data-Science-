# Development environment

## Supported baseline

The first modeling environment supports CPython 3.12 through 3.14. It was
resolved and validated on Windows with CPython 3.14.6. Current package versions
were checked against official PyPI metadata on 2026-08-08; the numeric stack
had native CPython 3.14 Windows wheels, so no source builds were used.

The runtime lock contains only the packages needed for the CPU-based
scikit-learn baseline and FastAPI service plus their resolved transitive
dependencies. Ruff and the HTTPX2 API test client are development-only. Web,
deep-learning, experiment-tracking, and deployment dependencies will be added
only when their phases begin.

## Setup on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip==26.1.2
.\.venv\Scripts\python.exe -m pip install -r requirements\dev.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

The exact runtime graph is recorded in `requirements/runtime.lock`; the
development lock includes it and adds Ruff. `pyproject.toml` separately records
the direct project dependencies and build metadata.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

The isolated Windows environment used for initial validation exposed a CPython
3.14 `tempfile` ACL incompatibility during `ensurepip`. Installation was
validated in the same fresh `.venv` with a temporary-directory shim scoped to
the bootstrap process. This was an environment-specific constraint; the
versioned setup remains the standard `venv` and `pip` workflow shown above.
The [clean-room reproduction audit](clean_room_reproduction.md) used the same
bounded workaround before the locked install, full verification matrix, and
ordinary wheel build passed.
