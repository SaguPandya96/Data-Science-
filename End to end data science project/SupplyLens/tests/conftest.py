"""Shared fixtures backed by the downloaded public dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from supplylens.config import load_config, resolve_path
from supplylens.data import clean_shipments, read_raw_data


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def raw_path(config: dict) -> Path:
    path = resolve_path(config["data"]["raw_path"])
    if not path.exists():
        pytest.fail("Public raw data is missing; run `python scripts/download_data.py` first")
    return path


@pytest.fixture(scope="session")
def raw_data(raw_path: Path) -> pd.DataFrame:
    return read_raw_data(raw_path)


@pytest.fixture(scope="session")
def shipments(raw_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    return clean_shipments(raw_data, int(config["target"]["delay_days_threshold"]))
