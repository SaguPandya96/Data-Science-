from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import resolve_path

TRAIN_REQUIRED = {
    "Store",
    "DayOfWeek",
    "Date",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
}
STORE_REQUIRED = {
    "Store",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2",
    "PromoInterval",
}


class DataValidationError(ValueError):
    """Raised when source data violates the pipeline contract."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str | None) -> None:
    """Fail loudly when a configured source snapshot has changed."""
    if not expected_sha256:
        return
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise DataValidationError(
            f"Checksum mismatch for {path}. Expected {expected_sha256}, found {actual}. "
            "Remove the file or use --force-download only after reviewing the source change."
        )


def download_file(
    url: str,
    destination: Path,
    force: bool = False,
    expected_sha256: str | None = None,
) -> Path:
    """Download and checksum a source file atomically, preserving valid caches."""
    if destination.exists() and not force:
        verify_checksum(destination, expected_sha256)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(
        url,
        timeout=(15, 180),
        stream=True,
        headers={"User-Agent": "store-revenue-forecasting/1.0"},
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    try:
        verify_checksum(temporary, expected_sha256)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {sorted(missing)}")


def validate_inputs(sales: pd.DataFrame, stores: pd.DataFrame) -> None:
    """Validate keys, schemas, dates, and target values before feature creation."""
    _require_columns(sales, TRAIN_REQUIRED, "sales data")
    _require_columns(stores, STORE_REQUIRED, "store data")

    if sales.empty or stores.empty:
        raise DataValidationError("Sales and store inputs must both contain rows.")
    if sales[["Store", "Date"]].isna().any().any():
        raise DataValidationError("Sales data contains missing Store or Date keys.")
    if stores["Store"].isna().any():
        raise DataValidationError("Store data contains missing Store keys.")
    if sales.duplicated(["Store", "Date"]).any():
        raise DataValidationError("Sales data contains duplicate Store/Date observations.")
    if stores["Store"].duplicated().any():
        raise DataValidationError("Store metadata must have exactly one row per Store.")
    if sales["Date"].isna().any():
        raise DataValidationError("Sales data contains unparseable dates.")
    if (sales["Sales"] < 0).any():
        raise DataValidationError("Sales values must be non-negative.")


def read_inputs(train_path: Path, store_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the Rossmann source tables using stable types for categorical fields."""
    sales = pd.read_csv(
        train_path,
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False,
    )
    stores = pd.read_csv(store_path, dtype={"PromoInterval": "string"}, low_memory=False)
    validate_inputs(sales, stores)
    return sales, stores


def load_inputs(
    data_config: dict[str, Any],
    project_root: Path,
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Resolve, optionally download, and read both source tables."""
    train_path = resolve_path(project_root, data_config["train_path"])
    store_path = resolve_path(project_root, data_config["store_path"])
    download_if_missing = bool(data_config.get("download_if_missing", False))

    if force_download or (download_if_missing and not train_path.exists()):
        url = data_config.get("train_url")
        if not url:
            raise FileNotFoundError(f"No train file or train_url is available: {train_path}")
        download_file(
            url,
            train_path,
            force=force_download,
            expected_sha256=data_config.get("train_sha256"),
        )
    if force_download or (download_if_missing and not store_path.exists()):
        url = data_config.get("store_url")
        if not url:
            raise FileNotFoundError(f"No store file or store_url is available: {store_path}")
        download_file(
            url,
            store_path,
            force=force_download,
            expected_sha256=data_config.get("store_sha256"),
        )

    if not train_path.exists() or not store_path.exists():
        raise FileNotFoundError(
            "Source files are missing. Generate sample data or enable download_if_missing. "
            f"Expected {train_path} and {store_path}."
        )

    verify_checksum(train_path, data_config.get("train_sha256"))
    verify_checksum(store_path, data_config.get("store_sha256"))
    sales, stores = read_inputs(train_path, store_path)
    return sales, stores, train_path, store_path


def merge_inputs(sales: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Attach store context to daily outcomes with a many-to-one merge contract."""
    merged = sales.merge(stores, on="Store", how="left", validate="many_to_one")
    if merged["StoreType"].isna().any():
        missing = sorted(merged.loc[merged["StoreType"].isna(), "Store"].unique().tolist())
        raise DataValidationError(f"Store metadata is missing for Store values: {missing[:10]}")
    return merged.sort_values(["Store", "Date"]).reset_index(drop=True)
