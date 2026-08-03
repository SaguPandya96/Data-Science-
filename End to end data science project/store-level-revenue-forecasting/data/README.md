# Data

## Source

The full pipeline uses the Rossmann Store Sales tables referenced by the original notebook:

- `train.csv`: daily store outcomes and known operating/calendar fields;
- `store.csv`: one row of store format, competition, and extended-promotion metadata per store.

The configured URLs point to a community GitHub mirror of the Kaggle dataset. The mirror is suitable for a portfolio reproduction but is not an approved production source. Raw downloads are cached in `data/raw/`, excluded from Git, and verified against the SHA-256 values in `config.yaml` before every read. An organizational deployment should still use a governed snapshot with source approval, license review, retention policy, and access controls.

## Directories

- `raw/`: downloaded source files;
- `sample/`: synthetic Rossmann-shaped history, store metadata, and next-day plan created by `scripts/generate_sample_data.py`;
- `processed/`: generated holdout predictions.

The sample data contains no real store activity. It exists only to exercise the code path quickly and must not be used for performance claims.

See [../docs/DATA_DICTIONARY.md](../docs/DATA_DICTIONARY.md) for field definitions, [../docs/FORECASTING_CONTRACT.md](../docs/FORECASTING_CONTRACT.md) for timing assumptions, and [../docs/SCORING_CONTRACT.md](../docs/SCORING_CONTRACT.md) for the future-plan interface.
