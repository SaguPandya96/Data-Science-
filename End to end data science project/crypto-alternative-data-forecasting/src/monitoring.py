"""Small production monitoring checks."""
from __future__ import annotations
import pandas as pd

def data_health(df:pd.DataFrame, required:list[str], date_col:str="date")->dict[str,object]:
    dates=pd.to_datetime(df[date_col],utc=True)
    return {"missing_columns":sorted(set(required)-set(df.columns)),"duplicate_dates":int(dates.duplicated().sum()),"stale_days":int((pd.Timestamp.now(tz="UTC").floor("D")-dates.max().floor("D")).days),"missing_fraction":df[required].isna().mean().to_dict() if set(required)<=set(df.columns) else {}}
