import pandas as pd
from src.features import build_market_features

def test_target_and_rolling_are_causal():
    x=pd.DataFrame({"date":pd.date_range("2024-01-01",periods=40,tz="UTC"),"open":range(1,41),"high":range(2,42),"low":range(1,41),"close":range(1,41),"volume":[10]*40})
    out=build_market_features(x); assert out.loc[0,"target"]==1; assert pd.isna(out.loc[39,"target"]); assert pd.isna(out.loc[6,"rolling_mean_7"])
