import pandas as pd
from src.features import merge_point_in_time
from src.models import chronological_splits
from src.evaluation import moving_block_accuracy_interval

def test_timestamp_and_split_ordering():
    m=pd.DataFrame({"date":pd.date_range("2024-01-01",periods=20,tz="UTC")}); n=pd.DataFrame({"date":m.date,"sentiment_mean":range(20)})
    out=merge_point_in_time(m,n); assert (out.feature_timestamp<out.target_timestamp).all(); assert out.loc[1,"sentiment_mean"]==0
    tr,va,te=chronological_splits(20); assert max(tr)<min(va)<max(va)<min(te)

def test_target_not_feature():
    features=["return_lag_1","sentiment_mean"]; assert "target" not in features and "next_day_return" not in features

def test_block_interval_is_ordered():
    low,high=moving_block_accuracy_interval([0,1,1,0,1,0,1,1],[0,1,0,0,1,1,1,0],block_size=2,repetitions=100)
    assert 0 <= low <= high <= 1
