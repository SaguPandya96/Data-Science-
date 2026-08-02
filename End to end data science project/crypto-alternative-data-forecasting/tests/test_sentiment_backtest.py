import pandas as pd
from src.sentiment import aggregate_daily
from src.backtesting import backtest

def test_sentiment_aggregation():
    x=pd.DataFrame({"feature_date":[pd.Timestamp("2024-01-01",tz="UTC")]*2,"headline":["a","b"],"sentiment_label":["positive","negative"],"compound_sentiment":[.5,-.5],"positive_probability":[.7,.1],"negative_probability":[.1,.8]})
    out=aggregate_daily(x); assert out.article_volume.iloc[0]==2 and out.positive_count.iloc[0]==1

def test_execution_lag():
    out,_=backtest(pd.date_range("2024-01-01",periods=3),[.9,.1,.1],[.1,.1,.1],threshold=.5,cost_bps=0,slippage_bps=0)
    assert out.position.tolist()==[0,1,0]
    assert out.strategy_return.tolist()==[0,.1,0]
