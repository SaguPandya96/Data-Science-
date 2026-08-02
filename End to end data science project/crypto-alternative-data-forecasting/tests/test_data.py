import pandas as pd
from src.data_cleaning import clean_market

def test_invalid_market_rows_removed():
    x=pd.DataFrame({"date":["2024-01-01","2024-01-01","2024-01-02"],"open":[1,1,-1],"high":[2,2,2],"low":[.5,.5,.5],"close":[1.5,1.5,1],"volume":[1,1,1]})
    out=clean_market(x); assert len(out)==1 and not out.date.duplicated().any()
