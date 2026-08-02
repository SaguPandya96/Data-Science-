"""Long/cash simulation with explicit signal timing and trading frictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(
    dates: pd.Series,
    probabilities: np.ndarray,
    realised_returns: pd.Series,
    threshold: float = 0.55,
    cost_bps: float = 10,
    slippage_bps: float = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply yesterday's forecast to today's realised close-to-close return.

    A probability recorded for date *t* forecasts the return ending on *t+1*.
    Shifting the signal once therefore aligns it with the realised daily return
    on *t+1*. Passing ``next_day_return`` here would add an unintended extra day.
    """
    output = pd.DataFrame(
        {
            "date": pd.Series(pd.to_datetime(dates)).reset_index(drop=True),
            "probability": np.asarray(probabilities),
            "realised_return": pd.Series(realised_returns).reset_index(drop=True),
        }
    )
    output["signal"] = (output["probability"] > threshold).astype(float)
    output["position"] = output["signal"].shift(1).fillna(0.0)
    output["turnover"] = output["position"].diff().abs().fillna(
        output["position"].abs()
    )

    friction_rate = (cost_bps + slippage_bps) / 10_000
    output["cost"] = output["turnover"] * friction_rate
    output["strategy_return"] = (
        output["position"] * output["realised_return"] - output["cost"]
    )
    output["strategy_equity"] = (1 + output["strategy_return"]).cumprod()
    output["buy_hold_equity"] = (1 + output["realised_return"]).cumprod()

    n_days = max(len(output), 1)
    cumulative_return = output["strategy_equity"].iloc[-1] - 1
    annualised_return = output["strategy_equity"].iloc[-1] ** (365 / n_days) - 1
    annualised_volatility = output["strategy_return"].std() * np.sqrt(365)
    peak = output["strategy_equity"].cummax()
    active = output["position"] > 0

    summary = pd.Series(
        {
            "cumulative_return": cumulative_return,
            "annualized_return": annualised_return,
            "annualized_volatility": annualised_volatility,
            "sharpe_ratio": (
                annualised_return / annualised_volatility
                if annualised_volatility > 0
                else np.nan
            ),
            "maximum_drawdown": (output["strategy_equity"] / peak - 1).min(),
            "hit_rate": (output.loc[active, "strategy_return"] > 0).mean(),
            "turnover": output["turnover"].sum(),
            "transaction_costs": output["cost"].sum(),
            "number_of_trades": int((output["turnover"] > 0).sum()),
            "average_active_day_return": output.loc[active, "strategy_return"].mean(),
        }
    )
    return output, summary
