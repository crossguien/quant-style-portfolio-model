from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(equity: pd.Series, returns: pd.Series, bars_per_year: float) -> dict:
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else 0.0
    cagr = (1 + total_ret) ** (bars_per_year / max(len(returns), 1)) - 1 if len(returns) else 0.0

    ann_vol = float(returns.std() * np.sqrt(bars_per_year)) if len(returns) else 0.0
    sharpe = float((returns.mean() * bars_per_year) / (returns.std() * np.sqrt(bars_per_year) + 1e-12)) if len(returns) else 0.0

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    return {
        "total_return": total_ret,
        "cagr": float(cagr),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }
