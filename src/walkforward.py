from __future__ import annotations

import pandas as pd
from .portfolio import backtest_portfolio


def walk_forward_portfolio(windows: list[dict[str, pd.DataFrame]], cfg: dict) -> dict:
    results = []
    for i, symdfs in enumerate(windows):
        port, met = backtest_portfolio(symdfs, cfg)
        if "error" in met:
            continue
        m = met["portfolio"]
        m["window"] = i
        m["start"] = str(port.index[0]) if len(port) else ""
        m["end"] = str(port.index[-1]) if len(port) else ""
        results.append(m)

    if not results:
        return {"error": "no windows produced results"}

    res = pd.DataFrame(results)
    return {
        "windows": int(res.shape[0]),
        "avg_sharpe": float(res["sharpe"].mean()),
        "avg_cagr": float(res["cagr"].mean()),
        "avg_max_dd": float(res["max_drawdown"].mean()),
        "per_window": results,
    }
