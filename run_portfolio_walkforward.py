from __future__ import annotations

import argparse
import json
import re

from src.data import fetch_ltf_htf
from src.config import default_config
from src.portfolio import prepare_symbol
from src.walkforward import walk_forward_portfolio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["SPY","NVDA","META","MSFT","AMZN"])
    p.add_argument("--ltf", default="30m")
    p.add_argument("--htf", default="60m")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--data_provider", default="yahoo", choices=["yahoo","alpha_vantage","alpaca","fmp","marketstack"])
    p.add_argument("--api_key", default=None)
    p.add_argument("--api_secret", default=None, help="Needed by alpaca")
    p.add_argument("--train_days", type=int, default=20)
    p.add_argument("--test_days", type=int, default=5)
    p.add_argument("--risk_weights", default="vol_inverse", choices=["equal","vol_inverse"])
    args = p.parse_args()

    cfg = default_config()
    cfg["risk_weights"] = args.risk_weights

    full = {}
    for t in args.tickers:
        ltf, htf = fetch_ltf_htf(
            t,
            args.ltf,
            args.htf,
            args.lookback,
            provider=args.data_provider,
            api_key=args.api_key,
            api_secret=args.api_secret,
        )
        full[t] = prepare_symbol(ltf, htf, cfg)

    # common index
    common = None
    for df in full.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) < 300:
        print(json.dumps({"error": "not enough common bars"}, indent=2))
        return

    for t in full:
        full[t] = full[t].reindex(common).dropna()

    if args.ltf.endswith("m"):
        m = re.match(r"^(\d+)m$", args.ltf)
        mins = int(m.group(1)) if m else 5
        bars_per_day = max(1, int((6.5 * 60) // mins))
    else:
        bars_per_day = 1
    train_bars = int(args.train_days * bars_per_day)
    test_bars = int(args.test_days * bars_per_day)

    idx = full[args.tickers[0]].index
    n = len(idx)
    i = 0
    windows = []
    while i + train_bars + test_bars <= n:
        test_idx = idx[i+train_bars:i+train_bars+test_bars]
        windows.append({t: full[t].loc[test_idx].copy() for t in full})
        i += test_bars

    res = walk_forward_portfolio(windows, cfg)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
