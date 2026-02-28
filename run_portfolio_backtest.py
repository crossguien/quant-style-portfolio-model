from __future__ import annotations

import argparse
import json

from src.data import fetch_ltf_htf
from src.config import default_config
from src.portfolio import prepare_symbol, backtest_portfolio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["SPY","NVDA","META","MSFT","AMZN"])
    p.add_argument("--ltf", default="30m")
    p.add_argument("--htf", default="60m")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--data_provider", default="yahoo", choices=["yahoo","alpha_vantage","alpaca","fmp","marketstack"])
    p.add_argument("--api_key", default=None)
    p.add_argument("--api_secret", default=None, help="Needed by alpaca")
    p.add_argument("--risk_weights", default="vol_inverse", choices=["equal","vol_inverse"])
    p.add_argument("--cash_buffer", type=float, default=None)
    p.add_argument("--max_gross", type=float, default=None)
    p.add_argument("--max_net", type=float, default=None)
    p.add_argument("--corr_penalty", type=float, default=None)
    args = p.parse_args()

    cfg = default_config()
    cfg["risk_weights"] = args.risk_weights
    if args.cash_buffer is not None: cfg["cash_buffer"] = args.cash_buffer
    if args.max_gross is not None: cfg["max_gross"] = args.max_gross
    if args.max_net is not None: cfg["max_net"] = args.max_net
    if args.corr_penalty is not None: cfg["corr_penalty"] = args.corr_penalty

    symbol_data = {}
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
        symbol_data[t] = prepare_symbol(ltf, htf, cfg)

    port, met = backtest_portfolio(symbol_data, cfg)
    print(json.dumps(met, indent=2))


if __name__ == "__main__":
    main()
