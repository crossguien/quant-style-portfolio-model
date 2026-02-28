from __future__ import annotations

import argparse
import json

from src.config import default_config
from src.data import fetch_ltf_htf
from src.options.backtest import backtest_options_model, default_options_model_config
from src.portfolio import prepare_symbol


def main():
    p = argparse.ArgumentParser(description="Backtest an options-model MVP (Black-Scholes proxy).")
    p.add_argument("--tickers", nargs="+", default=["SPY", "NVDA", "META", "MSFT", "AMZN"])
    p.add_argument("--ltf", default="30m")
    p.add_argument("--htf", default="60m")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--data_provider", default="yahoo", choices=["yahoo", "alpha_vantage", "alpaca", "fmp", "marketstack"])
    p.add_argument("--api_key", default=None)
    p.add_argument("--api_secret", default=None, help="Needed by alpaca")
    p.add_argument("--risk_weights", default="vol_inverse", choices=["equal", "vol_inverse"])
    p.add_argument("--profile", default="balanced", choices=["balanced", "strict"])

    # Options-model params
    p.add_argument("--option_dte_days", type=int, default=45)
    p.add_argument("--target_otm", type=float, default=0.00)
    p.add_argument("--capital_at_risk", type=float, default=0.15)
    p.add_argument("--fee_bps", type=float, default=8.0)
    p.add_argument("--stop_loss_pct", type=float, default=0.25)
    p.add_argument("--take_profit_pct", type=float, default=0.40)
    p.add_argument("--iv_scale", type=float, default=0.8)
    p.add_argument("--iv_floor", type=float, default=0.10)
    p.add_argument("--iv_cap", type=float, default=1.00)
    p.add_argument("--risk_free_rate", type=float, default=0.04)
    p.add_argument("--min_score_abs", type=float, default=None)
    p.add_argument("--min_adx", type=float, default=None)
    p.add_argument("--side_mode", default=None, choices=["both", "long_only", "short_only"])
    args = p.parse_args()

    cfg = default_config()
    cfg["risk_weights"] = args.risk_weights

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

    opt_cfg = default_options_model_config()
    profile_defaults = {
        "balanced": {"min_score_abs": 0.50, "min_adx": 20.0, "side_mode": "long_only"},
        "strict": {"min_score_abs": 0.55, "min_adx": 25.0, "side_mode": "long_only"},
    }[args.profile]
    opt_cfg.update(
        {
            "option_dte_days": args.option_dte_days,
            "target_otm": args.target_otm,
            "capital_at_risk": args.capital_at_risk,
            "fee_bps": args.fee_bps,
            "stop_loss_pct": args.stop_loss_pct,
            "take_profit_pct": args.take_profit_pct,
            "iv_scale": args.iv_scale,
            "iv_floor": args.iv_floor,
            "iv_cap": args.iv_cap,
            "risk_free_rate": args.risk_free_rate,
            "min_score_abs": args.min_score_abs if args.min_score_abs is not None else profile_defaults["min_score_abs"],
            "min_adx": args.min_adx if args.min_adx is not None else profile_defaults["min_adx"],
            "side_mode": args.side_mode if args.side_mode is not None else profile_defaults["side_mode"],
        }
    )

    _, met = backtest_options_model(symbol_data, cfg, opt_cfg=opt_cfg)
    print(json.dumps(met, indent=2))


if __name__ == "__main__":
    main()
