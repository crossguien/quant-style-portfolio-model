from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import default_config
from src.data import fetch_ltf_htf
from src.portfolio import (
    _apply_portfolio_constraints,
    _base_risk_weights,
    _corr_penalized_weights,
    prepare_symbol,
)
from src.risk import size_vol_target


def side_from_pos(pos: int) -> str:
    if pos > 0:
        return "long"
    if pos < 0:
        return "short"
    return "flat"


def action_from_pos(pos: int) -> str:
    if pos > 0:
        return "enter_or_hold_long"
    if pos < 0:
        return "enter_or_hold_short"
    return "stay_flat_or_exit"


def signal_quality(score: float, pos: int, entry_thr: float) -> str:
    if pos == 0:
        if abs(score) >= 0.9 * entry_thr:
            return "watchlist_near_trigger"
        return "neutral"
    ratio = abs(score) / max(entry_thr, 1e-9)
    if ratio >= 1.5:
        return "strong"
    if ratio >= 1.15:
        return "moderate"
    return "weak"


def stop_levels(close: float, atr: float, pos: int, cfg: dict) -> dict:
    if pos == 0 or pd.isna(atr) or atr <= 0:
        return {"stop": None, "take_profit": None, "trail_hint": None}

    stop_mult = float(cfg["stop_atr"])
    tp_mult = float(cfg["tp_atr"])
    trail_mult = float(cfg["trail_atr"])

    if pos > 0:
        return {
            "stop": close - stop_mult * atr,
            "take_profit": close + tp_mult * atr,
            "trail_hint": close - trail_mult * atr,
        }
    return {
        "stop": close + stop_mult * atr,
        "take_profit": close - tp_mult * atr,
        "trail_hint": close + trail_mult * atr,
    }


def main():
    p = argparse.ArgumentParser(description="Generate daily trade plan from latest model signals.")
    p.add_argument("--tickers", nargs="+", default=["SPY", "NVDA", "META", "MSFT", "AMZN"])
    p.add_argument("--ltf", default="30m")
    p.add_argument("--htf", default="60m")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--data_provider", default="yahoo", choices=["yahoo","alpha_vantage","alpaca","fmp","marketstack"])
    p.add_argument("--api_key", default=None)
    p.add_argument("--api_secret", default=None, help="Needed by alpaca")
    p.add_argument("--risk_weights", default="vol_inverse", choices=["equal", "vol_inverse"])
    p.add_argument("--cash_buffer", type=float, default=None)
    p.add_argument("--max_gross", type=float, default=None)
    p.add_argument("--max_net", type=float, default=None)
    p.add_argument("--corr_penalty", type=float, default=None)
    p.add_argument("--out", default=None, help="Optional path to write JSON plan.")
    args = p.parse_args()

    cfg = default_config()
    cfg["risk_weights"] = args.risk_weights
    if args.cash_buffer is not None:
        cfg["cash_buffer"] = args.cash_buffer
    if args.max_gross is not None:
        cfg["max_gross"] = args.max_gross
    if args.max_net is not None:
        cfg["max_net"] = args.max_net
    if args.corr_penalty is not None:
        cfg["corr_penalty"] = args.corr_penalty

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
        sym = prepare_symbol(ltf, htf, cfg)
        if sym.empty:
            raise RuntimeError(
                f"No usable feature rows for {t}. Increase --lookback or use a lower timeframe with more bars."
            )
        symbol_data[t] = sym

    base_w = _base_risk_weights(symbol_data, cfg)
    risk_w = _corr_penalized_weights(symbol_data, base_w, cfg)

    raw_exposure = {}
    symbol_rows = {}
    for t, df in symbol_data.items():
        row = df.iloc[-1]
        pos = int(row["target_pos"])
        size_raw = float(size_vol_target(df["atr_pct"], cfg).iloc[-1])
        raw_exp = float(risk_w[t] * size_raw * pos)

        symbol_rows[t] = {
            "timestamp": str(df.index[-1]),
            "close": float(row["close"]),
            "score": float(row["score"]),
            "target_pos": pos,
            "side": side_from_pos(pos),
            "action": action_from_pos(pos),
            "signal_quality": signal_quality(float(row["score"]), pos, float(cfg["entry_thr"])),
            "risk_weight": float(risk_w[t]),
            "size_raw": size_raw,
            "raw_exposure": raw_exp,
            "atr_pct": float(row["atr_pct"]),
            "adx": float(row["adx"]),
            "htf_bull": bool(row["htf_bull"]),
            "htf_bear": bool(row["htf_bear"]),
            "stops": stop_levels(float(row["close"]), float(row["atr"]), pos, cfg),
        }
        raw_exposure[t] = raw_exp

    adjusted = _apply_portfolio_constraints(pd.Series(raw_exposure), cfg)
    for t in symbol_rows:
        symbol_rows[t]["adj_exposure"] = float(adjusted[t])

    gross = float(adjusted.abs().sum())
    net = float(adjusted.sum())
    longs = sum(1 for s in symbol_rows.values() if s["target_pos"] > 0)
    shorts = sum(1 for s in symbol_rows.values() if s["target_pos"] < 0)
    flats = sum(1 for s in symbol_rows.values() if s["target_pos"] == 0)

    plan = {
        "config": {
            "tickers": args.tickers,
            "ltf": args.ltf,
            "htf": args.htf,
            "lookback": args.lookback,
            "data_provider": args.data_provider,
            "risk_weights": cfg["risk_weights"],
            "cash_buffer": cfg["cash_buffer"],
            "max_gross": cfg["max_gross"],
            "max_net": cfg["max_net"],
            "corr_penalty": cfg["corr_penalty"],
        },
        "portfolio_summary": {
            "gross_exposure": gross,
            "net_exposure": net,
            "n_longs": longs,
            "n_shorts": shorts,
            "n_flats": flats,
        },
        "symbols": symbol_rows,
    }

    text = json.dumps(plan, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
