from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from src.config import default_config
from src.data import fetch_ltf_htf
from src.portfolio import prepare_symbol


def _dte(expiry: str) -> int:
    exp = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc).date()
    now = datetime.now(timezone.utc).date()
    return (exp - now).days


def _pick_option_contract(
    ticker: str,
    side: str,
    spot: float,
    min_dte: int,
    max_dte: int,
    target_moneyness: float,
    min_oi: int,
    min_volume: int,
    max_spread_pct: float,
):
    tk = yf.Ticker(ticker)
    expiries = [e for e in tk.options if min_dte <= _dte(e) <= max_dte]
    if not expiries:
        return None, "no_expiry_in_dte_range"

    best = None
    target_strike = spot * (1.0 + target_moneyness if side == "long" else 1.0 - target_moneyness)

    for exp in expiries:
        chain = tk.option_chain(exp)
        table = chain.calls if side == "long" else chain.puts
        if table is None or table.empty:
            continue

        df = table.copy()
        if side == "long":
            df = df[df["strike"] >= spot]
        else:
            df = df[df["strike"] <= spot]

        for col in ["bid", "ask", "lastPrice", "strike", "openInterest", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["bid", "ask", "strike"])
        df = df[(df["ask"] > 0) & (df["bid"] >= 0)]
        if "openInterest" in df.columns:
            df = df[df["openInterest"].fillna(0) >= min_oi]
        if "volume" in df.columns:
            df = df[df["volume"].fillna(0) >= min_volume]
        if df.empty:
            continue

        df["mid"] = (df["bid"] + df["ask"]) / 2.0
        df = df[df["mid"] > 0]
        if df.empty:
            continue

        df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
        df = df[df["spread_pct"] <= max_spread_pct]
        if df.empty:
            continue

        df["dist"] = (df["strike"] - target_strike).abs()
        df = df.sort_values(["dist", "spread_pct", "openInterest"], ascending=[True, True, False])
        row = df.iloc[0]
        candidate = {
            "ticker": ticker,
            "expiry": exp,
            "dte": _dte(exp),
            "contract_symbol": row.get("contractSymbol"),
            "side": "call_buy" if side == "long" else "put_buy",
            "strike": float(row["strike"]),
            "bid": float(row["bid"]),
            "ask": float(row["ask"]),
            "mid": float(row["mid"]),
            "spread_pct": float(row["spread_pct"]),
            "last_price": float(row.get("lastPrice", 0.0)),
            "implied_volatility": float(row.get("impliedVolatility", 0.0)),
            "open_interest": int(row.get("openInterest", 0) or 0),
            "volume": int(row.get("volume", 0) or 0),
        }
        if best is None or (candidate["dte"], candidate["spread_pct"]) < (best["dte"], best["spread_pct"]):
            best = candidate

    if best is None:
        return None, "no_contract_passed_filters"
    return best, None


def main():
    p = argparse.ArgumentParser(description="Build an options trade plan from stock model direction.")
    p.add_argument("--tickers", nargs="+", default=["SPY", "NVDA", "META", "MSFT", "AMZN"])
    p.add_argument("--ltf", default="30m")
    p.add_argument("--htf", default="60m")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--data_provider", default="yahoo", choices=["yahoo", "alpha_vantage", "alpaca", "fmp", "marketstack"])
    p.add_argument("--api_key", default=None)
    p.add_argument("--api_secret", default=None, help="Needed by alpaca")
    p.add_argument("--risk_weights", default="vol_inverse", choices=["equal", "vol_inverse"])
    p.add_argument("--min_dte", type=int, default=14)
    p.add_argument("--max_dte", type=int, default=45)
    p.add_argument("--target_moneyness", type=float, default=0.02, help="0.02 means ~2% OTM target strike.")
    p.add_argument("--min_oi", type=int, default=100)
    p.add_argument("--min_volume", type=int, default=10)
    p.add_argument("--max_spread_pct", type=float, default=0.25)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = default_config()
    cfg["risk_weights"] = args.risk_weights

    results = {}
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
        sig = prepare_symbol(ltf, htf, cfg)
        if sig.empty:
            results[t] = {"action": "skip", "reason": "no_signal_rows"}
            continue

        row = sig.iloc[-1]
        pos = int(row["target_pos"])
        side = "long" if pos > 0 else "short" if pos < 0 else "flat"
        base = {
            "timestamp": str(sig.index[-1]),
            "spot": float(row["close"]),
            "score": float(row["score"]),
            "target_pos": pos,
            "side": side,
        }
        if side == "flat":
            results[t] = {**base, "action": "no_options_trade", "reason": "flat_stock_signal"}
            continue

        contract, err = _pick_option_contract(
            ticker=t,
            side=side,
            spot=float(row["close"]),
            min_dte=args.min_dte,
            max_dte=args.max_dte,
            target_moneyness=args.target_moneyness,
            min_oi=args.min_oi,
            min_volume=args.min_volume,
            max_spread_pct=args.max_spread_pct,
        )
        if err:
            results[t] = {**base, "action": "no_options_trade", "reason": err}
            continue
        results[t] = {**base, "action": "candidate_contract", "contract": contract}

    payload = {
        "note": "This is an options idea generator from stock-direction signals. It is not an execution engine.",
        "config": {
            "tickers": args.tickers,
            "ltf": args.ltf,
            "htf": args.htf,
            "lookback": args.lookback,
            "data_provider": args.data_provider,
            "risk_weights": args.risk_weights,
            "min_dte": args.min_dte,
            "max_dte": args.max_dte,
            "target_moneyness": args.target_moneyness,
            "min_oi": args.min_oi,
            "min_volume": args.min_volume,
            "max_spread_pct": args.max_spread_pct,
        },
        "symbols": results,
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
