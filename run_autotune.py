from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass

from src.config import default_config
from src.data import fetch_ltf_htf
from src.portfolio import backtest_portfolio, prepare_symbol


@dataclass(frozen=True)
class SearchSpace:
    risk_weights: tuple[str, ...]
    entry_thr: tuple[float, ...]
    adx_min: tuple[float, ...]
    require_adx: tuple[bool, ...]
    stop_atr: tuple[float, ...]
    tp_atr: tuple[float, ...]
    trail_atr: tuple[float, ...]
    cash_buffer: tuple[float, ...]
    max_gross: tuple[float, ...]
    max_net: tuple[float, ...]
    corr_penalty: tuple[float, ...]


def parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in value.split(",") if x.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def parse_csv_bools(value: str) -> tuple[bool, ...]:
    out = []
    for x in value.split(","):
        v = x.strip().lower()
        if not v:
            continue
        if v in {"true", "t", "1", "yes", "y"}:
            out.append(True)
        elif v in {"false", "f", "0", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid bool token: {x}")
    return tuple(out)


def bars_per_year_from_ltf(ltf: str) -> float:
    if ltf.endswith("m"):
        try:
            mins = int(ltf[:-1])
        except ValueError:
            mins = 5
        bars_per_day = max(1, int((6.5 * 60) // mins))
        return float(bars_per_day * 252)
    return float(252)


def random_candidate(space: SearchSpace, rng: random.Random) -> dict:
    return {
        "risk_weights": rng.choice(space.risk_weights),
        "entry_thr": rng.choice(space.entry_thr),
        "adx_min": rng.choice(space.adx_min),
        "require_adx": rng.choice(space.require_adx),
        "stop_atr": rng.choice(space.stop_atr),
        "tp_atr": rng.choice(space.tp_atr),
        "trail_atr": rng.choice(space.trail_atr),
        "cash_buffer": rng.choice(space.cash_buffer),
        "max_gross": rng.choice(space.max_gross),
        "max_net": rng.choice(space.max_net),
        "corr_penalty": rng.choice(space.corr_penalty),
    }


def candidate_key(c: dict) -> tuple:
    return (
        c["risk_weights"],
        c["entry_thr"],
        c["adx_min"],
        c["require_adx"],
        c["stop_atr"],
        c["tp_atr"],
        c["trail_atr"],
        c["cash_buffer"],
        c["max_gross"],
        c["max_net"],
        c["corr_penalty"],
    )


def score_tuple(metrics: dict, objective: str) -> tuple[float, float]:
    if objective == "sharpe":
        return float(metrics["sharpe"]), float(metrics["total_return"])
    if objective == "cagr":
        return float(metrics["cagr"]), float(metrics["sharpe"])
    return float(metrics["total_return"]), float(metrics["sharpe"])


def main():
    p = argparse.ArgumentParser(description="Auto-tune strategy parameters using random search.")
    p.add_argument("--tickers", nargs="+", default=["SPY", "NVDA", "META", "MSFT", "AMZN"])
    p.add_argument("--ltf", default="30m")
    p.add_argument("--htf", default="60m")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--data_provider", default="yahoo", choices=["yahoo","alpha_vantage","alpaca","fmp","marketstack"])
    p.add_argument("--api_key", default=None)
    p.add_argument("--api_secret", default=None, help="Needed by alpaca")
    p.add_argument("--max_evals", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--objective", choices=["ret_then_sharpe", "sharpe", "cagr"], default="ret_then_sharpe")
    p.add_argument("--out", default=None, help="Optional path to write JSON results.")

    p.add_argument("--risk_weights", default="equal,vol_inverse")
    p.add_argument("--entry_thr", default="0.40,0.45,0.50,0.55")
    p.add_argument("--adx_min", default="12,15,18")
    p.add_argument("--require_adx", default="true,false")
    p.add_argument("--stop_atr", default="1.5,2.0,2.5")
    p.add_argument("--tp_atr", default="2.0,3.0,4.0")
    p.add_argument("--trail_atr", default="2.0,2.5,3.0")
    p.add_argument("--cash_buffer", default="0.05,0.10")
    p.add_argument("--max_gross", default="1.5,1.8")
    p.add_argument("--max_net", default="0.6,0.8")
    p.add_argument("--corr_penalty", default="0.3,0.5")
    args = p.parse_args()

    space = SearchSpace(
        risk_weights=parse_csv_strings(args.risk_weights),
        entry_thr=parse_csv_floats(args.entry_thr),
        adx_min=parse_csv_floats(args.adx_min),
        require_adx=parse_csv_bools(args.require_adx),
        stop_atr=parse_csv_floats(args.stop_atr),
        tp_atr=parse_csv_floats(args.tp_atr),
        trail_atr=parse_csv_floats(args.trail_atr),
        cash_buffer=parse_csv_floats(args.cash_buffer),
        max_gross=parse_csv_floats(args.max_gross),
        max_net=parse_csv_floats(args.max_net),
        corr_penalty=parse_csv_floats(args.corr_penalty),
    )

    rng = random.Random(args.seed)
    base_cfg = default_config()
    base_cfg["bars_per_year"] = bars_per_year_from_ltf(args.ltf)

    raw = {}
    for t in args.tickers:
        raw[t] = fetch_ltf_htf(
            t,
            args.ltf,
            args.htf,
            args.lookback,
            provider=args.data_provider,
            api_key=args.api_key,
            api_secret=args.api_secret,
        )

    tried = set()
    results = []

    for _ in range(max(args.max_evals, 1)):
        c = random_candidate(space, rng)
        k = candidate_key(c)
        if k in tried:
            continue
        tried.add(k)

        cfg = dict(base_cfg)
        cfg.update(c)
        symbol_data = {}
        for t in args.tickers:
            ltf, htf = raw[t]
            symbol_data[t] = prepare_symbol(ltf, htf, cfg)

        _, met = backtest_portfolio(symbol_data, cfg)
        if "portfolio" not in met:
            continue
        pm = met["portfolio"]
        results.append(
            {
                "score": score_tuple(pm, args.objective),
                "portfolio": pm,
                "params": c,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[: max(args.top_k, 1)]
    best = top[0] if top else None

    payload = {
        "objective": args.objective,
        "evaluated": len(results),
        "search_space": asdict(space),
        "tickers": args.tickers,
        "ltf": args.ltf,
        "htf": args.htf,
        "lookback": args.lookback,
        "data_provider": args.data_provider,
        "best": best,
        "top": top,
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
