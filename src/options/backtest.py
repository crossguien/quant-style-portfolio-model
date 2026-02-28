from __future__ import annotations

import math

import pandas as pd

from src.metrics import compute_metrics
from src.options.black_scholes import bs_price


def default_options_model_config() -> dict:
    return {
        "option_dte_days": 45,
        "target_otm": 0.00,
        "risk_free_rate": 0.04,
        "iv_scale": 0.8,
        "iv_floor": 0.10,
        "iv_cap": 1.00,
        "capital_at_risk": 0.15,      # total portfolio premium budget across symbols
        "fee_bps": 8.0,               # per entry/exit on premium notional
        "stop_loss_pct": 0.25,        # exit if premium down 25% from entry
        "take_profit_pct": 0.40,      # exit if premium up 40% from entry
        "exit_on_signal_flip": True,
        # Entry filters (to reduce low-conviction option trades)
        "min_score_abs": 0.50,
        "min_adx": 20.0,
        "side_mode": "long_only",     # "both", "long_only", "short_only"
    }


def _annual_vol_from_atr_pct(atr_pct: float, bars_per_year: float, iv_scale: float, iv_floor: float, iv_cap: float) -> float:
    # ATR%-based rough annualized vol proxy (MVP; replace with IV surface in production).
    v = float(atr_pct) * math.sqrt(max(float(bars_per_year), 1.0)) * float(iv_scale)
    return min(max(v, float(iv_floor)), float(iv_cap))


def _option_mark(spot: float, strike: float, t_years: float, vol: float, rate: float, side: int) -> float:
    is_call = side > 0
    return max(bs_price(spot, strike, t_years, rate, vol, is_call), 0.01)


def backtest_options_model(symbol_data: dict[str, pd.DataFrame], cfg: dict, opt_cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    if opt_cfg is None:
        opt_cfg = default_options_model_config()
    else:
        tmp = default_options_model_config()
        tmp.update(opt_cfg)
        opt_cfg = tmp

    common = None
    for df in symbol_data.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) < 50:
        return pd.DataFrame(), {"error": "not enough common timestamps"}

    aligned = {s: symbol_data[s].reindex(common).dropna().copy() for s in symbol_data}
    syms = [s for s in aligned if not aligned[s].empty]
    if not syms:
        return pd.DataFrame(), {"error": "no aligned symbol data"}

    n_syms = len(syms)
    cap_total = float(opt_cfg["capital_at_risk"])
    fee = float(opt_cfg["fee_bps"]) / 10000.0
    rf = float(opt_cfg["risk_free_rate"])
    bars_per_year = float(cfg.get("bars_per_year", 252.0))
    dt_years = 1.0 / max(bars_per_year, 1.0)
    dte_years_init = float(opt_cfg["option_dte_days"]) / 365.0
    target_otm = float(opt_cfg["target_otm"])
    stop_loss_pct = float(opt_cfg["stop_loss_pct"])
    take_profit_pct = float(opt_cfg["take_profit_pct"])
    exit_on_flip = bool(opt_cfg["exit_on_signal_flip"])

    # Per-symbol state.
    state = {
        s: {
            "open": False,
            "side": 0,  # +1 call, -1 put
            "strike": 0.0,
            "ttm": 0.0,
            "entry_mark": 0.0,
            "last_mark": 0.0,
            "contracts": 0.0,
            "trades": 0,
            "wins": 0,
            "pnl_dollars": 0.0,
        }
        for s in syms
    }

    port = pd.DataFrame(index=common)
    port["ret_port"] = 0.0
    port["equity"] = 1.0

    for i, t in enumerate(common):
        prev_equity = float(port["equity"].iloc[i - 1]) if i > 0 else 1.0
        pnl_total = 0.0
        fees_total = 0.0

        for s in syms:
            d = aligned[s]
            row = d.loc[t]
            desired = int(row["target_pos"]) if not pd.isna(row["target_pos"]) else 0
            score = float(row["score"]) if "score" in row else 0.0
            adx = float(row["adx"]) if "adx" in row else 0.0
            if abs(score) < float(opt_cfg["min_score_abs"]) or adx < float(opt_cfg["min_adx"]):
                desired = 0
            side_mode = str(opt_cfg.get("side_mode", "both"))
            if side_mode == "long_only" and desired < 0:
                desired = 0
            elif side_mode == "short_only" and desired > 0:
                desired = 0
            spot = float(row["close"])
            atr_pct = float(row["atr_pct"]) if not pd.isna(row["atr_pct"]) else 0.01
            vol = _annual_vol_from_atr_pct(
                atr_pct, bars_per_year, opt_cfg["iv_scale"], opt_cfg["iv_floor"], opt_cfg["iv_cap"]
            )
            st = state[s]

            if st["open"]:
                st["ttm"] = max(st["ttm"] - dt_years, 1e-6)
                mark = _option_mark(spot, st["strike"], st["ttm"], vol, rf, st["side"])
                pnl_bar = st["contracts"] * 100.0 * (mark - st["last_mark"])
                st["pnl_dollars"] += pnl_bar
                st["last_mark"] = mark
                pnl_total += pnl_bar

                pnl_pct_vs_entry = (mark / max(st["entry_mark"], 1e-9)) - 1.0
                exit_now = False
                if pnl_pct_vs_entry <= -stop_loss_pct:
                    exit_now = True
                if pnl_pct_vs_entry >= take_profit_pct:
                    exit_now = True
                if st["ttm"] <= dt_years:
                    exit_now = True
                if exit_on_flip and desired != 0 and desired != st["side"]:
                    exit_now = True
                if desired == 0:
                    exit_now = True

                if exit_now:
                    notional = st["contracts"] * 100.0 * mark
                    f = fee * abs(notional)
                    fees_total += f
                    st["pnl_dollars"] -= f
                    if st["pnl_dollars"] > 0:
                        st["wins"] += 1
                    st["open"] = False
                    st["side"] = 0
                    st["strike"] = 0.0
                    st["ttm"] = 0.0
                    st["entry_mark"] = 0.0
                    st["last_mark"] = 0.0
                    st["contracts"] = 0.0
                    st["pnl_dollars"] = 0.0

            # New entry after potential exit.
            if (not st["open"]) and desired != 0:
                side = 1 if desired > 0 else -1
                strike = spot * (1.0 + target_otm if side > 0 else 1.0 - target_otm)
                mark = _option_mark(spot, strike, dte_years_init, vol, rf, side)
                alloc_dollars = prev_equity * cap_total / max(n_syms, 1)
                contracts = alloc_dollars / max(mark * 100.0, 1e-9)
                if contracts > 0:
                    st["open"] = True
                    st["side"] = side
                    st["strike"] = strike
                    st["ttm"] = dte_years_init
                    st["entry_mark"] = mark
                    st["last_mark"] = mark
                    st["contracts"] = contracts
                    st["trades"] += 1
                    f = fee * abs(contracts * 100.0 * mark)
                    fees_total += f
                    st["pnl_dollars"] -= f

        ret_bar = 0.0 if prev_equity <= 0 else (pnl_total - fees_total) / prev_equity
        port.at[t, "ret_port"] = ret_bar
        port.at[t, "equity"] = prev_equity * (1.0 + ret_bar)

    met = compute_metrics(port["equity"], port["ret_port"].fillna(0.0), bars_per_year)
    trades = sum(state[s]["trades"] for s in syms)
    wins = sum(state[s]["wins"] for s in syms)
    met["model"] = "options_proxy_bs"
    met["trades"] = int(trades)
    met["win_rate"] = float(wins / trades) if trades > 0 else 0.0
    met["capital_at_risk"] = cap_total
    met["option_dte_days"] = int(opt_cfg["option_dte_days"])
    met["target_otm"] = float(opt_cfg["target_otm"])

    by_symbol = {
        s: {
            "trades": int(state[s]["trades"]),
            "wins": int(state[s]["wins"]),
            "win_rate": float(state[s]["wins"] / state[s]["trades"]) if state[s]["trades"] > 0 else 0.0,
        }
        for s in syms
    }
    return port, {"portfolio": met, "symbols": by_symbol, "assumptions": opt_cfg}
