from __future__ import annotations

import pandas as pd
import numpy as np

from .features import build_features
from .signal import build_signal
from .risk import size_vol_target, apply_stops
from .metrics import compute_metrics


def _base_risk_weights(symbol_dfs: dict[str, pd.DataFrame], cfg: dict) -> dict[str, float]:
    mode = cfg.get("risk_weights", "equal")
    syms = list(symbol_dfs.keys())
    if not syms:
        return {}

    if mode == "equal":
        w = 1.0 / len(syms)
        return {s: w for s in syms}

    if mode == "vol_inverse":
        inv = {}
        for s, df in symbol_dfs.items():
            v = float(df["atr_pct"].dropna().median()) if "atr_pct" in df.columns else 0.01
            inv[s] = 1.0 / max(v, 1e-6)
        tot = sum(inv.values()) if inv else 1.0
        return {s: inv[s] / tot for s in inv}

    raise ValueError("risk_weights must be 'equal' or 'vol_inverse'")


def _corr_penalized_weights(symbol_dfs: dict[str, pd.DataFrame], base_w: dict[str, float], cfg: dict) -> dict[str, float]:
    """
    Downweight symbols that are highly correlated with the rest of the basket.
    Penalty factor: w_i' = w_i / (1 + corr_penalty * avg_corr_i)
    avg_corr_i is average of positive correlations of i with others over corr_lookback.
    """
    syms = list(symbol_dfs.keys())
    if len(syms) <= 1:
        return base_w

    look = int(cfg.get("corr_lookback", 200))
    penalty = float(cfg.get("corr_penalty", 0.5))

    rets = {}
    for s in syms:
        r = symbol_dfs[s]["close"].pct_change().dropna()
        rets[s] = r.tail(look)

    ret_df = pd.DataFrame(rets).dropna()
    if ret_df.shape[0] < 20:
        return base_w

    corr = ret_df.corr().fillna(0.0)
    avg_pos = {}
    for s in syms:
        vals = corr.loc[s].drop(labels=[s]).clip(lower=0.0)
        avg_pos[s] = float(vals.mean()) if len(vals) else 0.0

    adj = {}
    for s in syms:
        adj[s] = base_w.get(s, 0.0) / (1.0 + penalty * avg_pos[s])

    tot = sum(adj.values()) if adj else 1.0
    return {s: adj[s] / tot for s in adj}


def prepare_symbol(ltf: pd.DataFrame, htf: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    feat = build_features(ltf, htf, cfg)
    sig = build_signal(feat, cfg)
    return sig.dropna()


def _apply_portfolio_constraints(exposure: pd.Series, cfg: dict) -> pd.Series:
    """
    exposure is per-symbol exposure at a bar (can be +/-), in leverage units.
    Constraints:
      - cash_buffer: scale exposures by (1 - cash_buffer)
      - max_gross: sum(|exposure|) <= max_gross
      - max_net: |sum(exposure)| <= max_net
    """
    exp = exposure.copy()
    cash = float(cfg.get("cash_buffer", 0.0))
    exp = exp * max(0.0, 1.0 - cash)

    gross = float(exp.abs().sum())
    max_gross = float(cfg.get("max_gross", 1e9))
    if gross > max_gross and gross > 0:
        exp = exp * (max_gross / gross)

    net = float(exp.sum())
    max_net = float(cfg.get("max_net", 1e9))
    if abs(net) > max_net and abs(net) > 0:
        exp = exp * (max_net / abs(net))

    return exp


def backtest_portfolio(symbol_data: dict[str, pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, dict]:
    # Common index
    common = None
    for df in symbol_data.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) < 30:
        return pd.DataFrame(), {"error": "not enough common timestamps"}

    aligned = {s: df.reindex(common).copy() for s, df in symbol_data.items()}

    base_w = _base_risk_weights(aligned, cfg)
    w = _corr_penalized_weights(aligned, base_w, cfg)

    cost = (cfg["slippage_bps"] + cfg["fee_bps"]) / 10000.0

    # Precompute per-symbol raw sizes and positions
    per = {}
    for s, df in aligned.items():
        d = df.copy()
        d["w"] = w[s]
        d["size_raw"] = size_vol_target(d["atr_pct"], cfg)  # per-symbol cap
        d["pos"] = d["target_pos"].shift(1).fillna(0).astype(int)
        d = apply_stops(d, cfg)
        d.loc[d["exit_flag"], "pos"] = 0
        d["pos_prev"] = d["pos"].shift(1).fillna(0).astype(int)
        d["trade"] = d["pos"] - d["pos_prev"]
        per[s] = d

    # Portfolio loop to apply constraints each bar
    port = pd.DataFrame(index=common)
    sym_cols = []
    for s in aligned:
        port[f"ret_{s}"] = 0.0
        sym_cols.append(s)

    for t in common:
        # raw exposures: w * size_raw * pos_prev
        exp = pd.Series({s: per[s].at[t, "w"] * per[s].at[t, "size_raw"] * per[s].at[t, "pos_prev"] for s in sym_cols})
        exp_adj = _apply_portfolio_constraints(exp, cfg)

        for s in sym_cols:
            d = per[s]
            # returns for that bar (use pos_prev and px change)
            px_ret = float(d.at[t, "close"] / d["close"].shift(1).at[t] - 1.0) if not pd.isna(d["close"].shift(1).at[t]) else 0.0
            # cost on trade, scaled by exposure weight magnitude
            trade = float(abs(d.at[t, "trade"]))
            ret_gross = exp_adj[s] * px_ret
            ret_cost = (trade > 0) * cost * abs(exp_adj[s])
            port.at[t, f"ret_{s}"] = ret_gross - ret_cost

    port["ret_port"] = port[[c for c in port.columns if c.startswith("ret_")]].sum(axis=1)
    port["equity"] = (1.0 + port["ret_port"].fillna(0.0)).cumprod()

    met_port = compute_metrics(port["equity"], port["ret_port"].fillna(0.0), cfg["bars_per_year"])
    met_port["risk_weights_mode"] = cfg.get("risk_weights", "equal")
    met_port["cash_buffer"] = float(cfg.get("cash_buffer", 0.0))
    met_port["max_gross"] = float(cfg.get("max_gross", 0.0))
    met_port["max_net"] = float(cfg.get("max_net", 0.0))
    met_port["corr_penalty"] = float(cfg.get("corr_penalty", 0.0))

    sym_meta = {s: {"risk_weight": float(w[s]), "median_atr_pct": float(aligned[s]["atr_pct"].median())} for s in aligned}

    return port, {"portfolio": met_port, "symbols": sym_meta}
