from __future__ import annotations

import numpy as np
import pandas as pd


def build_signal(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()

    trend1 = np.sign(out["close"] - out["sma_slow"])
    trend2 = np.sign(out["ema_fast"] - out["sma_slow"])
    vwap_loc = np.sign(out["close"] - out["vwap"])

    mfi_mom = (out["mfi"] - 50.0) / 50.0
    rsi_mom = (out["rsi"] - 50.0) / 50.0

    reg = (1.0 - out["squeeze_on"].astype(float)) * np.tanh(10 * out["atr_slope"].fillna(0))
    strength = np.tanh((out["adx"] - cfg["adx_trend"]) / 10.0)
    part = np.tanh((out["vol_ratio"] - 1.0) * 2.0)

    raw = (
        cfg["w_trend"] * (0.6 * trend1 + 0.4 * trend2)
        + cfg["w_vwap"] * vwap_loc
        + cfg["w_mom"] * (0.6 * mfi_mom + 0.4 * rsi_mom)
        + cfg["w_reg"] * reg
        + cfg["w_adx"] * strength
        + cfg["w_vol"] * part
    )

    out["score"] = np.tanh(raw)

    out["target_pos"] = 0
    out.loc[(out["score"] >= cfg["entry_thr"]) & out["htf_bull"], "target_pos"] = 1
    out.loc[(out["score"] <= -cfg["entry_thr"]) & out["htf_bear"], "target_pos"] = -1

    if cfg.get("require_adx", True):
        out.loc[out["adx"] < cfg["adx_min"], "target_pos"] = 0

    return out
