from __future__ import annotations

import numpy as np
import pandas as pd


def size_vol_target(atr_pct: pd.Series, cfg: dict) -> pd.Series:
    atr_pct = atr_pct.clip(lower=1e-6)
    size = cfg["target_vol_per_bar"] / atr_pct
    return size.clip(0, cfg["max_leverage_per_symbol"])


def apply_stops(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    pos = out["pos"].values
    close = out["close"].values
    atr = out["atr"].values

    stop_mult = cfg["stop_atr"]
    tp_mult = cfg["tp_atr"]
    trail_mult = cfg["trail_atr"]

    stop_px = np.full(n, np.nan)
    tp_px = np.full(n, np.nan)
    trail_px = np.full(n, np.nan)

    entry_px = np.nan
    peak = -np.inf
    trough = np.inf

    for i in range(n):
        if pos[i] == 0:
            entry_px = np.nan
            peak = -np.inf
            trough = np.inf
            continue

        if i == 0 or pos[i] != pos[i-1]:
            entry_px = close[i]
            peak = close[i]
            trough = close[i]

        peak = max(peak, close[i])
        trough = min(trough, close[i])

        if pos[i] > 0:
            stop_px[i] = entry_px - stop_mult * atr[i]
            tp_px[i] = entry_px + tp_mult * atr[i]
            trail_px[i] = peak - trail_mult * atr[i]
        else:
            stop_px[i] = entry_px + stop_mult * atr[i]
            tp_px[i] = entry_px - tp_mult * atr[i]
            trail_px[i] = trough + trail_mult * atr[i]

    out["stop_px"] = stop_px
    out["tp_px"] = tp_px
    out["trail_px"] = trail_px

    exit_flag = np.zeros(n, dtype=bool)
    for i in range(n):
        if pos[i] == 0:
            continue
        if pos[i] > 0:
            if close[i] <= stop_px[i] or close[i] <= trail_px[i] or close[i] >= tp_px[i]:
                exit_flag[i] = True
        else:
            if close[i] >= stop_px[i] or close[i] >= trail_px[i] or close[i] <= tp_px[i]:
                exit_flag[i] = True

    out["exit_flag"] = exit_flag
    return out
