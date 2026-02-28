from __future__ import annotations

import pandas as pd
import numpy as np
from . import indicators as ind


def align_htf_to_ltf(htf: pd.DataFrame, ltf_index: pd.DatetimeIndex) -> pd.DataFrame:
    h = htf.copy().sort_index()
    h = h.reindex(h.index.union(ltf_index)).sort_index().ffill()
    return h.reindex(ltf_index)


def build_features(ltf: pd.DataFrame, htf: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = ltf.copy()

    df["vwap"] = ind.vwap_session(df)
    df["ema_fast"] = ind.ema(df["close"], cfg["ema_fast"])
    df["sma_slow"] = ind.sma(df["close"], cfg["sma_slow"])

    df["atr"] = ind.atr(df, cfg["atr_n"])
    df["atr_pct"] = df["atr"] / df["close"]
    df["atr_slope"] = df["atr"].diff().rolling(cfg["atr_slope_n"]).mean()

    df["mfi"] = ind.mfi(df, cfg["mfi_n"])
    df["rsi"] = ind.rsi(df["close"], cfg["rsi_n"])
    df["adx"] = ind.adx(df, cfg["adx_n"])
    df["squeeze_on"] = ind.squeeze_on(df, cfg["bb_n"], cfg["bb_k"], cfg["kc_n"], cfg["kc_m"])

    vavg = df["volume"].rolling(cfg["vol_n"]).mean()
    df["vol_ratio"] = df["volume"] / vavg.replace(0, np.nan)

    h = htf.copy()
    h["htf_ema"] = ind.ema(h["close"], cfg["htf_ema"])
    h["htf_sma"] = ind.sma(h["close"], cfg["htf_sma"])
    h = align_htf_to_ltf(h[["close","htf_ema","htf_sma"]], df.index)

    df["htf_close"] = h["close"]
    df["htf_ema"] = h["htf_ema"]
    df["htf_sma"] = h["htf_sma"]
    df["htf_bull"] = (df["htf_close"] > df["htf_sma"]) & (df["htf_ema"] > df["htf_sma"])
    df["htf_bear"] = (df["htf_close"] < df["htf_sma"]) & (df["htf_ema"] < df["htf_sma"])

    return df.dropna()
