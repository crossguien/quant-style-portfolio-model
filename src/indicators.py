from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def vwap_session(df: pd.DataFrame) -> pd.Series:
    tp = typical_price(df)
    pv = tp * df["volume"]
    out = pd.Series(index=df.index, dtype="float64")
    dates = df.index.date
    for d in np.unique(dates):
        m = (dates == d)
        v = df.loc[m, "volume"].replace(0, np.nan)
        out.loc[m] = pv.loc[m].cumsum() / v.cumsum()
    return out


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean()
    rd = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = typical_price(df)
    raw = tp * df["volume"]
    dirn = tp.diff()
    pos = raw.where(dirn > 0, 0.0)
    neg = raw.where(dirn < 0, 0.0).abs()
    ps = pos.rolling(n).sum()
    ns = neg.rolling(n).sum()
    mr = ps / ns.replace(0, np.nan)
    return 100 - (100 / (1 + mr))


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(),
                    (df["high"]-pc).abs(),
                    (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std()
    up = mid + k * sd
    lo = mid - k * sd
    return mid, up, lo


def keltner(df: pd.DataFrame, n: int = 20, m: float = 1.5):
    mid = ema(df["close"], n)
    a = atr(df, n)
    up = mid + m * a
    lo = mid - m * a
    return mid, up, lo


def squeeze_on(df: pd.DataFrame, bb_n: int = 20, bb_k: float = 2.0, kc_n: int = 20, kc_m: float = 1.5) -> pd.Series:
    _, bbu, bbl = bollinger(df["close"], bb_n, bb_k)
    _, kcu, kcl = keltner(df, kc_n, kc_m)
    return (bbu < kcu) & (bbl > kcl)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l = df["high"], df["low"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)

    tr = true_range(df)
    atr_w = tr.ewm(alpha=1/n, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_w.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_w.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()
