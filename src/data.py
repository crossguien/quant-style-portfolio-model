from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    out = df[needed].dropna().copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    return out


def _yahoo_fetch_ohlcv(ticker: str, interval: str, lookback_days: int) -> pd.DataFrame:
    period = f"{lookback_days}d"
    df = yf.download(
        tickers=ticker,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}, interval={interval}, period={period}")

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return MultiIndex columns like ("Close", "SPY")
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df = df.rename(columns={c: str(c).lower() for c in df.columns})

    return _normalize_ohlcv(df)


def _interval_alpha_vantage(interval: str) -> str:
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min"}
    if interval not in mapping:
        raise ValueError(f"Alpha Vantage interval not supported: {interval}")
    return mapping[interval]


def _alpha_vantage_fetch_ohlcv(ticker: str, interval: str, lookback_days: int, api_key: str) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("Alpha Vantage requires api_key (or ALPHAVANTAGE_API_KEY env var).")
    iv = _interval_alpha_vantage(interval)
    r = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker,
            "interval": iv,
            "outputsize": "full",
            "adjusted": "false",
            "apikey": api_key,
        },
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    key = f"Time Series ({iv})"
    if key not in payload:
        note = payload.get("Note") or payload.get("Error Message") or payload.get("Information") or str(payload)
        raise RuntimeError(f"Alpha Vantage fetch failed for {ticker}: {note}")
    df = pd.DataFrame(payload[key]).T
    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
    )
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    return _normalize_ohlcv(df[df.index >= cutoff])


def _interval_alpaca(interval: str) -> str:
    mapping = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min", "60m": "1Hour", "1d": "1Day"}
    if interval not in mapping:
        raise ValueError(f"Alpaca interval not supported: {interval}")
    return mapping[interval]


def _alpaca_fetch_ohlcv(
    ticker: str, interval: str, lookback_days: int, api_key: str, api_secret: str, feed: str = "iex"
) -> pd.DataFrame:
    if not api_key or not api_secret:
        raise RuntimeError("Alpaca requires api_key and api_secret (or ALPACA_API_KEY / ALPACA_API_SECRET).")
    tf = _interval_alpaca(interval)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    r = requests.get(
        f"https://data.alpaca.markets/v2/stocks/{ticker}/bars",
        params={
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframe": tf,
            "limit": 10000,
            "adjustment": "raw",
            "feed": feed,
            "sort": "asc",
        },
        headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    bars = payload.get("bars", [])
    if not bars:
        raise RuntimeError(f"Alpaca returned no bars for {ticker}")
    df = pd.DataFrame(bars)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"})
    df = df.set_index("timestamp")
    return _normalize_ohlcv(df)


def _interval_fmp(interval: str) -> str:
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "1hour"}
    if interval not in mapping:
        raise ValueError(f"FMP interval not supported: {interval}")
    return mapping[interval]


def _fmp_fetch_ohlcv(ticker: str, interval: str, lookback_days: int, api_key: str) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("FMP requires api_key (or FMP_API_KEY env var).")
    iv = _interval_fmp(interval)
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days)
    r = requests.get(
        f"https://financialmodelingprep.com/api/v3/historical-chart/{iv}/{ticker}",
        params={"from": str(start), "to": str(end), "apikey": api_key},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"FMP returned no bars for {ticker}: {payload}")
    df = pd.DataFrame(payload)
    df = df.rename(columns={"date": "timestamp"})
    df = df.set_index("timestamp")
    return _normalize_ohlcv(df)


def _interval_marketstack(interval: str) -> str:
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "1hour"}
    if interval not in mapping:
        raise ValueError(f"Marketstack interval not supported: {interval}")
    return mapping[interval]


def _marketstack_fetch_ohlcv(ticker: str, interval: str, lookback_days: int, api_key: str) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("Marketstack requires api_key (or MARKETSTACK_API_KEY env var).")
    iv = _interval_marketstack(interval)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    r = requests.get(
        "http://api.marketstack.com/v1/intraday",
        params={
            "access_key": api_key,
            "symbols": ticker,
            "interval": iv,
            "date_from": start.date().isoformat(),
            "date_to": end.date().isoformat(),
            "limit": 1000,
            "sort": "ASC",
        },
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"Marketstack fetch failed for {ticker}: {payload['error']}")
    rows = payload.get("data", [])
    if not rows:
        raise RuntimeError(f"Marketstack returned no bars for {ticker}")
    df = pd.DataFrame(rows)
    df = df.rename(columns={"date": "timestamp"})
    df = df.set_index("timestamp")
    return _normalize_ohlcv(df)


def fetch_ohlcv(
    ticker: str,
    interval: str = "5m",
    lookback_days: int = 60,
    provider: str = "yahoo",
    api_key: str | None = None,
    api_secret: str | None = None,
) -> pd.DataFrame:
    provider = provider.lower()
    if provider == "yahoo":
        return _yahoo_fetch_ohlcv(ticker, interval, lookback_days)
    if provider == "alpha_vantage":
        return _alpha_vantage_fetch_ohlcv(
            ticker, interval, lookback_days, api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")
        )
    if provider == "alpaca":
        return _alpaca_fetch_ohlcv(
            ticker,
            interval,
            lookback_days,
            api_key or os.getenv("ALPACA_API_KEY", ""),
            api_secret or os.getenv("ALPACA_API_SECRET", ""),
        )
    if provider == "fmp":
        return _fmp_fetch_ohlcv(ticker, interval, lookback_days, api_key or os.getenv("FMP_API_KEY", ""))
    if provider == "marketstack":
        return _marketstack_fetch_ohlcv(
            ticker, interval, lookback_days, api_key or os.getenv("MARKETSTACK_API_KEY", "")
        )
    raise ValueError(f"Unsupported data provider: {provider}")


def fetch_ltf_htf(
    ticker: str,
    ltf: str = "5m",
    htf: str = "60m",
    lookback_days: int = 60,
    provider: str = "yahoo",
    api_key: str | None = None,
    api_secret: str | None = None,
):
    return (
        fetch_ohlcv(ticker, ltf, lookback_days, provider=provider, api_key=api_key, api_secret=api_secret),
        fetch_ohlcv(ticker, htf, lookback_days, provider=provider, api_key=api_key, api_secret=api_secret),
    )
