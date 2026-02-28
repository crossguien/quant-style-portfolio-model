from __future__ import annotations


def default_config() -> dict:
    return {
        # Features
        "ema_fast": 50,
        "sma_slow": 200,
        "atr_n": 14,
        "atr_slope_n": 10,
        "mfi_n": 14,
        "rsi_n": 14,
        "adx_n": 14,
        "bb_n": 20,
        "bb_k": 2.0,
        "kc_n": 20,
        "kc_m": 1.5,
        "vol_n": 20,
        "htf_ema": 50,
        "htf_sma": 200,

        # Signal weights
        "w_trend": 0.22,
        "w_vwap": 0.20,
        "w_mom": 0.22,
        "w_reg": 0.14,
        "w_adx": 0.12,
        "w_vol": 0.10,

        # Thresholds
        "entry_thr": 0.45,
        "adx_trend": 20.0,
        "adx_min": 15.0,
        "require_adx": True,

        # Execution costs
        "slippage_bps": 1.5,
        "fee_bps": 0.5,

        # Risk and sizing (per ticker)
        "target_vol_per_bar": 0.0010,
        "max_leverage_per_symbol": 2.5,

        # Stops (per ticker)
        "stop_atr": 2.0,
        "tp_atr": 3.0,
        "trail_atr": 2.5,

        # Portfolio risk budget
        "risk_weights": "vol_inverse",  # "equal" or "vol_inverse"

        # Portfolio constraints
        "cash_buffer": 0.10,      # keep 10% in cash
        "max_gross": 1.50,        # sum(|exposure|) cap
        "max_net": 0.60,          # |sum(exposure)| cap

        # Correlation penalty
        "corr_lookback": 200,     # bars
        "corr_penalty": 0.50,     # higher -> stronger downweighting

        # Annualization (5m approx)
        "bars_per_year": 78 * 252,
    }
