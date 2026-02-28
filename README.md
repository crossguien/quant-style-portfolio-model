# Quant-Style Portfolio Trading System v2 (Python)

Adds:
- Default tickers include **SPY NVDA META MSFT AMZN**
- Portfolio constraints:
  - Max gross leverage
  - Max net exposure
  - Cash buffer (keeps some capital unallocated)
  - Correlation penalty (downweights highly correlated symbols)

## Quick start
```bash
python -m venv .venv
pip install -r requirements.txt

python run_portfolio_backtest.py --tickers SPY NVDA META MSFT AMZN --ltf 30m --htf 60m --lookback 60 --risk_weights vol_inverse
python run_portfolio_walkforward.py --tickers SPY NVDA META MSFT AMZN --ltf 30m --htf 60m --lookback 60 --risk_weights vol_inverse
```

## Data provider toggle
All scripts support:
- `--data_provider yahoo|alpha_vantage|alpaca|fmp|marketstack`
- `--api_key ...`
- `--api_secret ...` (alpaca only)

Examples:
```bash
# Yahoo (default, no key)
python run_daily_trade_plan.py --data_provider yahoo

# Alpha Vantage
export ALPHAVANTAGE_API_KEY="your_key"
python run_daily_trade_plan.py --data_provider alpha_vantage

# Alpaca (free IEX feed with API key + secret)
export ALPACA_API_KEY="your_key_id"
export ALPACA_API_SECRET="your_secret"
python run_daily_trade_plan.py --data_provider alpaca

# FMP
export FMP_API_KEY="your_key"
python run_daily_trade_plan.py --data_provider fmp

# Marketstack
export MARKETSTACK_API_KEY="your_key"
python run_daily_trade_plan.py --data_provider marketstack
```

Notes:
- Free tiers may be delayed, rate-limited, or not include full intraday history.
- Alpaca free typically uses IEX feed; full market-wide feed is paid.
- If a provider/plan does not support your interval (`5m`, `30m`, etc.), the script will raise an error.

## Auto-tune
```bash
python run_autotune.py \
  --tickers SPY NVDA META MSFT AMZN \
  --ltf 30m --htf 60m --lookback 60 \
  --data_provider yahoo \
  --max_evals 120 --objective ret_then_sharpe \
  --out autotune_results.json
```

The tuner runs random search over a parameter grid and prints JSON with `best` and `top` candidates.

## Daily trade plan
```bash
python run_daily_trade_plan.py \
  --tickers SPY NVDA META MSFT AMZN \
  --ltf 30m --htf 60m --lookback 60 \
  --data_provider yahoo \
  --risk_weights vol_inverse \
  --out daily_trade_plan.json
```

This prints per-symbol model actions (`enter_or_hold_long`, `enter_or_hold_short`, `stay_flat_or_exit`), risk weights, suggested exposure, and stop/take-profit hints.
It also includes `signal_quality`: `strong`, `moderate`, `weak`, `watchlist_near_trigger`, or `neutral`.

## Options trade plan (separate script)
```bash
python run_options_trade_plan.py \
  --tickers SPY NVDA META MSFT AMZN \
  --ltf 30m --htf 60m --lookback 60 \
  --data_provider yahoo \
  --min_dte 14 --max_dte 45 \
  --target_moneyness 0.02 \
  --out options_trade_plan.json
```

This keeps options logic isolated from stock logic. It maps stock direction to option ideas:
- stock `long` signal -> candidate call
- stock `short` signal -> candidate put
- stock `flat` signal -> no options trade

## Options model backtest (MVP)
```bash
python run_options_model_backtest.py \
  --tickers SPY NVDA META MSFT AMZN \
  --ltf 30m --htf 60m --lookback 60 \
  --data_provider yahoo \
  --profile balanced \
  --option_dte_days 45 \
  --target_otm 0.00 \
  --capital_at_risk 0.15 \
  --stop_loss_pct 0.25 \
  --take_profit_pct 0.40
```

This is a separate options model backtest that uses:
- stock-direction signals from your core model
- Black-Scholes pricing proxy (synthetic options PnL)
- configurable DTE, OTM target, risk budget, and stop/TP logic

Important: this is an MVP proxy model, not a full historical options-chain backtester.
For production-grade options research, you should use historical options chain/greeks data.

Filter profiles:
- balanced (more trades): `--profile balanced`
- strict (fewer trades, higher win rate): `--profile strict`

You can still override profile defaults explicitly:
`--min_score_abs ... --min_adx ... --side_mode ...`

Tip: tightening filters usually reduces trade count but improves quality.

## Daily workflow
1. `python run_daily_trade_plan.py`  
   Use this for today's directional and sizing plan.
2. `python run_portfolio_backtest.py`  
   Check if recent regime quality is still acceptable.
3. `python run_portfolio_walkforward.py`  
   Confirm rolling-window robustness (less overfit than single-window backtest).
4. `python run_autotune.py` (weekly or bi-weekly, not every day)  
   Refit parameter choices, then validate before adopting.

## Reading outputs
- Good signs:
  - `total_return > 0`
  - `sharpe > 1.0` (stronger above ~1.5)
  - shallow `max_drawdown` (closer to 0)
  - trade plan has coherent exposure (not maxed out in one direction constantly)
- Warning signs:
  - negative `total_return` for multiple consecutive runs
  - `sharpe <= 0`
  - deepening `max_drawdown`
  - unstable behavior where tiny parameter changes flip results wildly

Daily plan field notes:
- `score`: model confidence-like signal in approximately `[-1, 1]`
- `signal_quality`:
  - `strong`: score is well beyond entry threshold
  - `moderate`: score is above threshold with margin
  - `weak`: score just above threshold
  - `watchlist_near_trigger`: not in position, but close to threshold
  - `neutral`: no edge

## Risk budgeting modes
- equal
- vol_inverse

Correlation penalty is applied on top of the chosen budgeting mode.
