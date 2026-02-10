# AShareGPT

**AShareGPT** is an automated alpha factor mining and backtesting system for the Chinese A-share market. It combines **Proximal Policy Optimization (PPO)** with a **Looped Transformer** to discover readable, formulaic alpha factors in Reverse Polish Notation (RPN).

Unlike black-box models that predict prices directly, AShareGPT is a **white-box agent**: it generates interpretable mathematical formulas (e.g., `CLOSE / SMA_20 - 1`) that are rigorously backtested with realistic A-share market rules.

## ✨ Key Features

- **Symbolic Alpha Discovery** — PPO-trained Transformer generates human-readable RPN formulas
- **60+ Technical Indicators** — via `pandas_ta` Strategy API (RSI, MACD, Bollinger Bands, OBV, etc.)
- **Dual Decision Frequency** — `daily` (aggregated bars) or `1min` (raw minute-level decisions)
- **A-share Market Rules** — T+1 settlement enforcement, tick-rounded price-limit (涨跌停) detection, tradable masks, optional liquidity constraints, T+0 whitelisting
- **Walk-Forward Optimization** — rolling train/val/test windows to reduce overfitting
- **Adjust Factor Support** — automatic 前复权 price adjustment with code alias fallback
- **GPU Acceleration** — all tensor operations on CUDA when available

## 📂 Project Structure

```text
AShareGPT/
├── model_core/                  # Core library
│   ├── config.py                # All configuration (env-var driven)
│   ├── model.py                 # Looped Transformer + SwiGLU + RMSNorm
│   ├── training.py              # PPO loop, reward orchestration, walk-forward
│   ├── vm.py                    # Stack-based VM for RPN formula execution
│   ├── ops.py                   # Symbolic operators (freq-adaptive windows)
│   ├── factors.py               # Feature engineering (60+ indicators)
│   ├── data_loader.py           # Minute CSV → tensors pipeline
│   ├── backtest.py              # Vectorized backtester with execution rules
│   ├── code_alias.py            # Old→new code mapping for adj factors
│   ├── market/
│   │   └── cn_rules.py          # T+1, price-limit, session-id logic
│   ├── data/
│   │   └── io.py                # Encoding-robust CSV I/O utilities
│   └── application/
│       └── services/            # Compatibility API wrapper
├── run_cn_train.py              # Entry point: alpha mining
├── run_cn_backtest.py           # Entry point: strategy backtest
├── clean_adj_factors.py         # Utility: normalize adjust factor CSVs
├── unify_data.py                # Utility: merge raw downloads into per-code files
├── scripts/
│   └── backfill_adj_by_alias.py # Utility: fill missing adj factors via alias map
├── tests/                       # Pytest suite
├── .env.example                 # Full configuration reference
└── data/                        # Data root (not tracked in git)
    ├── 2025/                    # Year folders with minute CSVs
    │   ├── 000001.SZ.csv
    │   └── ...
    └── 复权因子/                 # Adjust factor CSVs (optional)
        ├── 000001.SZ.csv
        └── ...
```

## 📊 Data Format

### Minute Data (`data/YYYY/<code>.csv`)

Raw minute-level OHLCV bars.

| Column | Type | Description |
|:---|:---|:---|
| `trade_time` | string | `YYYY-MM-DD HH:MM:SS` |
| `open` | float | Open price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Close price |
| `vol` | float | Volume |
| `amount` | float | Turnover |

### Adjust Factors (`data/复权因子/<code>.csv`)

| Column | Type | Description |
|:---|:---|:---|
| `code` | string | Security code |
| `date` | string | Date (`YYYYMMDD` or `YYYY-MM-DD`) |
| `adj_factor` | float | Cumulative adjust factor |

If a file is missing, the factor defaults to `1.0`. Code aliases (e.g., old code → new code after restructuring) are resolved via `code_alias_map.csv`.

## 🚀 Quick Start

### 1. Install

```bash
pip install -r requirements.txt
pip install pandas-ta-classic  # preferred; pandas_ta also works as fallback
```

### 2. Configure

Copy and customize the environment file:

```bash
cp .env.example .env
# Edit .env to set data paths, decision frequency, etc.
```

### 3. Train (Mine Alphas)

```bash
python run_cn_train.py
```

- Discovers formulaic alpha factors via PPO reinforcement learning
- Best formula saved to `best_cn_strategy.json`
- Supports walk-forward optimization (`CN_WALK_FORWARD=1`)
- CUDA auto-detected

### 4. Backtest

```bash
python run_cn_backtest.py --strategy best_cn_strategy.json
```

Options:
- `--symbols 000001.SZ,600519.SH` — restrict to specific codes
- `--start-date 2025-01-01` / `--end-date 2025-06-01` — date range
- `--curve-out equity.csv` — export equity curve
- `--no-adj` — disable adjust factor application

Key metrics: Sharpe Ratio, Sortino Ratio, Annual Return, Max Drawdown, Win Rate.

## ⚙️ Configuration

All settings are driven by environment variables (see `.env.example` for the full list). Key parameters:

### Data & Symbols

| Variable | Default | Description |
|:---|:---|:---|
| `CN_MINUTE_DATA_ROOT` | `data` | Root directory for minute CSVs |
| `CN_MINUTE_YEARS` | *(auto)* | Comma-separated years to load |
| `CN_CODES` | *(auto)* | Comma-separated codes; empty = auto-discover |
| `CN_MAX_CODES` | `50` | Max symbols to load |
| `CN_MINUTE_DAYS` | `120` | Rolling window when no end date set |

### Decision Frequency & Returns

| Variable | Default | Description |
|:---|:---|:---|
| `CN_DECISION_FREQ` | `daily` | `daily` or `1min` |
| `CN_BAR_STYLE` | `daily` | `daily` (full OHLCV) or `signal_snapshot` |
| `CN_TARGET_RET_MODE` | `close_to_close` | `close_to_close` or `signal_to_exit` |
| `CN_HOLD_DAYS` | `1` | Hold period for daily frequency |
| `CN_HOLD_BARS` | `1` | Hold period for 1min frequency |

### A-share Market Rules

| Variable | Default | Description |
|:---|:---|:---|
| `CN_ENFORCE_T_PLUS_ONE` | `1` | Enable T+1 same-day sell blocking |
| `CN_T0_ALLOWED_CODES_FILE` | `cn_t0_allowed_codes.csv` | CSV whitelist loaded when `CN_T0_ALLOWED_CODES` is empty |
| `CN_T0_ALLOWED_CODES` | *(empty)* | Comma-separated T+0 exempt codes override (e.g., ETFs); if env+file both resolve empty, all symbols are treated as T+1 |
| `CN_LIMIT_HIT_TOL` | `0.001` | Price-limit detection tolerance |
| `CN_TICK_SIZE` | `0.01` | Minimum tick size used for limit-price rounding |
| `CN_LOT_SIZE` | `100` | Lot size used by liquidity constraints / volume impact |
| `CN_ENFORCE_TRADING_HOURS` | `1` | Filter minute data to continuous trading hours |
| `CN_TRADABLE_REQUIRE_LIQUIDITY` | `1` | Infer `tradable=0` when `volume==0` and `amount==0` |
| `CN_ENABLE_LIQUIDITY_CONSTRAINTS` | `0` | Enable partial-fill style max-trade clamp per bar |
| `CN_LIQUIDITY_PARTICIPATION_RATE` | `0.05` | Participation cap for `max_trade` (fraction of bar volume) |
| `CN_VOLUME_IMPACT` | `0.0` | Extra slippage term based on trade size vs volume |
| `CN_VOLUME_IMPACT_ALPHA` | `0.5` | Exponent for volume impact |
| `CN_STAMP_TAX_RATE` | `0.0` | Sell-side stamp tax rate (applied uniformly) |
| `COST_RATE_BUY` | *(unset)* | Optional buy-side commission override (else `COST_RATE`) |
| `COST_RATE_SELL` | *(unset)* | Optional sell-side commission override (else `COST_RATE`) |
| `CN_LIMIT_EXEMPT_FILE` | *(unset)* | Optional CSV to disable limit hits for given code/date ranges |

### Training (PPO)

| Variable | Default | Description |
|:---|:---|:---|
| `TRAIN_STEPS` | `400` | PPO training iterations |
| `BATCH_SIZE` | `1024` | Formulas sampled per step |
| `MAX_FORMULA_LEN` | `8` | Max tokens per formula |
| `PPO_EPOCHS` | `4` | PPO update epochs per step |

### Walk-Forward Optimization

| Variable | Default | Description |
|:---|:---|:---|
| `CN_WALK_FORWARD` | `0` | Set to `1` to enable |
| `CN_WFO_TRAIN_DAYS` | `60` | Training window |
| `CN_WFO_VAL_DAYS` | `20` | Validation window |
| `CN_WFO_TEST_DAYS` | `20` | Test window |
| `CN_WFO_STEP_DAYS` | `20` | Step size between folds |

## 🧩 Backtest Assumptions & Simplifications

Some A-share rules require **instrument metadata** (security type, ST flag, listing date, trading-status flags) or **L2/LOB** data. With minute OHLCV only, AShareGPT uses the following simplifications:

- **ST/*ST 5% price limit**: not detected automatically. ST stocks are treated as normal board codes (i.e. code-prefix-based limits only).
- **ETF vs Stock differences**: only modeled via whitelist (T+0 vs T+1). This repo ships a snapshot file `cn_t0_allowed_codes.csv` (SSE ETF category + SZSE ETF list, fetched on 2026-02-10), loaded by default through `CN_T0_ALLOWED_CODES_FILE`. If env+file yield no codes, it falls back to stock-style simplification (all symbols treated as T+1). Fees/taxes are not instrument-type aware unless you run separate universes or keep `CN_STAMP_TAX_RATE=0`.
- **New listing / special IPO limit rules**: not inferred from listing dates. If needed, provide `CN_LIMIT_EXEMPT_FILE` to exempt known date ranges.
- **Order book / queueing / limit-up封单**: not modeled. Instead, you can optionally enable an approximate execution cap (`CN_ENABLE_LIQUIDITY_CONSTRAINTS=1`) and volume-based impact (`CN_VOLUME_IMPACT>0`) to reduce “ideal fills”.

## 🧪 Data Utilities

### Clean Adjust Factors

Normalize date formats, remove duplicates, and validate adjust factor CSVs:

```bash
python clean_adj_factors.py data/复权因子/
```

### Unify Raw Downloads

Merge bulk-downloaded CSVs into per-code files:

```bash
python unify_data.py --mode all
```

### Backfill Adj Factors by Alias

Fill missing old-code adjust factor files using new-code mappings:

```bash
python scripts/backfill_adj_by_alias.py
```

## 🧠 Architecture Overview

```text
┌─────────────────────────────────────────────────┐
│              run_cn_train.py                    │
│                    │                            │
│     ┌──────────────▼─────────────┐              │
│     │     training.py (PPO)      │              │
│     │  ┌─────────┐ ┌──────────┐  │              │
│     │  │ Model   │ │ Reward   │  │              │
│     │  │ (Looped │ │ Orchestr.│  │              │
│     │  │ Transf.)│ │          │  │              │
│     │  └────┬────┘ └─────┬────┘  │              │
│     │       │            │       │              │
│     │  ┌────▼────┐  ┌────▼────┐  │              │
│     │  │ StackVM │  │Backtest │  │              │
│     │  │ (ops.py)│  │ Engine  │  │              │
│     │  └─────────┘  └─────────┘  │              │
│     └────────────────────────────┘              │
│                    │                            │
│     ┌──────────────▼─────────────┐              │
│     │   data_loader.py           │              │
│     │   factors.py (60+ feats)   │              │
│     │   cn_rules.py (T+1/涨跌停) │              │
│     └────────────────────────────┘              │
└─────────────────────────────────────────────────┘
```

1. **NeuralSymbolicAlphaGenerator** (`model.py`) — Looped Transformer with SwiGLU FFN and RMSNorm that generates RPN token sequences
2. **StackVM** (`vm.py`) — Executes formulas on GPU tensors using frequency-adaptive operators
3. **PPO Training** (`training.py`) — Reinforcement learning loop with reward = backtest Sharpe ratio
4. **ChinaBacktest** (`backtest.py`) — Vectorized backtester enforcing T+1 settlement, price-limit blocking, and tradability masks
5. **FeatureEngineer** (`factors.py`) — 60+ technical indicators computed via `pandas_ta` Strategy API
6. **ChinaMarketRules** (`cn_rules.py`) — Session-id tracking, per-code T+1/T+0 classification, 涨跌停 detection

## 📜 License

See [LICENSE](LICENSE) for details.
