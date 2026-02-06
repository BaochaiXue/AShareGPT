# AShareGPT

**AShareGPT** is a specialized Alpha mining and backtesting system for the Chinese A-share market. It combines **Reinforcement Learning (PPO)** with a **Symbolic Transformer (NeuralSymbolicAlphaGenerator)** to automatically generate, validate, and optimize formulaic alpha factors.

Unlike traditional "black box" deep learning models that predict prices directly, AShareGPT operates as a "White Box" Agent: it writes readable mathematical formulas (e.g., `(OPEN / MA_20) - 1`) that are then rigorously backtested.

## 📂 Data Layout

The system expects data in the following structure under the project root:

```text
AShareGPT/
├── data/
│   ├── 2025/               <-- Year Folders (Minute CSVs)
│   │   ├── 000001.SZ.csv
│   │   ├── 600519.SH.csv
│   │   └── ...
│   ├── 复权因子/           <-- Adjust Factors (Optional but Recommended)
│   │   ├── 000001.SZ.csv
│   │   └── ...
│   └── ...
```

### 1. Minute Data (`data/YYYY/<code>.csv`)
Raw minute-level OHLCV data.
- **Required Columns**: `trade_time`, `open`, `high`, `low`, `close`, `vol`, `amount`
- **Logic**: The loader downsamples this to daily signals. By default, it uses `10:00` for entry and `15:00` for exit (configurable).

### 2. Adjust Factors (`data/复权因子/<code>.csv`)
Used to handle splits and dividends.
- **Required Columns**: `date`, `adj_factor`
- **Fallback**: If missing, factor defaults to `1.0`.

## 🚀 Quick Start

### 1. Requirements
Ensure you have the necessary dependencies. Note that `pandas_ta` is required but might be missing from `requirements.txt`.

```bash
pip install -r requirements.txt
pip install pandas_ta  # or pandas_ta_classic
```

### 2. Train (Mine Alphas)
Start the PPO training loop to discover new formulas.

```bash
python run_cn_train.py
```
- **Output**: The best discovered formula is saved to `best_cn_strategy.json`.
- **Compute**: Supports CUDA if available.

### 3. Backtest
Validate a discovered strategy on historical data.

```bash
python run_cn_backtest.py --strategy best_cn_strategy.json
```
- **Key Metrics**: Sharpe Ratio, Sortino Ratio, Annual Returns, Max Drawdown.
- **Curve Output**: Use `--curve-out equity.csv` to save the capital curve.

## ⚙️ Configuration

Key settings in `model_core/config.py` can be overridden via Environment Variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CN_MINUTE_YEARS` | *(Last 2 Years)* | Comma-separated years to load (e.g., `2020,2021,2022`). |
| `CN_MAX_CODES` | `50` | Max number of stocks to load (set higher for full market). |
| `CN_SIGNAL_TIME` | `10:00` | Intraday time to generate signal/enter trade. |
| `CN_EXIT_TIME` | `15:00` | Intraday time to close position. |
| `TRAIN_STEPS` | `400` | Number of PPO training iterations. |
| `CN_WALK_FORWARD`| `0` | Set to `1` to enable Walk-Forward Optimization. |

## 🧠 System Architecture

- **NeuralSymbolicAlphaGenerator (`model_core/neural_symbolic_alpha_generator.py`)**: A Looped Transformer that generates Reverse Polish Notation (RPN) formulas.
- **StackVM (`model_core/vm.py`)**: A vectorized stack machine that executes these formulas on GPU tensors.
- **Engine (`model_core/engine.py`)**: The PPO Reinforcement Learning loop that rewards the model based on Backtest Sharpe Ratio.
- **Data Loader (`model_core/data_loader.py`)**: Handles complex A-share data alignment, robust normalization, and feature engineering (60+ technical indicators).

## Migration Notes
- **v0.6.0**: `model_core.infrastructure.legacy` removed. Use adapters if needed.
- **Pandas TA**: Ensure you have a compatible version installed for feature generation (`RSI`, `MACD`, etc.).
