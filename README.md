# AShareGPT

A-share (China) factor mining and backtesting using **minute CSV** data.

## Data Layout
```
AShareGPT/
  data/
    2025/
      000001.SZ.csv
      600519.SH.csv
```
Each CSV must contain columns:
`trade_time, open, high, low, close, vol, amount`

## Quick Start
```bash
cd AShareGPT
python run_cn_train.py
python run_cn_backtest.py --strategy best_cn_strategy.json
```

## Environment Variables
See `.env.example` for settings (signal minute, exit minute, date filters).

## Notes
- Default backtest window is the **latest 7 trading days** unless you set `CN_MINUTE_END_DATE`.
- Features are the 5 basic factors: `RET, RET5, VOL_CHG, V_RET, TREND`.
