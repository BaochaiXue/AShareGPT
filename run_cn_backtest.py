#!/usr/bin/env python
"""
A-share Strategy Backtest Script (Minute CSV only)

Usage:
    python run_cn_backtest.py --strategy best_cn_strategy.json
    python run_cn_backtest.py --symbols 000001.SZ,600519.SH
"""
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from model_core.config import ModelConfig
from model_core.data_loader import ChinaMinuteDataLoader
from model_core.backtest import ChinaBacktest
from model_core.vm import StackVM


def load_strategy(filepath: str) -> list:
    with open(filepath, "r") as f:
        return json.load(f)


def _fmt(v: float) -> str:
    return f"{v:.2%}"


def print_metrics(title: str, result) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)
    print(f"Sortino Score: {result.score:.4f}")
    print(f"Mean Return: {result.mean_return:.4%}")
    if not result.metrics:
        return
    m = result.metrics
    print(f"CAGR: {_fmt(m['cagr'])} | Annual Vol: {_fmt(m['annual_vol'])} | Sharpe: {m['sharpe']:.2f}")
    print(f"Max Drawdown: {_fmt(m['max_drawdown'])} | Calmar: {m['calmar']:.2f} | Win Rate: {_fmt(m['win_rate'])}")
    print(f"Profit Factor: {m['profit_factor']:.2f} | Expectancy: {_fmt(m['expectancy'])}")
    print(f"Avg Turnover: {_fmt(m['avg_turnover'])} | Long: {_fmt(m['long_ratio'])} | Short: {_fmt(m['short_ratio'])} | Flat: {_fmt(m['flat_ratio'])}")


def save_equity_curve(path: str, dates, result) -> None:
    if result.equity_curve is None or result.portfolio_returns is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "date": dates.astype("datetime64[ns]"),
        "equity": result.equity_curve.tolist() if hasattr(result.equity_curve, "tolist") else result.equity_curve,
        "return": result.portfolio_returns.tolist() if hasattr(result.portfolio_returns, "tolist") else result.portfolio_returns,
    })
    df.to_csv(out, index=False)
    print(f"📈 Equity curve saved: {out}")


def run_backtest(
    formula: list,
    symbols: Optional[list[str]] = None,
    split: bool = False,
    walk_forward: bool = False,
    curve_out: Optional[str] = None,
):
    print("Loading minute CSV data...")
    loader = ChinaMinuteDataLoader()
    loader.load_data(
        codes=symbols or ModelConfig.CN_CODES or None,
        years=ModelConfig.CN_MINUTE_YEARS or None,
        start_date=ModelConfig.CN_MINUTE_START_DATE,
        end_date=ModelConfig.CN_MINUTE_END_DATE,
        signal_time=ModelConfig.CN_SIGNAL_TIME,
        exit_time=ModelConfig.CN_EXIT_TIME,
        pool_size=ModelConfig.CN_POOL_SIZE,
    )

    vm = StackVM()
    bt = ChinaBacktest()

    print(f"Data shape: {loader.feat_tensor.shape}")
    print(f"Symbols: {len(loader.symbols)}")
    print(f"Date range: {loader.dates.min()} to {loader.dates.max()}")
    print("\nExecuting strategy formula...")

    factors = vm.execute(formula, loader.feat_tensor)
    if factors is None:
        print("❌ Invalid formula - execution failed.")
        return None

    import torch
    if torch.std(factors) < 1e-4:
        print("⚠️  Warning: Factor has near-zero variance (trivial formula).")

    print("Running backtest...\n")

    if walk_forward:
        folds = loader.walk_forward_splits()
        if not folds:
            print("⚠️  Walk-forward disabled: not enough data.")
            return None
        v_scores, t_scores = [], []
        for i, fold in enumerate(folds, 1):
            for label, sl in [("Val", fold.val), ("Test", fold.test)]:
                if sl.end_idx <= sl.start_idx:
                    continue
                sig = factors[:, sl.start_idx:sl.end_idx]
                r = bt.evaluate(sig, sl.raw_data_cache, sl.target_ret, return_details=True)
                print_metrics(f"Fold {i} - {label}", r)
                (v_scores if label == "Val" else t_scores).append(float(r.score.item()))
        if v_scores:
            print(f"\nAvg Val Score: {sum(v_scores)/len(v_scores):.4f}")
        if t_scores:
            print(f"Avg Test Score: {sum(t_scores)/len(t_scores):.4f}")
        return None

    if split:
        splits = loader.train_val_test_split()
        for name in ("train", "val", "test"):
            sl = splits.get(name)
            if sl is None or sl.end_idx <= sl.start_idx:
                continue
            sig = factors[:, sl.start_idx:sl.end_idx]
            r = bt.evaluate(sig, sl.raw_data_cache, sl.target_ret, return_details=True)
            print_metrics(f"{name.capitalize()} Results", r)
            if curve_out:
                out_path = Path(curve_out)
                save_equity_curve(str(out_path.with_name(f"{out_path.stem}_{name}{out_path.suffix or '.csv'}")),
                                  sl.dates, r)
        return None

    # Full sample
    r = bt.evaluate(factors, loader.raw_data_cache, loader.target_ret, return_details=True)
    print("=" * 60)
    print("📊 Backtest Results")
    print("=" * 60)
    print_metrics("Full Sample", r)
    if curve_out:
        save_equity_curve(curve_out, loader.dates, r)
    return {"score": float(r.score.item()), "mean_return": r.mean_return,
            "avg_turnover": r.metrics["avg_turnover"] if r.metrics else 0.0}


def main():
    parser = argparse.ArgumentParser(description="Run A-share minute backtest")
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--formula", type=str, default=None)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--curve-out", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("🇨🇳 AShareGPT Backtest (Minute CSV)")
    print("=" * 60)

    if args.formula:
        formula = json.loads(args.formula)
    elif args.strategy:
        formula = load_strategy(args.strategy)
    elif os.path.exists(ModelConfig.STRATEGY_FILE):
        formula = load_strategy(ModelConfig.STRATEGY_FILE)
        print(f"Using strategy file: {ModelConfig.STRATEGY_FILE}")
    else:
        print(f"❌ No strategy file found at {ModelConfig.STRATEGY_FILE}")
        sys.exit(1)

    print(f"Formula: {formula}")
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    if symbols:
        print(f"Testing on symbols: {symbols}")

    try:
        run_backtest(formula, symbols, split=args.split,
                     walk_forward=args.walk_forward, curve_out=args.curve_out)
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
