#!/usr/bin/env python
"""
A-share Strategy Backtest Script (Minute CSV only)

Usage:
    python run_cn_backtest.py --strategy best_cn_strategy.json
    python run_cn_backtest.py --symbols 000001.SZ,600519.SH
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import torch

from model_core.config import ModelConfig
from model_core.data_loader import ChinaMinuteDataLoader
from model_core.backtest import ChinaBacktest
from model_core.vm import StackVM


def load_strategy(filepath: str) -> list:
    with open(filepath, 'r') as f:
        return json.load(f)


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def print_metrics(title: str, result) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)
    print(f"Sortino Score: {result.score.item():.4f}")
    print(f"Mean Return: {result.mean_return:.4%}")
    if not result.metrics:
        return
    m = result.metrics
    print(f"CAGR: {_format_pct(m['cagr'])} | Annual Vol: {_format_pct(m['annual_vol'])} | Sharpe: {m['sharpe']:.2f}")
    print(f"Max Drawdown: {_format_pct(m['max_drawdown'])} | Calmar: {m['calmar']:.2f} | Win Rate: {_format_pct(m['win_rate'])}")
    print(f"Profit Factor: {m['profit_factor']:.2f} | Expectancy: {_format_pct(m['expectancy'])}")
    print(f"Avg Turnover: {_format_pct(m['avg_turnover'])} | Long: {_format_pct(m['long_ratio'])} | Short: {_format_pct(m['short_ratio'])} | Flat: {_format_pct(m['flat_ratio'])}")


def save_equity_curve(path: str, dates, result) -> None:
    if result.equity_curve is None or result.portfolio_returns is None:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "date": dates.astype("datetime64[ns]"),
            "equity": result.equity_curve.numpy(),
            "return": result.portfolio_returns.numpy(),
        }
    )
    df.to_csv(out_path, index=False)
    print(f"📈 Equity curve saved: {out_path}")


def run_backtest(
    formula: list,
    symbols: list | None = None,
    split: bool = False,
    walk_forward: bool = False,
    curve_out: str | None = None,
):
    print("Loading minute CSV data...")
    loader = ChinaMinuteDataLoader()
    loader.load_data(
        codes=symbols or ModelConfig.CN_CODES,
        years=ModelConfig.CN_MINUTE_YEARS,
        start_date=ModelConfig.CN_MINUTE_START_DATE,
        end_date=ModelConfig.CN_MINUTE_END_DATE,
        signal_time=ModelConfig.CN_SIGNAL_TIME,
        exit_time=ModelConfig.CN_EXIT_TIME,
        limit_codes=ModelConfig.CN_MAX_CODES,
    )

    print(f"Data shape: {loader.feat_tensor.shape}")
    print(f"Symbols: {len(loader.symbols) if loader.symbols else 'N/A'}")
    print(f"Date range: {loader.dates.min()} to {loader.dates.max()}")

    print("\nExecuting strategy formula...")
    vm = StackVM()
    factors = vm.execute(formula, loader.feat_tensor)

    if factors is None:
        print("❌ Invalid formula - execution failed")
        return

    if factors.std() < 1e-4:
        print("⚠️  Warning: Factor has near-zero variance (trivial formula)")

    print("\nRunning backtest...")
    bt = ChinaBacktest()

    if walk_forward:
        folds = loader.walk_forward_splits()
        if not folds:
            print("⚠️  Walk-forward disabled: not enough data for configured windows.")
        else:
            val_scores = []
            test_scores = []
            for idx, fold in enumerate(folds, 1):
                if fold.val.end_idx > fold.val.start_idx:
                    res_val = factors[:, fold.val.start_idx:fold.val.end_idx]
                    val_result = bt.evaluate(
                        res_val,
                        fold.val.raw_data_cache,
                        fold.val.target_ret,
                        return_details=True,
                    )
                    print_metrics(f"Fold {idx} - Validation", val_result)
                    val_scores.append(val_result.score.item())
                if fold.test.end_idx > fold.test.start_idx:
                    res_test = factors[:, fold.test.start_idx:fold.test.end_idx]
                    test_result = bt.evaluate(
                        res_test,
                        fold.test.raw_data_cache,
                        fold.test.target_ret,
                        return_details=True,
                    )
                    print_metrics(f"Fold {idx} - Test", test_result)
                    test_scores.append(test_result.score.item())
            if val_scores:
                print(f"\nWalk-forward Avg Val Score: {sum(val_scores) / len(val_scores):.4f}")
            if test_scores:
                print(f"Walk-forward Avg Test Score: {sum(test_scores) / len(test_scores):.4f}")
        return None

    if split:
        splits = loader.train_val_test_split()
        for name in ("train", "val", "test"):
            if name not in splits:
                continue
            split_slice = splits[name]
            res_slice = factors[:, split_slice.start_idx:split_slice.end_idx]
            result = bt.evaluate(
                res_slice,
                split_slice.raw_data_cache,
                split_slice.target_ret,
                return_details=True,
            )
            print_metrics(f"{name.capitalize()} Results", result)
            if curve_out:
                suffix = f"_{name}"
                out_path = Path(curve_out)
                out_file = out_path.with_name(f"{out_path.stem}{suffix}{out_path.suffix or '.csv'}")
                save_equity_curve(str(out_file), split_slice.dates, result)
        return None

    result = bt.evaluate(
        factors,
        loader.raw_data_cache,
        loader.target_ret,
        return_details=True,
    )

    print("\n" + "=" * 60)
    print("📊 Backtest Results")
    print("=" * 60)
    print_metrics("Full Sample", result)

    if curve_out:
        save_equity_curve(curve_out, loader.dates, result)

    return {
        'score': result.score.item(),
        'mean_return': result.mean_return,
        'avg_turnover': result.metrics["avg_turnover"] if result.metrics else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run A-share minute backtest")
    parser.add_argument("--strategy", type=str, default=None, help="Path to strategy JSON file")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated list of symbols")
    parser.add_argument("--formula", type=str, default=None, help="Formula as JSON string")
    parser.add_argument("--split", action="store_true", help="Report train/val/test metrics")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--curve-out", type=str, default=None, help="Save equity curve CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("🇨🇳 AShareGPT Backtest (Minute CSV)")
    print("=" * 60)

    if args.formula:
        formula = json.loads(args.formula)
    elif args.strategy:
        formula = load_strategy(args.strategy)
    else:
        if os.path.exists(ModelConfig.STRATEGY_FILE):
            formula = load_strategy(ModelConfig.STRATEGY_FILE)
            print(f"Using strategy file: {ModelConfig.STRATEGY_FILE}")
        else:
            print(f"❌ No strategy file found at {ModelConfig.STRATEGY_FILE}")
            print("   Train a model first: python run_cn_train.py")
            sys.exit(1)

    print(f"Formula: {formula}")

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
        print(f"Testing on symbols: {symbols}")

    try:
        run_backtest(
            formula,
            symbols,
            split=args.split,
            walk_forward=args.walk_forward,
            curve_out=args.curve_out,
        )
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
