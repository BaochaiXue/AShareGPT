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

from model_core.config import ModelConfig
from model_core.entrypoints import create_backtest_use_case


def load_strategy(filepath: str) -> list:
    with open(filepath, 'r') as f:
        return json.load(f)


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def print_metrics(title: str, result) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)
    print(f"Sortino Score: {result.score:.4f}")
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
            "equity": result.equity_curve,
            "return": result.portfolio_returns,
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
    mode = "walk_forward" if walk_forward else ("split" if split else "full")
    backtest_use_case, _ = create_backtest_use_case()
    use_case_result = backtest_use_case.run(
        formula=formula,
        mode=mode,
        symbols=symbols,
        limit_codes=ModelConfig.CN_MAX_CODES,
        return_details=True,
    )
    if not use_case_result.ok:
        print(f"❌ {use_case_result.message}")
        return None

    payload = use_case_result.payload or {}
    dates = payload.get("dates")
    symbols_loaded = payload.get("symbols") or []
    feat_shape = payload.get("feat_shape")
    warnings = payload.get("warnings") or []

    if dates is None:
        print("❌ Backtest use-case returned incomplete payload")
        return None

    print(f"Data shape: {feat_shape}")
    print(f"Symbols: {len(symbols_loaded) if symbols_loaded else 'N/A'}")
    print(f"Date range: {dates.min()} to {dates.max()}")
    print("\nExecuting strategy formula...")
    for warning in warnings:
        print(f"⚠️  Warning: {warning}")
    print("\nRunning backtest...")

    if mode == "walk_forward":
        folds = payload.get("folds") or []
        if not folds:
            print(f"⚠️  {use_case_result.message}")
            return None
        for fold in folds:
            idx = fold.get("index")
            val_result = fold.get("val")
            test_result = fold.get("test")
            if val_result is not None:
                print_metrics(f"Fold {idx} - Validation", val_result)
            if test_result is not None:
                print_metrics(f"Fold {idx} - Test", test_result)
        avg_val_score = payload.get("avg_val_score")
        avg_test_score = payload.get("avg_test_score")
        if avg_val_score is not None:
            print(f"\nWalk-forward Avg Val Score: {avg_val_score:.4f}")
        if avg_test_score is not None:
            print(f"Walk-forward Avg Test Score: {avg_test_score:.4f}")
        return None

    if mode == "split":
        split_results = payload.get("splits") or {}
        for name in ("train", "val", "test"):
            out = split_results.get(name)
            if not out:
                continue
            result = out.get("result")
            split_dates = out.get("dates")
            if result is None or split_dates is None:
                continue
            print_metrics(f"{name.capitalize()} Results", result)
            if curve_out:
                suffix = f"_{name}"
                out_path = Path(curve_out)
                out_file = out_path.with_name(f"{out_path.stem}{suffix}{out_path.suffix or '.csv'}")
                save_equity_curve(str(out_file), split_dates, result)
        return None

    result = payload.get("result")
    if result is None:
        print("❌ Backtest use-case returned no full-sample result")
        return None

    print("\n" + "=" * 60)
    print("📊 Backtest Results")
    print("=" * 60)
    print_metrics("Full Sample", result)

    if curve_out:
        save_equity_curve(curve_out, dates, result)

    return {
        'score': result.score,
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
