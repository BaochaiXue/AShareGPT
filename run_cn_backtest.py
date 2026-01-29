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

import torch

from model_core.config import ModelConfig
from model_core.data_loader import ChinaMinuteDataLoader
from model_core.backtest import ChinaBacktest
from model_core.vm import StackVM


def load_strategy(filepath: str) -> list:
    with open(filepath, 'r') as f:
        return json.load(f)


def run_backtest(formula: list, symbols: list | None = None):
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
    score, mean_ret = bt.evaluate(
        factors.unsqueeze(0),
        loader.raw_data_cache,
        loader.target_ret,
    )

    print("\n" + "=" * 60)
    print("📊 Backtest Results")
    print("=" * 60)
    print(f"Sortino Score: {score.item():.4f}")
    print(f"Mean Return: {mean_ret:.4%}")

    signal = torch.tanh(factors)
    position = torch.sign(signal)

    turnover = torch.abs(position - torch.roll(position, 1, dims=1))
    turnover[:, 0] = 0
    avg_turnover = turnover.mean().item()

    long_pct = (position > 0).float().mean().item()
    short_pct = (position < 0).float().mean().item()
    flat_pct = (position == 0).float().mean().item()

    print("\n📈 Strategy Statistics:")
    print(f"Avg Turnover: {avg_turnover:.2%}")
    print(f"Long Positions: {long_pct:.1%}")
    print(f"Short Positions: {short_pct:.1%}")
    print(f"Flat Positions: {flat_pct:.1%}")

    return {
        'score': score.item(),
        'mean_return': mean_ret,
        'avg_turnover': avg_turnover,
    }


def main():
    parser = argparse.ArgumentParser(description="Run A-share minute backtest")
    parser.add_argument("--strategy", type=str, default=None, help="Path to strategy JSON file")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated list of symbols")
    parser.add_argument("--formula", type=str, default=None, help="Formula as JSON string")
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
        run_backtest(formula, symbols)
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
