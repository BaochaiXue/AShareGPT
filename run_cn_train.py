#!/usr/bin/env python
"""
A-share Alpha Factor Mining Training Script (Minute CSV only)

Usage:
    python run_cn_train.py

Requirements:
    - Local minute CSVs under ./data/YYYY/<code>.csv
"""

import os
import sys

from model_core.engine import AlphaEngine


def main():
    print("=" * 60)
    print("🇨🇳 AShareGPT Training (Minute CSV)")
    print("=" * 60)
    print(f"Strategy Output: {os.environ.get('STRATEGY_FILE', 'best_cn_strategy.json')}")
    print()

    try:
        engine = AlphaEngine(use_lord_regularization=True)
        engine.train()

        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)

        if engine.best_formula:
            print(f"Best Score: {engine.best_score:.4f}")
            print(f"Best Formula: {engine.best_formula}")
        else:
            print("⚠️  No valid formula found. Try increasing TRAIN_STEPS.")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPossible causes:")
        print("  1. Minute CSVs not found - check ./data/YYYY/")
        print("  2. Date filters exclude all data")
        print("  3. CN_CODES/CN_MINUTE_YEARS not set")
        sys.exit(1)


if __name__ == "__main__":
    main()
