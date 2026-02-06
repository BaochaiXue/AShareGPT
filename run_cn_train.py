#!/usr/bin/env python
"""
A-share Alpha Factor Mining Training Script (Minute CSV only)

Usage:
    python run_cn_train.py
"""
import os
import sys

from model_core.config import ModelConfig
from model_core.training import create_training_workflow


def main():
    print("=" * 60)
    print("🇨🇳 AShareGPT Training (Minute CSV)")
    print("=" * 60)
    print(f"Strategy Output: {ModelConfig.STRATEGY_FILE}")
    print(f"Decision Freq:   {ModelConfig.CN_DECISION_FREQ}")
    print()

    try:
        wf = create_training_workflow(use_lord=True)

        mode_label = "PPO + LoRD" if wf.use_lord else "PPO"
        print(f"🚀 Starting Alpha Mining with {mode_label}...")
        for line in wf.window_descriptions():
            print(line)

        def _on_new_best(score, mean_ret, formula):
            from tqdm import tqdm
            tqdm.write(f"[!] New King: Score {score:.2f} | Ret {mean_ret:.2%} | Formula {formula}")

        result = wf.run(
            strategy_path=ModelConfig.STRATEGY_FILE,
            on_new_best=_on_new_best,
        )

        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)

        if result.best_formula:
            print(f"Best Score: {result.best_score:.4f}")
            print(f"Best Formula: {result.best_formula}")
            for snap in result.evaluations:
                print(
                    f"  {snap.label}: Score {snap.score:.4f} | "
                    f"MeanRet {snap.mean_return:.2%} | Sharpe {snap.sharpe:.2f} | "
                    f"MaxDD {snap.max_drawdown:.2%}"
                )
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
