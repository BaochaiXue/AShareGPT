import os
import torch

class ModelConfig:
    """Configuration for A-share minute backtest/training."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training defaults (A-share)
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1024"))
    TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "400"))
    MAX_FORMULA_LEN = int(os.getenv("MAX_FORMULA_LEN", "8"))
    INPUT_DIM = 5

    # China market settings
    COST_RATE = float(os.getenv("COST_RATE", "0.0005"))
    ALLOW_SHORT = os.getenv("ALLOW_SHORT", "0") == "1"
    STRATEGY_FILE = os.getenv("STRATEGY_FILE", "best_cn_strategy.json")
    CN_USE_MINUTE = os.getenv("CN_USE_MINUTE", "1") == "1"
    CN_MINUTE_DATA_ROOT = os.getenv("CN_MINUTE_DATA_ROOT", "data")
    CN_MINUTE_START_DATE = os.getenv("CN_MINUTE_START_DATE", "")
    CN_MINUTE_END_DATE = os.getenv("CN_MINUTE_END_DATE", "")
    CN_SIGNAL_TIME = os.getenv("CN_SIGNAL_TIME", "10:00")
    CN_EXIT_TIME = os.getenv("CN_EXIT_TIME", "15:00")
    CN_MAX_CODES = int(os.getenv("CN_MAX_CODES", "50"))
    CN_MINUTE_DAYS = int(os.getenv("CN_MINUTE_DAYS", "7"))
    _CN_CODES_RAW = os.getenv("CN_CODES", "")
    CN_CODES = [c.strip() for c in _CN_CODES_RAW.split(",") if c.strip()]
    _CN_MINUTE_YEARS_RAW = os.getenv("CN_MINUTE_YEARS", "")
    CN_MINUTE_YEARS = [
        int(y.strip()) for y in _CN_MINUTE_YEARS_RAW.split(",") if y.strip().isdigit()
    ]
