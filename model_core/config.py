import os
import torch

class ModelConfig:
    """Configuration for A-share minute backtest/training."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training defaults (A-share)
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1024"))
    TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "400"))
    MAX_FORMULA_LEN = int(os.getenv("MAX_FORMULA_LEN", "8"))
    INPUT_DIM = 58  # Updated for pandas_ta features
    PPO_EPOCHS = int(os.getenv("PPO_EPOCHS", "4"))
    PPO_CLIP_EPS = float(os.getenv("PPO_CLIP_EPS", "0.2"))
    PPO_VALUE_COEF = float(os.getenv("PPO_VALUE_COEF", "0.5"))
    PPO_ENTROPY_COEF = float(os.getenv("PPO_ENTROPY_COEF", "0.01"))
    PPO_MAX_GRAD_NORM = float(os.getenv("PPO_MAX_GRAD_NORM", "1.0"))

    # China market settings
    COST_RATE = float(os.getenv("COST_RATE", "0.0005"))
    SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0001"))
    SLIPPAGE_IMPACT = float(os.getenv("SLIPPAGE_IMPACT", "0.0"))
    ALLOW_SHORT = os.getenv("ALLOW_SHORT", "0") == "1"
    SIGNAL_LAG = int(os.getenv("CN_SIGNAL_LAG", "1"))
    ANNUALIZATION_FACTOR = int(os.getenv("ANNUALIZATION_FACTOR", "252"))
    STRATEGY_FILE = os.getenv("STRATEGY_FILE", "best_cn_strategy.json")
    CN_USE_MINUTE = os.getenv("CN_USE_MINUTE", "1") == "1"
    CN_MINUTE_DATA_ROOT = os.getenv("CN_MINUTE_DATA_ROOT", "data")
    CN_USE_ADJ_FACTOR = os.getenv("CN_USE_ADJ_FACTOR", "1") == "1"
    CN_ADJ_FACTOR_DIR = os.getenv("CN_ADJ_FACTOR_DIR", "复权因子")
    CN_CODE_ALIAS_FILE = os.getenv("CN_CODE_ALIAS_FILE", "code_alias_map.csv")
    CN_MINUTE_START_DATE = os.getenv("CN_MINUTE_START_DATE", "")
    CN_MINUTE_END_DATE = os.getenv("CN_MINUTE_END_DATE", "")
    CN_SIGNAL_TIME = os.getenv("CN_SIGNAL_TIME", "10:00")
    CN_EXIT_TIME = os.getenv("CN_EXIT_TIME", "15:00")
    CN_MAX_CODES = int(os.getenv("CN_MAX_CODES", "50"))
    CN_MINUTE_DAYS = int(os.getenv("CN_MINUTE_DAYS", "7"))
    CN_TRAIN_RATIO = float(os.getenv("CN_TRAIN_RATIO", "0.7"))
    CN_VAL_RATIO = float(os.getenv("CN_VAL_RATIO", "0.15"))
    CN_TEST_RATIO = float(os.getenv("CN_TEST_RATIO", "0.15"))
    CN_TRAIN_DAYS = int(os.getenv("CN_TRAIN_DAYS", "0"))
    CN_VAL_DAYS = int(os.getenv("CN_VAL_DAYS", "0"))
    CN_TEST_DAYS = int(os.getenv("CN_TEST_DAYS", "0"))
    CN_WALK_FORWARD = os.getenv("CN_WALK_FORWARD", "0") == "1"
    CN_WFO_TRAIN_DAYS = int(os.getenv("CN_WFO_TRAIN_DAYS", "60"))
    CN_WFO_VAL_DAYS = int(os.getenv("CN_WFO_VAL_DAYS", "20"))
    CN_WFO_TEST_DAYS = int(os.getenv("CN_WFO_TEST_DAYS", "20"))
    CN_WFO_STEP_DAYS = int(os.getenv("CN_WFO_STEP_DAYS", "20"))
    _CN_CODES_RAW = os.getenv("CN_CODES", "")
    CN_CODES = [c.strip() for c in _CN_CODES_RAW.split(",") if c.strip()]
    _CN_MINUTE_YEARS_RAW = os.getenv("CN_MINUTE_YEARS", "")
    CN_MINUTE_YEARS = [
        int(y.strip()) for y in _CN_MINUTE_YEARS_RAW.split(",") if y.strip().isdigit()
    ]
