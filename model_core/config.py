import csv
import os
from pathlib import Path

import torch


def _parse_code_list(raw_codes: str) -> list[str]:
    return [code.strip() for code in raw_codes.split(",") if code.strip()]


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _load_codes_from_csv(path_value: str) -> list[str]:
    if not path_value:
        return []
    path = _resolve_repo_path(path_value)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            field_map = {name.strip().lower(): name for name in reader.fieldnames}
            code_col = (
                field_map.get("code")
                or field_map.get("fund_code")
                or field_map.get("security_code")
            )
            if code_col:
                parsed_codes = [(row.get(code_col) or "").strip() for row in reader]
            else:
                parsed_codes = [(next(iter(row.values()), "") or "").strip() for row in reader]
        else:
            handle.seek(0)
            parsed_codes = [line.strip().split(",")[0] for line in handle if line.strip()]

    seen: set[str] = set()
    unique_codes: list[str] = []
    for code in parsed_codes:
        if not code or code in seen:
            continue
        seen.add(code)
        unique_codes.append(code)
    return unique_codes


class ModelConfig:
    """Configuration for A-share minute backtest/training."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training defaults (A-share)
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1024"))
    TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "400"))
    MAX_FORMULA_LEN = int(os.getenv("MAX_FORMULA_LEN", "8"))
    PPO_EPOCHS = int(os.getenv("PPO_EPOCHS", "4"))
    PPO_CLIP_EPS = float(os.getenv("PPO_CLIP_EPS", "0.2"))
    PPO_VALUE_COEF = float(os.getenv("PPO_VALUE_COEF", "0.5"))
    PPO_ENTROPY_COEF = float(os.getenv("PPO_ENTROPY_COEF", "0.01"))
    PPO_MAX_GRAD_NORM = float(os.getenv("PPO_MAX_GRAD_NORM", "1.0"))

    # China market settings
    COST_RATE = float(os.getenv("COST_RATE", "0.0005"))
    # Optional side-specific commission rates. When unset/empty, fall back to COST_RATE.
    _COST_RATE_BUY_RAW = os.getenv("COST_RATE_BUY", "").strip()
    COST_RATE_BUY = float(_COST_RATE_BUY_RAW) if _COST_RATE_BUY_RAW else None
    _COST_RATE_SELL_RAW = os.getenv("COST_RATE_SELL", "").strip()
    COST_RATE_SELL = float(_COST_RATE_SELL_RAW) if _COST_RATE_SELL_RAW else None
    SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0001"))
    SLIPPAGE_IMPACT = float(os.getenv("SLIPPAGE_IMPACT", "0.0"))
    ALLOW_SHORT = os.getenv("ALLOW_SHORT", "0") == "1"
    SIGNAL_LAG = int(os.getenv("CN_SIGNAL_LAG", "1"))
    # Annualization: auto-compute for 1min (240 bars/day * 252 trading days)
    _CN_DECISION_FREQ_RAW = os.getenv("CN_DECISION_FREQ", "daily").strip().lower()
    _DEFAULT_ANN = "60480" if _CN_DECISION_FREQ_RAW == "1min" else "252"
    ANNUALIZATION_FACTOR = int(os.getenv("ANNUALIZATION_FACTOR", _DEFAULT_ANN))
    # Price-limit (涨跌停) detection tolerance
    CN_LIMIT_HIT_TOL = float(os.getenv("CN_LIMIT_HIT_TOL", "0.001"))
    CN_TICK_SIZE = float(os.getenv("CN_TICK_SIZE", "0.01"))
    CN_LOT_SIZE = int(os.getenv("CN_LOT_SIZE", "100"))
    # Tax/fee asymmetry (generic; instrument-specific exemptions require metadata)
    CN_STAMP_TAX_RATE = float(os.getenv("CN_STAMP_TAX_RATE", "0.0"))
    # Optional CSV for listing-day / special limit exemptions:
    # code,start_date[,end_date]
    CN_LIMIT_EXEMPT_FILE = os.getenv("CN_LIMIT_EXEMPT_FILE", "")
    STRATEGY_FILE = os.getenv("STRATEGY_FILE", "best_cn_strategy.json")
    CN_MINUTE_DATA_ROOT = os.getenv("CN_MINUTE_DATA_ROOT", "data")
    CN_USE_ADJ_FACTOR = os.getenv("CN_USE_ADJ_FACTOR", "1") == "1"
    CN_ADJ_FACTOR_DIR = os.getenv("CN_ADJ_FACTOR_DIR", "复权因子")
    CN_CODE_ALIAS_FILE = os.getenv("CN_CODE_ALIAS_FILE", "code_alias_map.csv")
    CN_MINUTE_START_DATE = os.getenv("CN_MINUTE_START_DATE", "")
    CN_MINUTE_END_DATE = os.getenv("CN_MINUTE_END_DATE", "")
    CN_SIGNAL_TIME = os.getenv("CN_SIGNAL_TIME", "10:00")
    CN_EXIT_TIME = os.getenv("CN_EXIT_TIME", "15:00")
    # Trading time filter (1min bars). When enabled, bars outside continuous trading
    # hours are treated as non-tradable.
    CN_ENFORCE_TRADING_HOURS = os.getenv("CN_ENFORCE_TRADING_HOURS", "1") == "1"
    # Tradable inference from minute OHLCV (volume/amount == 0 treated as non-tradable).
    CN_TRADABLE_REQUIRE_LIQUIDITY = os.getenv("CN_TRADABLE_REQUIRE_LIQUIDITY", "1") == "1"
    # Liquidity/partial-fill constraints (approximate). Uses participation rate × volume.
    CN_ENABLE_LIQUIDITY_CONSTRAINTS = os.getenv("CN_ENABLE_LIQUIDITY_CONSTRAINTS", "0") == "1"
    CN_LIQUIDITY_PARTICIPATION_RATE = float(os.getenv("CN_LIQUIDITY_PARTICIPATION_RATE", "0.05"))
    # Additional slippage term based on trade size / bar volume:
    # slip += turnover * CN_VOLUME_IMPACT * (trade_shares / volume) ** CN_VOLUME_IMPACT_ALPHA
    CN_VOLUME_IMPACT = float(os.getenv("CN_VOLUME_IMPACT", "0.0"))
    CN_VOLUME_IMPACT_ALPHA = float(os.getenv("CN_VOLUME_IMPACT_ALPHA", "0.5"))
    # Execution rule controls:
    # - CN_ENFORCE_T_PLUS_ONE=1 applies same-day sell blocking by default.
    # - CN_T0_ALLOWED_CODES can whitelist symbols that are allowed to sell intraday.
    CN_ENFORCE_T_PLUS_ONE = os.getenv("CN_ENFORCE_T_PLUS_ONE", "1") == "1"
    CN_T0_ALLOWED_CODES_FILE = os.getenv("CN_T0_ALLOWED_CODES_FILE", "cn_t0_allowed_codes.csv").strip()
    _CN_T0_ALLOWED_CODES_RAW = os.getenv("CN_T0_ALLOWED_CODES", "")
    CN_T0_ALLOWED_CODES = _parse_code_list(_CN_T0_ALLOWED_CODES_RAW)
    if not CN_T0_ALLOWED_CODES:
        CN_T0_ALLOWED_CODES = _load_codes_from_csv(CN_T0_ALLOWED_CODES_FILE)
    # Decision frequency:
    # - daily: aggregate minute data to one bar per day.
    # - 1min: keep minute bars as the decision timeline.
    CN_DECISION_FREQ = os.getenv("CN_DECISION_FREQ", "daily").strip().lower()
    # Bar/return semantics:
    # - CN_BAR_STYLE=daily         -> full-session OHLCV bars.
    # - CN_BAR_STYLE=signal_snapshot -> legacy single-minute snapshot bars.
    CN_BAR_STYLE = os.getenv("CN_BAR_STYLE", "daily").strip().lower()
    # - CN_TARGET_RET_MODE=close_to_close -> close[t]/close[t-hold_days]-1 (T+1-friendly by default).
    # - CN_TARGET_RET_MODE=signal_to_exit -> legacy same-day signal_time->exit_time return.
    CN_TARGET_RET_MODE = os.getenv("CN_TARGET_RET_MODE", "close_to_close").strip().lower()
    CN_HOLD_DAYS = int(os.getenv("CN_HOLD_DAYS", "1"))
    CN_HOLD_BARS = int(os.getenv("CN_HOLD_BARS", "1"))
    CN_POOL_SIZE = int(os.getenv("CN_POOL_SIZE", "50"))
    CN_MINUTE_DAYS = int(os.getenv("CN_MINUTE_DAYS", "120"))
    CN_TRAIN_RATIO = float(os.getenv("CN_TRAIN_RATIO", "0.7"))
    CN_VAL_RATIO = float(os.getenv("CN_VAL_RATIO", "0.0"))
    CN_TEST_RATIO = float(os.getenv("CN_TEST_RATIO", "0.3"))
    CN_TRAIN_DAYS = int(os.getenv("CN_TRAIN_DAYS", "0"))
    CN_VAL_DAYS = int(os.getenv("CN_VAL_DAYS", "0"))
    CN_TEST_DAYS = int(os.getenv("CN_TEST_DAYS", "0"))
    CN_WALK_FORWARD = os.getenv("CN_WALK_FORWARD", "0") == "1"
    CN_WFO_TRAIN_DAYS = int(os.getenv("CN_WFO_TRAIN_DAYS", "60"))
    CN_WFO_VAL_DAYS = int(os.getenv("CN_WFO_VAL_DAYS", "20"))
    CN_WFO_TEST_DAYS = int(os.getenv("CN_WFO_TEST_DAYS", "20"))
    CN_WFO_STEP_DAYS = int(os.getenv("CN_WFO_STEP_DAYS", "20"))
    CN_FEATURE_NORM = os.getenv("CN_FEATURE_NORM", "train").strip().lower()
    CN_FEATURE_CLIP = float(os.getenv("CN_FEATURE_CLIP", "5.0"))
    CN_REWARD_MODE = os.getenv("CN_REWARD_MODE", "train").strip().lower()
    CN_STRICT_FEATURE_INDICATORS = os.getenv("CN_STRICT_FEATURE_INDICATORS", "1") == "1"
    CN_FEATURE_NEAR_ZERO_STD_TOL = float(os.getenv("CN_FEATURE_NEAR_ZERO_STD_TOL", "1e-6"))
    _CN_CODES_RAW = os.getenv("CN_CODES", "")
    CN_CODES = [c.strip() for c in _CN_CODES_RAW.split(",") if c.strip()]
    _CN_MINUTE_YEARS_RAW = os.getenv("CN_MINUTE_YEARS", "")
    CN_MINUTE_YEARS = [
        int(y.strip()) for y in _CN_MINUTE_YEARS_RAW.split(",") if y.strip().isdigit()
    ]
