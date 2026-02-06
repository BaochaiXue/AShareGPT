from __future__ import annotations

from datetime import time
from pathlib import Path

import pandas as pd
import torch

from model_core.config import ModelConfig
from model_core.data_loader import ChinaMinuteDataLoader
from model_core.factors import FeatureEngineer


def _write_minute_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _fake_features(
    raw_dict: dict[str, torch.Tensor],
    *,
    normalize: bool = True,
    norm_stats=None,
    clip: float = 5.0,
    strict_indicator_mapping: bool = True,
    near_zero_std_tol: float = 1e-6,
) -> torch.Tensor:
    # Stable lightweight feature stub for loader pipeline tests.
    close = raw_dict["close"]
    return torch.stack([close, close * 0.5], dim=1)


def _base_config(monkeypatch) -> None:
    monkeypatch.setattr(ModelConfig, "CN_USE_ADJ_FACTOR", False)
    monkeypatch.setattr(ModelConfig, "CN_MINUTE_DAYS", 0)
    monkeypatch.setattr(ModelConfig, "CN_DECISION_FREQ", "daily")
    monkeypatch.setattr(ModelConfig, "CN_BAR_STYLE", "daily")
    monkeypatch.setattr(ModelConfig, "CN_TARGET_RET_MODE", "close_to_close")
    monkeypatch.setattr(ModelConfig, "CN_HOLD_DAYS", 1)
    monkeypatch.setattr(ModelConfig, "CN_HOLD_BARS", 1)
    monkeypatch.setattr(ModelConfig, "CN_ENFORCE_T_PLUS_ONE", True)
    monkeypatch.setattr(ModelConfig, "CN_T0_ALLOWED_CODES", [])
    monkeypatch.setattr(ModelConfig, "CN_FEATURE_NORM", "none")
    monkeypatch.setattr(ModelConfig, "CN_STRICT_FEATURE_INDICATORS", True)
    monkeypatch.setattr(ModelConfig, "CN_FEATURE_NEAR_ZERO_STD_TOL", 0.0)
    monkeypatch.setattr(ModelConfig, "CN_TRAIN_DAYS", 0)
    monkeypatch.setattr(ModelConfig, "CN_VAL_DAYS", 0)
    monkeypatch.setattr(ModelConfig, "CN_TEST_DAYS", 0)
    monkeypatch.setattr(FeatureEngineer, "compute_features", staticmethod(_fake_features))


def test_load_data_matches_helper_pipeline_snapshot(tmp_path: Path, monkeypatch) -> None:
    _base_config(monkeypatch)
    monkeypatch.setattr(ModelConfig, "CN_TRAIN_RATIO", 0.6)
    monkeypatch.setattr(ModelConfig, "CN_VAL_RATIO", 0.2)

    rows_000001 = [
        {"trade_time": "2024-01-02 10:00:00", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "vol": 100.0, "amount": 1000.0},
        {"trade_time": "2024-01-02 15:00:00", "open": 10.2, "high": 10.3, "low": 10.1, "close": 10.25, "vol": 80.0, "amount": 900.0},
        {"trade_time": "2024-01-03 10:00:00", "open": 10.3, "high": 10.4, "low": 10.2, "close": 10.35, "vol": 90.0, "amount": 920.0},
        {"trade_time": "2024-01-03 15:00:00", "open": 10.4, "high": 10.6, "low": 10.3, "close": 10.5, "vol": 88.0, "amount": 940.0},
    ]
    rows_000002 = [
        {"trade_time": "2024-01-02 10:00:00", "open": 20.0, "high": 20.2, "low": 19.9, "close": 20.1, "vol": 120.0, "amount": 2000.0},
        {"trade_time": "2024-01-02 15:00:00", "open": 20.2, "high": 20.3, "low": 20.0, "close": 20.22, "vol": 110.0, "amount": 2020.0},
        {"trade_time": "2024-01-03 10:00:00", "open": 20.1, "high": 20.3, "low": 20.0, "close": 20.15, "vol": 105.0, "amount": 2010.0},
        {"trade_time": "2024-01-03 15:00:00", "open": 20.2, "high": 20.4, "low": 20.1, "close": 20.28, "vol": 106.0, "amount": 2030.0},
    ]
    _write_minute_csv(tmp_path / "2024" / "000001.csv", rows_000001)
    _write_minute_csv(tmp_path / "2024" / "000002.csv", rows_000002)

    loader = ChinaMinuteDataLoader(data_root=str(tmp_path))
    loader.load_data(codes=["000001", "000002"], years=[2024], signal_time="10:00", exit_time="15:00")

    helper_loader = ChinaMinuteDataLoader(data_root=str(tmp_path))
    sig_time = helper_loader._parse_time("10:00") or time(10, 0)
    exit_t = helper_loader._parse_time("15:00")
    bar_style = helper_loader._resolve_bar_style()
    target_ret_mode, hold_days = helper_loader._resolve_target_ret_mode()
    start_dt, end_dt, end_dt_exclusive = helper_loader._resolve_time_bounds("", "")
    per_code = helper_loader._build_per_code_frames(
        codes=["000001", "000002"],
        years=[2024],
        start_dt=start_dt,
        end_dt=end_dt,
        end_dt_exclusive=end_dt_exclusive,
        sig_time=sig_time,
        exit_t=exit_t,
        bar_style=bar_style,
        target_ret_mode=target_ret_mode,
        hold_days=hold_days,
    )
    helper_loader._apply_recent_day_cutoff(per_code, end_dt=end_dt)
    pivots = helper_loader._build_pivots(per_code)
    index = pivots["close"].index
    columns = pivots["close"].columns
    train_len, val_len, test_len = helper_loader._resolve_split_sizes(len(index))
    helper_loader._validate_split_order(index, train_len, val_len, test_len)
    raw_data_cache, target_tensor = helper_loader._build_tensors_from_pivots(
        pivots,
        index=index,
        columns=columns,
    )
    raw_feat = FeatureEngineer.compute_features(
        raw_data_cache,
        normalize=False,
        strict_indicator_mapping=ModelConfig.CN_STRICT_FEATURE_INDICATORS,
        near_zero_std_tol=ModelConfig.CN_FEATURE_NEAR_ZERO_STD_TOL,
    )
    feat_tensor = helper_loader._normalize_features(raw_feat, train_len=train_len)

    assert loader.dates is not None
    assert loader.symbols is not None
    assert loader.feat_tensor is not None
    assert loader.target_ret is not None
    assert loader.raw_data_cache is not None

    assert list(loader.dates) == list(index)
    assert loader.symbols == list(columns)
    assert torch.allclose(loader.feat_tensor, feat_tensor, equal_nan=True)
    assert torch.allclose(loader.target_ret, target_tensor, equal_nan=True)
    assert torch.allclose(loader.raw_data_cache["close"], raw_data_cache["close"], equal_nan=True)


def test_load_data_split_lengths(tmp_path: Path, monkeypatch) -> None:
    _base_config(monkeypatch)
    monkeypatch.setattr(ModelConfig, "CN_TRAIN_RATIO", 0.5)
    monkeypatch.setattr(ModelConfig, "CN_VAL_RATIO", 0.25)

    rows = []
    for day, close in (
        ("2024-01-02", 10.1),
        ("2024-01-03", 10.2),
        ("2024-01-04", 10.3),
        ("2024-01-05", 10.4),
    ):
        rows.extend(
            [
                {"trade_time": f"{day} 10:00:00", "open": 10.0, "high": 10.2, "low": 9.9, "close": close, "vol": 100.0, "amount": 1000.0},
                {"trade_time": f"{day} 15:00:00", "open": 10.1, "high": 10.3, "low": 10.0, "close": close + 0.1, "vol": 80.0, "amount": 900.0},
            ]
        )
    _write_minute_csv(tmp_path / "2024" / "000001.csv", rows)

    loader = ChinaMinuteDataLoader(data_root=str(tmp_path))
    loader.load_data(codes=["000001"], years=[2024], signal_time="10:00", exit_time="15:00")
    splits = loader.train_val_test_split()

    assert len(splits["train"].dates) == 2
    assert len(splits["val"].dates) == 1
    assert len(splits["test"].dates) == 1


def test_target_return_missing_mask_preserved(tmp_path: Path, monkeypatch) -> None:
    _base_config(monkeypatch)
    monkeypatch.setattr(ModelConfig, "CN_TRAIN_RATIO", 0.7)
    monkeypatch.setattr(ModelConfig, "CN_VAL_RATIO", 0.0)

    rows_000001 = [
        {"trade_time": "2024-01-02 10:00:00", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "vol": 100.0, "amount": 1000.0},
        {"trade_time": "2024-01-02 15:00:00", "open": 10.1, "high": 10.2, "low": 10.0, "close": 10.1, "vol": 80.0, "amount": 900.0},
        {"trade_time": "2024-01-03 10:00:00", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "vol": 100.0, "amount": 1000.0},
        {"trade_time": "2024-01-03 15:00:00", "open": 10.2, "high": 10.3, "low": 10.1, "close": 10.2, "vol": 80.0, "amount": 900.0},
    ]
    rows_000002 = [
        {"trade_time": "2024-01-02 10:00:00", "open": 20.0, "high": 20.1, "low": 19.9, "close": 20.0, "vol": 100.0, "amount": 2000.0},
        {"trade_time": "2024-01-02 15:00:00", "open": 20.1, "high": 20.2, "low": 20.0, "close": 20.2, "vol": 90.0, "amount": 2100.0},
    ]
    _write_minute_csv(tmp_path / "2024" / "000001.csv", rows_000001)
    _write_minute_csv(tmp_path / "2024" / "000002.csv", rows_000002)

    loader = ChinaMinuteDataLoader(data_root=str(tmp_path))
    loader.load_data(codes=["000001", "000002"], years=[2024], signal_time="10:00", exit_time="15:00")

    assert loader.target_ret is not None
    assert loader.symbols is not None
    idx_000002 = loader.symbols.index("000002")
    # Second day for 000002 is missing, and should remain NaN in target_ret.
    assert torch.isnan(loader.target_ret[idx_000002, 1]).item()


def test_adj_factor_pipeline_updates_prices(tmp_path: Path, monkeypatch) -> None:
    _base_config(monkeypatch)
    monkeypatch.setattr(ModelConfig, "CN_USE_ADJ_FACTOR", True)
    monkeypatch.setattr(ModelConfig, "CN_ADJ_FACTOR_DIR", "adj")
    monkeypatch.setattr(ModelConfig, "CN_TRAIN_RATIO", 1.0)
    monkeypatch.setattr(ModelConfig, "CN_VAL_RATIO", 0.0)

    rows = [
        {"trade_time": "2024-01-02 10:00:00", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1, "vol": 100.0, "amount": 1000.0},
        {"trade_time": "2024-01-02 15:00:00", "open": 10.2, "high": 10.4, "low": 10.1, "close": 10.3, "vol": 80.0, "amount": 900.0},
    ]
    _write_minute_csv(tmp_path / "2024" / "000001.csv", rows)
    adj_path = tmp_path / "adj" / "000001.csv"
    adj_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"date": "2024-01-02", "adj_factor": 2.0}]).to_csv(adj_path, index=False)

    loader = ChinaMinuteDataLoader(data_root=str(tmp_path))
    loader.load_data(codes=["000001"], years=[2024], signal_time="10:00", exit_time="15:00")

    assert loader.raw_data_cache is not None
    assert torch.allclose(loader.raw_data_cache["open"][0, 0], torch.tensor(20.0))
    assert "adj_factor" in loader.raw_data_cache
    assert torch.allclose(loader.raw_data_cache["adj_factor"][0, 0], torch.tensor(2.0))


def test_daily_bar_aggregates_full_session_ohlcv(tmp_path: Path, monkeypatch) -> None:
    _base_config(monkeypatch)
    monkeypatch.setattr(ModelConfig, "CN_TRAIN_RATIO", 1.0)
    monkeypatch.setattr(ModelConfig, "CN_VAL_RATIO", 0.0)

    rows = [
        {"trade_time": "2024-01-02 09:31:00", "open": 10.0, "high": 10.3, "low": 9.9, "close": 10.2, "vol": 100.0, "amount": 1000.0},
        {"trade_time": "2024-01-02 10:00:00", "open": 10.2, "high": 10.4, "low": 10.1, "close": 10.3, "vol": 90.0, "amount": 950.0},
        {"trade_time": "2024-01-02 15:00:00", "open": 10.35, "high": 10.6, "low": 10.2, "close": 10.5, "vol": 110.0, "amount": 1200.0},
    ]
    _write_minute_csv(tmp_path / "2024" / "000001.csv", rows)

    loader = ChinaMinuteDataLoader(data_root=str(tmp_path))
    loader.load_data(codes=["000001"], years=[2024], signal_time="10:00", exit_time="15:00")

    assert loader.raw_data_cache is not None
    assert torch.allclose(loader.raw_data_cache["open"][0, 0], torch.tensor(10.0))
    assert torch.allclose(loader.raw_data_cache["high"][0, 0], torch.tensor(10.6))
    assert torch.allclose(loader.raw_data_cache["low"][0, 0], torch.tensor(9.9))
    assert torch.allclose(loader.raw_data_cache["close"][0, 0], torch.tensor(10.5))
    assert torch.allclose(loader.raw_data_cache["volume"][0, 0], torch.tensor(300.0))
    assert torch.allclose(loader.raw_data_cache["amount"][0, 0], torch.tensor(3150.0))


def test_close_to_close_return_is_t_plus_one(tmp_path: Path, monkeypatch) -> None:
    _base_config(monkeypatch)
    monkeypatch.setattr(ModelConfig, "CN_TRAIN_RATIO", 1.0)
    monkeypatch.setattr(ModelConfig, "CN_VAL_RATIO", 0.0)

    rows = [
        {"trade_time": "2024-01-02 10:00:00", "open": 9.8, "high": 10.0, "low": 9.7, "close": 9.9, "vol": 100.0, "amount": 1000.0},
        {"trade_time": "2024-01-02 15:00:00", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0, "vol": 80.0, "amount": 900.0},
        {"trade_time": "2024-01-03 10:00:00", "open": 10.1, "high": 10.3, "low": 10.0, "close": 10.2, "vol": 90.0, "amount": 920.0},
        {"trade_time": "2024-01-03 15:00:00", "open": 10.4, "high": 10.6, "low": 10.3, "close": 10.5, "vol": 88.0, "amount": 940.0},
    ]
    _write_minute_csv(tmp_path / "2024" / "000001.csv", rows)

    loader = ChinaMinuteDataLoader(data_root=str(tmp_path))
    loader.load_data(codes=["000001"], years=[2024], signal_time="10:00", exit_time="15:00")

    assert loader.target_ret is not None
    # Day 1 has no previous close -> NaN; day 2 uses close[2]/close[1]-1 = 10.5/10.0-1.
    assert torch.isnan(loader.target_ret[0, 0]).item()
    assert torch.allclose(loader.target_ret[0, 1], torch.tensor(0.05), atol=1e-6)


def test_minute_decision_frequency_uses_minute_timeline_and_hold_bars(
    tmp_path: Path, monkeypatch
) -> None:
    _base_config(monkeypatch)
    monkeypatch.setattr(ModelConfig, "CN_DECISION_FREQ", "1min")
    monkeypatch.setattr(ModelConfig, "CN_HOLD_BARS", 2)
    monkeypatch.setattr(ModelConfig, "CN_TRAIN_RATIO", 1.0)
    monkeypatch.setattr(ModelConfig, "CN_VAL_RATIO", 0.0)

    rows = [
        {"trade_time": "2024-01-02 09:30:00", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "vol": 10.0, "amount": 100.0},
        {"trade_time": "2024-01-02 09:31:00", "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0, "vol": 10.0, "amount": 110.0},
        {"trade_time": "2024-01-02 09:32:00", "open": 12.0, "high": 12.0, "low": 12.0, "close": 12.0, "vol": 10.0, "amount": 120.0},
        {"trade_time": "2024-01-02 09:33:00", "open": 13.0, "high": 13.0, "low": 13.0, "close": 13.0, "vol": 10.0, "amount": 130.0},
        {"trade_time": "2024-01-02 09:34:00", "open": 14.0, "high": 14.0, "low": 14.0, "close": 14.0, "vol": 10.0, "amount": 140.0},
    ]
    _write_minute_csv(tmp_path / "2024" / "000001.csv", rows)

    loader = ChinaMinuteDataLoader(data_root=str(tmp_path))
    loader.load_data(codes=["000001"], years=[2024], signal_time="10:00", exit_time="15:00")

    assert loader.dates is not None
    assert loader.target_ret is not None
    assert loader.raw_data_cache is not None
    assert len(loader.dates) == 5
    assert torch.isnan(loader.target_ret[0, 0]).item()
    assert torch.isnan(loader.target_ret[0, 1]).item()
    assert torch.allclose(loader.target_ret[0, 2], torch.tensor(0.2), atol=1e-6)
    assert torch.allclose(loader.target_ret[0, 3], torch.tensor(13.0 / 11.0 - 1.0), atol=1e-6)
    assert torch.all(loader.raw_data_cache["tradable"][0] == 1.0).item()


def test_minute_target_alignment_uses_past_bars_not_future(tmp_path: Path, monkeypatch) -> None:
    _base_config(monkeypatch)
    monkeypatch.setattr(ModelConfig, "CN_DECISION_FREQ", "1min")
    monkeypatch.setattr(ModelConfig, "CN_HOLD_BARS", 1)
    monkeypatch.setattr(ModelConfig, "CN_TRAIN_RATIO", 1.0)
    monkeypatch.setattr(ModelConfig, "CN_VAL_RATIO", 0.0)

    rows = [
        {"trade_time": "2024-01-02 09:30:00", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "vol": 10.0, "amount": 100.0},
        {"trade_time": "2024-01-02 09:31:00", "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0, "vol": 10.0, "amount": 110.0},
        {"trade_time": "2024-01-02 09:32:00", "open": 12.0, "high": 12.0, "low": 12.0, "close": 12.0, "vol": 10.0, "amount": 120.0},
    ]
    _write_minute_csv(tmp_path / "2024" / "000001.csv", rows)

    loader = ChinaMinuteDataLoader(data_root=str(tmp_path))
    loader.load_data(codes=["000001"], years=[2024], signal_time="10:00", exit_time="15:00")

    assert loader.target_ret is not None
    # Backward-looking label: return[t] = close[t] / close[t-1] - 1
    assert torch.isnan(loader.target_ret[0, 0]).item()
    assert torch.allclose(loader.target_ret[0, 1], torch.tensor(0.1), atol=1e-6)
    assert torch.allclose(loader.target_ret[0, 2], torch.tensor(12.0 / 11.0 - 1.0), atol=1e-6)


def test_loader_auto_generates_t_plus_one_masks(tmp_path: Path, monkeypatch) -> None:
    _base_config(monkeypatch)
    monkeypatch.setattr(ModelConfig, "CN_DECISION_FREQ", "1min")
    monkeypatch.setattr(ModelConfig, "CN_HOLD_BARS", 1)
    monkeypatch.setattr(ModelConfig, "CN_TRAIN_RATIO", 1.0)
    monkeypatch.setattr(ModelConfig, "CN_VAL_RATIO", 0.0)

    rows = [
        {"trade_time": "2024-01-02 09:30:00", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "vol": 10.0, "amount": 100.0},
        {"trade_time": "2024-01-02 09:31:00", "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0, "vol": 10.0, "amount": 110.0},
        {"trade_time": "2024-01-03 09:30:00", "open": 12.0, "high": 12.0, "low": 12.0, "close": 12.0, "vol": 10.0, "amount": 120.0},
        {"trade_time": "2024-01-03 09:31:00", "open": 13.0, "high": 13.0, "low": 13.0, "close": 13.0, "vol": 10.0, "amount": 130.0},
    ]
    _write_minute_csv(tmp_path / "2024" / "000001.csv", rows)

    loader = ChinaMinuteDataLoader(data_root=str(tmp_path))
    loader.load_data(codes=["000001"], years=[2024])

    assert loader.raw_data_cache is not None
    expected_session = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    expected_required = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    expected_block = torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(loader.raw_data_cache["session_id"].cpu(), expected_session)
    assert torch.allclose(loader.raw_data_cache["t_plus_one_required"].cpu(), expected_required)
    assert torch.allclose(loader.raw_data_cache["t_plus_one_sell_block"].cpu(), expected_block)


def test_loader_t0_whitelist_disables_t_plus_one_block(tmp_path: Path, monkeypatch) -> None:
    _base_config(monkeypatch)
    monkeypatch.setattr(ModelConfig, "CN_DECISION_FREQ", "1min")
    monkeypatch.setattr(ModelConfig, "CN_HOLD_BARS", 1)
    monkeypatch.setattr(ModelConfig, "CN_TRAIN_RATIO", 1.0)
    monkeypatch.setattr(ModelConfig, "CN_VAL_RATIO", 0.0)
    monkeypatch.setattr(ModelConfig, "CN_T0_ALLOWED_CODES", ["510300"])

    rows_stock = [
        {"trade_time": "2024-01-02 09:30:00", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "vol": 10.0, "amount": 100.0},
        {"trade_time": "2024-01-02 09:31:00", "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0, "vol": 10.0, "amount": 110.0},
    ]
    rows_t0 = [
        {"trade_time": "2024-01-02 09:30:00", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0, "vol": 10.0, "amount": 200.0},
        {"trade_time": "2024-01-02 09:31:00", "open": 21.0, "high": 21.0, "low": 21.0, "close": 21.0, "vol": 10.0, "amount": 210.0},
    ]
    _write_minute_csv(tmp_path / "2024" / "000001.csv", rows_stock)
    _write_minute_csv(tmp_path / "2024" / "510300.csv", rows_t0)

    loader = ChinaMinuteDataLoader(data_root=str(tmp_path))
    loader.load_data(codes=["000001", "510300"], years=[2024])

    assert loader.raw_data_cache is not None
    assert loader.symbols is not None
    idx_stock = loader.symbols.index("000001")
    idx_t0 = loader.symbols.index("510300")
    assert torch.all(loader.raw_data_cache["t_plus_one_required"][idx_stock] == 1.0).item()
    assert torch.all(loader.raw_data_cache["t_plus_one_required"][idx_t0] == 0.0).item()
    assert torch.allclose(
        loader.raw_data_cache["t_plus_one_sell_block"][idx_stock].cpu(),
        torch.tensor([0.0, 1.0], dtype=torch.float32),
    )
    assert torch.all(loader.raw_data_cache["t_plus_one_sell_block"][idx_t0] == 0.0).item()
