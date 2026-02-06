from __future__ import annotations

import torch

from model_core.backtest import ChinaBacktest


def _make_raw_data(open_: torch.Tensor, high: torch.Tensor, low: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "open": open_,
        "high": high,
        "low": low,
    }


def test_summary_and_detail_scores_are_consistent() -> None:
    bt = ChinaBacktest()
    bt.allow_short = False
    bt.signal_lag = 1
    bt.slippage_rate = 0.0
    bt.slippage_impact = 0.0

    factors = torch.tensor(
        [
            [1.0, -1.0, 0.5, 0.2],
            [0.1, 0.2, -0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    target_ret = torch.tensor(
        [
            [0.01, -0.02, 0.03, 0.01],
            [0.02, 0.01, -0.01, 0.03],
        ],
        dtype=torch.float32,
    )
    raw_data = _make_raw_data(
        open_=torch.ones_like(target_ret) * 10.0,
        high=torch.ones_like(target_ret) * 10.2,
        low=torch.ones_like(target_ret) * 9.8,
    )

    summary = bt.evaluate(factors, raw_data, target_ret, return_details=False)
    detail = bt.evaluate(factors, raw_data, target_ret, return_details=True)

    assert torch.allclose(summary.score, detail.score)
    assert abs(summary.mean_return - detail.mean_return) < 1e-12
    assert detail.equity_curve is not None
    assert detail.portfolio_returns is not None
    assert detail.equity_curve.shape[0] == factors.shape[1]
    assert detail.portfolio_returns.shape[0] == factors.shape[1]


def test_reentry_after_missing_return_still_charges_turnover_cost() -> None:
    bt = ChinaBacktest()
    bt.allow_short = False
    bt.signal_lag = 0
    bt.cost_rate = 0.1
    bt.slippage_rate = 0.0
    bt.slippage_impact = 0.0

    factors = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float32)
    target_ret = torch.tensor([[0.01, float("nan"), 0.01]], dtype=torch.float32)
    raw_data = _make_raw_data(
        open_=torch.ones_like(target_ret) * 10.0,
        high=torch.ones_like(target_ret) * 10.0,
        low=torch.ones_like(target_ret) * 10.0,
    )

    result = bt.evaluate(factors, raw_data, target_ret, return_details=True)
    assert result.portfolio_returns is not None
    # Day 1: 0.01 - 0.1 * 1; Day 2 missing -> 0; Day 3 re-entry also charges 1 turnover.
    expected = torch.tensor([-0.09, 0.0, -0.09], dtype=torch.float32)
    assert torch.allclose(result.portfolio_returns, expected, atol=1e-6)
