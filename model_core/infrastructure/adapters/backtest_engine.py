from __future__ import annotations

from typing import Optional

import torch

from model_core.backtest import ChinaBacktest
from model_core.domain.models import BacktestEvaluation


class ChinaBacktestEngineAdapter:
    """Adapter from `ChinaBacktest` to `BacktestEnginePort`."""

    def __init__(self, backtest: Optional[ChinaBacktest] = None):
        self._backtest = backtest or ChinaBacktest()

    @property
    def backtest(self) -> ChinaBacktest:
        return self._backtest

    def evaluate(
        self,
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> BacktestEvaluation:
        raw = self._backtest.evaluate(
            factors,
            raw_data,
            target_ret,
            return_details=return_details,
        )
        return BacktestEvaluation(
            score=float(raw.score.item()),
            mean_return=float(raw.mean_return),
            metrics=dict(raw.metrics or {}),
            equity_curve=raw.equity_curve.tolist() if raw.equity_curve is not None else None,
            portfolio_returns=raw.portfolio_returns.tolist() if raw.portfolio_returns is not None else None,
        )
