from __future__ import annotations

from typing import Optional

import torch

from model_core.backtest import ChinaBacktest


class LegacyBacktestEngine:
    """Adapter from legacy `ChinaBacktest` to the backtest engine port."""

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
    ):
        return self._backtest.evaluate(
            factors,
            raw_data,
            target_ret,
            return_details=return_details,
        )

