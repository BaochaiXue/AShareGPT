from __future__ import annotations

from typing import Optional, Protocol

import torch

from model_core.domain.models import (
    BacktestEvaluation,
    DataBundle,
    DatasetSlice,
    Formula,
    TrainingArtifact,
    WalkForwardBundle,
)


class DataGatewayPort(Protocol):
    """Data access contract for loading and slicing market data."""

    def load(
        self,
        *,
        codes: Optional[list[str]] = None,
        years: Optional[list[int]] = None,
        start_date: str = "",
        end_date: str = "",
        signal_time: str = "",
        exit_time: str = "",
        limit_codes: int = 50,
    ) -> None:
        ...

    def bundle(self) -> DataBundle:
        ...

    def train_val_test_split(self) -> dict[str, DatasetSlice]:
        ...

    def walk_forward_splits(self) -> list[WalkForwardBundle]:
        ...


class FormulaExecutorPort(Protocol):
    """Formula execution contract."""

    def execute(self, formula: Formula, feat_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        ...


class BacktestEnginePort(Protocol):
    """Backtest evaluation contract."""

    def evaluate(
        self,
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> BacktestEvaluation:
        ...


class TrainerPort(Protocol):
    """Training orchestration contract."""

    def train(self) -> TrainingArtifact:
        ...
