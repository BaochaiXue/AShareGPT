from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch

Formula = list[int]


@dataclass
class DataBundle:
    """Canonical in-memory market data payload."""

    feat_tensor: torch.Tensor
    raw_data_cache: dict[str, torch.Tensor]
    target_ret: torch.Tensor
    dates: pd.DatetimeIndex
    symbols: list[str]


@dataclass
class DatasetSlice:
    """Windowed view over the canonical market payload."""

    feat_tensor: torch.Tensor
    raw_data_cache: dict[str, torch.Tensor]
    target_ret: torch.Tensor
    dates: pd.DatetimeIndex
    symbols: list[str]
    start_idx: int
    end_idx: int


@dataclass
class WalkForwardBundle:
    """One walk-forward fold containing train/val/test windows."""

    train: DatasetSlice
    val: DatasetSlice
    test: DatasetSlice


@dataclass
class TrainingArtifact:
    """Minimal training output used by application use-cases."""

    best_formula: Optional[Formula]
    best_score: float
    strategy_path: Optional[str] = None

