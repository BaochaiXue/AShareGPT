from __future__ import annotations

from typing import Any, Optional

import torch

from model_core.alphagpt import AlphaGPT
from model_core.backtest import ChinaBacktest
from model_core.config import ModelConfig
from model_core.data_loader import ChinaMinuteDataLoader
from model_core.engine import AlphaEngine
from model_core.vm import StackVM


def _default_data_kwargs() -> dict[str, Any]:
    return {
        "codes": ModelConfig.CN_CODES,
        "years": ModelConfig.CN_MINUTE_YEARS,
        "start_date": ModelConfig.CN_MINUTE_START_DATE,
        "end_date": ModelConfig.CN_MINUTE_END_DATE,
        "signal_time": ModelConfig.CN_SIGNAL_TIME,
        "exit_time": ModelConfig.CN_EXIT_TIME,
        "limit_codes": ModelConfig.CN_MAX_CODES,
    }


def create_training_engine(
    *,
    use_lord_regularization: bool = True,
    lord_decay_rate: float = 1e-3,
    lord_num_iterations: int = 5,
    data_kwargs: Optional[dict[str, Any]] = None,
) -> AlphaEngine:
    """
    Build the training runtime from the composition root.

    This keeps dependency assembly out of `AlphaEngine`.
    """

    loader = ChinaMinuteDataLoader()
    loader.load_data(**(data_kwargs or _default_data_kwargs()))

    model = AlphaGPT().to(ModelConfig.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    vm = StackVM()
    backtest = ChinaBacktest()

    return AlphaEngine(
        use_lord_regularization=use_lord_regularization,
        lord_decay_rate=lord_decay_rate,
        lord_num_iterations=lord_num_iterations,
        loader=loader,
        model=model,
        optimizer=optimizer,
        vm=vm,
        backtest=backtest,
        auto_load_data=False,
    )

