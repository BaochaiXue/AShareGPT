from __future__ import annotations

from typing import Any, Optional

import torch

from model_core.neural_symbolic_alpha_generator import (
    NeuralSymbolicAlphaGenerator,
    NewtonSchulzLowRankDecay,
    StableRankMonitor,
)
from model_core.application.services import TrainingWorkflowService, build_token_tables
from model_core.backtest import ChinaBacktest
from model_core.config import ModelConfig
from model_core.data_loader import ChinaMinuteDataLoader
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


def create_training_workflow_service_from_components(
    *,
    use_lord_regularization: bool = True,
    lord_decay_rate: float = 1e-3,
    lord_num_iterations: int = 5,
    loader: Optional[ChinaMinuteDataLoader] = None,
    model: Optional[NeuralSymbolicAlphaGenerator] = None,
    optimizer=None,
    vm: Optional[StackVM] = None,
    backtest: Optional[ChinaBacktest] = None,
    auto_load_data: bool = True,
    data_kwargs: Optional[dict[str, Any]] = None,
) -> TrainingWorkflowService:
    """Build training workflow from optional runtime components."""

    loader = loader or ChinaMinuteDataLoader()
    if auto_load_data and loader.feat_tensor is None:
        loader.load_data(**(data_kwargs or _default_data_kwargs()))
    if loader.dates is None:
        raise ValueError("Data not loaded. Provide a loaded loader or enable auto_load_data.")

    model = model or NeuralSymbolicAlphaGenerator().to(ModelConfig.DEVICE)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-3)
    vm = vm or StackVM()
    backtest = backtest or ChinaBacktest()
    token_arity, token_delta = build_token_tables(
        vocab_size=model.vocab_size,
        feat_offset=vm.feat_offset,
        arity_map=vm.arity_map,
    )

    use_lord = bool(use_lord_regularization)
    if use_lord:
        lord_opt = NewtonSchulzLowRankDecay(
            model.named_parameters(),
            decay_rate=lord_decay_rate,
            num_iterations=lord_num_iterations,
            target_keywords=["q_proj", "k_proj", "attention"],
        )
        rank_monitor = StableRankMonitor(
            model,
            target_keywords=["attention", "in_proj", "out_proj"],
        )
    else:
        lord_opt = None
        rank_monitor = None

    return TrainingWorkflowService(
        loader=loader,
        model=model,
        optimizer=optimizer,
        vm=vm,
        backtest_engine=backtest,
        bos_id=model.bos_id,
        token_arity=token_arity.to(ModelConfig.DEVICE),
        token_delta=token_delta.to(ModelConfig.DEVICE),
        device=ModelConfig.DEVICE,
        use_lord=use_lord,
        lord_opt=lord_opt,
        rank_monitor=rank_monitor,
        train_steps=ModelConfig.TRAIN_STEPS,
        batch_size=ModelConfig.BATCH_SIZE,
        max_formula_len=ModelConfig.MAX_FORMULA_LEN,
        ppo_epochs=ModelConfig.PPO_EPOCHS,
        ppo_clip_eps=ModelConfig.PPO_CLIP_EPS,
        ppo_value_coef=ModelConfig.PPO_VALUE_COEF,
        ppo_entropy_coef=ModelConfig.PPO_ENTROPY_COEF,
        ppo_max_grad_norm=ModelConfig.PPO_MAX_GRAD_NORM,
        rank_every=100,
    )


def create_training_workflow_service(
    *,
    use_lord_regularization: bool = True,
    lord_decay_rate: float = 1e-3,
    lord_num_iterations: int = 5,
    data_kwargs: Optional[dict[str, Any]] = None,
) -> TrainingWorkflowService:
    """Build the application-level training workflow from the composition root."""

    return create_training_workflow_service_from_components(
        use_lord_regularization=use_lord_regularization,
        lord_decay_rate=lord_decay_rate,
        lord_num_iterations=lord_num_iterations,
        data_kwargs=data_kwargs,
    )
