from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from model_core.training import _PpoLoop


@dataclass
class TrainingRunState:
    """Compatibility training output for callers expecting the legacy service."""

    best_score: float
    best_formula: Optional[list[int]]
    history: dict[str, list[Any]]


class PpoTrainingService:
    """Compatibility wrapper over the unified PPO loop implementation."""

    def __init__(
        self,
        *,
        model,
        optimizer,
        bos_id: int,
        token_arity: torch.Tensor,
        token_delta: torch.Tensor,
        device: torch.device,
        reward_orchestrator,
        use_lord: bool = False,
        lord_opt=None,
    ):
        self._loop = _PpoLoop(
            model=model,
            optimizer=optimizer,
            bos_id=bos_id,
            token_arity=token_arity,
            token_delta=token_delta,
            device=device,
            reward_orch=reward_orchestrator,
            use_lord=use_lord,
            lord_opt=lord_opt,
        )

    def train(
        self,
        *,
        full_feat: torch.Tensor,
        train_steps: int,
        batch_size: int,
        max_formula_len: int,
        ppo_epochs: int,
        ppo_clip_eps: float,
        ppo_value_coef: float,
        ppo_entropy_coef: float,
        ppo_max_grad_norm: float,
        rank_monitor=None,
        rank_every: int = 100,
        on_new_best: Optional[Callable[[float, float, list[int]], None]] = None,
    ) -> TrainingRunState:
        best_score, best_formula, history = self._loop.run(
            full_feat=full_feat,
            train_steps=train_steps,
            batch_size=batch_size,
            max_len=max_formula_len,
            ppo_epochs=ppo_epochs,
            clip_eps=ppo_clip_eps,
            value_coef=ppo_value_coef,
            entropy_coef=ppo_entropy_coef,
            max_grad_norm=ppo_max_grad_norm,
            rank_monitor=rank_monitor,
            rank_every=rank_every,
            on_new_best=on_new_best,
        )
        return TrainingRunState(
            best_score=float(best_score),
            best_formula=best_formula,
            history=history,
        )

