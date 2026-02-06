from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from model_core.config import ModelConfig
from model_core.data_loader import ChinaMinuteDataLoader, DataSlice
from .ppo_training_service import PpoTrainingService
from .reward_orchestrator import FormulaRewardOrchestrator


def build_token_tables(
    *,
    vocab_size: int,
    feat_offset: int,
    arity_map: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build token arity/delta tables consumed by PPO sampling."""

    token_arity = torch.zeros(vocab_size, dtype=torch.long)
    token_arity[:feat_offset] = 0

    token_delta = torch.ones(vocab_size, dtype=torch.long)
    for token, arity in arity_map.items():
        if 0 <= token < vocab_size:
            arity_int = int(arity)
            token_arity[token] = arity_int
            token_delta[token] = 1 - arity_int
    return token_arity, token_delta


@dataclass
class EvaluationSnapshot:
    """Post-training evaluation line for one dataset window."""

    label: str
    score: float
    mean_return: float
    sharpe: float
    max_drawdown: float


@dataclass
class TrainingWorkflowResult:
    """Structured output of a training run."""

    best_score: float
    best_formula: Optional[list[int]]
    history: dict[str, list[Any]]
    evaluations: list[EvaluationSnapshot]


class TrainingWorkflowService:
    """High-level training workflow extracted from the compatibility engine."""

    def __init__(
        self,
        *,
        loader: ChinaMinuteDataLoader,
        model,
        optimizer,
        vm,
        backtest_engine,
        bos_id: int,
        token_arity: torch.Tensor,
        token_delta: torch.Tensor,
        device: torch.device,
        use_lord: bool,
        lord_opt=None,
        rank_monitor=None,
        train_steps: int,
        batch_size: int,
        max_formula_len: int,
        ppo_epochs: int,
        ppo_clip_eps: float,
        ppo_value_coef: float,
        ppo_entropy_coef: float,
        ppo_max_grad_norm: float,
        rank_every: int = 100,
    ):
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.vm = vm
        self.backtest_engine = backtest_engine
        self.bos_id = int(bos_id)
        self.token_arity = token_arity
        self.token_delta = token_delta
        self.device = device

        self.use_lord = bool(use_lord)
        self.lord_opt = lord_opt
        self.rank_monitor = rank_monitor

        self.train_steps = int(train_steps)
        self.batch_size = int(batch_size)
        self.max_formula_len = int(max_formula_len)
        self.ppo_epochs = int(ppo_epochs)
        self.ppo_clip_eps = float(ppo_clip_eps)
        self.ppo_value_coef = float(ppo_value_coef)
        self.ppo_entropy_coef = float(ppo_entropy_coef)
        self.ppo_max_grad_norm = float(ppo_max_grad_norm)
        self.rank_every = int(rank_every)

        self.splits = self.loader.train_val_test_split()
        self.train_slice = self.splits.get("train")
        self.val_slice = self.splits.get("val")
        self.test_slice = self.splits.get("test")
        self.walk_forward_folds = self.loader.walk_forward_splits() if ModelConfig.CN_WALK_FORWARD else []
        self.use_wfo = ModelConfig.CN_WALK_FORWARD and len(self.walk_forward_folds) > 0

    def run(
        self,
        *,
        strategy_path: str,
        history_path: str = "training_history.json",
        on_new_best: Optional[Callable[[float, float, list[int]], None]] = None,
    ) -> TrainingWorkflowResult:
        full_feat = self.loader.feat_tensor
        if full_feat is None or self.loader.raw_data_cache is None or self.loader.target_ret is None:
            raise ValueError("Data not loaded. Check data loader.")

        train_slice = self.train_slice
        if train_slice is None:
            train_slice = self.loader.get_slice(0, full_feat.shape[-1])

        reward_orchestrator = FormulaRewardOrchestrator(
            vm=self.vm,
            backtest_engine=self.backtest_engine,
            train_slice=train_slice,
            val_slice=self.val_slice,
            walk_forward_folds=self.walk_forward_folds,
            use_wfo=self.use_wfo,
        )
        ppo_service = PpoTrainingService(
            model=self.model,
            optimizer=self.optimizer,
            bos_id=self.bos_id,
            token_arity=self.token_arity,
            token_delta=self.token_delta,
            device=self.device,
            reward_orchestrator=reward_orchestrator,
            use_lord=self.use_lord,
            lord_opt=self.lord_opt,
        )

        run_state = ppo_service.train(
            full_feat=full_feat,
            train_steps=self.train_steps,
            batch_size=self.batch_size,
            max_formula_len=self.max_formula_len,
            ppo_epochs=self.ppo_epochs,
            ppo_clip_eps=self.ppo_clip_eps,
            ppo_value_coef=self.ppo_value_coef,
            ppo_entropy_coef=self.ppo_entropy_coef,
            ppo_max_grad_norm=self.ppo_max_grad_norm,
            rank_monitor=self.rank_monitor if self.use_lord else None,
            rank_every=self.rank_every,
            on_new_best=on_new_best,
        )

        strategy_file = Path(strategy_path)
        if run_state.best_formula is not None:
            with strategy_file.open("w", encoding="utf-8") as handle:
                json.dump(run_state.best_formula, handle)
        elif strategy_file.exists():
            strategy_file.unlink()
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(run_state.history, handle)

        evaluations = self._evaluate_best_formula(run_state.best_formula, full_feat)
        return TrainingWorkflowResult(
            best_score=run_state.best_score,
            best_formula=run_state.best_formula,
            history=run_state.history,
            evaluations=evaluations,
        )

    def _evaluate_best_formula(
        self,
        best_formula: Optional[list[int]],
        full_feat: torch.Tensor,
    ) -> list[EvaluationSnapshot]:
        if not best_formula or self.use_wfo:
            return []

        res = self.vm.execute(best_formula, full_feat)
        if res is None:
            return []

        snapshots: list[EvaluationSnapshot] = []
        for label, data_slice in (
            ("Train", self.train_slice),
            ("Val", self.val_slice),
            ("Test", self.test_slice),
        ):
            if data_slice is None:
                continue
            snapshots.append(self._evaluate_slice(label=label, signal=res, data_slice=data_slice))
        return snapshots

    def _evaluate_slice(
        self,
        *,
        label: str,
        signal: torch.Tensor,
        data_slice: DataSlice,
    ) -> EvaluationSnapshot:
        sig_slice = signal[:, data_slice.start_idx : data_slice.end_idx]
        result = self.backtest_engine.evaluate(
            sig_slice,
            data_slice.raw_data_cache,
            data_slice.target_ret,
            return_details=True,
        )
        metrics = result.metrics or {}
        return EvaluationSnapshot(
            label=label,
            score=float(result.score.item()),
            mean_return=float(result.mean_return),
            sharpe=float(metrics.get("sharpe", float("nan"))),
            max_drawdown=float(metrics.get("max_drawdown", float("nan"))),
        )

    def train_window_descriptions(self) -> list[str]:
        """Return printable window descriptions for CLI compatibility logs."""

        if self.use_wfo:
            return [f"   Walk-forward validation: {len(self.walk_forward_folds)} folds"]

        lines: list[str] = []
        if self.train_slice is not None:
            lines.append(
                f"   Train window: {self.train_slice.dates.min()} -> {self.train_slice.dates.max()}"
            )
        if self.val_slice is not None:
            lines.append(f"   Val window:   {self.val_slice.dates.min()} -> {self.val_slice.dates.max()}")
        if self.test_slice is not None:
            lines.append(
                f"   Test window:  {self.test_slice.dates.min()} -> {self.test_slice.dates.max()}"
            )
        return lines
