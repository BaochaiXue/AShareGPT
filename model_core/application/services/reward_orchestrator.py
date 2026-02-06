from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from model_core.data_loader import DataSlice, WalkForwardFold


@dataclass
class FormulaEvaluation:
    """Reward and selection metrics for one candidate formula."""

    reward: float
    selection_score: Optional[float]
    mean_return: float
    train_score: Optional[float] = None
    val_score: Optional[float] = None


class FormulaRewardOrchestrator:
    """
    Pure orchestration for formula scoring.

    This service extracts reward logic from the training loop so it can be tested
    without running PPO end-to-end.
    """

    def __init__(
        self,
        *,
        vm,
        backtest_engine,
        train_slice: DataSlice,
        val_slice: Optional[DataSlice],
        walk_forward_folds: list[WalkForwardFold],
        use_wfo: bool,
        reward_mode: str = "selection",
    ):
        self._vm = vm
        self._backtest_engine = backtest_engine
        self._train_slice = train_slice
        self._val_slice = val_slice
        self._walk_forward_folds = walk_forward_folds
        self._use_wfo = use_wfo
        mode = reward_mode.strip().lower()
        if mode not in {"train", "selection"}:
            raise ValueError(f"Unsupported reward_mode={reward_mode!r}; expected 'train' or 'selection'.")
        self._reward_mode = mode

    @torch.no_grad()
    def evaluate_formula(self, formula: list[int], full_feat: torch.Tensor) -> FormulaEvaluation:
        res = self._vm.execute(formula, full_feat)
        if res is None:
            return FormulaEvaluation(reward=-5.0, selection_score=None, mean_return=0.0)
        if res.std() < 1e-4:
            return FormulaEvaluation(reward=-2.0, selection_score=None, mean_return=0.0)

        if self._use_wfo:
            return self._evaluate_wfo(res)
        return self._evaluate_train_val(res)

    def _evaluate_wfo(self, res: torch.Tensor) -> FormulaEvaluation:
        fold_scores: list[torch.Tensor] = []
        fold_returns: list[float] = []
        for fold in self._walk_forward_folds:
            if fold.val.end_idx <= fold.val.start_idx:
                continue
            res_val = res[:, fold.val.start_idx : fold.val.end_idx]
            if res_val.numel() == 0:
                continue
            result = self._backtest_engine.evaluate(
                res_val,
                fold.val.raw_data_cache,
                fold.val.target_ret,
            )
            fold_scores.append(result.score)
            fold_returns.append(result.mean_return)

        if not fold_scores:
            return FormulaEvaluation(reward=-2.0, selection_score=None, mean_return=0.0)

        reward = float(torch.stack(fold_scores).mean().item())
        mean_return = float(sum(fold_returns) / len(fold_returns))
        return FormulaEvaluation(
            reward=reward,
            selection_score=reward,
            mean_return=mean_return,
            train_score=None,
            val_score=reward,
        )

    def _evaluate_train_val(self, res: torch.Tensor) -> FormulaEvaluation:
        res_train = res[:, self._train_slice.start_idx : self._train_slice.end_idx]
        if res_train.numel() == 0 or res_train.std() < 1e-4:
            return FormulaEvaluation(reward=-2.0, selection_score=None, mean_return=0.0)

        train_result = self._backtest_engine.evaluate(
            res_train,
            self._train_slice.raw_data_cache,
            self._train_slice.target_ret,
        )
        train_score = float(train_result.score.item())
        selection_score = train_score
        mean_return = float(train_result.mean_return)
        val_score: Optional[float] = None

        if self._val_slice and self._val_slice.end_idx > self._val_slice.start_idx:
            res_val = res[:, self._val_slice.start_idx : self._val_slice.end_idx]
            if res_val.numel() > 0:
                val_result = self._backtest_engine.evaluate(
                    res_val,
                    self._val_slice.raw_data_cache,
                    self._val_slice.target_ret,
                )
                selection_score = float(val_result.score.item())
                mean_return = float(val_result.mean_return)
                val_score = selection_score

        reward = train_score if self._reward_mode == "train" else selection_score

        return FormulaEvaluation(
            reward=reward,
            selection_score=selection_score,
            mean_return=mean_return,
            train_score=train_score,
            val_score=val_score,
        )
