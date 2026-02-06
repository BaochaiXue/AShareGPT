from __future__ import annotations

from typing import Any, Optional
from warnings import warn

from tqdm import tqdm

from .application.services import TrainingWorkflowService
from .bootstrap.factories import create_training_workflow_service_from_components
from .config import ModelConfig


class AlphaEngine:
    """
    Compatibility facade over application-level training workflow.

    Deprecated path: prefer wiring through `TrainAlphaUseCase`.
    """

    def __init__(
        self,
        use_lord_regularization: bool = True,
        lord_decay_rate: float = 1e-3,
        lord_num_iterations: int = 5,
        *,
        workflow: Optional[TrainingWorkflowService] = None,
        loader=None,
        model=None,
        optimizer=None,
        vm=None,
        backtest=None,
        auto_load_data: bool = True,
        data_kwargs: Optional[dict[str, Any]] = None,
    ):
        warn(
            "AlphaEngine is a compatibility wrapper; prefer application/use_cases APIs.",
            DeprecationWarning,
            stacklevel=2,
        )

        if workflow is not None:
            self.workflow = workflow
        else:
            self.workflow = create_training_workflow_service_from_components(
                use_lord_regularization=use_lord_regularization,
                lord_decay_rate=lord_decay_rate,
                lord_num_iterations=lord_num_iterations,
                loader=loader,
                model=model,
                optimizer=optimizer,
                vm=vm,
                backtest=backtest,
                auto_load_data=auto_load_data,
                data_kwargs=data_kwargs,
            )

        # Compatibility attributes exposed for older callers.
        self.loader = self.workflow.loader
        self.model = self.workflow.model
        self.opt = self.workflow.optimizer
        self.vm = self.workflow.vm
        self.bt = self.workflow.backtest_engine
        self.token_arity = self.workflow.token_arity
        self.token_delta = self.workflow.token_delta
        self.lord_opt = self.workflow.lord_opt
        self.rank_monitor = self.workflow.rank_monitor
        self.use_lord = self.workflow.use_lord

        self.best_score: float = -float("inf")
        self.best_formula: Optional[list[int]] = None
        self.training_history: dict[str, list[Any]] = {
            "step": [],
            "avg_reward": [],
            "best_score": [],
            "stable_rank": [],
            "avg_val_score": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }

    def train(self) -> None:
        print(
            "🚀 Starting Alpha Mining with PPO + LoRD..."
            if self.use_lord
            else "🚀 Starting Alpha Mining with PPO..."
        )
        if self.use_lord:
            print("   LoRD Regularization enabled")
            print("   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        for line in self.workflow.train_window_descriptions():
            print(line)

        def _on_new_best(score: float, mean_return: float, formula: list[int]) -> None:
            tqdm.write(f"[!] New King: Score {score:.2f} | Ret {mean_return:.2%} | Formula {formula}")

        result = self.workflow.run(
            strategy_path=ModelConfig.STRATEGY_FILE,
            history_path="training_history.json",
            on_new_best=_on_new_best,
        )

        self.best_score = result.best_score
        self.best_formula = result.best_formula
        self.training_history = result.history

        print("\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")
        for snapshot in result.evaluations:
            print(
                f"  {snapshot.label}: Score {snapshot.score:.4f} | "
                f"MeanRet {snapshot.mean_return:.2%} | Sharpe {snapshot.sharpe:.2f} | "
                f"MaxDD {snapshot.max_drawdown:.2%}"
            )
