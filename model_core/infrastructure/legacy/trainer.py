from __future__ import annotations

from typing import Callable, Optional

from tqdm import tqdm

from model_core.application.services import TrainingWorkflowService
from model_core.config import ModelConfig
from model_core.domain.models import TrainingArtifact


class LegacyAlphaTrainer:
    """Adapter exposing the application workflow as a legacy trainer port."""

    def __init__(
        self,
        *,
        use_lord_regularization: bool = True,
        lord_decay_rate: float = 1e-3,
        lord_num_iterations: int = 5,
        workflow_factory: Optional[Callable[..., TrainingWorkflowService]] = None,
    ):
        if workflow_factory is None:
            raise ValueError("workflow_factory is required for LegacyAlphaTrainer.")
        self._workflow_factory = workflow_factory
        self._workflow_kwargs = {
            "use_lord_regularization": use_lord_regularization,
            "lord_decay_rate": lord_decay_rate,
            "lord_num_iterations": lord_num_iterations,
        }
        self._workflow: Optional[TrainingWorkflowService] = None

    @property
    def workflow(self) -> Optional[TrainingWorkflowService]:
        return self._workflow

    def train(self) -> TrainingArtifact:
        self._workflow = self._workflow_factory(**self._workflow_kwargs)

        print(
            "🚀 Starting Alpha Mining with PPO + LoRD..."
            if self._workflow.use_lord
            else "🚀 Starting Alpha Mining with PPO..."
        )
        if self._workflow.use_lord:
            print("   LoRD Regularization enabled")
            print("   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        for line in self._workflow.train_window_descriptions():
            print(line)

        def _on_new_best(score: float, mean_return: float, formula: list[int]) -> None:
            tqdm.write(f"[!] New King: Score {score:.2f} | Ret {mean_return:.2%} | Formula {formula}")

        result = self._workflow.run(
            strategy_path=ModelConfig.STRATEGY_FILE,
            history_path="training_history.json",
            on_new_best=_on_new_best,
        )

        print("\n✓ Training completed!")
        print(f"  Best score: {result.best_score:.4f}")
        print(f"  Best formula: {result.best_formula}")
        for snapshot in result.evaluations:
            print(
                f"  {snapshot.label}: Score {snapshot.score:.4f} | "
                f"MeanRet {snapshot.mean_return:.2%} | Sharpe {snapshot.sharpe:.2f} | "
                f"MaxDD {snapshot.max_drawdown:.2%}"
            )

        return TrainingArtifact(
            best_formula=result.best_formula,
            best_score=float(result.best_score),
            strategy_path=ModelConfig.STRATEGY_FILE,
        )
