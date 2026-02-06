from __future__ import annotations

from typing import Callable, Optional

from model_core.config import ModelConfig
from model_core.domain.models import TrainingArtifact
from model_core.engine import AlphaEngine


class LegacyAlphaTrainer:
    """Adapter from legacy `AlphaEngine` to the trainer port."""

    def __init__(
        self,
        *,
        use_lord_regularization: bool = True,
        lord_decay_rate: float = 1e-3,
        lord_num_iterations: int = 5,
        engine_factory: Optional[Callable[..., AlphaEngine]] = None,
    ):
        self._engine_factory = engine_factory or AlphaEngine
        self._engine_kwargs = {
            "use_lord_regularization": use_lord_regularization,
            "lord_decay_rate": lord_decay_rate,
            "lord_num_iterations": lord_num_iterations,
        }
        self._engine: Optional[AlphaEngine] = None

    @property
    def engine(self) -> Optional[AlphaEngine]:
        return self._engine

    def train(self) -> TrainingArtifact:
        self._engine = self._engine_factory(**self._engine_kwargs)
        self._engine.train()
        return TrainingArtifact(
            best_formula=self._engine.best_formula,
            best_score=float(self._engine.best_score),
            strategy_path=ModelConfig.STRATEGY_FILE,
        )
