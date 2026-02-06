from __future__ import annotations

from model_core.domain.models import TrainingArtifact
from model_core.ports.interfaces import TrainerPort


class TrainAlphaUseCase:
    """Application service for model training orchestration."""

    def __init__(self, trainer: TrainerPort):
        self._trainer = trainer

    def run(self) -> TrainingArtifact:
        return self._trainer.train()

