from __future__ import annotations

from dataclasses import dataclass

from model_core.application.use_cases import BacktestFormulaUseCase, TrainAlphaUseCase
from model_core.infrastructure.legacy import (
    LegacyAlphaTrainer,
    LegacyBacktestEngine,
    LegacyChinaDataGateway,
    LegacyStackVmExecutor,
)
from .factories import create_training_workflow_service


@dataclass
class LegacyContainer:
    """
    Composition root for the compatibility layer.

    This keeps old implementations but exposes them through explicit ports.
    """

    data_gateway: LegacyChinaDataGateway
    formula_executor: LegacyStackVmExecutor
    backtest_engine: LegacyBacktestEngine
    trainer: LegacyAlphaTrainer

    def backtest_use_case(self) -> BacktestFormulaUseCase:
        return BacktestFormulaUseCase(
            data_gateway=self.data_gateway,
            executor=self.formula_executor,
            backtest_engine=self.backtest_engine,
        )

    def train_use_case(self) -> TrainAlphaUseCase:
        return TrainAlphaUseCase(trainer=self.trainer)


def create_legacy_container(
    *,
    use_lord_regularization: bool = True,
    lord_decay_rate: float = 1e-3,
    lord_num_iterations: int = 5,
) -> LegacyContainer:
    return LegacyContainer(
        data_gateway=LegacyChinaDataGateway(),
        formula_executor=LegacyStackVmExecutor(),
        backtest_engine=LegacyBacktestEngine(),
        trainer=LegacyAlphaTrainer(
            use_lord_regularization=use_lord_regularization,
            lord_decay_rate=lord_decay_rate,
            lord_num_iterations=lord_num_iterations,
            workflow_factory=create_training_workflow_service,
        ),
    )
