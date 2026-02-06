from __future__ import annotations

from model_core.bootstrap.container import LegacyContainer
from model_core.bootstrap import create_legacy_container
from model_core.application.use_cases import BacktestFormulaUseCase, TrainAlphaUseCase


def create_app_container(
    *,
    use_lord_regularization: bool = True,
    lord_decay_rate: float = 1e-3,
    lord_num_iterations: int = 5,
) -> LegacyContainer:
    """Composition-root helper for new use-case based integration."""

    return create_legacy_container(
        use_lord_regularization=use_lord_regularization,
        lord_decay_rate=lord_decay_rate,
        lord_num_iterations=lord_num_iterations,
    )


def create_train_use_case(
    *,
    use_lord_regularization: bool = True,
    lord_decay_rate: float = 1e-3,
    lord_num_iterations: int = 5,
) -> tuple[TrainAlphaUseCase, LegacyContainer]:
    """Create training use-case with wired dependencies."""

    container = create_app_container(
        use_lord_regularization=use_lord_regularization,
        lord_decay_rate=lord_decay_rate,
        lord_num_iterations=lord_num_iterations,
    )
    return container.train_use_case(), container


def create_backtest_use_case() -> tuple[BacktestFormulaUseCase, LegacyContainer]:
    """Create backtest use-case with wired dependencies."""

    container = create_app_container()
    return container.backtest_use_case(), container
