"""Legacy adapters that wrap existing model_core implementations."""

from .backtest_engine import LegacyBacktestEngine
from .data_gateway import LegacyChinaDataGateway
from .formula_executor import LegacyStackVmExecutor
from .trainer import LegacyAlphaTrainer

__all__ = [
    "LegacyAlphaTrainer",
    "LegacyBacktestEngine",
    "LegacyChinaDataGateway",
    "LegacyStackVmExecutor",
]
