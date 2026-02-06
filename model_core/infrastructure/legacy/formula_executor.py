from __future__ import annotations

from typing import Optional

import torch

from model_core.domain.models import Formula
from model_core.vm import StackVM


class LegacyStackVmExecutor:
    """Adapter from legacy `StackVM` to the formula executor port."""

    def __init__(self, vm: Optional[StackVM] = None):
        self._vm = vm or StackVM()

    @property
    def vm(self) -> StackVM:
        return self._vm

    def execute(self, formula: Formula, feat_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        return self._vm.execute(formula, feat_tensor)

