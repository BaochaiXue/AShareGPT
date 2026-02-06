"""
Symbolic operators for the Stack VM.

Provides two operator sets:
- OPS_CONFIG_DAILY:  windows tuned for daily bars (5, 20 days)
- OPS_CONFIG_1MIN:   windows tuned for minute bars (5, 20, 60, 240 bars)

The active configuration is selected via `get_ops_config()` based on
`ModelConfig.CN_DECISION_FREQ`.
"""
from __future__ import annotations

from typing import Callable

import torch

from .config import ModelConfig


@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 0:
        return x
    t = x.shape[1]
    if d >= t:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], d), device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, : t - d]], dim=1)

@torch.jit.script
def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    return x - _ts_delay(x, d)

@torch.jit.script
def _ts_zscore(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    mean = windows.mean(dim=-1)
    std = windows.std(dim=-1) + 1e-6
    return (x - mean) / std

@torch.jit.script
def _ts_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1:
        return x
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    w = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    w = w / w.sum()
    return (windows * w).sum(dim=-1)

@torch.jit.script
def _ts_std(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.std(dim=-1, unbiased=False)

@torch.jit.script
def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    last = windows[:, :, -1].unsqueeze(-1)
    less_equal = (windows <= last).to(x.dtype).sum(dim=-1)
    return (less_equal - 1.0) / float(d - 1)

@torch.jit.script
def _safe_div(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = torch.where(y >= 0, y + eps, y - eps)
    return x / denom


# ---------------------------------------------------------------------------
#  Operator tables
# ---------------------------------------------------------------------------

# Shared base operators (arity-2 arithmetic + arity-1 unary)
_BASE_OPS: list[tuple[str, Callable[..., torch.Tensor], int]] = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', _safe_div, 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
]

# Daily-scale time-series ops (the original set)
OPS_CONFIG_DAILY: list[tuple[str, Callable[..., torch.Tensor], int]] = _BASE_OPS + [
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('MA20', lambda x: _ts_decay_linear(x, 20), 1),
    ('STD20', lambda x: _ts_std(x, 20), 1),
    ('TS_RANK20', lambda x: _ts_rank(x, 20), 1),
]

# Minute-scale time-series ops (richer set for intraday)
OPS_CONFIG_1MIN: list[tuple[str, Callable[..., torch.Tensor], int]] = _BASE_OPS + [
    ('DELTA1', lambda x: _ts_delta(x, 1), 1),      # 1-bar momentum
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),      # 5-bar momentum
    ('DELTA20', lambda x: _ts_delta(x, 20), 1),    # ~5 min change
    ('MA5', lambda x: _ts_decay_linear(x, 5), 1),
    ('MA20', lambda x: _ts_decay_linear(x, 20), 1),
    ('MA60', lambda x: _ts_decay_linear(x, 60), 1),    # ~15 min
    ('MA240', lambda x: _ts_decay_linear(x, 240), 1),  # ~1 trading day
    ('STD20', lambda x: _ts_std(x, 20), 1),
    ('STD60', lambda x: _ts_std(x, 60), 1),
    ('TS_RANK20', lambda x: _ts_rank(x, 20), 1),
    ('ZSCORE20', lambda x: _ts_zscore(x, 20), 1),
]


def get_ops_config() -> list[tuple[str, Callable[..., torch.Tensor], int]]:
    """Return the appropriate OPS_CONFIG based on CN_DECISION_FREQ."""
    if ModelConfig.CN_DECISION_FREQ == "1min":
        return OPS_CONFIG_1MIN
    return OPS_CONFIG_DAILY


# Default export for backward compatibility
OPS_CONFIG = get_ops_config()
