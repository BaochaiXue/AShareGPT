
from typing import Callable
import torch

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
    # Keep denominator away from zero while preserving sign.
    denom = torch.where(y >= 0, y + eps, y - eps)
    return x / denom

OPS_CONFIG: list[tuple[str, Callable[..., torch.Tensor], int]] = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', _safe_div, 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('MA20', lambda x: _ts_decay_linear(x, 20), 1),
    ('STD20', lambda x: _ts_std(x, 20), 1),
    ('TS_RANK20', lambda x: _ts_rank(x, 20), 1),
]
