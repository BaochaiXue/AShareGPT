import math
from dataclasses import dataclass
from typing import Optional

import torch

from .config import ModelConfig


@dataclass
class BacktestResult:
    score: torch.Tensor
    mean_return: float
    metrics: Optional[dict[str, float]] = None
    equity_curve: Optional[torch.Tensor] = None
    portfolio_returns: Optional[torch.Tensor] = None


class ChinaBacktest:
    """
    Vectorized backtest for China A-share/ETF.
    Uses open-to-open returns with turnover-based transaction cost,
    signal lag, and optional slippage.
    """

    def __init__(self):
        self.cost_rate = ModelConfig.COST_RATE
        self.slippage_rate = ModelConfig.SLIPPAGE_RATE
        self.slippage_impact = ModelConfig.SLIPPAGE_IMPACT
        self.allow_short = ModelConfig.ALLOW_SHORT
        self.signal_lag = max(0, ModelConfig.SIGNAL_LAG)
        self.annualization_factor = max(1, ModelConfig.ANNUALIZATION_FACTOR)

    def _compute_slippage(
        self,
        turnover: torch.Tensor,
        raw_data: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if (self.slippage_rate <= 0) and (self.slippage_impact <= 0):
            return torch.zeros_like(turnover)

        slip = turnover * self.slippage_rate
        if (
            self.slippage_impact > 0
            and raw_data
            and {"high", "low", "open"}.issubset(raw_data.keys())
        ):
            hl_range = (raw_data["high"] - raw_data["low"]).abs() / (raw_data["open"].abs() + 1e-6)
            slip = slip + turnover * self.slippage_impact * hl_range
        return slip

    def _compute_risk_metrics(
        self,
        portfolio_ret: torch.Tensor,
        equity_curve: torch.Tensor,
        turnover: torch.Tensor,
        position: torch.Tensor,
    ) -> dict[str, float]:
        eps = 1e-12
        n = portfolio_ret.numel()
        if n == 0:
            return {}

        mean = portfolio_ret.mean()
        std = portfolio_ret.std(unbiased=False)
        ann_factor = float(self.annualization_factor)

        ann_return = torch.pow(torch.clamp(1.0 + mean, min=eps), ann_factor) - 1.0
        ann_vol = std * math.sqrt(ann_factor)
        sharpe = (mean / (std + eps)) * math.sqrt(ann_factor)

        downside = torch.clamp(portfolio_ret, max=0.0)
        down_std = downside.std(unbiased=False)
        sortino = (mean / (down_std + eps)) * math.sqrt(ann_factor)

        equity_end = equity_curve[-1]
        total_return = equity_end - 1.0
        years = n / ann_factor if ann_factor > 0 else 0.0
        if years > 0:
            cagr = torch.pow(torch.clamp(equity_end, min=eps), 1.0 / years) - 1.0
        else:
            cagr = torch.tensor(0.0, device=portfolio_ret.device)

        peak = torch.cummax(equity_curve, dim=0)[0]
        drawdown = equity_curve / peak - 1.0
        max_drawdown = drawdown.min()
        calmar = cagr / (max_drawdown.abs() + eps)

        pos = portfolio_ret[portfolio_ret > 0]
        neg = portfolio_ret[portfolio_ret < 0]
        win_rate = (portfolio_ret > 0).float().mean()
        avg_win = pos.mean() if pos.numel() > 0 else torch.tensor(0.0, device=portfolio_ret.device)
        avg_loss = neg.mean() if neg.numel() > 0 else torch.tensor(0.0, device=portfolio_ret.device)
        profit_factor = pos.sum() / (neg.abs().sum() + eps) if neg.numel() > 0 else torch.tensor(float("inf"))
        expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss

        centered = portfolio_ret - mean
        m3 = (centered ** 3).mean()
        m4 = (centered ** 4).mean()
        skew = m3 / (std ** 3 + eps)
        kurtosis = m4 / (std ** 4 + eps) - 3.0

        avg_turnover = turnover.mean()
        gross_exposure = position.abs().mean()
        long_ratio = (position > 0).float().mean()
        short_ratio = (position < 0).float().mean()
        flat_ratio = (position == 0).float().mean()

        def to_float(value: torch.Tensor) -> float:
            return float(value.detach().cpu().item())

        return {
            "total_return": to_float(total_return),
            "cagr": to_float(cagr),
            "annual_return": to_float(ann_return),
            "annual_vol": to_float(ann_vol),
            "sharpe": to_float(sharpe),
            "sortino": to_float(sortino),
            "max_drawdown": to_float(max_drawdown),
            "calmar": to_float(calmar),
            "win_rate": to_float(win_rate),
            "profit_factor": float(to_float(profit_factor)) if torch.isfinite(profit_factor).item() else float("inf"),
            "avg_win": to_float(avg_win),
            "avg_loss": to_float(avg_loss),
            "expectancy": to_float(expectancy),
            "skew": to_float(skew),
            "kurtosis": to_float(kurtosis),
            "avg_turnover": to_float(avg_turnover),
            "gross_exposure": to_float(gross_exposure),
            "long_ratio": to_float(long_ratio),
            "short_ratio": to_float(short_ratio),
            "flat_ratio": to_float(flat_ratio),
        }

    def evaluate(
        self,
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> BacktestResult:
        if factors.numel() == 0:
            return BacktestResult(score=torch.tensor(-2.0, device=target_ret.device), mean_return=0.0)

        signal = torch.tanh(factors)

        if self.allow_short:
            position = torch.sign(signal)
        else:
            position = (signal > 0).float()

        if self.signal_lag > 0:
            position = torch.roll(position, self.signal_lag, dims=1)
            position[:, : self.signal_lag] = 0.0

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)

        slippage = self._compute_slippage(turnover, raw_data)
        pnl = position * target_ret - turnover * self.cost_rate - slippage

        mu = pnl.mean(dim=1)
        std = pnl.std(dim=1) + 1e-6

        neg_mask = pnl < 0
        neg_count = neg_mask.sum(dim=1)
        downside = torch.where(neg_mask, pnl, torch.zeros_like(pnl))
        down_mean = downside.sum(dim=1) / torch.clamp(neg_count, min=1)
        down_var = (neg_mask.float() * (downside - down_mean.unsqueeze(1)) ** 2).sum(dim=1)
        down_var = down_var / torch.clamp(neg_count - 1, min=1)
        down_std = torch.sqrt(down_var + 1e-6)

        use_down = neg_count > 5
        sortino = torch.where(use_down, mu / down_std, mu / std) * math.sqrt(self.annualization_factor)

        sortino = torch.where(mu < 0, torch.full_like(sortino, -2.0), sortino)
        sortino = torch.where(turnover.mean(dim=1) > 0.5, sortino - 1.0, sortino)
        sortino = torch.where(position.abs().sum(dim=1) == 0, torch.full_like(sortino, -2.0), sortino)

        sortino = torch.clamp(sortino, -3.0, 5.0)
        final_fitness = torch.median(sortino)
        mean_return = pnl.mean(dim=1).mean().item()

        if not return_details:
            return BacktestResult(score=final_fitness, mean_return=mean_return)

        portfolio_ret = pnl.mean(dim=0)
        equity_curve = torch.cumprod(torch.clamp(1.0 + portfolio_ret, min=1e-6), dim=0)
        metrics = self._compute_risk_metrics(portfolio_ret, equity_curve, turnover, position)

        return BacktestResult(
            score=final_fitness,
            mean_return=mean_return,
            metrics=metrics,
            equity_curve=equity_curve.detach().cpu(),
            portfolio_returns=portfolio_ret.detach().cpu(),
        )
