import torch
from .config import ModelConfig


class ChinaBacktest:
    """
    Vectorized backtest for China A-share/ETF.
    Uses open-to-open returns with turnover-based transaction cost.
    """

    def __init__(self):
        self.cost_rate = ModelConfig.COST_RATE
        self.allow_short = ModelConfig.ALLOW_SHORT

    def evaluate(
        self,
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if factors.numel() == 0:
            return torch.tensor(-2.0, device=target_ret.device), 0.0

        signal = torch.tanh(factors)

        if self.allow_short:
            position = torch.sign(signal)
        else:
            position = (signal > 0).float()

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)

        pnl = position * target_ret - turnover * self.cost_rate

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
        sortino = torch.where(use_down, mu / down_std, mu / std) * 15.87

        sortino = torch.where(mu < 0, torch.full_like(sortino, -2.0), sortino)
        sortino = torch.where(turnover.mean(dim=1) > 0.5, sortino - 1.0, sortino)
        sortino = torch.where(position.abs().sum(dim=1) == 0, torch.full_like(sortino, -2.0), sortino)

        sortino = torch.clamp(sortino, -3.0, 5.0)
        final_fitness = torch.median(sortino)
        mean_return = pnl.mean(dim=1).mean().item()

        return final_fitness, mean_return
