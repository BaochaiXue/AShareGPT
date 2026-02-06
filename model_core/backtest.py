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


@dataclass
class TradingPath:
    valid: torch.Tensor
    valid_f: torch.Tensor
    tradable_f: torch.Tensor
    position: torch.Tensor
    turnover: torch.Tensor
    pnl: torch.Tensor


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
            # Missing OHLC should not poison pnl with NaN slippage.
            hl_range = torch.nan_to_num(hl_range, nan=0.0, posinf=0.0, neginf=0.0)
            slip = slip + turnover * self.slippage_impact * hl_range
        return torch.nan_to_num(slip, nan=0.0, posinf=0.0, neginf=0.0)

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

    def _build_position(self, signal: torch.Tensor) -> torch.Tensor:
        if self.allow_short:
            position = torch.sign(signal)
        else:
            position = (signal > 0).float()

        if self.signal_lag > 0:
            position = torch.roll(position, self.signal_lag, dims=1)
            position[:, : self.signal_lag] = 0.0

        return position

    def _apply_execution_constraints(
        self,
        desired_position: torch.Tensor,
        tradable_f: torch.Tensor,
        raw_data: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        position = torch.zeros_like(desired_position)
        prev_pos = torch.zeros(
            desired_position.shape[0],
            dtype=desired_position.dtype,
            device=desired_position.device,
        )
        locked_buy = torch.zeros_like(prev_pos)

        sell_block = None
        t_plus_one_required = None
        session_id = None
        if raw_data is not None:
            sell_block = raw_data.get("t_plus_one_sell_block")
            t_plus_one_required = raw_data.get("t_plus_one_required")
            session_id = raw_data.get("session_id")
        if sell_block is not None and sell_block.shape != desired_position.shape:
            raise ValueError(
                "raw_data['t_plus_one_sell_block'] must match signal shape [assets, time]."
            )
        if t_plus_one_required is not None and t_plus_one_required.shape != desired_position.shape:
            raise ValueError(
                "raw_data['t_plus_one_required'] must match signal shape [assets, time]."
            )
        if session_id is not None and session_id.shape != desired_position.shape:
            raise ValueError(
                "raw_data['session_id'] must match signal shape [assets, time]."
            )

        # Price-limit masks
        limit_up = None
        limit_down = None
        if raw_data is not None:
            limit_up = raw_data.get("limit_up")
            limit_down = raw_data.get("limit_down")

        if t_plus_one_required is None and sell_block is not None:
            per_asset_required = (sell_block.max(dim=1, keepdim=True).values > 0).float()
            t_plus_one_required = per_asset_required.expand_as(desired_position)
        if t_plus_one_required is None:
            t_plus_one_required = torch.zeros_like(desired_position)
        if session_id is None:
            session_id = torch.zeros_like(desired_position)

        prev_session = session_id[:, 0].clone()

        for t in range(desired_position.shape[1]):
            current_session = session_id[:, t]
            if t == 0:
                session_changed = torch.ones_like(current_session, dtype=torch.bool)
            else:
                session_changed = current_session != prev_session
            locked_buy = torch.where(session_changed, torch.zeros_like(locked_buy), locked_buy)

            desired_t = desired_position[:, t]
            tradable_t = tradable_f[:, t] > 0
            t_plus_one_t = t_plus_one_required[:, t] > 0

            if sell_block is not None:
                blocked_t = sell_block[:, t] > 0
            else:
                blocked_t = t_plus_one_t & (~session_changed)

            min_allowed = locked_buy
            reduce_long = desired_t < min_allowed
            desired_t = torch.where(blocked_t & reduce_long, min_allowed, desired_t)

            # --- Price-limit enforcement ---
            # 涨停: cannot buy (increase long position)
            if limit_up is not None:
                up_hit = limit_up[:, t] > 0
                buy_increase = desired_t > prev_pos
                desired_t = torch.where(up_hit & buy_increase, prev_pos, desired_t)

            # 跌停: cannot sell (decrease long position)
            if limit_down is not None:
                dn_hit = limit_down[:, t] > 0
                sell_decrease = desired_t < prev_pos
                desired_t = torch.where(dn_hit & sell_decrease, prev_pos, desired_t)

            current_pos = torch.where(tradable_t, desired_t, prev_pos)

            buy_increase = torch.clamp(current_pos - prev_pos, min=0.0)
            locked_buy = torch.where(t_plus_one_t, locked_buy + buy_increase, torch.zeros_like(locked_buy))
            locked_buy = torch.minimum(locked_buy, torch.clamp(current_pos, min=0.0))

            position[:, t] = current_pos
            prev_pos = current_pos
            prev_session = current_session

        return position

    def _compute_turnover(self, position: torch.Tensor, tradable_f: torch.Tensor) -> torch.Tensor:
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        # Charge turnover only when market is tradable.
        return turnover * tradable_f

    def _compute_pnl(
        self,
        *,
        position: torch.Tensor,
        turnover: torch.Tensor,
        target_ret: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        has_return_f: torch.Tensor,
    ) -> torch.Tensor:
        slippage = self._compute_slippage(turnover, raw_data)
        safe_target = torch.nan_to_num(target_ret, nan=0.0, posinf=0.0, neginf=0.0)
        pnl = position * safe_target - turnover * self.cost_rate - slippage
        return pnl * has_return_f

    def _build_trading_path(
        self,
        *,
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
    ) -> TradingPath:
        valid = torch.isfinite(target_ret)
        has_return_f = valid.float()
        if "tradable" in raw_data:
            tradable_f = torch.nan_to_num(raw_data["tradable"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            tradable_f = torch.ones_like(has_return_f)
        signal = torch.tanh(factors)
        desired_position = self._build_position(signal)
        position = self._apply_execution_constraints(desired_position, tradable_f, raw_data)
        turnover = self._compute_turnover(position, tradable_f)
        pnl = self._compute_pnl(
            position=position,
            turnover=turnover,
            target_ret=target_ret,
            raw_data=raw_data,
            has_return_f=has_return_f,
        )
        return TradingPath(
            valid=valid,
            valid_f=has_return_f,
            tradable_f=tradable_f,
            position=position,
            turnover=turnover,
            pnl=pnl,
        )

    def _compute_score(self, path: TradingPath) -> tuple[torch.Tensor, float]:
        valid = path.valid
        valid_f = path.valid_f
        pnl = path.pnl
        position = path.position
        turnover = path.turnover

        valid_count = valid_f.sum(dim=1)
        has_obs = valid_count > 0
        valid_count_safe = torch.clamp(valid_count, min=1.0)

        mu = pnl.sum(dim=1) / valid_count_safe
        centered = (pnl - mu.unsqueeze(1)) * valid_f
        std = torch.sqrt((centered ** 2).sum(dim=1) / valid_count_safe + 1e-6)

        neg_mask = (pnl < 0) & valid
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
        sortino = torch.where(has_obs, sortino, torch.full_like(sortino, -2.0))
        sortino = torch.clamp(sortino, -3.0, 5.0)

        final_fitness = torch.median(sortino)
        total_valid = torch.clamp(valid_f.sum(), min=1.0)
        mean_return = (pnl.sum() / total_valid).item()
        return final_fitness, mean_return

    @staticmethod
    def _compute_portfolio_curve(path: TradingPath) -> tuple[torch.Tensor, torch.Tensor]:
        valid_per_t = path.valid_f.sum(dim=0)
        portfolio_ret = torch.where(
            valid_per_t > 0,
            path.pnl.sum(dim=0) / torch.clamp(valid_per_t, min=1.0),
            torch.zeros_like(valid_per_t),
        )
        equity_curve = torch.cumprod(torch.clamp(1.0 + portfolio_ret, min=1e-6), dim=0)
        return portfolio_ret, equity_curve

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

        path = self._build_trading_path(factors=factors, raw_data=raw_data, target_ret=target_ret)
        final_fitness, mean_return = self._compute_score(path)

        if not return_details:
            return BacktestResult(score=final_fitness, mean_return=mean_return)

        portfolio_ret, equity_curve = self._compute_portfolio_curve(path)
        metrics = self._compute_risk_metrics(portfolio_ret, equity_curve, path.turnover, path.position)

        return BacktestResult(
            score=final_fitness,
            mean_return=mean_return,
            metrics=metrics,
            equity_curve=equity_curve.detach().cpu(),
            portfolio_returns=portfolio_ret.detach().cpu(),
        )
