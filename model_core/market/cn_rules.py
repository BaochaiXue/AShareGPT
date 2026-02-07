from __future__ import annotations

from dataclasses import dataclass

import torch

from model_core.config import ModelConfig


def _normalize_code(code: str) -> str:
    """Normalize to bare code without exchange suffix. '510300.SH' -> '510300'."""
    return code.strip().upper().split(".")[0]


def _limit_pct_for_code(code: str) -> float:
    """Return price limit percentage for a given A-share code.

    Main board (SH 60xxxx / SZ 00xxxx): ±10%
    ChiNext (SZ 300xxx/301xxx):         ±20%
    STAR (SH 688xxx):                   ±20%
    BSE (8xxxxx and common NEEQ roots): ±30%
    Legacy delisted board (400/420):    ±5%

    Note: listing-day no-limit exemptions need external listing metadata and are
    handled via optional runtime masks, not just code prefix.
    """
    c = _normalize_code(code)
    if c.startswith("400") or c.startswith("420"):
        return 0.05
    if c.startswith("300") or c.startswith("301") or c.startswith("688"):
        return 0.20
    if c.startswith("8") or c.startswith("43") or c.startswith("83") or c.startswith("87"):
        return 0.30
    return 0.10


@dataclass(frozen=True)
class ChinaMarketRules:
    """
    Execution-rule layer for CN equities/ETFs:
    - T+1 settlement (same-day sell blocking)
    - Price limit (涨跌停) detection
    """

    enforce_t_plus_one: bool
    t0_allowed_codes: frozenset[str]
    limit_hit_tol: float

    @classmethod
    def from_config(cls) -> "ChinaMarketRules":
        return cls(
            enforce_t_plus_one=bool(ModelConfig.CN_ENFORCE_T_PLUS_ONE),
            t0_allowed_codes=frozenset(_normalize_code(c) for c in ModelConfig.CN_T0_ALLOWED_CODES),
            limit_hit_tol=ModelConfig.CN_LIMIT_HIT_TOL,
        )

    def is_t_plus_one(self, code: str) -> bool:
        if not self.enforce_t_plus_one:
            return False
        return _normalize_code(code) not in self.t0_allowed_codes

    # -----------------------------------------------------------------
    #  T+1 matrices
    # -----------------------------------------------------------------

    def t_plus_one_required_matrix(
        self,
        *,
        symbols: list[str],
        num_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        per_symbol = torch.tensor(
            [1.0 if self.is_t_plus_one(code) else 0.0 for code in symbols],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        if num_steps <= 0:
            return per_symbol[:, :0]
        return per_symbol.expand(-1, num_steps)

    @staticmethod
    def build_session_id_matrix(
        *,
        dates,
        num_assets: int,
        device: torch.device,
    ) -> torch.Tensor:
        session_codes, _ = dates.normalize().factorize(sort=False)
        session_1d = torch.tensor(session_codes, dtype=torch.float32, device=device).unsqueeze(0)
        if num_assets <= 0:
            return session_1d[:0]
        return session_1d.expand(num_assets, -1)

    @staticmethod
    def build_t_plus_one_sell_block(
        *,
        session_ids: torch.Tensor,
        t_plus_one_required: torch.Tensor,
        decision_freq: str,
    ) -> torch.Tensor:
        """Build sell-block matrix.

        - daily:  session changes every bar → same_session is all-zero → no
                  intra-step block.  The backtest loop itself prevents selling
                  shares bought in the same step via ``locked_buy``.
        - 1min:   within-session bars are blocked; cross-session = new day.
        """
        block = torch.zeros_like(t_plus_one_required)
        if t_plus_one_required.numel() == 0:
            return block
        same_session = (session_ids[:, 1:] == session_ids[:, :-1]).float()
        block[:, 1:] = same_session * t_plus_one_required[:, 1:]
        return block

    # -----------------------------------------------------------------
    #  Price-limit (涨跌停) matrices
    # -----------------------------------------------------------------

    @staticmethod
    def compute_prev_day_close(
        close: torch.Tensor,
        session_ids: torch.Tensor,
    ) -> torch.Tensor:
        """For each bar, return the closing price of the *previous* trading day.

        At session boundaries (new day), prev_day_close = close of last bar
        of the prior session.  Within a session, carry forward.
        """
        prev_day = close.clone()
        prev_day[:, 0] = close[:, 0]  # no prior day → use own close (no limit hit)
        for t in range(1, close.shape[1]):
            new_session = session_ids[:, t] != session_ids[:, t - 1]
            # New session → last bar's close IS the prev-day close
            # Same session → keep carrying forward
            prev_day[:, t] = torch.where(new_session, close[:, t - 1], prev_day[:, t - 1])
        return prev_day

    def build_limit_hit_masks(
        self,
        *,
        close: torch.Tensor,
        prev_day_close: torch.Tensor,
        symbols: list[str],
        limit_exempt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect limit-up and limit-down hits.

        Uses change **relative to previous day's close** (not adjacent bar),
        matching A-share 涨跌停 definition.

        Returns:
            limit_up:   Bool[assets, time] — 涨停, cannot BUY
            limit_down: Bool[assets, time] — 跌停, cannot SELL
        """
        pct_change = (close - prev_day_close) / (prev_day_close.abs() + 1e-8)

        thresholds = torch.tensor(
            [_limit_pct_for_code(s) for s in symbols],
            dtype=close.dtype,
            device=close.device,
        ).unsqueeze(1)

        tol = self.limit_hit_tol
        limit_up = pct_change >= (thresholds - tol)
        limit_down = pct_change <= (-thresholds + tol)

        if limit_exempt is not None:
            exempt = limit_exempt > 0
            limit_up = limit_up & (~exempt)
            limit_down = limit_down & (~exempt)

        # First bar of the dataset is never a limit hit
        limit_up[:, 0] = False
        limit_down[:, 0] = False
        return limit_up, limit_down
