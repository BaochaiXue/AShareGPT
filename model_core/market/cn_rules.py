from __future__ import annotations

from dataclasses import dataclass

import torch

from model_core.config import ModelConfig


def _normalize_code(code: str) -> str:
    return code.strip().upper()


def _limit_pct_for_code(code: str) -> float:
    """Return price limit percentage for a given A-share code.

    主板(SH 60xxxx / SZ 00xxxx):  ±10%
    创业板(SZ 300xxx):            ±20%
    科创板(SH 688xxx):            ±20%
    北交所(BJ 8xxxxx/4xxxxx):     ±30%  (simplified)
    """
    c = _normalize_code(code).split(".")[0]
    if c.startswith("300") or c.startswith("688"):
        return 0.20
    if c.startswith("8") or c.startswith("4"):
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
    limit_hit_tol: float  # tolerance for limit detection (e.g. 0.001)

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

        For both daily and 1min freq:
        - daily:  any bar in the same calendar day as a buy cannot sell (effectively
                  means you can never sell on the same day you buy — whole row is
                  within one session).  We mark ALL bars that share a session with
                  their predecessor so that the backtest engine can block sells on
                  new-buy bars.
        - 1min:   within-session bars are blocked; cross-session boundary = new day
                  = can now sell yesterday's position.
        """
        block = torch.zeros_like(t_plus_one_required)
        if t_plus_one_required.numel() == 0:
            return block
        # For daily freq the session_id changes every row, so same_session is
        # all-zeros.  That is correct: the backtest engine separately enforces
        # that we cannot sell on the *buy-bar* itself by checking position
        # changes within the step.
        same_session = (session_ids[:, 1:] == session_ids[:, :-1]).float()
        block[:, 1:] = same_session * t_plus_one_required[:, 1:]
        return block

    # -----------------------------------------------------------------
    #  Price-limit (涨跌停) matrices
    # -----------------------------------------------------------------

    def build_limit_hit_masks(
        self,
        *,
        close: torch.Tensor,
        symbols: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect limit-up and limit-down hits.

        Returns:
            limit_up:   Bool[assets, time] — 涨停, cannot BUY  (price already at ceiling)
            limit_down: Bool[assets, time] — 跌停, cannot SELL (price already at floor)
        """
        # pct change vs prev bar
        prev_close = torch.zeros_like(close)
        prev_close[:, 1:] = close[:, :-1]
        prev_close[:, 0] = close[:, 0]  # no change on first bar

        pct_change = (close - prev_close) / (prev_close.abs() + 1e-8)

        # per-code limit thresholds
        thresholds = torch.tensor(
            [_limit_pct_for_code(s) for s in symbols],
            dtype=close.dtype,
            device=close.device,
        ).unsqueeze(1)

        tol = self.limit_hit_tol
        limit_up = pct_change >= (thresholds - tol)
        limit_down = pct_change <= (-thresholds + tol)

        # First bar is never a limit hit
        limit_up[:, 0] = False
        limit_down[:, 0] = False
        return limit_up, limit_down
