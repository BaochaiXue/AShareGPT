from __future__ import annotations

from dataclasses import dataclass

import torch

from model_core.config import ModelConfig


def _normalize_code(code: str) -> str:
    return code.strip().upper()


@dataclass(frozen=True)
class ChinaMarketRules:
    """
    Minimal execution-rule layer for CN equities/ETFs.

    The default is conservative: enforce T+1 for all symbols unless explicitly
    whitelisted in CN_T0_ALLOWED_CODES.
    """

    enforce_t_plus_one: bool
    t0_allowed_codes: frozenset[str]

    @classmethod
    def from_config(cls) -> "ChinaMarketRules":
        return cls(
            enforce_t_plus_one=bool(ModelConfig.CN_ENFORCE_T_PLUS_ONE),
            t0_allowed_codes=frozenset(_normalize_code(c) for c in ModelConfig.CN_T0_ALLOWED_CODES),
        )

    def is_t_plus_one(self, code: str) -> bool:
        if not self.enforce_t_plus_one:
            return False
        return _normalize_code(code) not in self.t0_allowed_codes

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
        # Session id is based on calendar day. This works for daily and minute timelines.
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
        block = torch.zeros_like(t_plus_one_required)
        if t_plus_one_required.numel() == 0:
            return block
        if decision_freq != "1min":
            return block

        same_session = (session_ids[:, 1:] == session_ids[:, :-1]).float()
        block[:, 1:] = same_session * t_plus_one_required[:, 1:]
        return block
