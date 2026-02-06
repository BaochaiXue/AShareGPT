from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from model_core.domain.models import Formula
from model_core.ports.interfaces import (
    BacktestEnginePort,
    DataGatewayPort,
    FormulaExecutorPort,
)


@dataclass
class BacktestUseCaseResult:
    """Result envelope for orchestrated backtest calls."""

    ok: bool
    message: str
    payload: Optional[dict] = None


class BacktestFormulaUseCase:
    """Application service that orchestrates load -> execute -> evaluate."""

    def __init__(
        self,
        data_gateway: DataGatewayPort,
        executor: FormulaExecutorPort,
        backtest_engine: BacktestEnginePort,
    ):
        self._data_gateway = data_gateway
        self._executor = executor
        self._backtest_engine = backtest_engine

    def run(
        self,
        *,
        formula: Formula,
        mode: str = "full",
        symbols: Optional[list[str]] = None,
        years: Optional[list[int]] = None,
        start_date: str = "",
        end_date: str = "",
        signal_time: str = "",
        exit_time: str = "",
        limit_codes: int = 50,
        return_details: bool = True,
    ) -> BacktestUseCaseResult:
        if mode not in {"full", "split", "walk_forward"}:
            return BacktestUseCaseResult(ok=False, message=f"Unsupported mode: {mode}")

        self._data_gateway.load(
            codes=symbols,
            years=years,
            start_date=start_date,
            end_date=end_date,
            signal_time=signal_time,
            exit_time=exit_time,
            limit_codes=limit_codes,
        )
        bundle = self._data_gateway.bundle()
        factors = self._executor.execute(formula, bundle.feat_tensor)
        if factors is None:
            return BacktestUseCaseResult(
                ok=False,
                message="Invalid formula - execution failed.",
            )

        warnings: list[str] = []
        if torch.std(factors) < 1e-4:
            warnings.append("Factor has near-zero variance (trivial formula).")

        payload: dict[str, Any] = {
            "mode": mode,
            "symbols": bundle.symbols,
            "dates": bundle.dates,
            "feat_shape": tuple(bundle.feat_tensor.shape),
            "warnings": warnings,
        }

        if mode == "full":
            result = self._backtest_engine.evaluate(
                factors,
                bundle.raw_data_cache,
                bundle.target_ret,
                return_details=return_details,
            )
            payload["result"] = result
            return BacktestUseCaseResult(
                ok=True,
                message="Backtest completed.",
                payload=payload,
            )

        if mode == "split":
            split_results: dict[str, Any] = {}
            splits = self._data_gateway.train_val_test_split()
            for name in ("train", "val", "test"):
                split_slice = splits.get(name)
                if split_slice is None:
                    continue
                if split_slice.end_idx <= split_slice.start_idx:
                    continue
                res_slice = factors[:, split_slice.start_idx : split_slice.end_idx]
                result = self._backtest_engine.evaluate(
                    res_slice,
                    split_slice.raw_data_cache,
                    split_slice.target_ret,
                    return_details=return_details,
                )
                split_results[name] = {
                    "result": result,
                    "dates": split_slice.dates,
                }
            payload["splits"] = split_results
            return BacktestUseCaseResult(
                ok=True,
                message="Split backtest completed.",
                payload=payload,
            )

        fold_results: list[dict[str, Any]] = []
        val_scores: list[float] = []
        test_scores: list[float] = []
        folds = self._data_gateway.walk_forward_splits()
        for idx, fold in enumerate(folds, 1):
            fold_out: dict[str, Any] = {"index": idx}
            if fold.val.end_idx > fold.val.start_idx:
                res_val = factors[:, fold.val.start_idx : fold.val.end_idx]
                val_result = self._backtest_engine.evaluate(
                    res_val,
                    fold.val.raw_data_cache,
                    fold.val.target_ret,
                    return_details=return_details,
                )
                fold_out["val"] = val_result
                val_scores.append(float(val_result.score))
            if fold.test.end_idx > fold.test.start_idx:
                res_test = factors[:, fold.test.start_idx : fold.test.end_idx]
                test_result = self._backtest_engine.evaluate(
                    res_test,
                    fold.test.raw_data_cache,
                    fold.test.target_ret,
                    return_details=return_details,
                )
                fold_out["test"] = test_result
                test_scores.append(float(test_result.score))
            if "val" in fold_out or "test" in fold_out:
                fold_results.append(fold_out)

        if not fold_results:
            payload["folds"] = []
            return BacktestUseCaseResult(
                ok=True,
                message="Walk-forward disabled: not enough data for configured windows.",
                payload=payload,
            )

        payload["folds"] = fold_results
        payload["avg_val_score"] = float(sum(val_scores) / len(val_scores)) if val_scores else None
        payload["avg_test_score"] = float(sum(test_scores) / len(test_scores)) if test_scores else None
        return BacktestUseCaseResult(
            ok=True,
            message="Walk-forward backtest completed.",
            payload=payload,
        )
