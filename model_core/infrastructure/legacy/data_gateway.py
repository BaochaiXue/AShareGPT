from __future__ import annotations

from typing import Optional

from model_core.data_loader import (
    ChinaMinuteDataLoader,
    DataSlice as LegacyDataSlice,
    WalkForwardFold as LegacyWalkForwardFold,
)
from model_core.domain.models import DataBundle, DatasetSlice, WalkForwardBundle


class LegacyChinaDataGateway:
    """Adapter from legacy `ChinaMinuteDataLoader` to the data gateway port."""

    def __init__(self, loader: Optional[ChinaMinuteDataLoader] = None):
        self._loader = loader or ChinaMinuteDataLoader()

    @property
    def loader(self) -> ChinaMinuteDataLoader:
        return self._loader

    def load(
        self,
        *,
        codes: Optional[list[str]] = None,
        years: Optional[list[int]] = None,
        start_date: str = "",
        end_date: str = "",
        signal_time: str = "",
        exit_time: str = "",
        limit_codes: int = 50,
    ) -> None:
        self._loader.load_data(
            codes=codes,
            years=years,
            start_date=start_date,
            end_date=end_date,
            signal_time=signal_time,
            exit_time=exit_time,
            limit_codes=limit_codes,
        )

    def bundle(self) -> DataBundle:
        if (
            self._loader.feat_tensor is None
            or self._loader.raw_data_cache is None
            or self._loader.target_ret is None
            or self._loader.dates is None
        ):
            raise ValueError("Data not loaded. Call load() first.")
        return DataBundle(
            feat_tensor=self._loader.feat_tensor,
            raw_data_cache=self._loader.raw_data_cache,
            target_ret=self._loader.target_ret,
            dates=self._loader.dates,
            symbols=self._loader.symbols or [],
        )

    def train_val_test_split(self) -> dict[str, DatasetSlice]:
        return {
            name: self._convert_slice(data_slice)
            for name, data_slice in self._loader.train_val_test_split().items()
        }

    def walk_forward_splits(self) -> list[WalkForwardBundle]:
        return [self._convert_fold(fold) for fold in self._loader.walk_forward_splits()]

    def _convert_slice(self, data_slice: LegacyDataSlice) -> DatasetSlice:
        return DatasetSlice(
            feat_tensor=data_slice.feat_tensor,
            raw_data_cache=data_slice.raw_data_cache,
            target_ret=data_slice.target_ret,
            dates=data_slice.dates,
            symbols=data_slice.symbols,
            start_idx=data_slice.start_idx,
            end_idx=data_slice.end_idx,
        )

    def _convert_fold(self, fold: LegacyWalkForwardFold) -> WalkForwardBundle:
        return WalkForwardBundle(
            train=self._convert_slice(fold.train),
            val=self._convert_slice(fold.val),
            test=self._convert_slice(fold.test),
        )
