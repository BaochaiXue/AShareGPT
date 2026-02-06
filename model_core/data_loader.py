from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from .config import ModelConfig
from .code_alias import load_code_alias_map
from .factors import FeatureEngineer


@dataclass
class DataSlice:
    feat_tensor: torch.Tensor
    raw_data_cache: dict[str, torch.Tensor]
    target_ret: torch.Tensor
    dates: pd.DatetimeIndex
    symbols: list[str]
    start_idx: int
    end_idx: int


@dataclass
class WalkForwardFold:
    train: DataSlice
    val: DataSlice
    test: DataSlice


class ChinaMinuteDataLoader:
    """
    Minute-level data loader for China A-share/ETF.
    Builds daily decision tensors from minute bars, then holds to exit minute.
    """

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = Path(data_root or ModelConfig.CN_MINUTE_DATA_ROOT)
        self.feat_tensor: Optional[torch.Tensor] = None
        self.raw_data_cache: Optional[dict[str, torch.Tensor]] = None
        self.target_ret: Optional[torch.Tensor] = None
        self.dates: Optional[pd.DatetimeIndex] = None
        self.symbols: Optional[list[str]] = None
        alias_path = Path(ModelConfig.CN_CODE_ALIAS_FILE)
        if not alias_path.is_absolute():
            alias_path = self.data_root.parent / alias_path
        self.code_alias_map = load_code_alias_map(alias_path)
        self._warned_missing_adj: set[str] = set()
        self._warned_alias_adj: set[str] = set()

    def _parse_time(self, value: str) -> Optional[time]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%H:%M").time()
        except ValueError:
            try:
                return datetime.strptime(value, "%H:%M:%S").time()
            except ValueError:
                return None

    def _resolve_years(self, years: Optional[list[int]]) -> list[int]:
        if years:
            return years
        if ModelConfig.CN_MINUTE_YEARS:
            return ModelConfig.CN_MINUTE_YEARS
        if not self.data_root.exists():
            return []
        available = [int(p.name) for p in self.data_root.iterdir() if p.is_dir() and p.name.isdigit()]
        available.sort()
        return available[-2:] if len(available) >= 2 else available

    def _resolve_codes(self, codes: Optional[list[str]], years: list[int], limit_codes: int) -> list[str]:
        if codes:
            return codes
        if ModelConfig.CN_CODES:
            return ModelConfig.CN_CODES[:limit_codes] if limit_codes else ModelConfig.CN_CODES
        candidates = []
        for year in years:
            year_dir = self.data_root / str(year)
            if not year_dir.exists():
                continue
            candidates.extend([p.stem for p in year_dir.glob("*.csv")])
            if candidates:
                break
        candidates = sorted(set(candidates))
        return candidates[:limit_codes] if limit_codes else candidates

    def _read_adj_factor_csv(self, path: Path) -> pd.DataFrame:
        encodings = ("utf-8", "utf-8-sig", "gbk", "gb18030")
        last_err: Optional[Exception] = None
        for enc in encodings:
            try:
                return pd.read_csv(
                    path,
                    usecols=lambda c: c in {"date", "adj_factor", "code", "证券代码"},
                    dtype={"date": "string"},
                    encoding=enc,
                )
            except Exception as exc:
                last_err = exc
        if last_err:
            raise last_err
        return pd.DataFrame()

    def _load_adj_factors(self, code: str) -> Optional[pd.DataFrame]:
        if not ModelConfig.CN_USE_ADJ_FACTOR:
            return None
        alias_code = self.code_alias_map.get(code)
        candidates = [code]
        if alias_code and alias_code != code:
            candidates.append(alias_code)

        for candidate in candidates:
            path = self.data_root / ModelConfig.CN_ADJ_FACTOR_DIR / f"{candidate}.csv"
            if not path.exists():
                continue
            try:
                df = self._read_adj_factor_csv(path)
            except Exception:
                continue
            if df.empty or "adj_factor" not in df.columns or "date" not in df.columns:
                continue
            df = df.loc[:, ["date", "adj_factor"]].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df["date"] = df["date"].dt.normalize()
            df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
            df = df.dropna(subset=["adj_factor"])
            if df.empty:
                continue
            df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            if candidate != code and code not in self._warned_alias_adj:
                print(f"[adj] alias applied: {code} -> {candidate}")
                self._warned_alias_adj.add(code)
            return df

        if code not in self._warned_missing_adj:
            print(f"[adj] missing adj_factor for {code}; fallback to 1.0")
            self._warned_missing_adj.add(code)
        return None

    def _apply_adj_factors(self, code: str, frame: pd.DataFrame) -> pd.DataFrame:
        adj = self._load_adj_factors(code)
        if adj is None or adj.empty:
            frame["adj_factor"] = 1.0
            return frame
        merged = frame.merge(adj, on="date", how="left").sort_values("date")
        merged["adj_factor"] = merged["adj_factor"].ffill().fillna(1.0)
        for col in ("open", "high", "low", "close"):
            merged[col] = merged[col].astype("float64") * merged["adj_factor"]
        return merged

    def _resolve_split_sizes(self, total_len: int) -> tuple[int, int, int]:
        if total_len <= 0:
            return 0, 0, 0
        if ModelConfig.CN_TRAIN_DAYS or ModelConfig.CN_VAL_DAYS or ModelConfig.CN_TEST_DAYS:
            train = max(0, ModelConfig.CN_TRAIN_DAYS)
            val = max(0, ModelConfig.CN_VAL_DAYS)
            test = max(0, ModelConfig.CN_TEST_DAYS)
            if train + val + test == 0:
                train = total_len
            if train + val + test < total_len:
                test += total_len - (train + val + test)
            if train + val + test > total_len:
                overflow = train + val + test - total_len
                test = max(0, test - overflow)
            if train + val + test > total_len:
                overflow = train + val + test - total_len
                val = max(0, val - overflow)
            if train + val + test > total_len:
                overflow = train + val + test - total_len
                train = max(0, train - overflow)
            return train, val, test
        train = int(total_len * ModelConfig.CN_TRAIN_RATIO)
        val = int(total_len * ModelConfig.CN_VAL_RATIO)
        if train <= 0:
            train = max(1, total_len - val)
        if train + val > total_len:
            val = max(0, total_len - train)
        test = max(0, total_len - train - val)
        return train, val, test

    def _slice_raw_data(self, start: int, end: int) -> dict[str, torch.Tensor]:
        if self.raw_data_cache is None:
            raise ValueError("raw_data_cache is empty. Call load_data() first.")
        return {k: v[:, start:end] for k, v in self.raw_data_cache.items()}

    def get_slice(self, start: int, end: int) -> DataSlice:
        if self.feat_tensor is None or self.target_ret is None or self.dates is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        start = max(0, start)
        end = max(start, min(end, self.feat_tensor.shape[2]))
        return DataSlice(
            feat_tensor=self.feat_tensor[:, :, start:end],
            raw_data_cache=self._slice_raw_data(start, end),
            target_ret=self.target_ret[:, start:end],
            dates=self.dates[start:end],
            symbols=self.symbols or [],
            start_idx=start,
            end_idx=end,
        )

    def train_val_test_split(self) -> dict[str, DataSlice]:
        if self.dates is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        total_len = len(self.dates)
        train_len, val_len, test_len = self._resolve_split_sizes(total_len)
        splits: dict[str, DataSlice] = {}
        cursor = 0
        if train_len > 0:
            splits["train"] = self.get_slice(cursor, cursor + train_len)
            cursor += train_len
        if val_len > 0:
            splits["val"] = self.get_slice(cursor, cursor + val_len)
            cursor += val_len
        if test_len > 0:
            splits["test"] = self.get_slice(cursor, cursor + test_len)
        return splits

    def walk_forward_splits(self) -> list[WalkForwardFold]:
        if self.dates is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        total_len = len(self.dates)
        train_len = max(0, ModelConfig.CN_WFO_TRAIN_DAYS)
        val_len = max(0, ModelConfig.CN_WFO_VAL_DAYS)
        test_len = max(0, ModelConfig.CN_WFO_TEST_DAYS)
        step_len = max(1, ModelConfig.CN_WFO_STEP_DAYS)
        window = train_len + val_len + test_len
        if window <= 0 or total_len < window:
            return []
        folds: list[WalkForwardFold] = []
        start = 0
        while start + window <= total_len:
            train_slice = self.get_slice(start, start + train_len)
            val_slice = self.get_slice(start + train_len, start + train_len + val_len)
            test_slice = self.get_slice(start + train_len + val_len, start + window)
            folds.append(WalkForwardFold(train=train_slice, val=val_slice, test=test_slice))
            start += step_len
        return folds

    def load_data(
        self,
        codes: Optional[list[str]] = None,
        years: Optional[list[int]] = None,
        start_date: str = "",
        end_date: str = "",
        signal_time: str = "",
        exit_time: str = "",
        limit_codes: int = 50,
    ) -> None:
        years = self._resolve_years(years)
        if not years:
            raise ValueError("No available year folders for minute data.")

        codes = self._resolve_codes(codes, years, limit_codes)
        if not codes:
            raise ValueError("No codes resolved for minute data.")

        sig_time = self._parse_time(signal_time or ModelConfig.CN_SIGNAL_TIME) or time(10, 0)
        exit_t = self._parse_time(exit_time or ModelConfig.CN_EXIT_TIME)

        start_dt = pd.to_datetime(start_date or ModelConfig.CN_MINUTE_START_DATE) if (start_date or ModelConfig.CN_MINUTE_START_DATE) else None
        end_dt = pd.to_datetime(end_date or ModelConfig.CN_MINUTE_END_DATE) if (end_date or ModelConfig.CN_MINUTE_END_DATE) else None

        per_code_frames: dict[str, pd.DataFrame] = {}

        for code in codes:
            records = []
            for year in years:
                path = self.data_root / str(year) / f"{code}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(
                    path,
                    usecols=["trade_time", "open", "high", "low", "close", "vol", "amount"],
                    dtype={"trade_time": "string"},
                )
                if df.empty:
                    continue
                df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
                df = df.dropna(subset=["trade_time"])
                if start_dt is not None:
                    df = df[df["trade_time"] >= start_dt]
                if end_dt is not None:
                    df = df[df["trade_time"] <= end_dt]
                if df.empty:
                    continue
                df["date"] = df["trade_time"].dt.normalize()

                for date, g in df.groupby("date"):
                    g = g.sort_values("trade_time")
                    time_series = g["trade_time"].dt.time
                    entry_candidates = g[time_series >= sig_time]
                    entry_row = entry_candidates.iloc[0] if not entry_candidates.empty else g.iloc[0]

                    if exit_t:
                        exit_candidates = g[time_series >= exit_t]
                        exit_row = exit_candidates.iloc[0] if not exit_candidates.empty else g.iloc[-1]
                    else:
                        exit_row = g.iloc[-1]

                    entry_open = float(entry_row["open"])
                    exit_close = float(exit_row["close"])
                    if entry_open == 0:
                        continue

                    records.append(
                        {
                            "date": date,
                            "open": float(entry_row["open"]),
                            "high": float(entry_row["high"]),
                            "low": float(entry_row["low"]),
                            "close": float(entry_row["close"]),
                            "volume": float(entry_row["vol"]),
                            "amount": float(entry_row["amount"]),
                            "target_ret": (exit_close / entry_open) - 1.0,
                        }
                    )

            if records:
                frame = pd.DataFrame(records)
                frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date")
                if ModelConfig.CN_USE_ADJ_FACTOR:
                    frame = self._apply_adj_factors(code, frame)
                per_code_frames[code] = frame

        if not per_code_frames:
            raise ValueError("No minute data loaded. Check codes/years/date filters.")

        if end_dt is None and ModelConfig.CN_MINUTE_DAYS:
            cutoff_days = ModelConfig.CN_MINUTE_DAYS
            for code, frame in list(per_code_frames.items()):
                frame = frame.sort_values("date")
                if len(frame) > cutoff_days:
                    frame = frame.iloc[-cutoff_days:]
                per_code_frames[code] = frame

        def build_pivot(field: str) -> pd.DataFrame:
            series_list = []
            for code, frame in per_code_frames.items():
                s = frame.set_index("date")[field].rename(code)
                series_list.append(s)
            pivot = pd.concat(series_list, axis=1).sort_index()
            pivot = pivot.ffill().fillna(0.0)
            return pivot

        open_df = build_pivot("open")
        high_df = build_pivot("high")
        low_df = build_pivot("low")
        close_df = build_pivot("close")
        volume_df = build_pivot("volume")
        amount_df = build_pivot("amount")
        target_df = build_pivot("target_ret")
        adj_df = build_pivot("adj_factor") if ModelConfig.CN_USE_ADJ_FACTOR else None

        index = close_df.index
        columns = close_df.columns

        def to_tensor(pivot: pd.DataFrame) -> torch.Tensor:
            pivot = pivot.reindex(index=index, columns=columns)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

        self.raw_data_cache = {
            "open": to_tensor(open_df),
            "high": to_tensor(high_df),
            "low": to_tensor(low_df),
            "close": to_tensor(close_df),
            "volume": to_tensor(volume_df),
            "amount": to_tensor(amount_df),
            "liquidity": to_tensor(amount_df),
            "fdv": to_tensor(amount_df),
        }
        if adj_df is not None:
            self.raw_data_cache["adj_factor"] = to_tensor(adj_df)

        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        self.target_ret = to_tensor(target_df)
        self.dates = index
        self.symbols = list(columns)

        print(f"CN Minute Data Ready. Shape: {self.feat_tensor.shape}")
