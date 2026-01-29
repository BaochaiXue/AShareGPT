from datetime import datetime, time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from .config import ModelConfig
from .factors import FeatureEngineer


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

        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        self.target_ret = to_tensor(target_df)
        self.dates = index
        self.symbols = list(columns)

        print(f"CN Minute Data Ready. Shape: {self.feat_tensor.shape}")
