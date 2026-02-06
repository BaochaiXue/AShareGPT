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
    Builds decision tensors from minute bars with configurable:
    - bar semantics (`daily` vs `signal_snapshot`)
    - return semantics (`close_to_close` vs `signal_to_exit`)
    """

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = Path(data_root or ModelConfig.CN_MINUTE_DATA_ROOT)
        self.feat_tensor: Optional[torch.Tensor] = None
        self.raw_data_cache: Optional[dict[str, torch.Tensor]] = None
        self.target_ret: Optional[torch.Tensor] = None
        self.dates: Optional[pd.DatetimeIndex] = None
        self.symbols: Optional[list[str]] = None
        self.feature_norm_info: dict[str, int | str | float] = {}
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

    @staticmethod
    def _resolve_bar_style() -> str:
        style = ModelConfig.CN_BAR_STYLE.strip().lower()
        if style not in {"daily", "signal_snapshot"}:
            raise ValueError(
                f"Unsupported CN_BAR_STYLE={ModelConfig.CN_BAR_STYLE!r}; "
                "expected 'daily' or 'signal_snapshot'."
            )
        return style

    @staticmethod
    def _resolve_target_ret_mode() -> tuple[str, int]:
        mode = ModelConfig.CN_TARGET_RET_MODE.strip().lower()
        if mode not in {"close_to_close", "signal_to_exit"}:
            raise ValueError(
                f"Unsupported CN_TARGET_RET_MODE={ModelConfig.CN_TARGET_RET_MODE!r}; "
                "expected 'close_to_close' or 'signal_to_exit'."
            )
        if mode == "signal_to_exit":
            return mode, 0

        hold_days = int(ModelConfig.CN_HOLD_DAYS)
        if hold_days < 1:
            raise ValueError("CN_HOLD_DAYS must be >= 1 when CN_TARGET_RET_MODE='close_to_close'.")
        return mode, hold_days

    def _is_date_only_literal(self, value: str) -> bool:
        """Return True when value is a date-only literal without clock time."""
        value = value.strip()
        if not value:
            return False
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        return False

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
            allocated = train + val + test
            if allocated < total_len:
                test += total_len - allocated

            overflow = train + val + test - total_len
            if overflow > 0:
                trim = min(test, overflow)
                test -= trim
                overflow -= trim
            if overflow > 0:
                trim = min(val, overflow)
                val -= trim
                overflow -= trim
            if overflow > 0:
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

    def _validate_split_order(
        self,
        dates: pd.DatetimeIndex,
        train_len: int,
        val_len: int,
        test_len: int,
    ) -> None:
        if len(dates) == 0:
            return
        if not dates.is_monotonic_increasing:
            raise ValueError("Date index is not sorted ascending.")
        if train_len + val_len + test_len > len(dates):
            raise ValueError("Split sizes exceed available data length.")

        # Hard check: train < val < test chronological boundary.
        if train_len > 0 and val_len > 0:
            if dates[train_len - 1] >= dates[train_len]:
                raise ValueError("Split order invalid: train end must be earlier than val start.")
        if val_len > 0 and test_len > 0:
            boundary = train_len + val_len
            if dates[boundary - 1] >= dates[boundary]:
                raise ValueError("Split order invalid: val end must be earlier than test start.")
        if val_len == 0 and train_len > 0 and test_len > 0:
            boundary = train_len
            if dates[boundary - 1] >= dates[boundary]:
                raise ValueError("Split order invalid: train end must be earlier than test start.")

    def _validate_target_ret_mask(self, target_df: pd.DataFrame, target_tensor: torch.Tensor) -> None:
        expected_nan = torch.tensor(
            target_df.isna().to_numpy().T,
            dtype=torch.bool,
            device=target_tensor.device,
        )
        actual_nan = torch.isnan(target_tensor)
        if not torch.equal(expected_nan, actual_nan):
            raise ValueError("target_ret NaN mask changed after tensor conversion.")

        # Hard check: missing target entries must not be copied from previous day.
        for col_idx, col in enumerate(target_df.columns):
            series = target_df[col]
            missing_idx = series.index[series.isna()]
            if len(missing_idx) == 0:
                continue
            pos = int(target_df.index.get_loc(missing_idx[0]))
            if pos == 0:
                continue
            prev = series.iloc[pos - 1]
            if pd.isna(prev):
                continue
            cur = target_tensor[col_idx, pos]
            if torch.isfinite(cur):
                if abs(float(cur.item()) - float(prev)) < 1e-12:
                    raise ValueError(f"target_ret forward-fill detected for {col}.")

    def _normalize_features(self, raw_feat: torch.Tensor, train_len: int) -> torch.Tensor:
        mode = ModelConfig.CN_FEATURE_NORM
        clip = ModelConfig.CN_FEATURE_CLIP
        total_len = raw_feat.shape[2]

        if mode == "none":
            self.feature_norm_info = {"mode": "none", "fit_len": 0, "clip": clip}
            return raw_feat
        if mode != "train":
            raise ValueError(f"Unsupported CN_FEATURE_NORM={mode}; expected 'none' or 'train'.")
        if train_len <= 0:
            raise ValueError("CN_FEATURE_NORM=train requires a non-empty train split.")
        if train_len > total_len:
            raise ValueError("Train split exceeds feature timeline length.")

        train_feat = raw_feat[:, :, :train_len]
        norm_stats = FeatureEngineer.fit_robust_stats(train_feat)
        self.feature_norm_info = {"mode": "train", "fit_len": train_len, "clip": clip}
        return FeatureEngineer.apply_robust_norm(raw_feat, norm_stats=norm_stats, clip=clip)

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

    def _resolve_time_bounds(
        self,
        start_date: str,
        end_date: str,
    ) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        start_raw = start_date or ModelConfig.CN_MINUTE_START_DATE
        end_raw = end_date or ModelConfig.CN_MINUTE_END_DATE
        start_dt = pd.to_datetime(start_raw) if start_raw else None
        end_dt = pd.to_datetime(end_raw) if end_raw else None
        end_dt_exclusive: Optional[pd.Timestamp] = None
        if end_dt is not None and self._is_date_only_literal(end_raw):
            end_dt_exclusive = end_dt.normalize() + pd.Timedelta(days=1)
        return start_dt, end_dt, end_dt_exclusive

    def _load_daily_records_for_code(
        self,
        *,
        code: str,
        years: list[int],
        start_dt: Optional[pd.Timestamp],
        end_dt: Optional[pd.Timestamp],
        end_dt_exclusive: Optional[pd.Timestamp],
        sig_time: time,
        exit_t: Optional[time],
        bar_style: str,
        target_ret_mode: str,
    ) -> list[dict[str, float | pd.Timestamp]]:
        records: list[dict[str, float | pd.Timestamp]] = []
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
            if end_dt_exclusive is not None:
                # Date-only end bounds are inclusive of the whole end day.
                df = df[df["trade_time"] < end_dt_exclusive]
            elif end_dt is not None:
                df = df[df["trade_time"] <= end_dt]
            if df.empty:
                continue
            df["date"] = df["trade_time"].dt.normalize()

            for date, day_frame in df.groupby("date"):
                day_frame = day_frame.sort_values("trade_time")
                time_series = day_frame["trade_time"].dt.time
                entry_candidates = day_frame[time_series >= sig_time]
                entry_row = entry_candidates.iloc[0] if not entry_candidates.empty else day_frame.iloc[0]

                if exit_t:
                    exit_candidates = day_frame[time_series >= exit_t]
                    exit_row = exit_candidates.iloc[0] if not exit_candidates.empty else day_frame.iloc[-1]
                else:
                    exit_row = day_frame.iloc[-1]

                if bar_style == "daily":
                    rec = {
                        "date": date,
                        "open": float(day_frame.iloc[0]["open"]),
                        "high": float(day_frame["high"].max()),
                        "low": float(day_frame["low"].min()),
                        "close": float(day_frame.iloc[-1]["close"]),
                        "volume": float(day_frame["vol"].sum()),
                        "amount": float(day_frame["amount"].sum()),
                    }
                else:
                    rec = {
                        "date": date,
                        "open": float(entry_row["open"]),
                        "high": float(entry_row["high"]),
                        "low": float(entry_row["low"]),
                        "close": float(entry_row["close"]),
                        "volume": float(entry_row["vol"]),
                        "amount": float(entry_row["amount"]),
                    }

                if target_ret_mode == "signal_to_exit":
                    entry_open = float(entry_row["open"])
                    exit_close = float(exit_row["close"])
                    if entry_open == 0:
                        continue
                    rec["target_ret"] = (exit_close / entry_open) - 1.0

                records.append(rec)
        return records

    def _build_per_code_frames(
        self,
        *,
        codes: list[str],
        years: list[int],
        start_dt: Optional[pd.Timestamp],
        end_dt: Optional[pd.Timestamp],
        end_dt_exclusive: Optional[pd.Timestamp],
        sig_time: time,
        exit_t: Optional[time],
        bar_style: str,
        target_ret_mode: str,
        hold_days: int,
    ) -> dict[str, pd.DataFrame]:
        per_code_frames: dict[str, pd.DataFrame] = {}
        for code in codes:
            records = self._load_daily_records_for_code(
                code=code,
                years=years,
                start_dt=start_dt,
                end_dt=end_dt,
                end_dt_exclusive=end_dt_exclusive,
                sig_time=sig_time,
                exit_t=exit_t,
                bar_style=bar_style,
                target_ret_mode=target_ret_mode,
            )
            if not records:
                continue
            frame = pd.DataFrame(records)
            frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            if ModelConfig.CN_USE_ADJ_FACTOR:
                frame = self._apply_adj_factors(code, frame)
            if target_ret_mode == "close_to_close":
                frame["target_ret"] = (frame["close"] / frame["close"].shift(hold_days)) - 1.0
            per_code_frames[code] = frame
        return per_code_frames

    def _apply_recent_day_cutoff(
        self,
        per_code_frames: dict[str, pd.DataFrame],
        *,
        end_dt: Optional[pd.Timestamp],
    ) -> None:
        if end_dt is not None or not ModelConfig.CN_MINUTE_DAYS:
            return
        cutoff_days = ModelConfig.CN_MINUTE_DAYS
        for code, frame in list(per_code_frames.items()):
            frame = frame.sort_values("date")
            if len(frame) > cutoff_days:
                frame = frame.iloc[-cutoff_days:]
            per_code_frames[code] = frame

    def _build_pivots(self, per_code_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        def build_pivot(
            field: str,
            *,
            ffill: bool = True,
            fill_value: Optional[float] = 0.0,
        ) -> pd.DataFrame:
            series_list = []
            for code, frame in per_code_frames.items():
                s = frame.set_index("date")[field].rename(code)
                series_list.append(s)
            pivot = pd.concat(series_list, axis=1).sort_index()
            if ffill:
                pivot = pivot.ffill()
            if fill_value is not None:
                pivot = pivot.fillna(fill_value)
            return pivot

        pivot_specs: dict[str, tuple[bool, Optional[float]]] = {
            "open": (True, None),
            "high": (True, None),
            "low": (True, None),
            "close": (True, None),
            "volume": (False, 0.0),
            "amount": (False, 0.0),
            "target_ret": (False, None),
        }
        pivots = {
            field: build_pivot(field, ffill=ffill, fill_value=fill_value)
            for field, (ffill, fill_value) in pivot_specs.items()
        }
        if ModelConfig.CN_USE_ADJ_FACTOR:
            pivots["adj_factor"] = build_pivot("adj_factor", ffill=True, fill_value=1.0)
        return pivots

    @staticmethod
    def _pivot_to_tensor(
        pivot: pd.DataFrame,
        *,
        index: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> torch.Tensor:
        aligned = pivot.reindex(index=index, columns=columns)
        return torch.tensor(aligned.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

    def _build_tensors_from_pivots(
        self,
        pivots: dict[str, pd.DataFrame],
        *,
        index: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        target_tensor = self._pivot_to_tensor(pivots["target_ret"], index=index, columns=columns)
        self._validate_target_ret_mask(pivots["target_ret"], target_tensor)

        raw_data_cache = {
            "open": self._pivot_to_tensor(pivots["open"], index=index, columns=columns),
            "high": self._pivot_to_tensor(pivots["high"], index=index, columns=columns),
            "low": self._pivot_to_tensor(pivots["low"], index=index, columns=columns),
            "close": self._pivot_to_tensor(pivots["close"], index=index, columns=columns),
            "volume": self._pivot_to_tensor(pivots["volume"], index=index, columns=columns),
            "amount": self._pivot_to_tensor(pivots["amount"], index=index, columns=columns),
            "liquidity": self._pivot_to_tensor(pivots["amount"], index=index, columns=columns),
            "fdv": self._pivot_to_tensor(pivots["amount"], index=index, columns=columns),
        }
        if "adj_factor" in pivots:
            raw_data_cache["adj_factor"] = self._pivot_to_tensor(
                pivots["adj_factor"],
                index=index,
                columns=columns,
            )
        return raw_data_cache, target_tensor

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
        bar_style = self._resolve_bar_style()
        target_ret_mode, hold_days = self._resolve_target_ret_mode()
        start_dt, end_dt, end_dt_exclusive = self._resolve_time_bounds(start_date, end_date)
        per_code_frames = self._build_per_code_frames(
            codes=codes,
            years=years,
            start_dt=start_dt,
            end_dt=end_dt,
            end_dt_exclusive=end_dt_exclusive,
            sig_time=sig_time,
            exit_t=exit_t,
            bar_style=bar_style,
            target_ret_mode=target_ret_mode,
            hold_days=hold_days,
        )

        if not per_code_frames:
            raise ValueError("No minute data loaded. Check codes/years/date filters.")

        self._apply_recent_day_cutoff(per_code_frames, end_dt=end_dt)
        pivots = self._build_pivots(per_code_frames)

        index = pivots["close"].index
        columns = pivots["close"].columns
        train_len, val_len, test_len = self._resolve_split_sizes(len(index))
        self._validate_split_order(index, train_len, val_len, test_len)
        self.raw_data_cache, target_tensor = self._build_tensors_from_pivots(
            pivots,
            index=index,
            columns=columns,
        )

        raw_feat = FeatureEngineer.compute_features(
            self.raw_data_cache,
            normalize=False,
            strict_indicator_mapping=ModelConfig.CN_STRICT_FEATURE_INDICATORS,
            near_zero_std_tol=ModelConfig.CN_FEATURE_NEAR_ZERO_STD_TOL,
        )
        self.feat_tensor = self._normalize_features(raw_feat, train_len=train_len)
        self.target_ret = target_tensor
        self.dates = index
        self.symbols = list(columns)

        print(f"CN Minute Data Ready. Shape: {self.feat_tensor.shape}")
        if target_ret_mode == "close_to_close":
            print(f"[ret] mode=close_to_close hold_days={hold_days}")
        else:
            print(
                f"[ret] mode=signal_to_exit signal_time={sig_time.strftime('%H:%M:%S')} "
                f"exit_time={(exit_t.strftime('%H:%M:%S') if exit_t else 'eod')}"
            )
        print(f"[bar] style={bar_style}")
        if self.feature_norm_info:
            print(
                "[norm] mode={mode} fit_len={fit_len} clip={clip}".format(
                    mode=self.feature_norm_info.get("mode"),
                    fit_len=self.feature_norm_info.get("fit_len"),
                    clip=self.feature_norm_info.get("clip"),
                )
            )
