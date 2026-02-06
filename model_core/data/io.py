from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "gbk", "gb18030")


def read_csv_any_encoding(
    path: Path,
    *,
    usecols: Any = None,
    dtype: Any = None,
    encodings: Iterable[str] = DEFAULT_ENCODINGS,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read CSV by trying common encodings in order."""

    last_err: Optional[Exception] = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, usecols=usecols, dtype=dtype, encoding=encoding, **kwargs)
        except Exception as exc:  # noqa: BLE001 - fallback reader intentionally retries broadly
            last_err = exc
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"failed to read csv: {path}")


def safe_to_datetime(series: pd.Series, *, utc: bool = False) -> pd.Series:
    """Convert to datetime with invalid values coerced to NaT."""

    try:
        return pd.to_datetime(series, errors="coerce", utc=utc, format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce", utc=utc)


def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric with invalid values coerced to NaN."""

    return pd.to_numeric(series, errors="coerce")


def atomic_write_csv(path: Path, df: pd.DataFrame, *, index: bool = False, **kwargs: Any) -> None:
    """Write CSV atomically via a temporary file in the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=index, **kwargs)
    tmp_path.replace(path)


def read_last_row_token(path: Path, key_col: str) -> Optional[str]:
    """Return the last non-empty value for `key_col` from a UTF-8 CSV file."""

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or key_col not in header:
            return None
        key_idx = header.index(key_col)

        last_value: Optional[str] = None
        for row in reader:
            if key_idx >= len(row):
                continue
            value = row[key_idx].strip()
            if value:
                last_value = value

    return last_value
