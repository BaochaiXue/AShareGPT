from __future__ import annotations

from pathlib import Path

import pandas as pd

from model_core.data.io import (
    atomic_write_csv,
    read_csv_any_encoding,
    read_last_row_token,
    safe_to_datetime,
    safe_to_numeric,
)


def test_read_csv_any_encoding_supports_gb18030(tmp_path: Path) -> None:
    path = tmp_path / "adj.csv"
    content = "code,date,adj_factor\n000001.SZ,20240102,1.23\n"
    path.write_text(content, encoding="gb18030")

    df = read_csv_any_encoding(path, dtype={"date": "string"})
    assert list(df.columns) == ["code", "date", "adj_factor"]
    assert df.iloc[0]["code"] == "000001.SZ"


def test_atomic_write_csv_and_read_last_row_token(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    df = pd.DataFrame(
        {
            "trade_time": ["2024-01-02 09:30:00", "2024-01-03 09:30:00"],
            "value": [1, 2],
        }
    )
    atomic_write_csv(path, df, index=False)

    token = read_last_row_token(path, "trade_time")
    assert token == "2024-01-03 09:30:00"


def test_safe_converters_coerce_invalid_values() -> None:
    dt = safe_to_datetime(pd.Series(["2024-01-02", "bad"]))
    num = safe_to_numeric(pd.Series(["1.23", "bad"]))

    assert str(dt.iloc[0].date()) == "2024-01-02"
    assert pd.isna(dt.iloc[1])
    assert num.iloc[0] == 1.23
    assert pd.isna(num.iloc[1])
