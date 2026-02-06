from __future__ import annotations

from pathlib import Path

import pandas as pd

from unify_data import append_group


def test_append_group_uses_datetime_comparison_for_minute_data(tmp_path: Path) -> None:
    out = tmp_path / "000001.SZ.csv"
    out.write_text(
        "trade_time,open,high,low,close,vol,amount\n"
        "2024-01-10 09:30:00,1,1,1,1,100,1000\n",
        encoding="utf-8",
    )

    group = pd.DataFrame(
        {
            "trade_time": ["2024/1/9 09:30:00", "2024-01-11 09:30:00"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "vol": [100, 200],
            "amount": [1000, 2000],
        }
    )

    rows = append_group(
        out,
        group,
        time_col="trade_time",
        cols=["trade_time", "open", "high", "low", "close", "vol", "amount"],
        expect_time=True,
        dry_run=False,
    )
    assert rows == 1

    out_df = pd.read_csv(out)
    assert len(out_df) == 2
    assert out_df.iloc[-1]["trade_time"] == "2024-01-11 09:30:00"


def test_append_group_uses_datetime_comparison_for_adj_dates(tmp_path: Path) -> None:
    out = tmp_path / "000001.SZ.adj.csv"
    out.write_text("code,date,adj_factor\n000001.SZ,20240110,1.1\n", encoding="utf-8")

    group = pd.DataFrame(
        {
            "code": ["000001.SZ", "000001.SZ"],
            "date": ["2024-1-9", "2024/01/11"],
            "adj_factor": [1.0, 1.2],
        }
    )

    rows = append_group(
        out,
        group,
        time_col="date",
        cols=["code", "date", "adj_factor"],
        expect_time=False,
        dry_run=False,
    )
    assert rows == 1

    out_df = pd.read_csv(out, dtype={"date": "string"})
    assert len(out_df) == 2
    assert out_df.iloc[-1]["date"] == "2024/01/11"
