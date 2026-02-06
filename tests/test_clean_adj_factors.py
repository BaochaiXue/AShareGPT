from __future__ import annotations

import csv
from pathlib import Path

from clean_adj_factors import normalize_date, process_file


def test_normalize_date_handles_common_formats() -> None:
    assert normalize_date("2024-01-02") == "20240102"
    assert normalize_date("2024/01/02 00:00:00") == "20240102"
    assert normalize_date("20240102") == "20240102"


def test_process_file_dedupes_and_sorts_by_normalized_date(tmp_path: Path) -> None:
    path = tmp_path / "000001.SZ.csv"
    path.write_text(
        "code,date,adj_factor\n"
        "000001.SZ,2024-01-03,1.30\n"
        "000001.SZ,20240102,1.20\n"
        "000001.SZ,2024/01/03,1.31\n",
        encoding="utf-8",
    )

    stats = process_file(path, dry_run=False)

    assert stats.changed is True
    assert stats.duplicate_dates == 1
    assert stats.rows_out == 2

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["code", "date", "adj_factor"]
    assert rows[1] == ["000001.SZ", "20240102", "1.20"]
    assert rows[2] == ["000001.SZ", "20240103", "1.31"]
