#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def read_last_line(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return ""
        block = 4096
        offset = min(size, block)
        while True:
            f.seek(size - offset)
            chunk = f.read(offset)
            if b"\n" in chunk or size == offset:
                lines = chunk.splitlines()
                return lines[-1].decode("utf-8", errors="ignore") if lines else ""
            offset = min(size, offset * 2)


def get_last_token(path: Path, expect_time: bool, time_col: str) -> Optional[str]:
    if not path.exists():
        return None
    line = read_last_line(path)
    if not line:
        return None
    col_idx = 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        header = next(csv.reader(handle), None)
        if header and time_col in header:
            col_idx = header.index(time_col)
    row = next(csv.reader([line]), [])
    if col_idx >= len(row):
        return None
    token = row[col_idx].strip()
    if expect_time:
        if len(token) >= 10 and token[:4].isdigit():
            return token
        return None
    if token.isdigit() and len(token) >= 8:
        return token
    return None


def read_csv_fallback(path: Path, *, dtype: dict, usecols) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, dtype=dtype, usecols=usecols, encoding=enc)
        except Exception as exc:
            last_err = exc
    raise last_err if last_err else RuntimeError(f"read_csv failed for {path}")


def append_group(
    output_path: Path,
    group: pd.DataFrame,
    time_col: str,
    cols: list[str],
    expect_time: bool,
    dry_run: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    group = group.loc[:, cols].copy()
    group = group.dropna(subset=[time_col])
    group = group.drop_duplicates(subset=[time_col], keep="last")

    last_token = get_last_token(output_path, expect_time=expect_time, time_col=time_col)
    if last_token:
        group = group[group[time_col] > last_token]

    if group.empty:
        return 0

    group = group.sort_values(time_col)
    header = not output_path.exists()

    if not dry_run:
        group.to_csv(output_path, mode="a", header=header, index=False)

    return len(group)


def iter_csv_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.csv"))


def process_minute_data(minute_root: Path, data_root: Path, dry_run: bool, max_files: int) -> None:
    files = list(iter_csv_files(minute_root))
    if max_files:
        files = files[:max_files]

    total_files = 0
    total_rows = 0

    for path in files:
        total_files += 1
        rel = path.relative_to(minute_root)
        year = rel.parts[0] if rel.parts else path.stem.split("-")[0]

        df = read_csv_fallback(
            path,
            dtype={"证券代码": "string", "code": "string", "trade_time": "string"},
            usecols=lambda c: c in {"证券代码", "code", "trade_time", "open", "high", "low", "close", "vol", "amount"},
        )
        if "code" not in df.columns and "证券代码" not in df.columns:
            print(f"[minute] missing code column: {path}")
            continue
        if "code" not in df.columns:
            df = df.rename(columns={"证券代码": "code"})

        for code, group in df.groupby("code", sort=False):
            output_path = data_root / year / f"{code}.csv"
            rows = append_group(
                output_path,
                group,
                time_col="trade_time",
                cols=["trade_time", "open", "high", "low", "close", "vol", "amount"],
                expect_time=True,
                dry_run=dry_run,
            )
            total_rows += rows

        print(f"[minute] {path} -> year {year}")

    print(f"[minute] files={total_files} rows_appended={total_rows}")


def process_adj_factors(adj_root: Path, data_root: Path, dry_run: bool, max_files: int) -> None:
    files = list(iter_csv_files(adj_root))
    if max_files:
        files = files[:max_files]

    total_files = 0
    total_rows = 0

    for path in files:
        total_files += 1

        df = read_csv_fallback(
            path,
            dtype={"证券代码": "string", "code": "string", "date": "string"},
            usecols=lambda c: c in {"证券代码", "code", "date", "adj_factor"},
        )
        if "code" not in df.columns and "证券代码" not in df.columns:
            print(f"[adj] missing code column: {path}")
            continue
        if "code" not in df.columns:
            df = df.rename(columns={"证券代码": "code"})
        else:
            if "证券代码" in df.columns:
                df["code"] = df["code"].fillna(df["证券代码"])

        for code, group in df.groupby("code", sort=False):
            output_path = data_root / "复权因子" / f"{code}.csv"
            rows = append_group(
                output_path,
                group,
                time_col="date",
                cols=["code", "date", "adj_factor"],
                expect_time=False,
                dry_run=dry_run,
            )
            total_rows += rows

        print(f"[adj] {path}")

    print(f"[adj] files={total_files} rows_appended={total_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify raw downloads into standard per-code files.")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--raw-root", default=str(Path(__file__).resolve().parent / "data" / "raw_downloads"))
    parser.add_argument("--mode", choices=["all", "minute", "adj"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    raw_root = Path(args.raw_root)

    minute_root = raw_root / "a股1分钟"
    adj_root = raw_root / "复权因子"

    if args.mode in ("all", "minute"):
        process_minute_data(minute_root, data_root, args.dry_run, args.max_files)

    if args.mode in ("all", "adj"):
        process_adj_factors(adj_root, data_root, args.dry_run, args.max_files)


if __name__ == "__main__":
    main()
