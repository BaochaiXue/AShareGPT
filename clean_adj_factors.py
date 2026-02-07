#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from model_core.data.io import DEFAULT_ENCODINGS, atomic_write_csv, read_csv_any_encoding


@dataclass
class FileStats:
    rows_in: int = 0
    rows_out: int = 0
    duplicate_dates: int = 0
    order_issues: int = 0
    invalid_rows: int = 0
    code_mismatch: int = 0
    changed: bool = False


def normalize_date(value: str) -> str:
    text = value.strip()
    if not text:
        return ""

    fmts = (
        "%Y%m%d",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(text, fmt).strftime("%Y%m%d")
        except ValueError:
            continue

    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.strptime(digits[:8], "%Y%m%d").strftime("%Y%m%d")
        except ValueError:
            pass
    raise ValueError(f"Cannot parse date: {text!r}")


def is_header(row: List[str]) -> bool:
    if len(row) < 2:
        return False
    c0 = row[0].strip().lower()
    c1 = row[1].strip().lower()
    return c0 in {"code"} and c1 in {"date"}


def read_csv_rows(path: Path) -> Iterable[List[str]]:
    df = read_csv_any_encoding(
        path,
        header=None,
        dtype=str,
        keep_default_na=False,
        encodings=DEFAULT_ENCODINGS,
    )
    if df.empty:
        return []
    return df.fillna("").values.tolist()


def write_csv_rows(path: Path, rows: Iterable[Tuple[str, str, str]]) -> None:
    out = pd.DataFrame(rows, columns=["code", "date", "adj_factor"])
    atomic_write_csv(path, out, index=False)


def process_file(path: Path, dry_run: bool) -> FileStats:
    stats = FileStats()
    file_code = path.stem.strip()

    rows = read_csv_rows(path)
    if not rows:
        return stats

    data: Dict[str, Tuple[str, str, str]] = {}
    start_idx = 1 if is_header(rows[0]) else 0
    prev_date = None

    for row in rows[start_idx:]:
        if not row:
            continue
        stats.rows_in += 1
        if len(row) < 3:
            stats.invalid_rows += 1
            continue
        code = row[0].strip()
        date_raw = row[1].strip()
        factor = row[2].strip()
        if not date_raw or not factor:
            stats.invalid_rows += 1
            continue

        try:
            date = normalize_date(date_raw)
        except ValueError:
            stats.invalid_rows += 1
            continue
        if prev_date is not None and date < prev_date:
            stats.order_issues += 1
        prev_date = date

        if date in data:
            stats.duplicate_dates += 1
        if code and file_code and code != file_code:
            stats.code_mismatch += 1

        out_code = file_code if file_code else code
        data[date] = (out_code, date, factor)

    if not data:
        return stats

    sorted_dates = sorted(data.keys())
    rows_out = [data[date] for date in sorted_dates]
    stats.rows_out = len(rows_out)

    stats.changed = (
        stats.duplicate_dates > 0
        or stats.order_issues > 0
        or stats.invalid_rows > 0
        or stats.code_mismatch > 0
        or stats.rows_out != stats.rows_in
    )

    if not dry_run and stats.changed:
        write_csv_rows(path, rows_out)

    return stats


def iter_csv_files(root: Path, max_files: int) -> Iterable[Path]:
    count = 0
    for path in sorted(root.glob("*.csv")):
        yield path
        count += 1
        if max_files and count >= max_files:
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean adj_factor CSVs: dedupe by date and sort ascending."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "复权因子",
        help="Directory containing per-code adj_factor CSV files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only, do not modify files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write per-file cleanup stats to CSV.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of files to process.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N files.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    totals = FileStats()
    files = 0
    changed_files = 0
    report_writer = None
    report_handle = None

    if args.report:
        report_handle = args.report.open("w", encoding="utf-8", newline="")
        report_writer = csv.writer(report_handle, lineterminator="\n")
        report_writer.writerow(
            [
                "file",
                "rows_in",
                "rows_out",
                "rows_fixed",
                "duplicate_dates",
                "order_issues",
                "invalid_rows",
                "code_mismatch",
                "changed",
            ]
        )

    try:
        for path in iter_csv_files(root, args.max_files):
            files += 1
            stats = process_file(path, args.dry_run)
            totals.rows_in += stats.rows_in
            totals.rows_out += stats.rows_out
            totals.duplicate_dates += stats.duplicate_dates
            totals.order_issues += stats.order_issues
            totals.invalid_rows += stats.invalid_rows
            totals.code_mismatch += stats.code_mismatch
            if stats.changed:
                changed_files += 1

            if report_writer:
                report_writer.writerow(
                    [
                        str(path),
                        stats.rows_in,
                        stats.rows_out,
                        stats.rows_in - stats.rows_out,
                        stats.duplicate_dates,
                        stats.order_issues,
                        stats.invalid_rows,
                        stats.code_mismatch,
                        int(stats.changed),
                    ]
                )

            if args.progress_every and files % args.progress_every == 0:
                print(f"processed {files} files...", flush=True)
                if report_handle:
                    report_handle.flush()
    finally:
        if report_handle:
            report_handle.close()

    print("done")
    print(f"files: {files}")
    print(f"changed_files: {changed_files}")
    print(f"rows_in: {totals.rows_in}")
    print(f"rows_out: {totals.rows_out}")
    print(f"duplicate_dates: {totals.duplicate_dates}")
    print(f"order_issues: {totals.order_issues}")
    print(f"invalid_rows: {totals.invalid_rows}")
    print(f"code_mismatch: {totals.code_mismatch}")
    if args.dry_run:
        print("dry_run: no files modified")


if __name__ == "__main__":
    main()
