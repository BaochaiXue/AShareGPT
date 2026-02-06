#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model_core.code_alias import load_code_alias_map  # noqa: E402


ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "gb18030")


def read_adj_csv(path: Path) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(
                path,
                usecols=lambda c: c in {"code", "date", "adj_factor", "证券代码"},
                dtype={"date": "string"},
                encoding=enc,
            )
        except Exception as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    return pd.DataFrame()


def collect_minute_codes(data_root: Path) -> set[str]:
    codes: set[str] = set()
    for year_dir in sorted(data_root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for path in year_dir.glob("*.csv"):
            codes.add(path.stem)
    return codes


def normalize_adj(df: pd.DataFrame, old_code: str) -> pd.DataFrame:
    if df.empty:
        return df
    if "code" not in df.columns and "证券代码" in df.columns:
        df = df.rename(columns={"证券代码": "code"})
    for col in ("date", "adj_factor"):
        if col not in df.columns:
            return pd.DataFrame()

    out = df.loc[:, ["date", "adj_factor"]].copy()
    out["date"] = pd.to_numeric(out["date"], errors="coerce").astype("Int64").astype("string")
    out["adj_factor"] = pd.to_numeric(out["adj_factor"], errors="coerce")
    out = out.dropna(subset=["date", "adj_factor"])
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    if out.empty:
        return out
    out.insert(0, "code", old_code)
    return out


def write_adj(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing old-code adj files from mapped new-code adj files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Data root containing year folders and adj factor folder.",
    )
    parser.add_argument(
        "--alias-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "code_alias_map.csv",
        help="CSV with old_code,new_code columns.",
    )
    parser.add_argument(
        "--adj-dir",
        type=str,
        default="复权因子",
        help="Adj factor directory name under data root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing old-code adj files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only, do not write files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional CSV report output path.",
    )
    args = parser.parse_args()

    data_root = args.data_root
    adj_root = data_root / args.adj_dir
    alias_file = args.alias_file

    alias_map = load_code_alias_map(alias_file)
    if not alias_map:
        raise SystemExit(f"No valid alias mapping found: {alias_file}")

    minute_codes = collect_minute_codes(data_root)
    stats = {
        "created": 0,
        "overwritten": 0,
        "skipped_exists": 0,
        "skipped_no_minute": 0,
        "missing_source": 0,
        "invalid_source": 0,
    }

    report_handle = None
    report_writer = None
    if args.report is not None:
        report_handle = args.report.open("w", encoding="utf-8", newline="")
        report_writer = csv.writer(report_handle, lineterminator="\n")
        report_writer.writerow(["old_code", "new_code", "status", "rows"])

    try:
        for old_code, new_code in sorted(alias_map.items()):
            status = ""
            rows = 0
            dst = adj_root / f"{old_code}.csv"
            src = adj_root / f"{new_code}.csv"
            dst_exists_before = dst.exists()

            if old_code not in minute_codes:
                status = "skipped_no_minute"
                stats[status] += 1
            elif dst_exists_before and not args.overwrite:
                status = "skipped_exists"
                stats[status] += 1
            elif not src.exists():
                status = "missing_source"
                stats[status] += 1
            else:
                try:
                    df = read_adj_csv(src)
                    out = normalize_adj(df, old_code)
                except Exception:
                    out = pd.DataFrame()
                if out.empty:
                    status = "invalid_source"
                    stats[status] += 1
                else:
                    rows = int(len(out))
                    if not args.dry_run:
                        write_adj(dst, out)
                    status = "overwritten" if dst_exists_before else "created"
                    stats[status] += 1

            print(f"{old_code} <- {new_code}: {status} rows={rows}")
            if report_writer is not None:
                report_writer.writerow([old_code, new_code, status, rows])
    finally:
        if report_handle is not None:
            report_handle.close()

    print("done")
    for key in ("created", "overwritten", "skipped_exists", "skipped_no_minute", "missing_source", "invalid_source"):
        print(f"{key}: {stats[key]}")
    if args.dry_run:
        print("dry_run: no files modified")


if __name__ == "__main__":
    main()
