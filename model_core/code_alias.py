from __future__ import annotations

import csv
from pathlib import Path


def load_code_alias_map(path: Path) -> dict[str, str]:
    """Load code alias mapping from CSV: old_code,new_code."""
    if not path.exists():
        return {}

    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        old_col = field_map.get("old_code")
        new_col = field_map.get("new_code")
        if not old_col or not new_col:
            return {}

        for row in reader:
            old_code = (row.get(old_col) or "").strip()
            new_code = (row.get(new_col) or "").strip()
            if not old_code or not new_code:
                continue
            if old_code == new_code:
                continue
            mapping[old_code] = new_code
    return mapping
