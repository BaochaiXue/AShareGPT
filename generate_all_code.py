#!/usr/bin/env python3
"""Generate a Markdown snapshot of all Python, YAML, and shell source files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_IGNORE_DIRS: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    ".idea",
    ".vscode",
    ".venv",
    "env",
    "envs",
    "venv",
    "build",
    "dist",
    "node_modules",
    "tests",
)

SUFFIX_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
}


def collect_files(root: Path, suffixes: Iterable[str], ignore_dirs: Sequence[str]) -> list[Path]:
    """Return all files under root that match suffixes while respecting ignore directories."""
    suffix_set: set[str] = {suffix.lower() for suffix in suffixes}
    ignore_set: set[str] = set(ignore_dirs)
    matches: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Trim directories in-place so os.walk does not descend into ignored folders.
        dirnames[:] = [directory for directory in dirnames if directory not in ignore_set]

        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in suffix_set:
                continue
            matches.append(path)

    matches.sort()
    return matches


def write_markdown(files: Iterable[Path], root: Path, output_path: Path) -> None:
    """Write a Markdown file containing each source file in fenced code blocks."""
    root = root.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for path in files:
            rel_path = path.resolve().relative_to(root)
            language = SUFFIX_TO_LANG.get(path.suffix.lower(), "")
            handle.write(f"{rel_path.as_posix()}\n")
            handle.write(f"```{language}\n")
            text = path.read_text(encoding="utf-8", errors="replace")
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.write("```\n\n")


def main() -> None:
    """Entry point for generating the Markdown snapshot."""
    parser = argparse.ArgumentParser(
        description="Generate a Markdown file containing all Python, YAML, and shell source code."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to scan (defaults to current working directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("all_code.md"),
        help="Path to the Markdown file to generate (defaults to ./all_code.md).",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=list(DEFAULT_IGNORE_DIRS),
        metavar="DIR",
        help="Directory names to skip anywhere under the root.",
    )
    args = parser.parse_args()

    files = collect_files(args.root, SUFFIX_TO_LANG.keys(), args.ignore)
    write_markdown(files, args.root, args.output)


if __name__ == "__main__":
    main()
