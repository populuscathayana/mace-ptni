#!/usr/bin/env python
"""Move extxyz files into subfolders by element composition.

Examples:
  Pt-only structures   -> MACE_extxyz/Pt/
  Pt-Ni structures     -> MACE_extxyz/PtNi/
  Pt-Ni-O structures   -> MACE_extxyz/PtNiO/

By default the script performs a dry run. Add --apply to move files.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_ELEMENT_PRIORITY = [
    "Pt",
    "Ni",
    "H",
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
]


def parse_priority(text: str | None) -> list[str]:
    if not text:
        return DEFAULT_ELEMENT_PRIORITY
    return [item.strip() for item in text.split(",") if item.strip()]


def composition_label(symbols: list[str], priority: list[str], order: str) -> str:
    unique = set(symbols)
    if order == "appearance":
        ordered = []
        for symbol in symbols:
            if symbol not in ordered:
                ordered.append(symbol)
    elif order == "alphabetical":
        ordered = sorted(unique)
    elif order == "priority":
        ordered = [symbol for symbol in priority if symbol in unique]
        ordered.extend(sorted(unique.difference(ordered)))
    else:
        raise ValueError(f"unknown order: {order}")
    return "".join(ordered)


def read_first_frame_symbols(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().strip()
        if not first:
            raise ValueError("empty file")
        try:
            natoms = int(first)
        except ValueError as exc:
            raise ValueError(f"first line is not atom count: {first[:80]!r}") from exc

        comment = handle.readline().strip()
        atom_lines = [handle.readline().strip() for _ in range(natoms)]

    if len(atom_lines) != natoms or any(not line for line in atom_lines):
        raise ValueError(f"incomplete first frame: expected {natoms} atom lines")

    species_column = 0
    match = re.search(r'Properties=("[^"]+"|\S+)', comment)
    if match:
        properties = match.group(1).strip('"')
        tokens = properties.split(":")
        column = 0
        for i in range(0, len(tokens) - 2, 3):
            name = tokens[i]
            count = int(tokens[i + 2])
            if name in {"species", "symbols", "element"}:
                species_column = column
                break
            column += count

    symbols = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) <= species_column:
            raise ValueError(f"atom line has no species column: {line[:80]!r}")
        symbol = parts[species_column]
        if not re.fullmatch(r"[A-Z][a-z]?", symbol):
            raise ValueError(f"invalid element symbol {symbol!r} in line: {line[:80]!r}")
        symbols.append(symbol)

    return symbols


def unique_target_path(target: Path) -> Path:
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    index = 1
    while True:
        candidate = parent / f"{stem}__dup{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def is_inside(path: Path, possible_parent: Path) -> bool:
    try:
        path.resolve().relative_to(possible_parent.resolve())
        return True
    except ValueError:
        return False


def find_extxyz_files(input_dir: Path, recursive: bool, pattern: str) -> list[Path]:
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted(path.resolve() for path in iterator if path.is_file())


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "status",
        "source_path",
        "target_path",
        "composition_label",
        "formula_counts",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify extxyz files into folders by element composition."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("MACE_extxyz"),
        help="Directory containing extxyz files. Default: ./MACE_extxyz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination root. Default: same as --input-dir.",
    )
    parser.add_argument("--pattern", default="*.extxyz")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--order",
        choices=["priority", "appearance", "alphabetical"],
        default="priority",
        help="Element order used in folder names. Default: priority.",
    )
    parser.add_argument(
        "--priority",
        default=",".join(DEFAULT_ELEMENT_PRIORITY),
        help="Comma-separated priority order. Default favors PtNi catalyst data.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move/copy files. Without this flag, only a dry run is reported.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV report path. Default: <output-dir>/composition_classification_manifest.csv.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir
    manifest = (
        args.manifest.resolve()
        if args.manifest
        else output_dir / "composition_classification_manifest.csv"
    )
    priority = parse_priority(args.priority)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    files = find_extxyz_files(input_dir, args.recursive, args.pattern)
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} found in {input_dir}")

    rows = []
    label_counts = Counter()

    for source in files:
        try:
            if args.recursive and is_inside(source.parent, output_dir) and source.parent != input_dir:
                rows.append(
                    {
                        "status": "skipped_nested",
                        "source_path": str(source),
                        "target_path": "",
                        "composition_label": "",
                        "formula_counts": "",
                        "message": "already in a subdirectory",
                    }
                )
                continue

            symbols = read_first_frame_symbols(source)
            counts = Counter(symbols)
            label = composition_label(symbols, priority, args.order)
            target_dir = output_dir / label
            target = unique_target_path(target_dir / source.name)
            label_counts[label] += 1

            row = {
                "status": "planned",
                "source_path": str(source),
                "target_path": str(target),
                "composition_label": label,
                "formula_counts": ";".join(f"{k}:{counts[k]}" for k in sorted(counts)),
                "message": "",
            }

            if args.apply:
                target_dir.mkdir(parents=True, exist_ok=True)
                if args.copy:
                    shutil.copy2(source, target)
                    row["status"] = "copied"
                else:
                    shutil.move(source, target)
                    row["status"] = "moved"

            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "status": "failed",
                    "source_path": str(source),
                    "target_path": "",
                    "composition_label": "",
                    "formula_counts": "",
                    "message": repr(exc),
                }
            )

    write_manifest(manifest, rows)

    failed = sum(1 for row in rows if row["status"] == "failed")
    acted = sum(1 for row in rows if row["status"] in {"moved", "copied"})
    failure_counts = Counter(row["message"] for row in rows if row["status"] == "failed")

    print(f"Input files: {len(files)}")
    print(f"Composition groups: {dict(sorted(label_counts.items()))}")
    print(f"Failed: {failed}")
    if failure_counts:
        print("Top failure messages:")
        for message, count in failure_counts.most_common(5):
            print(f"  {count}: {message}")
    print(f"Manifest: {manifest}")
    if args.apply:
        action = "Copied" if args.copy else "Moved"
        print(f"{action}: {acted}")
    else:
        print("Dry run only. Add --apply to move files.")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
