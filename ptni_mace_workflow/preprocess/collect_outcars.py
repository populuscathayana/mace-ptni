#!/usr/bin/env python
"""Collect all OUTCAR files under a root directory into one OUTCARs folder.

Output filenames are based on each source file's absolute path, with characters
that are invalid on Windows replaced by underscores. A CSV manifest is written
so every copied file can be traced back to its original absolute path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


INVALID_FILENAME_CHARS = '<>:"/\\|?*'


def safe_path_token(path: Path) -> str:
    token = str(path.resolve())
    for char in INVALID_FILENAME_CHARS:
        token = token.replace(char, "_")
    token = token.replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("._")


def make_output_name(path: Path, max_len: int = 220) -> str:
    token = safe_path_token(path)
    filename = f"OUTCAR_{token}.OUTCAR"
    if len(filename) <= max_len:
        return filename

    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    keep = max_len - len("OUTCAR_") - len(f"_{digest}.OUTCAR")
    return f"OUTCAR_{token[:keep]}_{digest}.OUTCAR"


def discover_outcars(root: Path, output_dir: Path) -> list[Path]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    outcars = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name != "OUTCAR":
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(output_dir)
            continue
        except ValueError:
            pass
        outcars.append(resolved)
    return sorted(outcars)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy every OUTCAR under the current tree into an OUTCARs folder."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Directory to search recursively. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("OUTCARs"),
        help="Directory for copied OUTCAR files. Default: ./OUTCARs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV manifest path. Default: <output-dir>/manifest.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    manifest = args.manifest.resolve() if args.manifest else output_dir / "manifest.csv"
    outcars = discover_outcars(root, output_dir)

    if not outcars:
        print(f"No OUTCAR files found under {root}")
        return 0

    rows = []
    used_names = set()
    for source in outcars:
        name = make_output_name(source)
        if name in used_names:
            digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
            name = f"{Path(name).stem}_{digest}.OUTCAR"
        used_names.add(name)

        target = output_dir / name
        rows.append(
            {
                "source_abs_path": str(source),
                "target_abs_path": str(target),
                "target_name": name,
                "source_size_bytes": source.stat().st_size,
            }
        )

    if args.dry_run:
        for row in rows:
            print(f"{row['source_abs_path']} -> {row['target_abs_path']}")
        print(f"Dry run: {len(rows)} OUTCAR files would be copied.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        shutil.copy2(row["source_abs_path"], row["target_abs_path"])

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Copied {len(rows)} OUTCAR files to {output_dir}")
    print(f"Wrote manifest to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
