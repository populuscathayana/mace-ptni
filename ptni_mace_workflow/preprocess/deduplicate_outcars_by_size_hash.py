#!/usr/bin/env python
"""Remove byte-identical duplicate OUTCAR files.

Workflow:
1. Group candidate files by size.
2. Hash only files in same-size groups.
3. Keep the first sorted path in each identical (size, sha256) group.
4. Report and optionally delete all remaining duplicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def iter_files(root: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(path.resolve() for path in iterator if path.is_file())


def sha256_file(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deduplicate OUTCAR files by size first, then SHA-256 hash."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("OUTCARs"),
        help="Directory containing collected OUTCAR files. Default: ./OUTCARs.",
    )
    parser.add_argument(
        "--pattern",
        default="OUTCAR*",
        help="Candidate filename pattern. Default: OUTCAR*.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search root recursively instead of only the top level.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for CSV reports. Default: <root>/dedupe_reports/<timestamp>.",
    )
    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=32,
        help="Read size for hashing in MB. Default: 32.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete duplicates. Without this flag, only reports are written.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")
    if args.chunk_mb < 1:
        raise SystemExit("--chunk-mb must be >= 1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir is not None
        else root / "dedupe_reports" / timestamp
    )
    chunk_size = args.chunk_mb * 1024 * 1024

    files = iter_files(root, args.pattern, args.recursive)
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} found in {root}")

    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in files:
        by_size[path.stat().st_size].append(path)

    size_rows = []
    same_size_files = []
    for size, paths in sorted(by_size.items()):
        size_rows.append(
            {
                "size_bytes": size,
                "file_count": len(paths),
                "needs_hash": len(paths) > 1,
            }
        )
        if len(paths) > 1:
            same_size_files.extend(paths)

    write_csv(
        report_dir / "size_groups.csv",
        size_rows,
        ["size_bytes", "file_count", "needs_hash"],
    )

    hash_rows = []
    by_size_hash: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for index, path in enumerate(same_size_files, start=1):
        size = path.stat().st_size
        digest = sha256_file(path, chunk_size)
        by_size_hash[(size, digest)].append(path)
        hash_rows.append(
            {
                "index": index,
                "total_to_hash": len(same_size_files),
                "size_bytes": size,
                "sha256": digest,
                "path": str(path),
            }
        )
        if index % 50 == 0 or index == len(same_size_files):
            print(f"Hashed {index}/{len(same_size_files)} same-size files")

    write_csv(
        report_dir / "hashed_same_size_files.csv",
        hash_rows,
        ["index", "total_to_hash", "size_bytes", "sha256", "path"],
    )

    duplicate_rows = []
    delete_targets = []
    for (size, digest), paths in sorted(by_size_hash.items(), key=lambda item: (item[0][0], item[0][1])):
        if len(paths) <= 1:
            continue
        keep = sorted(paths)[0]
        for duplicate in sorted(paths)[1:]:
            delete_targets.append(duplicate)
            duplicate_rows.append(
                {
                    "size_bytes": size,
                    "sha256": digest,
                    "keep_path": str(keep),
                    "duplicate_path": str(duplicate),
                    "duplicate_size_bytes": duplicate.stat().st_size,
                }
            )

    write_csv(
        report_dir / "duplicates_to_delete.csv",
        duplicate_rows,
        ["size_bytes", "sha256", "keep_path", "duplicate_path", "duplicate_size_bytes"],
    )

    deleted_rows = []
    if args.delete:
        for path in delete_targets:
            size = path.stat().st_size
            path.unlink()
            deleted_rows.append({"deleted_path": str(path), "deleted_size_bytes": size})
        write_csv(
            report_dir / "deleted_files.csv",
            deleted_rows,
            ["deleted_path", "deleted_size_bytes"],
        )

    duplicate_bytes = sum(row["duplicate_size_bytes"] for row in duplicate_rows)
    print(f"Candidate files: {len(files)}")
    print(f"Size groups: {len(by_size)}")
    print(f"Same-size files hashed: {len(same_size_files)}")
    print(f"Duplicate files found: {len(delete_targets)}")
    print(f"Duplicate bytes: {duplicate_bytes}")
    print(f"Report directory: {report_dir}")
    if args.delete:
        print(f"Deleted files: {len(deleted_rows)}")
    else:
        print("Dry run only. Add --delete to remove duplicates.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
