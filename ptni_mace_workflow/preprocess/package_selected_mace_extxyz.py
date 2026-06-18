#!/usr/bin/env python
"""Combine selected composition folders into one MACE extxyz and package it.

MACE trains directly on extxyz. The tar.gz archive created here is only for
download/transfer convenience; unpack it before training, or train on the
combined extxyz directly if running on the server.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import tarfile
from datetime import datetime
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_extxyz(root: Path, groups: list[str]) -> list[tuple[str, Path]]:
    files = []
    for group in groups:
        group_dir = root / group
        if not group_dir.is_dir():
            print(f"[warn] missing group directory: {group_dir}")
            continue
        for path in sorted(group_dir.glob("*.extxyz")):
            if path.is_file():
                files.append((group, path.resolve()))
    return files


def concatenate(files: list[tuple[str, Path]], output: Path) -> list[dict]:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("wb") as write_handle:
        for index, (group, path) in enumerate(files, start=1):
            size = path.stat().st_size
            with path.open("rb") as read_handle:
                while True:
                    chunk = read_handle.read(32 * 1024 * 1024)
                    if not chunk:
                        break
                    write_handle.write(chunk)
            rows.append(
                {
                    "index": index,
                    "group": group,
                    "source_path": str(path),
                    "source_name": path.name,
                    "source_size_bytes": size,
                }
            )
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "group",
                "source_path",
                "source_name",
                "source_size_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_checksum(path: Path, entries: list[Path]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f"{sha256_file(entry)}  {entry.name}\n")


def make_archive(archive: Path, files: list[Path]) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a combined MACE extxyz and optional tar.gz for selected groups."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("MACE_extxyz"),
        help="Root containing composition folders. Default: ./MACE_extxyz.",
    )
    parser.add_argument(
        "--groups",
        default="PtNi,Pt,Ni",
        help="Comma-separated group folder names. Default: PtNi,Pt,Ni.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("MACE_selected_package"),
        help="Directory for combined extxyz and reports.",
    )
    parser.add_argument(
        "--name",
        default="ptni_pt_ni",
        help="Base output name. Default: ptni_pt_ni.",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Only create combined extxyz and reports, not tar.gz.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    if not groups:
        raise SystemExit("--groups cannot be empty")
    if not root.is_dir():
        raise SystemExit(f"Root does not exist: {root}")

    files = find_extxyz(root, groups)
    if not files:
        raise SystemExit(f"No extxyz files found for groups: {groups}")

    combined = output_dir / f"{args.name}_all.extxyz"
    manifest = output_dir / f"{args.name}_manifest.csv"
    checksum = output_dir / f"{args.name}_sha256.txt"
    archive = output_dir / f"{args.name}_mace_extxyz.tar.gz"

    rows = concatenate(files, combined)
    write_manifest(manifest, rows)

    checksum_targets = [combined, manifest]
    if not args.no_archive:
        make_archive(archive, [combined, manifest])
        checksum_targets.append(archive)
    write_checksum(checksum, checksum_targets)

    total_input_bytes = sum(path.stat().st_size for _, path in files)
    print(f"Groups: {groups}")
    print(f"Files combined: {len(files)}")
    print(f"Input bytes: {total_input_bytes}")
    print(f"Combined extxyz: {combined}")
    print(f"Manifest: {manifest}")
    print(f"SHA256: {checksum}")
    if not args.no_archive:
        print(f"Archive: {archive}")
    print(f"Finished at: {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
