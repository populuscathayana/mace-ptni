#!/usr/bin/env python
"""Filter collected OUTCAR files by POTCAR label.

Default use case for this project:
delete OUTCAR files that used exact Pt_pv but did not use Pt_pv_GW.

The script is dry-run by default. Add --apply to delete matching files.
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path


def iter_files(root: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(path.resolve() for path in iterator if path.is_file())


def compile_label_re(label: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])")


def label_present(text: str, label_re: re.Pattern) -> bool:
    return bool(label_re.search(text))


def scan_outcar(
    path: Path,
    forbidden_re: re.Pattern,
    required_re: re.Pattern,
    max_context: int,
    max_lines: int,
) -> dict:
    potcar_lines = []
    title_lines = []
    fallback_has_forbidden = False
    fallback_has_required = False

    lines_scanned = 0
    stopped_at_limit = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_lines > 0 and line_number > max_lines:
                stopped_at_limit = True
                break
            lines_scanned = line_number
            is_potcar_line = "POTCAR:" in line
            is_title_line = "TITEL" in line and "PAW" in line
            if is_potcar_line:
                potcar_lines.append(line.strip())
            if is_title_line:
                title_lines.append(line.strip())

            # Fallback is useful for unusual OUTCAR variants, but final decision
            # still prefers explicit POTCAR/TITEL evidence when available.
            if not fallback_has_forbidden and label_present(line, forbidden_re):
                fallback_has_forbidden = True
            if not fallback_has_required and label_present(line, required_re):
                fallback_has_required = True

    evidence_lines = potcar_lines + title_lines
    evidence_text = "\n".join(evidence_lines)
    if evidence_lines:
        has_forbidden = label_present(evidence_text, forbidden_re)
        has_required = label_present(evidence_text, required_re)
        evidence_source = "potcar_or_title_lines"
    else:
        has_forbidden = fallback_has_forbidden
        has_required = fallback_has_required
        evidence_source = "full_text_fallback"

    if has_required:
        decision = "keep_required_present"
    elif has_forbidden:
        decision = "delete_forbidden_without_required"
    else:
        decision = "keep_no_forbidden_label"

    context = evidence_lines[:max_context]
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "decision": decision,
        "has_forbidden_label": has_forbidden,
        "has_required_label": has_required,
        "evidence_source": evidence_source,
        "lines_scanned": lines_scanned,
        "stopped_at_line_limit": stopped_at_limit,
        "potcar_line_count": len(potcar_lines),
        "title_line_count": len(title_lines),
        "evidence_preview": " || ".join(context),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete collected OUTCAR files that used one POTCAR label but not another."
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
        "--forbidden-label",
        default="Pt_pv",
        help="Exact POTCAR label that is not allowed without the required label. Default: Pt_pv.",
    )
    parser.add_argument(
        "--required-label",
        default="Pt_pv_GW",
        help="Exact POTCAR label that protects a file from deletion. Default: Pt_pv_GW.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for CSV reports. Default: <root>/potcar_filter_reports/<timestamp>.",
    )
    parser.add_argument(
        "--max-context-lines",
        type=int,
        default=8,
        help="Number of POTCAR/TITEL evidence lines to keep in the manifest preview. Default: 8.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=100,
        help="Only scan the first N lines of each OUTCAR. Use 0 to scan full files. Default: 100.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matching OUTCAR files. Without this flag, only reports are written.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")
    if args.max_context_lines < 0:
        raise SystemExit("--max-context-lines must be >= 0")
    if args.max_lines < 0:
        raise SystemExit("--max-lines must be >= 0")
    if args.forbidden_label == args.required_label:
        raise SystemExit("--forbidden-label and --required-label must be different")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir is not None
        else root / "potcar_filter_reports" / timestamp
    )

    forbidden_re = compile_label_re(args.forbidden_label)
    required_re = compile_label_re(args.required_label)
    files = iter_files(root, args.pattern, args.recursive)
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} found in {root}")

    rows = []
    delete_targets = []
    for index, path in enumerate(files, start=1):
        row = scan_outcar(path, forbidden_re, required_re, args.max_context_lines, args.max_lines)
        row["index"] = index
        row["total"] = len(files)
        rows.append(row)
        if row["decision"] == "delete_forbidden_without_required":
            delete_targets.append(path)
        if index % 100 == 0 or index == len(files):
            print(f"Scanned {index}/{len(files)} OUTCAR files")

    fieldnames = [
        "index",
        "total",
        "path",
        "size_bytes",
        "decision",
        "has_forbidden_label",
        "has_required_label",
        "evidence_source",
        "lines_scanned",
        "stopped_at_line_limit",
        "potcar_line_count",
        "title_line_count",
        "evidence_preview",
    ]
    write_csv(report_dir / "potcar_filter_manifest.csv", rows, fieldnames)

    delete_rows = [row for row in rows if row["decision"] == "delete_forbidden_without_required"]
    write_csv(report_dir / "outcars_to_delete.csv", delete_rows, fieldnames)

    deleted_rows = []
    total_bytes = sum(path.stat().st_size for path in delete_targets)
    if args.apply:
        for path in delete_targets:
            size = path.stat().st_size
            path.unlink()
            deleted_rows.append({"deleted_path": str(path), "deleted_size_bytes": size})
        write_csv(
            report_dir / "deleted_files.csv",
            deleted_rows,
            ["deleted_path", "deleted_size_bytes"],
        )

    kept_required = sum(1 for row in rows if row["decision"] == "keep_required_present")
    kept_no_forbidden = sum(1 for row in rows if row["decision"] == "keep_no_forbidden_label")
    print(f"Candidate files: {len(files)}")
    print(f"Forbidden label: {args.forbidden_label}")
    print(f"Required label: {args.required_label}")
    print(f"Lines scanned per file: {'full file' if args.max_lines == 0 else args.max_lines}")
    print(f"Keep with required label: {kept_required}")
    print(f"Keep without forbidden label: {kept_no_forbidden}")
    print(f"Files marked for deletion: {len(delete_targets)}")
    print(f"Bytes marked for deletion: {total_bytes}")
    print(f"Report directory: {report_dir}")
    if args.apply:
        print(f"Deleted files: {len(deleted_rows)}")
    else:
        print("Dry run only. Inspect outcars_to_delete.csv, then add --apply to delete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
