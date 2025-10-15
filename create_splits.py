#!/usr/bin/env python3
"""
Make Waterbirds split replicas by reading metadata.csv and
copying/symlinking/linking images into split directories with class subfolders.

Default locations (change with CLI flags):
  --src  /dsi/scratch/users/eisenbr2/waterbird_complete95_forest2water2/
  --meta /dsi/scratch/users/eisenbr2/waterbird_complete95_forest2water2/metadata.csv
  --dest /dsi/dsai-lab/Ran/cbm/waterbirds/splits/

Layout created:
  {dest}/train/0/...  (landbird)
  {dest}/train/1/...  (waterbird)
  {dest}/val/0/...
  {dest}/val/1/...
  {dest}/test/0/...
  {dest}/test/1/...

Each split dir will also contain a filtered metadata.csv for that split.

Usage examples:
  python make_waterbirds_splits.py
  python make_waterbirds_splits.py --mode symlink
  python make_waterbirds_splits.py --mode link --overwrite
  python make_waterbirds_splits.py --src <SRC_ROOT> --meta <CSV> --dest <DEST_ROOT>
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SPLIT_MAP = {0: "train", 1: "val", 2: "test"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create per-split replicas of the Waterbirds dataset.")
    p.add_argument("--src", type=Path, default=Path("/dsi/scratch/users/eisenbr2/waterbird_complete95_forest2water2/"),
                   help="Source dataset root directory.")
    p.add_argument("--meta", type=Path, default=Path("/dsi/scratch/users/eisenbr2/waterbird_complete95_forest2water2/metadata.csv"),
                   help="Path to metadata.csv.")
    p.add_argument("--dest", type=Path, default=Path("/dsi/dsai-lab/Ran/cbm/waterbirds/splits/"),
                   help="Destination root for split replicas.")
    p.add_argument("--mode", choices=["copy", "symlink", "link"], default="copy",
                   help="How to place files in the split replicas: copy, symlink, or hardlink (link).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files if they already exist.")
    return p.parse_args()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def normalized_rel_path(rel: str) -> str:
    # Remove any leading slashes to keep it relative
    return rel.lstrip("/")

def resolve_source_file(src_root: Path, y: int, rel: str) -> Path:
    """
    Try two common layouts:
      1) <src_root>/<y>/<rel>
      2) <src_root>/<rel>
    Returns the first existing path, or the most likely path #1 if none exist.
    """
    rel = normalized_rel_path(rel)
    p1 = src_root / str(y) / rel
    if p1.exists():
        return p1
    p2 = src_root / rel
    if p2.exists():
        return p2
    # Fall back to p1 as the expected layout, even if missing (caller will log)
    return p1

def place_file(src: Path, dst: Path, mode: str, overwrite: bool) -> Tuple[bool, str]:
    """
    Place file from src -> dst using the selected mode.
    Returns (success, message).
    """
    try:
        if dst.exists():
            if overwrite:
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                else:
                    return False, f"Destination exists and is not a file: {dst}"
            else:
                return True, "exists, skipped"

        ensure_dir(dst.parent)

        if mode == "copy":
            import shutil
            shutil.copy2(src, dst)
        elif mode == "symlink":
            os.symlink(src, dst)
        elif mode == "link":
            os.link(src, dst)
        else:
            return False, f"Unknown mode: {mode}"
        return True, "ok"
    except FileNotFoundError:
        return False, f"missing source: {src}"
    except PermissionError as e:
        return False, f"permission error: {e}"
    except OSError as e:
        return False, f"os error: {e}"

def write_split_metadata(dest_split_dir: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    out_csv = dest_split_dir / "metadata.csv"
    ensure_dir(dest_split_dir)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main() -> None:
    args = parse_args()

    if not args.meta.exists():
        print(f"ERROR: metadata.csv not found at {args.meta}", file=sys.stderr)
        sys.exit(1)

    # Read metadata
    with args.meta.open("r", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        required = {"img_filename", "y", "split"}
        missing = required - set(header)
        if missing:
            print(f"ERROR: metadata.csv is missing required columns: {sorted(missing)}", file=sys.stderr)
            sys.exit(1)

        rows = list(reader)

    # Prepare groupings by split
    split_groups: Dict[int, List[Dict[str, str]]] = {0: [], 1: [], 2: []}
    for r in rows:
        try:
            split_code = int(r["split"])
        except ValueError:
            print(f"WARNING: bad split value {r['split']!r} for row with img {r.get('img_filename')}", file=sys.stderr)
            continue
        if split_code not in split_groups:
            print(f"WARNING: unknown split code {split_code} (expected 0/1/2). Skipping row.", file=sys.stderr)
            continue
        split_groups[split_code].append(r)

    # Stats and logs
    total_ok = 0
    total_missing = 0
    missing_log: List[str] = []

    for split_code, split_rows in split_groups.items():
        split_name = SPLIT_MAP[split_code]
        dest_split_dir = args.dest / split_name
        # Write filtered metadata for this split
        write_split_metadata(dest_split_dir, header, split_rows)

        # Place files into class subfolders under this split
        for r in split_rows:
            try:
                y = int(r["y"])
            except ValueError:
                print(f"WARNING: bad class value {r['y']!r} for img {r.get('img_filename')}", file=sys.stderr)
                continue

            rel = r["img_filename"]
            src_file = resolve_source_file(args.src, y, rel)
            rel_norm = normalized_rel_path(rel)

            # Keep the same relative structure under class dir
            dst_file = dest_split_dir / str(y) / rel_norm

            ok, msg = place_file(src_file, dst_file, args.mode, args.overwrite)
            if ok and msg == "ok":
                total_ok += 1
            elif ok and msg.startswith("exists"):
                # don't count as error
                pass
            else:
                total_missing += 1
                missing_log.append(f"{split_name},{y},{rel_norm} -> {msg}")

        # Write a per-split missing log if any
        if missing_log:
            with (dest_split_dir / "missing_files.log").open("w") as f:
                f.write("\n".join(missing_log))

    # Print a brief summary
    print("Done.")
    print(f"Placed OK: {total_ok}")
    print(f"Missing/Errors: {total_missing}")
    print("Dest root:", args.dest)
    print("Mode:", args.mode, "Overwrite:", args.overwrite)

if __name__ == "__main__":
    main()
