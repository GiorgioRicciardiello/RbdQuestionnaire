#!/usr/bin/env python3
import shutil
import csv
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

SRC_DIR = Path("/sc/arion/projects/sleeplab/ActigraphyUKBB")
DST_DIR = Path("/sc/arion/projects/sleeplab/ActigraphyUKBB/Data/raw_actigraphy")
LOG_FILE = Path("/sc/arion/projects/sleeplab/ActigraphyUKBB/Data/copy_log.csv")

# Ensure destination exists
DST_DIR.mkdir(parents=True, exist_ok=True)

def load_completed_files():
    """Load already-copied files from the log (resume support)."""
    completed = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("copied", "").lower() == "yes":
                    completed.add(row["file_name"])
    return completed

def copy_file(src_file: Path):
    """Copy a single .cwa file and return log row."""
    dst_file = DST_DIR / src_file.name
    row = {
        "file_name": src_file.name,
        "source_folder": str(src_file.parent),
        "destination_folder": str(dst_file),
        "copied": "no",
        "error": "",
    }

    try:
        if dst_file.exists():
            row["error"] = "File already exists"
        else:
            shutil.copy2(src_file, dst_file)
            row["copied"] = "yes"
    except Exception as e:
        row["error"] = str(e)

    return row

def main():
    # Detect CPU count from LSF or fallback
    try:
        n_cpus = int(os.environ.get("LSB_DJOB_NUMPROC", os.cpu_count() or 4))
    except Exception:
        n_cpus = 8
    print(f"Using {n_cpus} workers")

    # Load completed files
    completed_files = load_completed_files()
    print(f"Found {len(completed_files)} files already copied in log.")

    # Write header if log doesn't exist
    if not LOG_FILE.exists():
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "file_name", "source_folder", "destination_folder", "copied", "error"
            ])
            writer.writeheader()

    # Gather new files
    cwa_files = [f for f in SRC_DIR.rglob("*.cwa") if f.name not in completed_files]
    print(f"New files to copy: {len(cwa_files)}")

    stats = Counter()

    # Copy in parallel
    with ThreadPoolExecutor(max_workers=n_cpus * 2) as executor:
        futures = [executor.submit(copy_file, f) for f in cwa_files]

        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "file_name", "source_folder", "destination_folder", "copied", "error"
            ])

            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                writer.writerow(row)

                if row["copied"] == "yes":
                    stats["copied"] += 1
                elif row["error"]:
                    stats["errors"] += 1
                else:
                    stats["skipped"] += 1

                if i % 1000 == 0:
                    print(f"{i}/{len(cwa_files)} processed")

    # Print summary
    print("\n===== Copy Summary =====")
    print(f"Copied: {stats['copied']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print("========================")

if __name__ == "__main__":
    main()
