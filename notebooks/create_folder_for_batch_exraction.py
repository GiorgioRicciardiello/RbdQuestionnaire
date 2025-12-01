"""
Move validated CWA files for second batch processing.

This script reads the CSV file (generated previously) that lists missing/validated
raw actigraphy files, and moves them into a new folder for second batch processing.
"""

import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm

# === CONFIG ===
# Path to the CSV created in the previous script (files_for_second_batch.csv)
output_file = Path("/sc/arion/projects/sleeplab/ActigraphyUKBB/results/files_for_second_batch.csv")

# Destination folder for the second batch
second_batch_dir = output_file.parent.joinpath("data_for_second_batch")
second_batch_dir.mkdir(parents=True, exist_ok=True)

# === SCRIPT ===
if not output_file.exists():
    raise FileNotFoundError(f"❌ Could not find {output_file}")

# Load the validated missing files
df_missing = pd.read_csv(output_file)

print(f"✅ Loaded {df_missing.shape[0]} files from {output_file}")

# Only keep valid files if column exists
if "valid_cwa" in df_missing.columns:
    df_missing = df_missing[df_missing["valid_cwa"] == True].reset_index(drop=True)

print(f"📂 Copying {df_missing.shape[0]} validated files to {second_batch_dir}")

# Copy files
for _, row in tqdm(df_missing.iterrows(), total=df_missing.shape[0], desc="Copying"):
    src = Path(row["file_path"])
    dst = second_batch_dir.joinpath(row["file_name"])
    try:
        if src.exists():
            if not dst.exists():
                shutil.copy2(src, dst)
        else:
            print(f"⚠️ Source file not found: {src}")
    except Exception as e:
        print(f"⚠️ Failed to copy {src}: {e}")

print(f"✅ All files processed. Files now in {second_batch_dir}")
