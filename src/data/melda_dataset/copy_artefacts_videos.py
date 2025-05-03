#!/usr/bin/env python3
"""
Copy all usable swallow-study videos that contain patient artefacts
into a single folder.

Usage:  python copy_usable_videos.py
"""

import pandas as pd
from pathlib import Path
import shutil
import sys
from tqdm import tqdm

# ---------------------------------------------------------------
# ---- user-configurable paths ----------------------------------
CSV_PATH  = Path("melda_swallow_annotations_final.csv")
BASE_DIR  = Path(r"\\fauad.fau.de\shares\ANKI\Projects\Swallowing\Data\from_Melda")
DEST_DIR  = Path(r"D:\Martin\thesis\data\upload_labelbox_dataset\Melda\artefacts")          # will be created if needed
# ---------------------------------------------------------------
def main() -> None:
    # ---------- load & filter annotations ----------------------
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        sys.exit(f"CSV not found: {CSV_PATH}")
    n = len(df)
    print(f"{n} total videos found in the CSV")
    usable = df[df["is_usable"] == 1.0].copy()
    n = len(usable)
    print(f"{n} is_usable videos found in the CSV")
    usable_artifacted = usable[usable["artifact_detected"] == 1.0].copy()

    n = len(usable_artifacted)
    print(f"{n} is_usable & artifact_detected videos found in the CSV")
    # ---------- prepare destination folder ---------------------
    DEST_DIR.mkdir(exist_ok=True)

    # ---------- copy videos ------------------------------------
    missing = []             # collect missing source files for a summary at the end
    copied  = 0

    for _, row in tqdm(usable.iterrows(), total=len(usable)):
        # patient subfolder name, e.g. "46"
        patient_folder = str(int(row["PatientID"]))

        # file name; ensure it ends with .mp4 even if the CSV missed it
        video_name = str(row["ParsedFullVideoID"])
        if not video_name.lower().endswith(".mp4"):
            video_name += ".mp4"

        src = BASE_DIR / patient_folder / video_name
        dst = DEST_DIR  / video_name

        try:
            shutil.copy2(src, dst)
            copied += 1
        except FileNotFoundError:
            missing.append(src)

    print(f"✔ Copied {copied} video(s) to \"{DEST_DIR}\"")

    # ---------- report missing files, if any -------------------
    if missing:
        print("\n⚠  The following files were not found:")
        for path in missing:
            print(f"   {path}")
if __name__ == "__main__":
    main()
