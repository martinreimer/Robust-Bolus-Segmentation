#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
last used cmd:
python preprocess_labelbox_export.py --original_dataset_dir "D:\Martin\thesis\data\raw\labelbox_output_mbss_mar
tin_0514" --video_notes_csv "D:\Martin\thesis\data\video_notes.csv" --output_dataset_dir "D:\Martin\thesis\data\processed\labelbox_output_mbss_martin_0514_processed" --project_source "MBSS_Martin" --verbose --empty_mask_threshold 5 --exclude_artefacts --exclude_first_frame

"""
r"""
Script to process annotated video exports:

1. Load `data_overview.csv` and `video_notes.csv`.
2. Filter `video_notes` by `project_source`, `not_use == 0`.
3. Exclude/include rows based on `Patient_Artefact` via CLI flags.
4. Report exclusions, duplicates, and coverage.
5. Inner-join with overview by `shared_video_id` and `project_source`.
6. Optionally exclude first frame of each video (--exclude_first_frame).
7. Process frames: skip before/after desired frames, copy images and masks, resize masks, exclude white or missing masks.
8. Accumulate and print per-video exclusion reasons.
9. Analyze empty (all-black) masks per video: count stats, determine files to remove.
10. Prune rows corresponding to removed empty-mask frames, print before/after counts.
11. Save final `data_overview.csv` with only existing frames.
"""
import os
import argparse
from typing import Dict, Optional, List, Tuple, Set
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm


def load_overview(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find data_overview.csv at {path}")
    return pd.read_csv(path)


def load_video_notes(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find video_notes.csv at {path}")
    return pd.read_csv(path, sep=',')


def filter_notes(df: pd.DataFrame, source: str, verbose: bool) -> pd.DataFrame:
    df_src = df[df['source'] == source]
    excluded = df_src[df_src['not_use'] == 1]['video_id'].unique()
    print(f"Videos excluded by 'not_use' flag for source {source}: {len(excluded)}")
    if verbose and excluded.size:
        for vid in excluded:
            print(f"  {vid}")

    df_filt = df_src[df_src['not_use'] != 1]
    print(f"After filtering not_use==0: {len(df_filt)} rows")

    dup = df_filt['video_id'].value_counts().loc[lambda x: x > 1]
    if not dup.empty:
        print(f"Found {dup.sum()} duplicate rows across {len(dup)} video_ids:")
        for vid, cnt in dup.items(): print(f"  Video '{vid}' appears {cnt} times")
    else:
        print("No duplicate video_ids after filtering.")
    return df_filt


def build_frame_ranges(df_notes: pd.DataFrame) -> Tuple[Dict[str, Optional[int]], Dict[str, Optional[int]]]:
    first: Dict[str, Optional[int]] = {}
    last: Dict[str, Optional[int]] = {}

    for _, r in df_notes.iterrows():
        vid = r['video_id']
        dff = pd.to_numeric(r.get('Desired_first_frame', np.nan), errors='coerce')
        first[vid] = None if pd.isna(dff) or dff <= 0 else int(dff)
        dlf = pd.to_numeric(r.get('Desired_last_frame', np.nan), errors='coerce')
        last[vid] = None if pd.isna(dlf) or dlf == 0 else int(dlf)
    return first, last


def report_coverage(
    df_over: pd.DataFrame,
    df_notes: pd.DataFrame,
    source: str
) -> None:
    vids_notes = set(df_notes['video_id'])
    vids_over = set(df_over[df_over['project_source'] == source]['video_name'])

    missing_in_over = vids_notes - vids_over
    if missing_in_over:
        print(f"{len(missing_in_over)} IDs in notes but no frames:", missing_in_over)
    else:
        print("All video_ids from notes have matching frames.")

    missing_in_notes = vids_over - vids_notes
    if missing_in_notes:
        print(f"{len(missing_in_notes)} IDs in frames not in notes:")
        for vid in sorted(missing_in_notes): print(f"  {vid}")
    else:
        print("All frames covered by notes.")


def setup_output_dirs(out_dir: str) -> Tuple[str, str]:
    imgs = os.path.join(out_dir, 'imgs')
    masks = os.path.join(out_dir, 'masks')
    os.makedirs(imgs, exist_ok=True)
    os.makedirs(masks, exist_ok=True)
    return imgs, masks


def process_frames(
    df: pd.DataFrame,
    src_imgs: str,
    src_masks: str,
    dst_imgs: str,
    dst_masks: str,
    first: Dict[str, Optional[int]],
    last: Dict[str, Optional[int]]
) -> Tuple[List[pd.Series], Dict[str, Dict[str, int]], Dict[str, int]]:
    overview_rows: List[pd.Series] = []
    exclusion: Dict[str, Dict[str, int]] = {}
    stats = {'skipped': 0, 'missing_mask':0, 'resized':0, 'white':0}

    def add_exc(vid: str, reason: str):
        exclusion.setdefault(vid, {}).setdefault(reason, 0)
        exclusion[vid][reason] += 1

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Processing frames"):
        num = int(r['frame']); idx = int(r['frame_idx']); vid = r['shared_video_id']

        if first.get(vid) is not None and num < first[vid]:
            add_exc(vid, 'before_first'); stats['skipped']+=1; continue
        if last.get(vid) is not None and num > last[vid]:
            add_exc(vid, 'after_last'); stats['skipped']+=1; continue

        src_f = os.path.join(src_imgs, f"{idx}.png"); dst_f = os.path.join(dst_imgs, f"{idx}.png")
        src_m = os.path.join(src_masks, f"{idx}_bolus.png"); dst_m = os.path.join(dst_masks, f"{idx}_bolus.png")

        if not os.path.exists(src_f): add_exc(vid, 'frame_missing'); stats['skipped']+=1; continue
        img = cv2.imread(src_f, cv2.IMREAD_UNCHANGED)
        if img is None: add_exc(vid, 'frame_unreadable'); stats['skipped']+=1; continue
        cv2.imwrite(dst_f, img)

        if not os.path.exists(src_m):
            add_exc(vid, 'mask_missing'); stats['missing_mask']+=1
        else:
            m = cv2.imread(src_m, cv2.IMREAD_UNCHANGED)
            if m is None:
                add_exc(vid, 'mask_unreadable'); stats['missing_mask']+=1
            elif np.all(m == 255):
                add_exc(vid, 'mask_all_white'); stats['white']+=1
                os.remove(dst_f); stats['skipped']+=1; continue
            else:
                h1, w1 = img.shape[:2]; h2, w2 = m.shape[:2]
                if (h1, w1) != (h2, w2):
                    m = cv2.resize(m, (w1, h1), interpolation=cv2.INTER_NEAREST)
                    stats['resized'] += 1
                cv2.imwrite(dst_m, m)
        overview_rows.append(r)
    return overview_rows, exclusion, stats


def print_exclusions(ex_info: Dict[str, Dict[str, int]]) -> None:
    if not ex_info:
        return
    print("\nExclusion summary per video:")
    for vid, reasons in ex_info.items():
        smry = ", ".join(f"{k}:{v}" for k,v in reasons.items())
        print(f"Video {vid} -> {smry}")



def analyze_empty_masks(df_new: pd.DataFrame, masks_dir: str, imgs_dir: str, threshold: Optional[int] = None, remove_all: bool = False) -> Set[Tuple[str, int]]:
    print("\nEmpty-mask analysis:")
    empty_counts: List[int] = []
    to_remove_set: Set[Tuple[str, int]] = set()
    for vid, grp in df_new.groupby('shared_video_id'):
        empties: List[Tuple[str, int, str]] = []
        for _, r in grp.iterrows():
            idx = int(r['frame_idx'])
            mask_path = os.path.join(masks_dir, f"{idx}_bolus.png")
            if os.path.exists(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if m is not None and np.all(m == 0): empties.append((vid, idx, mask_path))
        count = len(empties); empty_counts.append(count)
        print(f"Video {vid}: {count} empty masks")
        if remove_all:
            for vid, idx, mask_path in empties:
                os.remove(mask_path)
                img_path = mask_path.replace(f"{idx}_bolus.png", f"{idx}.png").replace("masks", "imgs")
                if os.path.exists(img_path): os.remove(img_path)
                to_remove_set.add((vid, idx))
            print(f"  Removed all {count} empty masks and frames")
        elif threshold is not None and count > threshold:
            for vid, idx, mask_path in empties[threshold:]:
                os.remove(mask_path)
                img_path = mask_path.replace(f"{idx}_bolus.png", f"{idx}.png").replace("masks", "imgs")
                if os.path.exists(img_path): os.remove(img_path)
                to_remove_set.add((vid, idx))
            print(f"  Removed {len(empties) - threshold} empty masks and frames beyond threshold {threshold}")
    total_videos = len(empty_counts)
    videos_with = sum(1 for c in empty_counts if c > 0)
    videos_without = total_videos - videos_with
    mean_c = float(np.mean(empty_counts)) if empty_counts else 0.0
    min_c = int(np.min(empty_counts)) if empty_counts else 0
    max_c = int(np.max(empty_counts)) if empty_counts else 0
    print("\nEmpty-mask summary across videos:")
    print(f"  Videos processed: {total_videos}")
    print(f"  Videos with empty masks: {videos_with}")
    print(f"  Videos without empty masks: {videos_without}")
    print(f"  Empty masks per video -> mean: {mean_c:.2f}, min: {min_c}, max: {max_c}")
    return to_remove_set

def extract_patient_id(video_id: str) -> str:
    """
    From a video filename like
      "Pt2198_Visit1_fixed_003.mp4"
      "Example_Pt2335_Visit1_fixed_000_000.mp4"
    produce the numeric patient ID, e.g. "2198" or "2335".
    """
    s = video_id.lower()
    if s.startswith("example_"):
        s = s[len("example_"):]
    if s.startswith("pt"):
        s = s[len("pt"):]
    if s.startswith("p"):
        s = s[len("p"):]
    return s.split("_", 1)[0]



def save_overview(rows: List[pd.Series], out_csv: str) -> pd.DataFrame:
    df_new = pd.DataFrame(rows)

    # add Patient_Id column
    df_new["patient_id"] = df_new["shared_video_id"].apply(extract_patient_id)

    if 'dataset_name' in df_new.columns:
        df_new = df_new.drop(columns=['dataset_name'])
    df_new.to_csv(out_csv, index=False)
    return df_new



def process_dataset(
    original_dir: str,
    notes_csv: str,
    output_dir: str,
    source: str,
    verbose: bool,
    empty_threshold: Optional[int],
    remove_empty_all: bool,
    exclude_artefacts: bool,
    include_only_artefacts: bool,
    exclude_first_frame: bool
) -> None:
    overview_path = os.path.join(original_dir, 'data_overview.csv')
    df_over = load_overview(overview_path)

    df_notes = load_video_notes(notes_csv)

    print(f"Original: {len(df_over)} frames, {df_over['shared_video_id'].nunique()} videos.")
    df_notes_f = filter_notes(df_notes, source, verbose)

    # Artifact filtering
    if exclude_artefacts and include_only_artefacts:
        raise ValueError("Cannot use both --exclude_artefacts and --include_only_artefacts")
    if exclude_artefacts:
        to_exclude = df_notes_f['Patient_Artefact'] == 1
        cnt = to_exclude.sum()
        df_notes_f = df_notes_f[~to_exclude]
        print(f"Excluded {cnt} rows where Patient_Artefact == 1")
    if include_only_artefacts:
        to_include = df_notes_f['Patient_Artefact'] == 1
        cnt = to_include.sum()
        df_notes_f = df_notes_f[to_include]
        print(f"Included only {cnt} rows where Patient_Artefact == 1")


    first_map, last_map = build_frame_ranges(df_notes_f)

    report_coverage(df_over, df_notes_f, source)

    df_join = df_over[(df_over['project_source']==source) & df_over['shared_video_id'].isin(df_notes_f['video_id'])]
    print(f"After join: {len(df_join)} frames, {df_join['shared_video_id'].nunique()} videos.")

    # Exclude first frame per video if requested
    if exclude_first_frame:
        pre_ex = len(df_join)
        firsts = df_join.groupby('shared_video_id')['frame'].min().reset_index()
        exc_set = set((row['shared_video_id'], row['frame']) for _, row in firsts.iterrows())
        df_join = df_join[~df_join.apply(lambda r: (r['shared_video_id'], r['frame']) in exc_set, axis=1)]
        excl_cnt = pre_ex - len(df_join)
        print(f"Excluded {excl_cnt} first-frame rows (one per video) due to --exclude_first_frame")

    src_imgs = os.path.join(original_dir, 'imgs')
    src_masks = os.path.join(original_dir, 'masks')
    dst_imgs, dst_masks = setup_output_dirs(output_dir)

    rows, ex_info, stats = process_frames(
        df_join, src_imgs, src_masks, dst_imgs, dst_masks, first_map, last_map
    )
    print_exclusions(ex_info)

    # analyze and prune empty masks
    pre_prune = len(rows)
    remove_set = analyze_empty_masks(pd.DataFrame(rows), dst_masks, empty_threshold, remove_empty_all)
    rows_filtered = [r for r in rows if (r['shared_video_id'], int(r['frame_idx'])) not in remove_set]
    post_prune = len(rows_filtered)
    print(f"\nRows before empty-mask pruning: {pre_prune}, after: {post_prune}")

    df_new = save_overview(rows_filtered, os.path.join(output_dir, 'data_overview.csv'))

    print("\nProcessing complete!")
    print(f"Output at: {output_dir}")
    print(f"Frames final: {len(df_new)}, total pruned frames: {stats['skipped'] + (pre_prune - post_prune)}")
    print(f"Masks missing/unreadable: {stats['missing_mask']}, resized: {stats['resized']}, white removed: {stats['white']}")


def main():
    parser = argparse.ArgumentParser(
        description="Process annotated video exports and prune frames."
    )
    parser.add_argument("--original_dataset_dir", required=True)
    parser.add_argument("--video_notes_csv", required=True)
    parser.add_argument("--output_dataset_dir", required=True)
    parser.add_argument("--project_source", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--empty_mask_threshold", type=int, default=None,
                        help="Keep only this many empty-mask frames per video; remove extras.")
    parser.add_argument("--remove_all_empty_masks", action="store_true",
                        help="Remove all empty-mask frames for each video.")
    parser.add_argument("--exclude_artefacts", action="store_true",
                        help="Exclude rows where Patient_Artefact == 1.")
    parser.add_argument("--include_only_artefacts", action="store_true",
                        help="Include only rows where Patient_Artefact == 1.")
    parser.add_argument("--exclude_first_frame", action="store_true",
                        help="Exclude the first frame of each video before further filtering.")
    args = parser.parse_args()

    process_dataset(
        original_dir=args.original_dataset_dir,
        notes_csv=args.video_notes_csv,
        output_dir=args.output_dataset_dir,
        source=args.project_source,
        verbose=args.verbose,
        empty_threshold=args.empty_mask_threshold,
        remove_empty_all=args.remove_all_empty_masks,
        exclude_artefacts=args.exclude_artefacts,
        include_only_artefacts=args.include_only_artefacts,
        exclude_first_frame=args.exclude_first_frame
    )


if __name__ == '__main__':
    main()
