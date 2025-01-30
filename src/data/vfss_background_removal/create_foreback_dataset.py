import flammkuchen as fl
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_mask(mask_file_path):
    """ Load the .mask file and extract mask data. Convert boolean masks to uint8 (0 and 255). """
    mask_data = fl.load(mask_file_path)
    if 'mask' not in mask_data:
        raise ValueError(f"Mask data not found in {mask_file_path}")
    mask_array = mask_data['mask'].astype(np.uint8) * 255
    return mask_array

def process_videos(input_folder, output_folder, val_split=0.2, random_seed=42):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # Define directories for train, val, and test
    train_dir = output_folder / "train"
    val_dir = output_folder / "val"
    test_dir = output_folder / "test"

    for subdir in [train_dir, val_dir, test_dir]:
        (subdir / "imgs").mkdir(parents=True, exist_ok=True)
        (subdir / "masks").mkdir(parents=True, exist_ok=True)
        (subdir / "overlays").mkdir(parents=True, exist_ok=True)

    video_files = sorted(input_folder.glob("*.mp4"))
    mask_files = sorted(input_folder.glob("*.mask"))

    # Create a set of video names that have masks
    masked_video_names = set(mask_file.stem for mask_file in mask_files)

    # Separate videos into train_videos (with masks) and test_videos (without masks)
    train_videos = [video for video in video_files if video.stem in masked_video_names]
    test_videos = [video for video in video_files if video.stem not in masked_video_names]

    # Split train_videos into train and val
    train_videos_split, val_videos_split = train_test_split(
        train_videos, test_size=val_split, random_state=random_seed
    )

    print(f"Total videos: {len(video_files)}")
    print(f"Training videos: {len(train_videos_split)}")
    print(f"Validation videos: {len(val_videos_split)}")
    print(f"Testing videos: {len(test_videos)}")

    # Track all processed data
    data_records = []
    global_frame_count = 0

    def process_single_video(video_file, set_type):
        nonlocal global_frame_count
        video_name = video_file.stem  # Get video name without extension
        mask_path = input_folder / f"{video_name}.mask"
        has_mask = mask_path.exists()

        if has_mask:
            masks = load_mask(mask_path)
            mask_folder = (train_dir if set_type == "train" else val_dir) / "masks"
            overlay_folder = (train_dir if set_type == "train" else val_dir) / "overlays"
        else:
            masks = None
            mask_folder = None
            overlay_folder = None

        # Process video frames
        cap = cv2.VideoCapture(str(video_file))
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if has_mask:
                # Assuming masks shape: (num_frames, height, width)
                if frame_idx >= masks.shape[0]:
                    print(f"Warning: Frame index {frame_idx} exceeds mask frames for {video_file.name}")
                    break
                frame_mask = masks[frame_idx]
                frame_height, frame_width = frame_mask.shape
                frame_resized = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    (frame_width, frame_height)
                )
            else:
                frame_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Save frame with a unique ID
            new_frame_name = f"{global_frame_count:06d}.png"
            if set_type == "train":
                target_dir = train_dir / "frames"
            elif set_type == "val":
                target_dir = val_dir / "frames"
            else:
                target_dir = test_dir / "frames"
            frame_path = target_dir / new_frame_name
            cv2.imwrite(str(frame_path), frame_resized)

            # Save mask if available
            if has_mask:
                mask = masks[frame_idx]
                mask_path_save = mask_folder / new_frame_name
                cv2.imwrite(str(mask_path_save), mask)

                # Create overlay
                mask_uint8 = mask.astype(np.uint8)

                # Create purple overlay by merging channels with consistent type and dimensions
                purple_overlay = cv2.merge([
                    (mask_uint8 * (128 / 255)).astype(np.uint8),  # Blue channel
                    np.zeros_like(mask_uint8),                   # Green channel
                    (mask_uint8 * (128 / 255)).astype(np.uint8)  # Red channel
                ])

                overlayed_image = cv2.addWeighted(
                    cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR), 0.6,
                    purple_overlay, 0.4, 0
                )
                overlay_path = overlay_folder / new_frame_name
                cv2.imwrite(str(overlay_path), overlayed_image)

            # Record data for CSV
            data_records.append({
                "video_name": video_file.name,
                "has_mask": has_mask,
                "set_type": set_type,
                "total_frames": frame_idx + 1,
                "new_frame_names": new_frame_name
            })

            frame_idx += 1
            global_frame_count += 1

        cap.release()
        print(f"Processed {frame_idx} frames from {video_file.name} into {set_type} set.")

    # Process training videos
    for video in train_videos_split:
        process_single_video(video, "train")

    # Process validation videos
    for video in val_videos_split:
        process_single_video(video, "val")

    # Process testing videos
    for video in test_videos:
        process_single_video(video, "test")

    # Create CSV data overview
    df = pd.DataFrame(data_records)
    df.to_csv(output_folder / "data_overview.csv", index=False)
    print(f"Data overview saved to {output_folder / 'data_overview.csv'}")

# Run the processing
if __name__ == "__main__":
    input_dir = "../../../data/foreback/ForeBack"
    output_dir = "../../../data/foreback/processed"
    process_videos(input_dir, output_dir)
