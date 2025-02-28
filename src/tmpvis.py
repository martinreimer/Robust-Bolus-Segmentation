import os
import cv2
import pandas as pd
import numpy as np

def create_side_by_side_videos(csv_path,
                               raw_images_folder,
                               masks_folder,
                               output_folder="visualization",
                               fps=30,
                               alpha=0.4):
    """
    - csv_path: path to your CSV file with columns:
        [frame_idx, video_name, frame, new_frame_name, new_mask_name, source, resolution]
    - raw_images_folder: folder where raw frames (e.g. 0.png) actually reside
    - masks_folder: folder where the _bolus masks (e.g. 0_bolus.png) reside
    - output_folder: destination folder to store the mp4 side-by-side videos
    - fps: frames per second for output videos
    - alpha: blending factor for the segmentation overlay (0 = no overlay, 1 = fully opaque)
    """

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Group by video name so we can create one output MP4 per video
    for video_name, group_df in df.groupby("video_name"):
        # Sort by frame_idx to preserve order
        group_df = group_df.sort_values("frame_idx")

        # Read the first frame to get dimensions
        first_row = group_df.iloc[0]
        raw_file = os.path.join(raw_images_folder, first_row["new_frame_name"])
        first_frame = cv2.imread(raw_file)
        if first_frame is None:
            print(f"Warning: could not read first frame {raw_file}. Skipping {video_name}.")
            continue

        height, width, _ = first_frame.shape

        # Prepare output video writer:
        # side-by-side width = 2 * original width
        out_width = 2 * width
        out_height = height

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "avc1"
        out_path = os.path.join(output_folder, f"{video_name}_side_by_side.mp4")
        video_writer = cv2.VideoWriter(out_path, fourcc, fps, (out_width, out_height))

        # Loop through each frame in this video
        for idx, row in group_df.iterrows():
            # Read raw image
            raw_name = row["new_frame_name"]
            raw_path = os.path.join(raw_images_folder, raw_name)
            raw_img = cv2.imread(raw_path)
            if raw_img is None:
                print(f"Warning: could not read raw image {raw_path}.")
                continue

            # Read mask image (grayscale)
            mask_name = row["new_mask_name"]
            mask_path = os.path.join(masks_folder, mask_name)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask_img is None:
                # If no mask found, just show the raw frame on both sides
                side_by_side = np.concatenate((raw_img, raw_img), axis=1)
                video_writer.write(side_by_side)
                continue

            # Create a purple overlay only where mask > 0
            # 1) Create a color image with purple in the mask region
            colored_label = np.zeros_like(raw_img, dtype=np.uint8)
            # BGR = (255, 0, 255) is purple
            colored_label[mask_img > 0] = (255, 0, 255)

            # 2) Blend raw and colored label with alpha
            overlay_img = cv2.addWeighted(raw_img, 1 - alpha, colored_label, alpha, 0)

            # Combine left (raw) and right (overlay) horizontally
            side_by_side = np.concatenate((raw_img, overlay_img), axis=1)

            # Write frame to output video
            video_writer.write(side_by_side)

        video_writer.release()
        print(f"Created side-by-side video: {out_path}")






if __name__ == "__main__":
    # Adjust these paths for your environment
    base_folder = r"D:\Martin\thesis\data\processed\dataset_first_experiments"

    # CSV file containing the mapping (frame_idx, video_name, etc.)
    csv_path = os.path.join(base_folder, "data_overview.csv")

    # The folder containing raw frames (like 0.png, 1.png, etc.)
    raw_images_folder = os.path.join(base_folder, "imgs")

    # The folder containing mask frames (like 0_bolus.png, 1_bolus.png, etc.)
    masks_folder = os.path.join(base_folder, "masks")  # If they are in the same folder as raw


    # Output folder for the generated MP4 files
    output_folder = os.path.join(base_folder, "visualization")

    create_side_by_side_videos(
        csv_path=csv_path,
        raw_images_folder=raw_images_folder,
        masks_folder=masks_folder,
        output_folder=output_folder,
        fps=3,
        alpha=0.4
    )
