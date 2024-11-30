import os
import pandas as pd
from pathlib import Path
from skvideo.io import vread
import matplotlib.pyplot as plt
import time

# Parameters
videos_directory = '\\\\fauad.fau.de\\shares\\ANKI\\Projects\\Swallowing\\Data\\from_Kopp\\extracted_videos_and_images'
labels_csv_path = "kopp_video_labels.csv"

# Create labels CSV if it doesn't exist
if not os.path.exists(labels_csv_path):
    pd.DataFrame(columns=["video_id", "label"]).to_csv(labels_csv_path, index=False)

# Load existing labels
labels_df = pd.read_csv(labels_csv_path)
labeled_videos = set(labels_df["video_id"].tolist())

# Get all video files in the directory
video_files = [str(f) for f in Path(videos_directory).rglob("*") if f.suffix in ['.mp4', '.avi', '.mkv', '.mov']]

# Remove already labeled videos
video_files = [vf for vf in video_files if os.path.basename(vf) not in labeled_videos]

print(f"Number of videos to label: {len(video_files)}")

def label_video(video_path):
    """
    Display frames 3, 4, and 5 of the video side by side and prompt for a label.
    Save the label to the CSV file.
    """
    # Read the video
    video_id = os.path.basename(video_path)
    video = vread(video_path)

    # Check if video has at least 15 frames
    if len(video) < 15:
        print(f"Video {video_id} does not have enough frames. Skipping.")
        return

    # Extract frames 0, middle and last
    frame_indices = [0, int((len(video)/2)-1), len(video)-1]
    print(frame_indices)
    frames = [video[i] for i in frame_indices]

    # Display frames side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [f"Frame {i+1}" for i in frame_indices]
    for ax, frame, title in zip(axes, frames, titles):
        ax.imshow(frame)
        ax.axis('off')
        ax.set_title(title)

    plt.suptitle(f"Video: {video_id}\nPress 'Enter' to close the image and proceed to labeling.")

    def on_key(event):
        if event.key == 'enter':  # Close the plot when 'Enter' is pressed
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)  # Show the plot and wait for interaction

    # Prompt for label
    while True:
        try:
            label = int(input(f"Enter label for {video_id} (1-4): "))
            if label not in [1, 2, 3, 4]:
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 4.")

    # Save the label
    # if video_id already exists in labels_df, update the label
    if video_id in labels_df["video_id"].tolist():
        labels_df.loc[labels_df["video_id"] == video_id, "label"] = label
        labels_df.to_csv(labels_csv_path, index=False)
    else:
        new_entry = {"video_id": video_id, "label": label}
        new_df = pd.DataFrame([new_entry])
        if os.path.exists(labels_csv_path):
            new_df.to_csv(labels_csv_path, mode="a", header=False, index=False)
        else:
            new_df.to_csv(labels_csv_path, index=False)

    print(f"Saved label {label} for video {video_id}.")
    time.sleep(0.1)  # Pause for 2 seconds before moving to the next video


# label mode parameter
mode = "all" # all or relabel 1/2/3/4
if mode == "all":
    for video_path in video_files:
        label_video(video_path)
        print("Moving to the next video...\n")
elif mode == 4:
    print(f"Relabeling {len(labels_df[labels_df['label'] == 4])} videos...")
    # iterate labels_df
    for index, row in labels_df.iterrows():
        video_id = row["video_id"]
        label = row["label"]
        if label == 4:
            # create path with videos_directory
            video_path = os.path.join(videos_directory, video_id)
            label_video(video_path)
            print("Moving to the next video...\n")