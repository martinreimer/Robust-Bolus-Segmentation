# Labelbox Export

Quick Infos:
- MBSS_Martin Project ID: cm5qm2z0c0b8b07zgdc5f52jp

- MBS Project ID: cl2t3p57f1b5y0764dutw9c2t

Run command
`python labelbox_export/main.py -pi cl2t3p57f1b5y0764dutw9c2t -o D:\Martin\thesis\data\raw\labelbox_output_mbs_0312`

`python labelbox_export/main.py -pi cm5qm2z0c0b8b07zgdc5f52jp -o D:\Martin\thesis\data\raw\labelbox_output_mbss_martin_0328`



This repository contains a Python script for exporting annotated swallowing videos from a Labelbox Project. The script extracts the individual frames from the videos and saves the landmark points and the area feature masks for each frame.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Caveats](#caveats)

## Features
- The script accesses a Labelbox project and exports all of the annotated videos with workflow status `DONE` from the project.
- For each exported video:
  - The individual video frames are extracted as PNG images.
  - For each frame, landmarks of key anatomical points are saved as PNG images. The considered anatomical points are:
    - Arytenoids
    - Base of Vallecula
    - Cricoid
    - Epiglottis
    - Hyoid
    - Nasal Spine
  - For each frame, masks of area features are saved as PNG images. The considered area features are:
    - Airway
    - Esophagus
    - Oral Area
    - Pharynx
    - Spine Area
  - For each frame, all of the potentially multiple bolus masks are merged into one. The merged mask is saved as PNG image.
- Additionally, the exported data is also saved in an NDJSON file and a CSV file is created as overview of the extracted video frames.

## Installation
Clone this repository and install the required packages by following the steps below:
```
git clone https://github.com/ankilab/Labelbox_Export.git
cd Labelbox_Export
pip install -r requirements.txt
```
Alternatively, you can clone the repository via SSH: `git clone git@github.com:ankilab/Labelbox_Export.git`

## Usage
   
1. Run the script with command line parameters: `python main.py -ak <API_KEY> -pi <project_id> -o <output_directory>` or `python main.py --api-key <API_KEY> --project-id <project_id> --output-directory <output_directory>`
    - Replace `<API_KEY>` with your Labelbox API key.
    - Replace `<project_id>` with the ID of the project from which the data should be exported. This ID can be found in the URL of the project.
    - The parameter `-o <output_directory>` or `--output-directory <output_directory>` is optional. If no output directory is provided, the current directory will be used for the output.
2. The extracted data can be found in the directory `<output_directory>/output`.

## Output
After running the script, the created directory `output` will be structured as follows:
- `output/`: The main directory for the output.
   - `exported_data.ndjson`: An NDJSON file containing the exported data.
   - `data_overview.csv`: A CSV file giving an overview of which frame in which video each exported frame corresponds to.
   - `frames/`: The directory containing the extracted individual video frames as `1.png`, `2.png`, ...
   - `landmarks/`: The directory containing the landmarks of key anatomical points as `<frame_index>_<landmark_name>.png` for each individual video frame.
   - `masks/`: The directory containing the masks of the area features and of the bolus as `<frame_index>_<feature_name>.png` for each individual video frame. There might be multiple masks for the Spine Area per frame.

## Caveats
- The script only works for exporting swallowing data labeled the same way as in the MBS project and it only exports videos with workflow status `DONE`.
- The directory `output` should be moved to another location after the execution of the script. Otherwise, this directory will be overwritten by the next run of the program.
- If an error occurs during the execution of the script, simply try running it again as there might have been a problem when trying to connect to Labelbox. Optionally, you can specify the index of the frame at which the execution should be restarted by adapting `start_index` in the calls of `export_video_frames`, `save_landmark_points`, and `save_masks`.