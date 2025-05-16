# ETL Pipeline for Labelbox Exports

This pipeline demonstrates how to take data exported from Labelbox and perform a series of transformations (pruning, merging, mask sanitization, resizing, visualization, splitting) to produce a final dataset ready for model training or other data science tasks.

---

## Overview of Steps

1. **Labelbox Export**  
   Export labeled videos/frames from Labelbox via the Labelbox API script.

2. **Preprocess Exports (Pruning)**  
   Sometimes certain frames need to be removed (e.g., bad cut points). This step prunes those frames based on a curation CSV.

   then also visualizes frames as videos script as sanity check to check if in processing everything went as expected.

   ```bash
3. **Create Dataset from Multiple Labelbox Exports**  
   Combine multiple separate Labelbox export folders into a single consolidated dataset.

4. **Mask Sanitizer**  
   Fix or inspect mask images (e.g., remove weird artifacts, ensure binary format).

5. **Resize Images**  
   Resize frames and/or masks to a uniform size.

6. **Visualize Dataset**  
   Generate sample overlays to confirm that frames and masks align.

7. **Train/Test Split**  
   Split the combined dataset into train/val/test sets for modeling.

---

## Individual Script Usage

Below are example commands for each script. Adjust the file paths as needed.

### 1. Labelbox Export

Export labeled data from Labelbox. You typically specify the Labelbox project ID (`-pi`) and an output directory (`-o`). For instance:

```bash
python labelbox_export/main.py \
  -pi <PROJECT_ID> \
  -o D:\Martin\thesis\data\raw\labelbox_output_mbss_martin
  
python labelbox_export/main.py -pi cm5qm2z0c0b8b07zgdc5f52jp -o D:\Martin\thesis\data\raw\labelbox_output_mbss_martin
python labelbox_export/main.py -pi cl2t3p57f1b5y0764dutw9c2t -o D:\Martin\thesis\data\raw\labelbox_output_mbs

````

python check_shape_mismatches.py --path "D:/Martin/thesis/data/processed/labelbox_output_mbss_martin_frames_excluded"  --img_folder "frames" --mask_folder "masks" --mask_suffix "_bolus"


### 2. Preprocess Exports (Pruning)
If some videos aren’t cut correctly, remove the unwanted frames using a curation CSV:

```bash
python preprocess_labelbox_export.py \
  --original_dataset_dir "D:\Martin\thesis\data\raw\labelbox_output_mbss_martin\labelbox_output" \
  --curation_csv "D:\Martin\thesis\data\video_notes.csv" \
  --output_dataset_dir "D:\Martin\thesis\data\processed\labelbox_output_mbss_martin_frames_excluded"

python preprocess_labelbox_export.py --project_source MBSS_Martin --original_dataset_dir D:\Martin\thesis\data\raw\labelbox_output_mbss_martin_0328 --video_notes_csv D:\Martin\thesis\data\video_notes.csv --output_dataset_dir D:\Martin\thesis\data\processed\labelbox_output_mbss_martin_0328_frames_excluded
python preprocess_labelbox_export.py --project_source MBS_Luisa --original_dataset_dir D:\Martin\thesis\data\raw\labelbox_output_mbs_0328 --video_notes_csv D:\Martin\thesis\data\video_notes.csv --output_dataset_dir D:\Martin\thesis\data\processed\labelbox_output_mbs_0328_frames_excluded

```

### 3. Create Dataset from Multiple Labelbox Exports
Merge two or more processed exports into a single dataset:

```bash




  
python create_dataset_from_labelboxexports.py --input_paths "D:\Martin\thesis\data\processed\labelbox_output_mbss_martin_0328_test" --output_path "D:\Martin\thesis\data\processed\dataset_labelbox_export_test_2504"
  
python create_dataset_from_labelboxexports.py --input_paths "D:\Martin\thesis\data\processed\labelbox_output_mbss_martin_0328_frames_excluded" "D:\Martin\thesis\data\processed\labelbox_output_mbs_0328_frames_excluded" --output_path "D:\Martin\thesis\data\processed\labelbox_dataset_0328"
```

### 4. Mask Sanitizer (Optional)
Check and optionally fix mask images for consistency:

```bash
# Just inspect:
python mask_sanitizer.py -p D:\Martin\thesis\data\processed\labelbox_dataset_0328\masks

# Fix any issues:
python mask_sanitizer.py -p D:\Martin\thesis\data\processed\dataset_labelbox_export\masks -fix
```

### 5. Resize Images (Optional)
Resize frames and masks to a uniform size (e.g., 512×512), with optional padding:

```bash
python resize_images.py \
  -p D:\Martin\thesis\data\processed\dataset_labelbox_export \
  --folders frames masks \
  -size 512 \
  -m pad_resize \
  --in_place
  
python resize_images.py -p D:\Martin\thesis\data\processed\labelbox_dataset_0328 --folders imgs masks -size 512 -m pad_resize --in_place

python resize_images.py -p D:\Martin\thesis\data\processed\dataset_labelbox_export_test_2504 --folders imgs masks -size 512 -m pad_resize --in_place
python resize_images.py -p D:\Martin\thesis\data\processed\dataset_labelbox_export_test_2504_test_final_roi_crop --folders imgs masks -size 256 -m pad_resize --in_place


```

### 6. Visualize Dataset (Optional)
Visualize a few overlayed frames + masks for sanity checks:

```bash
python visualize_dataset.py \
  -p D:\Martin\thesis\data\processed\dataset_labelbox_export \
  -n 20 \
  --mask_suffix _bolus
 
python visualize_dataset.py -p D:\Martin\thesis\data\processed\labelbox_dataset_0328 -n 200 --mask_suffix _bolus
python visualize_dataset.py -p D:\Martin\thesis\data\raw\labelbox_output_mbss_martin -n 1000 --mask_suffix _bolus
python visualize_dataset.py -p D:\Martin\thesis\data\raw\labelbox_output_mbs -n 1000 --mask_suffix _bolus

python visualize_dataset.py -p D:\Martin\thesis\data\processed\dataset_labelbox_export_test_2504 -n 1000 --mask_suffix _bolus

```

### 7. Train/Test Split (Optional)
Split the dataset into train, val, and test sets:

```bash
python create_train_test_split.py \
  --input_dirs "D:\Martin\thesis\data\processed\dataset_labelbox_export" \
  --output_dir "D:\Martin\thesis\data\processed\dataset_train_val_test_split"

python create_train_test_split.py --input_dirs "D:\Martin\thesis\data\processed\dataset_labelbox_export_test_2504" --output_dir "D:\Martin\thesis\data\processed\dataset_labelbox_export_test_2504_test_final"
```