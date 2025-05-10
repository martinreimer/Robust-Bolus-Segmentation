# Thesis Project: Exploring Robustness in Videofluoroscopy Bolus Segmentation

This repository contains the code and data for my Master's thesis project titled **"Exploring Robustness in Videofluoroscopy Bolus Segmentation"**. The thesis focuses on segmenting the bolus in videofluoroscopy studies using advanced deep learning techniques. The project aims to develop robust models that can generalize across diverse data while evaluating the impact of data quality and model configurations.

![groundtruth_example](https://github.com/user-attachments/assets/2709240a-5efb-469f-8d00-4354555ace1a)

---

## Project Timeline
- **Duration**: 15 January 2025 – 15 July 2025

---

## Repository Structure

### Folders
1. **`src/data/`**  
   Contains scripts for preparing, cleaning, and analyzing datasets. Tasks include:
   - Selecting high-quality videos.
   - Extracting swallows from video frames.
   - Annotating segmentation labels.
   - Analyzing data quality using NIQE, PSNR, and contrast metrics.

2. **`src/models/`**  
   Contains scripts and configurations for training and evaluating machine learning models. Tasks include:
   - Training U-Net and its variations (e.g., small, medium, large, with/without attention).
   - Experimenting with additional architectures to improve robustness and explainability.
   - Evaluating model performance on diverse datasets.


### Includes Code from External Repos:
- labelbox-export
- PyTorch U-Net https://github.com/milesial/Pytorch-UNet
- niqe score
  - https://github.com/guptapraful/niqe
- PSNR; SSIM, ..https://sewar.readthedocs.io/en/latest/


# Setup
```bash
python 3.12.6
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torchinfo sewar moviepy==1.0.3 pandas numpy scikit-learn torch tqdm wandb albumentations opencv-python matplotlib  nvitop moviepy
pip install -U git+https://github.com/qubvel/segmentation_models.pytorch
```
