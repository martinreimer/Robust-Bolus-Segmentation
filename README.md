# Thesis Project: Exploring Robustness in Videofluoroscopy Bolus Segmentation

This repository contains the code and data for my Master's thesis project titled **"Exploring Robustness in Videofluoroscopy Bolus Segmentation"**. The thesis focuses on segmenting the bolus in videofluoroscopy studies using advanced deep learning techniques. The project aims to develop robust models that can generalize across datasets from different sources while evaluating the impact of data quality and model configurations.

---

## Project Timeline
- **Duration**: 15 January 2025 – 15 July 2025
- **Key Goals**:
  1. Create a comprehensive dataset combining sources from the United States, Germany, and international partners.
  2. Train and evaluate U-Net-like architectures.
  3. Assess the impact of data quality using metrics such as Natural Image Quality Evaluator (NIQE).
  4. Explore model robustness and generalizability across diverse datasets.
  5. Write and submit a paper based on the findings.

---

## Repository Structure

### Folders
1. **`data_processing_analysis/`**  
   Contains scripts for preparing, cleaning, and analyzing datasets. Tasks include:
   - Selecting high-quality videos.
   - Extracting swallows from video frames.
   - Annotating segmentation labels.
   - Analyzing data quality using NIQE, PSNR, and contrast metrics.

2. **`modeling/`**  
   Contains scripts and configurations for training and evaluating machine learning models. Tasks include:
   - Training U-Net and its variations (e.g., small, medium, large, with/without attention).
   - Experimenting with additional architectures to improve robustness and explainability.
   - Evaluating model performance on diverse datasets.

### Key Files
- `README.md`: Main repository description (this file).
- `requirements.txt`: Python dependencies for running the project.


