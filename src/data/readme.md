# Data Processing and Analysis

This folder contains scripts for:
1. **Processing**: Cleaning, annotating and preparing datasets for further use.
2. **Analysis**: Generating insights through statistics and visualizations.

## Contents
Each dataset as its own directory since each dataset has its own structure and has to be processed differently.

- kopp_dataset
  - pathological swallowing dataset
  - contains patient info
  - swallows need to be extracted from full recordings
- leonard_dataset
  - normal swallowing dataset
  - no patient info abailable
- melda_dataset
   - pathological swallowing dataset
   - contains patient info
   - swallows are (>50%) extracted from full recordings
   - 


Pipeline:
- get raw data
- run create data from labelexports, from leonard
- in processed folder:
- mask sanitize data
- resize images if needed
- create train test split