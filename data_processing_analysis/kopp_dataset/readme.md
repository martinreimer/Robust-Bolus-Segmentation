TLDR:
- we have 534 patients with lateral videos: 25 patients with no lateral videos
- 64 patients with videos from different acquisition dates
- todo: swallows need to be extracted


Analysis of Kopp Dataset (Erlangen / Disphagia)
TLDR: Disphagia dataset with uncut videos of swallows from different perspectives

Pros:
- patient information
- labels for conditions

Challenges / Todos:
- [ ] multiple perspectives (1-20, ~5) -> Identify correct perspectives (heuristic or CNN Classification)
    - perspectives: lateral, frontal, zoomed in lateral and movement doing scan
    - [ ] sometimes >1 video from correct perspective -> choose one
- [ ] uncut videos -> Identify swallows and cut videos
- [ ] Some videos have movements in them (of patient or camera) -> further analysis
- only disphagia, no normals

### Folder Structure:
- patient id dcm folders
   - 560 patient identifiers
  - every patient folder can have multiple subfolders/videos with dcm file
- Diagnosen und Prozeduren folder
   - Diagnosen_psuedonymisiert.csv
      -  Columns: PATIENT_ID	FALL_ID	ICD_CODE	DIAGNOSE_DATUM	GEWISSHEIT_ID
      - 3,742 rows
  - Prozeduren_pseudonymisiert.csv
      - Columns: PATIENT_ID	FALL_ID	OPS_CODE PROZEDUR_DATUM
      - 1,523 rows
- extracted_videos_and_images folder
   - extracted mp4 files from dcm files per video id
- Kopp_Data_Overview.xlsx
   - columns:
      - Birth Year
      - Age
      - Sex
      - Manufacturer
      - Manufacturer's Model Name
      - Device Serial Number
      - Image/ Video Identifier
      - Comments about the Image/ Video
    -  3,018 rows
    - prob. just extracted info from dicom files
- README.md
   - mentions test sample ids
