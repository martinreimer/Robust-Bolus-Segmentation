# Melda Dataset (US, Pathological Swallowing)
TLDR: 484 swallows of patients with dysphagia

Pros:
- swallows extracted
- patient information

- full swallows + split swallows mp4 files
- X-ray videos of pathological swallows from Dr. Melda Kunduc
- folder structure:
    - patient id mp4 folders
        - 484 identifiers
        - variants:
            - 1 full mp4 video
            - 1 full mp4 video + swallow split videos + excel explaining split +  sc file + crf
              - .sc files: swallow cut tmp
    - abnormals folder
        - 10 mp4/mpg files each having abnormal description in filename (long videos)
    - DICOM folder
        - 7 folders with id in name
                -  each has MKV files
    - pt1_anon - pt8_anon folders
        - video of split swallows
        - each has mp4 files, excel files with split info, some also mask files
    - De-identified_demographic_Sheet.xlsx
        - some information about diseases idk
    - folder_list.xlsx
        - list of patient id folders
    - from_melda.xlsx
        - maping of patient id to video ids and their metadata
        - columns: Patient ID	Video Name	Visit	Height	Width	if CRF	if Extracted
    - from_melda_events.xslx
        - mapping of video ids to video snippet ids + length info
        - columns: Patient ID	Video Name	Event Name	length	if annotated	swallow type	quality
    - from_melda_events_with_rating.xlsx
        - better than above
        - mapping of video ods to snippet ids + length + swallow type
        - columns: Patient ID	Video Name	Event Name	length	if annotated	swallow type	quality	comments
    - spreadsheet.ipynb

- Google Docs: Swallowing Dataset Annotation
    -  Patho VSSS Info


Q:
- patient info?
- Deidentified demo sheet: Is each row one patient (480 rows)? Is row id related to patient id/folder list id?
- whats pt1_anon - pt8_anon folders? Mapping in google doc?
- What are crf videos next to normal ones?
- sollen alle videos noch extracted und verwendet werden?
- 22x age missing, but birth year is there?
- Exlucde data where age or gender missing?
- Usage of dysphagia patients only or also including other pathologies?
- Why are in the execls more swallow splits recorded than in the video folders? Because these other swallows are not frontal videos?
- Why are there (4 times) negative frame counts?