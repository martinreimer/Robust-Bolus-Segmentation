

Pipeline:
- copy data from \\fauad.fau.de\shares\ANKI\Projects\Swallowing\Data\from_Leonard\ForeBack to data/foreback
- run create_foreback_dataset.py
- run src/data/resize_images.py on train and test to resolution 512x512
- run background_removal.py as training and prediction