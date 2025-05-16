from io import StringIO
import pandas as pd
csv_path = r"D:\Martin\thesis\data\raw\labelbox_output_mbss_martin_0514\data_overview.csv"
df = pd.read_csv(csv_path)
print(df.head())

# check for each unique video name how many times it has the value 0 for the column frame
unique_videos = df["video_name"].unique()
for video in unique_videos:
    count = df[df["video_name"] == video]["frame"].value_counts().get(0, 0)
    if count > 1:
        print(f"Video: {video}, Count of 0s in 'frame': {count}")

#for i in list(df["video_name"].unique()):
#    print(i)
'''
# Strip any leading/trailing whitespace
df['video_id'] = df['video_id'].str.strip()

# Extract patient ID as everything before "_V"
df['patient_id'] = df['video_id'].str.extract(r'^.*?([A-Za-z]*\d+)_V', expand=False)

# pd print out all rows no limit
# exlcude rows where not_use is 1
df = df[df["not_use"] != 1]
pd.set_option('display.max_rows', None)
print(df["patient_id"].value_counts())

# print how many have one count and how many have more than one
print("Number of patients with one video:", df["patient_id"].value_counts()[df["patient_id"].value_counts() == 1].count())
print("Number of patients with more than one video:", df["patient_id"].value_counts()[df["patient_id"].value_counts() > 1].count())

# how many unique patients?
print("Number of unique patients:", df["patient_id"].nunique())
#df.to_csv(r"D:\Martin\thesis\data\video_notes_new.csv", index=False)

'''