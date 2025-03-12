from pathlib import Path 
import pandas as pd 
from tqdm import tqdm
import shutil

path_root = Path('/ocean_storage/data/UKA/UKA_Thorax/public_export')
path_metadata = path_root / 'metadata'
path_data = path_root / 'data'

df_labels = pd.read_csv(path_metadata/'annotations.csv')
df_images = pd.read_csv(path_metadata/'metadata.csv')

df_images['UID'] = df_images['PatientName'].str.split('_').str[1]
df_remove = df_images[~df_images['UID'].isin(df_labels['UID'])]
df = df_images[df_images['UID'].isin(df_labels['UID'])]

assert len(df_remove)+len(df) == len(df_images), "Duplicates and non duplicates should cover all images"
assert not (df_remove['PatientID'].isin(df['PatientID'].values)).any(), "Can not remove entire patient"


# Remove exams with multiple images (DICOM)
for patientID in tqdm(df_remove['PatientID'].unique()):
    shutil.rmtree(path_data/patientID) 

# path_data = path_root / 'data_png'
# for patientID in tqdm(df_remove['UID'].unique()):
#     path_file = path_data/f'{patientID}.png'
#     if path_file.is_file():
#         path_file.unlink() 

df.to_csv(path_metadata/'metadata.csv', index=False)