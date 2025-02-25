from pathlib import Path 
import pandas as pd 
import shutil
from tqdm import tqdm

###########################################
# Remove exams with multiple images and it is not clear which one to use
###########################################


path_root = Path('/ocean_storage/data/UKA/UKA_Thorax/public_export')
path_data = path_root/'data'
path_metadata = path_root/'metadata'


df_images = pd.read_csv(path_metadata/'metadata.csv')

# Find duplicates and non duplicates
df = df_images.drop_duplicates(subset=['PatientName'], keep=False).reset_index(drop=True) # multiple images for the same exam
df_dub = df_images[df_images['PatientName'].duplicated(keep=False)].sort_values(by='PatientName')


assert len(df_dub)+len(df) == len(df_images), "Duplicates and non duplicates should cover all images"
assert not (df_dub['PatientID'].isin(df['PatientID'].values)).any(), "Can not remove entire patient"

# Remove exams with multiple images (DICOM)
for patientID in tqdm(df_dub['PatientID'].unique()):
    shutil.rmtree(path_data/patientID) 


df.to_csv(path_metadata/'metadata.csv', index=False)
          