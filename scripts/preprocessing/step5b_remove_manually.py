from pathlib import Path 
import pandas as pd 
from tqdm import tqdm
import shutil

path_root = Path('/ocean_storage/data/UKA/UKA_Thorax/public_export')
path_metadata = path_root / 'metadata'
path_data = path_root / 'data'

df_labels = pd.read_csv(path_metadata/'annotation.csv')
df_images = pd.read_csv(path_metadata/'metadata.csv')
df_images['UID'] = df_images['PatientName'].str.split('_').str[1]

remove_uids = [
    'd296a023636d911996dfa2e015b135c1269e0fcb48944a8ce45fb9015acdcb0b', # Abdomen left lateral position,
    'ba6bb23d58955b8d558fea23e2c80999512890a4e1522758fdc77b4c16dc093c', # empty image,
    'b8bcbd6631da75a2d8f35532c46b9be6192bcd3c9904f8f032ba2a5295daf0db', # black image probably wg left-sign thorax a.p.
    'b31bfe74b25ae28d4f31840c83cfe37526cba6dbcf78b9ac521707727f9543e2', # Ankle joint
    'a925b916d9c7716c0c3a1f38ce6106b16c223308092a066cef2654af860fe26a', # shoulder image
    '3df29468008172279c5bc92e1d0a1b0a85785b428efb0e917647d8dc60f5cd36', # Abdomen a.p.
    '077aeb0848604d7f7f7efddb5261ec529f861bc657d56a21eb7c62d06486eae0', # Thorax lateral
    'b3cb19f7322d76d23cc1b0b82f15cf91f4d0d5ce9ab6d550928d13d50692dce6', # Abdomen LSL

    '108e2d88e37b48ea8325f1de3878c4705b3d1b546a53e6ae024c85b2441d4e7b', # Text in image 
    '68c7c75bea6c0596e325762d055c97c985e6bbf65b14e86a0cbda1b18b7186e1', # Text in image
    '55be502d7a6f40a6a85649bbcf8da3e4dbd2a134588cb5ddeebf7adb42e05e17', # Text in image
    'ffe175be5c6bde735621aec70ab9444e7332d54564b5ed116bd84b935142e2f4', # Text in image
]

  
df_remove = df_images[df_images['UID'].isin(remove_uids)]
df_images = df_images[~df_images['UID'].isin(remove_uids)]
df_labels = df_labels[~df_labels['UID'].isin(remove_uids)]

assert not (df_remove['PatientID'].isin(df_images['PatientID'].values)).any(), "Can not remove entire patient"


# Remove exams with multiple images (DICOM)
for patientID in tqdm(df_remove['PatientID'].unique()):
    shutil.rmtree(path_data/patientID) 

# path_data = path_root / 'data_png_resize_512'
# for patientID in tqdm(df_remove['UID'].unique()):
#     path_file = path_data/f'{patientID}.png'
#     if path_file.is_file():
#         path_file.unlink() 

df_labels.to_csv(path_metadata/'annotation.csv', index=False)
df_images.to_csv(path_metadata/'metadata.csv', index=False)
