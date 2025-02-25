from pathlib import Path 
import pandas as pd 
import pydicom
import numpy as np
import PIL.Image as Image
import shutil


def read_img(path_dcm):
    # Read the DICOM file
    ds = pydicom.dcmread(path_dcm)
    img = ds.pixel_array

    # Normalize pixel values to 0-255 if needed
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype(np.uint8)

    # Convert to PIL image
    img = Image.fromarray(img)
    return img 
