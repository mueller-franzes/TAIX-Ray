from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil
import os 


def find_dicom_files(root_dir):
    """ Efficient recursive search for .dcm files using os.scandir() instead of rglob() """
    for entry in os.scandir(root_dir):
        if entry.is_dir(follow_symlinks=False):
            yield from find_dicom_files(entry.path)  # Recursively search subdirectories
        elif entry.is_file() and entry.name.endswith('.dcm'):
            yield Path(entry.path)  # Convert to Path object only when needed

def verify_xray(ds):
    modality = ds.get('Modality', '') == "CR"
    photometric = ds.get('PhotometricInterpretation', '') in ['MONOCHROME1', 'MONOCHROME2']
    samples = ds.get('SamplesPerPixel', 0) == 1
    sop = ds.get('SOPClassUID', '') == "1.2.840.10008.5.1.4.1.1.1"
    return modality and photometric and samples and sop

def maybe_convert(x):
    if isinstance(x, (pydicom.sequence.Sequence, pydicom.dataset.Dataset)):
        return None  # Don't store complex nested data
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    elif isinstance(x, pydicom.valuerep.DSfloat):
        return float(x) 
    elif isinstance(x, pydicom.valuerep.IS):
        return int(x)
    return x

def get(ds, key):
    keyword =  ds[key].keyword
    if keyword == "":
        return ds[key].name
    return keyword

def dataset2dict(ds, exclude=['PixelData', 'Overlay Data'] ):
    return {get(ds, key): maybe_convert(ds[key].value) 
            for key in ds.keys()
            if get(ds, key) not in exclude}

def read_metadata(path_dcm):
    meta_dict = {} # 'Path': str(path_dcm.relative_to(path_root_data)) # WARNING: path leak accession numbers

    try:
        # Try to read the DICOM file
        ds = pydicom.dcmread(path_dcm, stop_before_pixels=False)

        # Verify if it is a valid CR X-Ray
        is_xray = verify_xray(ds)
        if not is_xray:
            return None 
        
        # Extract metadata
        meta_dict.update(dataset2dict(ds))        

        # Check if the pixel data is valid
        try:
            pixel_array = ds.pixel_array  # Try to access pixel data
            if pixel_array.max() - pixel_array.min() < 1e-5:
                return None 
         
            # Copy the file
            path_out = path_root_out_data / ds.get('PatientID')/ds.get('StudyInstanceUID')/ds.get('SeriesInstanceUID')
            path_out.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path_dcm, path_out/path_dcm.name)
            return meta_dict
        except:
            return None 
    
    except Exception as e:
        return None 
    
    

if __name__ == "__main__":
    # Setting 
    path_root = Path('/ocean_storage/data/UKA/UKA_Thorax')
    path_root_data = path_root/'download/data'

    path_root_out = path_root/'public_export'
    path_root_out_data = path_root_out/'data'
    path_root_out_metadata = path_root_out/'metadata'

    path_root_out_data.mkdir(parents=True, exist_ok=True)
    path_root_out_metadata.mkdir(parents=True, exist_ok=True)

    paths_series = find_dicom_files(str(path_root_data))  
    metadata_list = []

    # Using ThreadPoolExecutor instead of Pool
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(read_metadata, path): path for path in paths_series}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing DICOM files"):
            try:
                metadata_list.append(future.result())
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")


    df = pd.DataFrame(metadata_list)
    df.to_csv(path_root_out_metadata / 'metadata.csv', index=False)

    print(f"Number Patients: {df['PatientID'].nunique()}")
    print(f"Number Studies: {df['StudyInstanceUID'].nunique()}")
    print(f"Number Series: {df['SeriesInstanceUID'].nunique()}")
    print(f"Number Images: {len(df)}")