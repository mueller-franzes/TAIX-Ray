from PIL import Image
import numpy as np 
from pathlib import Path 
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pydicom

def dicom_to_png(path_dcm):
    ds = pydicom.dcmread(path_dcm)

    image = ds.pixel_array
    assert image.dtype == np.uint16, "Assuming 16-bit DICOM image"

    # Apply Rescale Slope and Intercept
    # slope = float(ds.get('RescaleSlope', 1))
    # intercept = float(ds.get('RescaleIntercept', 0))
    # image = image * slope + intercept
    # image = image.astype(np.uint16) # WARNING: type cast might not be valid
    
    # Save as 16-bit PNG
    img = Image.fromarray(image, mode="I;16")
    uid = str(ds.get('PatientName')).split('_')[1]
    img.save(path_data_out/f'{uid}.png', format='PNG')


if __name__ == "__main__":
    # Setting 
    path_root = Path('/ocean_storage/data/UKA/UKA_Thorax/public_export')
    path_data = path_root/'data'
    path_metadata = path_root/'metadata'

    path_data_out = path_root/'data_png'
    path_data_out.mkdir(parents=True, exist_ok=True)


    paths_series = path_data.rglob('*.dcm')

    # Using ThreadPoolExecutor instead of Pool
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(dicom_to_png, path): path for path in paths_series}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing DICOM files"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    # Single thread
    # for path in paths_series:
    #     dicom_to_png(path)