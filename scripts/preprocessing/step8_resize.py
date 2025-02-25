from PIL import Image
import numpy as np 
from pathlib import Path 
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def resize(path_png):
    img = Image.open(path_png) # 16bit int

    # Determine new size while maintaining aspect ratio
    target_size = 1024
    w, h = img.size
    
    if h > w:
        new_h, new_w = target_size, int(w * target_size / h)
    else:
        new_w, new_h = target_size, int(h * target_size / w)
    
    # Resize image
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Save the resized image
    img.save(path_data_out/path_png.relative_to(path_data) , format='PNG')
    # return img 


if __name__ == "__main__":
    # Setting 
    path_root = Path('/ocean_storage/data/UKA/UKA_Thorax/public_export')
    path_data = path_root/'data_png'
    path_metadata = path_root/'metadata'
    
    path_data_out = path_root/'data_png_resize_1024'
    path_data_out.mkdir(parents=True, exist_ok=True)


    paths_series = path_data.rglob('*.png')

    # Using ThreadPoolExecutor instead of Pool
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(resize, path): path for path in paths_series}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    # Single thread
    # for path in paths_series:
    #     # img = Image.open(path)
    #     img = resize(path)
    #     img.save('test.png' , format='PNG')
