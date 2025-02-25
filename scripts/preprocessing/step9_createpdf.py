from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pydicom
from PIL import Image, ImageDraw
import pandas as pd

def load_dicom_image(path_file, max_image_size):
    """Load and process a single DICOM image."""
    ds = pydicom.dcmread(path_file)
    img = ds.pixel_array

    # Normalize pixel values to 0-255 
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype(np.uint8)

    # Convert to PIL image
    img = Image.fromarray(img)
    img.thumbnail((max_image_size, max_image_size))
    return img

    
def load_png(path_file, max_image_size):
    img = Image.open(path_file)
    # img.thumbnail((max_image_size, max_image_size))

    # Normalize pixel values to 0-255
    img = np.array(img)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype(np.uint8)

    return Image.fromarray(img) 


def create_pdf(paths_group, output_pdf_path, images_per_row, images_per_col, max_image_size):
    """Converts multiple DICOM files into a multi-image-per-page PDF."""
    images = []
    
    # Load images in parallel
    with ThreadPoolExecutor() as executor:
        images = list(tqdm(executor.map(lambda path: load_png(path, max_image_size), paths_group), total=len(paths_group)))
    
    
    grid_w, grid_h = images_per_row * max_image_size, images_per_col * max_image_size
    num_pages = (len(images) + (images_per_row * images_per_col) - 1) // (images_per_row * images_per_col)
    pdf_pages = []

    for i in range(num_pages):
        page = Image.new("L", (grid_w, grid_h), 255)
        for j in range(images_per_row * images_per_col):
            idx = i * images_per_row * images_per_col + j
            if idx >= len(images):
                break
            x = (j % images_per_row) * max_image_size
            y = (j // images_per_row) * max_image_size
            page.paste(images[idx], (x, y))
        pdf_pages.append(page)

    pdf_pages[0].save(output_pdf_path, "PDF", save_all=True, append_images=pdf_pages[1:])

if __name__ == "__main__":
    path_root = Path('/ocean_storage/data/UKA/UKA_Thorax')/'public_export'
    path_data = path_root/'data'
    path_data_png = path_root/'data_png_resize'
    path_meta = path_root/'metadata'
    path_out = path_root / 'pdfs'
    path_out.mkdir(parents=True, exist_ok=True)

    images_per_row = 10
    images_per_col = 10
    max_pages_per_pdf = 100
    max_image_size = 512

    # Option 1: DICOMs
    # df_meta = pd.read_csv(path_meta/'metadata.csv', usecols=['_Path'])
    # paths_series = [path_root.parent/p for p in df_meta['_Path'].tolist()]

    # Option 2: PNGs
    paths_series = list(path_data_png.iterdir())

    max_images_per_pdf = images_per_row * images_per_col * max_pages_per_pdf
    grouped_paths = [paths_series[i:i + max_images_per_pdf] for i in range(0, len(paths_series), max_images_per_pdf)]

    
    for i, group in enumerate(grouped_paths):
        create_pdf(group, path_out / f"batch_{i}.pdf", images_per_row, images_per_col, max_image_size)
