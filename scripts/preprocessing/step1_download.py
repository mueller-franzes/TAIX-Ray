from pathlib import Path
import pandas as pd
from io import BytesIO
import boto3
from boto3.s3.transfer import TransferConfig
import argparse
import pyzipper
import zipfile
import json
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np

def init_bucket(s3_config: str):
    try:
        with open(s3_config, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{s3_config}' not found.")
    
    session = boto3.Session(profile_name=config.get("profile_name"))
    s3 = session.resource('s3', endpoint_url=config.get("endpoint_url"))
    return s3.Bucket(config.get("bucket_name"))

def load_bytesio(bucket, path_file):
    config = TransferConfig()
    file_stream = BytesIO()
    start_time = time.time()
    bucket.download_fileobj(str(path_file), file_stream, Config=config)
    elapsed_time = time.time() - start_time
    file_size_mb = len(file_stream.getbuffer()) / (1024 * 1024)
    speed = file_size_mb / elapsed_time if elapsed_time > 0 else 0
    
    file_stream.seek(0)
    return file_stream, speed

def download(bucket, item, path_out, use_pseudo=False, pbar=None):
    folder_col = 'PseudoAccessionNumber' if use_pseudo else 'AccessionNumber'
    acc_num, zip_password, rel_path = item[1][[folder_col, 'Key', 'Path']]
    rel_path = Path(rel_path)
    file_stream, speed = load_bytesio(bucket, f'PACS_Export/{rel_path}')
    
    # Define save path
    zip_path = path_out / rel_path.parts[-1]
    extract_path = path_out / f"{acc_num}"
    
    # Save zip file
    with open(zip_path, "wb") as f:
        f.write(file_stream.getbuffer())
    
    # Unzip  
    with pyzipper.AESZipFile(zip_path, 'r') as zf:
        zf.setpassword(zip_password.encode())
        zf.extractall(extract_path)
    
    # Extract  
    inner_zip_path = extract_path / rel_path.stem
    with zipfile.ZipFile(inner_zip_path, 'r') as zf:
        zf.extractall(extract_path)
    
    # Remove the zip files after extraction
    inner_zip_path.unlink()
    zip_path.unlink()
    
    if pbar:
        pbar.set_postfix(speed=f"{speed:.2f} MB/s")
        pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Download and process files from S3.")
    parser.add_argument("--path_csv", default="/ocean_storage/data/UKA/UKA_Thorax/download/metadata/pseudo_table.csv", type=str, help="Path to CSV file")
    parser.add_argument("--path_out", default="/ocean_storage/data/UKA/UKA_Thorax/download/data", type=str, help="Output directory")
    parser.add_argument("--s3_config", default="s3_config.json", type=str, help="Path to S3 configuration file")
    parser.add_argument("--use_pseudo", default=False, type=bool, help="Store files under pseudo accession number")
    parser.add_argument("--parallelize", default=False, type=bool, help="Enable parallelized download")
    
    args = parser.parse_args()
    
    # Create output directory
    path_out = Path(args.path_out)
    path_out.mkdir(parents=True, exist_ok=True)
    
    # Read exams with valid images
    df = pd.read_csv(args.path_csv)
    df = df.drop_duplicates(subset=['AccessionNumber'], keep="first").reset_index(drop=True)

    # Read exams with valid report 
    df_reports = pd.read_excel('/ocean_storage/data/UKA/UKA_Thorax/download/metadata/reports.xlsx')
    df_reports = df_reports.dropna(subset=['Untersuchungsnummer'])
    df_reports = df_reports.drop_duplicates(subset=['Untersuchungsnummer'], keep="first")
    df_reports['Untersuchungsnummer'] = df_reports['Untersuchungsnummer'].str.replace('-', '0')

    # Merge
    df = df[df['AccessionNumber'].isin(df_reports["Untersuchungsnummer"])]

      
    # Init Bucket
    bucket = init_bucket(args.s3_config)

  
    # Start Download 
    errors = []
    with tqdm(total=len(df), desc="Downloading files", unit="file") as pbar:
        if args.parallelize:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(download, bucket, item, path_out, args.use_pseudo, pbar) for item in df.iterrows()]
                for future in futures:
                    future.result()
        else:
            for item in df.iterrows():
                try:
                    download(bucket, item, path_out, args.use_pseudo, pbar)
                except Exception as e:
                    print(f"------------------------ Error ------------------------")
                    print(item[1][['AccessionNumber', 'PseudoAccessionNumber']])
                    print("Error:", e)
                    errors.append({"AccessionNumber": item[1]['AccessionNumber'], "PseudoAccessionNumber": item[1]['PseudoAccessionNumber'], "Error": str(e)})
                    continue
    df = pd.DataFrame(errors)
    df.to_csv('errors.csv', index=False)
    
if __name__ == "__main__":
    main()
