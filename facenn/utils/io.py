import os
import requests
import gdown
import logging
from tqdm import tqdm

logger = logging.getLogger("facenn")

def download_file_from_url(url, dest_path):
    if os.path.exists(dest_path):
        return

    logger.info(f"Downloading {url} to {dest_path}...")
    
    # Check if google drive
    if "drive.google.com" in url:
        gdown.download(url, dest_path, quiet=False)
    else:
        response = requests.get(url, stream=True, allow_redirects=True)
        if response.status_code != 200:
             logger.error(f"Failed to download {url}: {response.status_code}")
             return
             
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            logger.error("ERROR, something went wrong")
