import os
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import create_directory_if_not_exists

CODENET_URL = "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz"

def download_and_preprocess_data(dataset_path="./dataset"):
    create_directory_if_not_exists(dataset_path)
    
    # Download the dataset
    print("Downloading Project CodeNet dataset...")
    response = requests.get(CODENET_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(f"{dataset_path}/Project_CodeNet.tar.gz", "wb") as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    
    # Extract the dataset
    print("Extracting the dataset...")
    os.system(f"tar -xzf {dataset_path}/Project_CodeNet.tar.gz -C {dataset_path}")
    
    # Process C++ files
    cpp_files = []
    for root, dirs, files in os.walk(f"{dataset_path}/Project_CodeNet"):
        for file in files:
            if file.endswith(".cpp"):
                cpp_files.append(os.path.join(root, file))
    
    # Read and preprocess C++ files
    preprocessed_data = []
    for file_path in tqdm(cpp_files, desc="Preprocessing C++ files"):
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
            preprocessed_data.append({
                "code": code,
                "file_path": file_path
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(preprocessed_data)
    
    # Save preprocessed data
    df.to_csv(f"{dataset_path}/preprocessed_cpp_data.csv", index=False)
    
    print(f"Preprocessed {len(df)} C++ files.")
    return df

if __name__ == "__main__":
    download_and_preprocess_data()
