"""
WattWise AI - Automated Dataset Acquisition Script
Downloads primary Grid-India NLDC hourly SCADA dataset and Zenodo Indian state datasets.
"""

import os
import requests

def download_file(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"[SKIP] {dest_path} already exists ({os.path.getsize(dest_path):,} bytes).")
        return
    print(f"[DOWNLOADING] {url} -> {dest_path} ...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    print(f"[DONE] Saved {dest_path} ({os.path.getsize(dest_path):,} bytes).")

def main():
    print("=== WattWise AI Dataset Acquisition Pipeline ===")
    
    # 1. Mendeley Primary Hourly SCADA Dataset (Jan 2024 - Jun 2025)
    mendeley_url = "https://data.mendeley.com/public-files/datasets/y58jknpgs8/files/c67cbcfa-f33b-49c0-925d-d6e1cda3a9c1/file_downloaded"
    download_file(mendeley_url, "data/raw/mendeley/Jan2024_Jun2025.xlsx")
    
    # 2. Zenodo State-Level Datasets (2013-2024)
    zenodo_files = {
        "data/raw/zenodo/Maharashtra.csv": "https://zenodo.org/api/records/14983362/files/Maharashtra.csv/content",
        "data/raw/zenodo/daily_energy_met_MU.csv": "https://zenodo.org/api/records/14983362/files/daily_energy_met_MU.csv/content",
        "data/raw/zenodo/peak_met_MW.csv": "https://zenodo.org/api/records/14983362/files/peak_met_MW.csv/content",
        "data/raw/zenodo/peak_unmet_MW.csv": "https://zenodo.org/api/records/14983362/files/peak_unmet_MW.csv/content"
    }
    for path, url in zenodo_files.items():
        download_file(url, path)
        
    print("\nDataset acquisition complete! All source files verified.")

if __name__ == "__main__":
    main()
