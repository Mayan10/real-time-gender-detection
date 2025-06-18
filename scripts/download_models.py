#!/usr/bin/env python
# scripts/download_models.py

import os
import sys
import requests
import argparse
from tqdm import tqdm
from pathlib import Path

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    dest_path = Path(destination)
    if not dest_path.parent.exists():
        dest_path.parent.mkdir(parents=True)
        
    progress_bar = tqdm(
        total=total_size, 
        unit='iB', 
        unit_scale=True, 
        desc=f"Downloading {os.path.basename(destination)}"
    )
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Download incomplete")
        return False
    return True

def download_yolo():
    """Download YOLOv4 model files"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # YOLOv4 weights
    weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
    weights_path = os.path.join(models_dir, "yolov4.weights")
    
    # YOLOv4 config
    config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
    config_path = os.path.join(models_dir, "yolov4.cfg")
    
    # COCO names file
    names_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    names_path = os.path.join(models_dir, "coco.names")
    
    print("Downloading YOLOv4 model files...")
    success = download_file(weights_url, weights_path)
    if success:
        success = download_file(config_url, config_path)
    if success:
        success = download_file(names_url, names_path)
    
    if success:
        print("YOLOv4 model files downloaded successfully!")
    else:
        print("Failed to download YOLOv4 model files.")

def main():
    parser = argparse.ArgumentParser(description="Download pre-trained models for gender detection")
    parser.add_argument("--yolo", action="store_true", help="Download YOLOv4 model")
    parser.add_argument("--all", action="store_true", help="Download all models")
    
    args = parser.parse_args()
    
    if args.all or args.yolo:
        download_yolo()
    elif len(sys.argv) == 1:
        # No arguments provided, show help
        parser.print_help()

if __name__ == "__main__":
    main()