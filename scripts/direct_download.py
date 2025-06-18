#!/usr/bin/env python

import os
import urllib.request
import sys

def download_file(url, destination):
    """Download a file with simple progress reporting"""
    print(f"Downloading {os.path.basename(destination)}...")
    
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    try:
        def report_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r{percent}% completed")
            sys.stdout.flush()
            
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

def main():
    # Define project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    
    # URLs for model files
    weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
    config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
    names_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    
    # Destination paths
    weights_path = os.path.join(models_dir, "yolov4.weights")
    config_path = os.path.join(models_dir, "yolov4.cfg")
    names_path = os.path.join(models_dir, "coco.names")
    
    # Download files
    success = download_file(weights_url, weights_path)
    if success:
        success = download_file(config_url, config_path)
    if success:
        success = download_file(names_url, names_path)
    
    if success:
        print("All YOLOv4 model files downloaded successfully!")
    else:
        print("Failed to download some YOLOv4 model files.")

if __name__ == "__main__":
    main()