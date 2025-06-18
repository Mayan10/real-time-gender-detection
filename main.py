#!/usr/bin/env python
# main.py

import os
import sys
import argparse
import cv2
from src.detector import GenderDetector

def check_camera_availability(camera_index):
    """Test if the camera at the given index is available"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False
    cap.release()
    return True

def main():
    parser = argparse.ArgumentParser(description="Gender detection using Vision Transformer")
    parser.add_argument(
        "--video", "-v", type=str, default="0",
        help="Path to video file or camera index (default: 0 for webcam)"
    )
    parser.add_argument(
        "--confidence", "-c", type=float, default=0.5,
        help="Confidence threshold for person detection (default: 0.5)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU usage even if Metal/GPU is available"
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="List available camera devices and exit"
    )
    
    args = parser.parse_args()
    
    # List available cameras if requested
    if args.list_cameras:
        print("Checking available cameras:")
        for i in range(10):  # Check camera indices 0-9
            if check_camera_availability(i):
                print(f"Camera index {i} is available")
        return 0
    
    # Set OpenCV to use Metal on macOS
    if not args.cpu:
        try:
            # Try to enable Metal backend for OpenCV on macOS
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.haveOpenCL():
                cv2.ocl.useOpenCL()
                print("OpenCV is using OpenCL acceleration (Metal on macOS)")
            else:
                print("OpenCL is not available, falling back to CPU")
        except Exception as e:
            print(f"Failed to enable Metal acceleration: {e}")
            print("Falling back to CPU")
    
    # Convert string "0" to integer 0 for webcam
    video_source = args.video
    if video_source.isdigit():
        video_source = int(video_source)
    
    # Check camera availability if using webcam
    if isinstance(video_source, int):
        if not check_camera_availability(video_source):
            print(f"Error: Camera at index {video_source} is not available.")
            print("Possible solutions:")
            print("1. Check camera permissions in System Preferences > Security & Privacy > Privacy > Camera")
            print("2. Close other applications that might be using the camera")
            print("3. Try a different camera index with --video <index>")
            print("4. Run with --list-cameras to see available cameras")
            return 1
    
    try:
        # Initialize gender detector
        detector = GenderDetector(
            confidence_threshold=args.confidence,
            use_cuda=not args.cpu
        )
        
        # Run detection on video source
        detector.run_on_video(video_source)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've downloaded the required model files.")
        print("Run: python scripts/download_models.py --yolo")
        return 1
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # On macOS, explicitly set environment variable to enable camera permissions dialog
    if sys.platform == 'darwin':
        os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    
    sys.exit(main())