#!/usr/bin/env python
# Our main command-line interface for gender detection

import os
import sys
import argparse
import cv2
from src.detector import GenderDetector

def check_camera_availability(camera_index):
    """Let's see if the camera at this index is available and working"""
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
    
    # If they want to see what cameras are available, let's show them
    if args.list_cameras:
        print("Let me check what cameras you have available:")
        for i in range(10):  # Check camera indices 0-9
            if check_camera_availability(i):
                print(f"Camera index {i} is available")
        return 0
    
    # On macOS, let's try to use Metal for better performance
    if not args.cpu:
        try:
            # Try to enable Metal backend for OpenCV on macOS
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.haveOpenCL():
                cv2.ocl.useOpenCL()
                print("Great! OpenCV is using OpenCL acceleration (Metal on macOS)")
            else:
                print("OpenCL isn't available, so we'll use the CPU instead")
        except Exception as e:
            print(f"Couldn't enable Metal acceleration: {e}")
            print("No worries, we'll use the CPU instead")
    
    # Convert string "0" to integer 0 for webcam
    video_source = args.video
    if video_source.isdigit():
        video_source = int(video_source)
    
    # Make sure the camera is actually available if we're trying to use one
    if isinstance(video_source, int):
        if not check_camera_availability(video_source):
            print(f"Oops! Camera at index {video_source} isn't available.")
            print("Here are some things to try:")
            print("1. Check camera permissions in System Preferences > Security & Privacy > Privacy > Camera")
            print("2. Close other applications that might be using the camera")
            print("3. Try a different camera index with --video <index>")
            print("4. Run with --list-cameras to see what cameras are available")
            return 1
    
    try:
        # Let's create our gender detector
        detector = GenderDetector(
            confidence_threshold=args.confidence,
            use_cuda=not args.cpu
        )
        
        # Now let's run it on the video source
        detector.run_on_video(video_source)
        
    except FileNotFoundError as e:
        print(f"Uh oh! {e}")
        print("Make sure you've downloaded the required model files.")
        print("Run: python scripts/download_models.py --yolo")
        return 1
    except KeyboardInterrupt:
        print("You interrupted the program - no problem!")
        return 0
    except Exception as e:
        print(f"Something went wrong: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # On macOS, we need to set some environment variables to make camera permissions work properly
    if sys.platform == 'darwin':
        os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    
    sys.exit(main())