#!/usr/bin/env python
# Some helpful utility functions for our gender detection project

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

def get_project_root():
    """Figure out where our project root directory is"""
    # This file is in src/, so we need to go up one level
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_model_path(model_filename):
    """Get the full path to a model file in our models directory"""
    return os.path.join(get_project_root(), "models", model_filename)

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize an image while keeping its proportions looking good"""
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        # Calculate the ratio based on the new height
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # Calculate the ratio based on the new width
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)

def display_image(image, title="Image", figsize=(10, 6)):
    """Show an image using matplotlib (useful for debugging)"""
    plt.figure(figsize=figsize)
    
    # Convert BGR to RGB if the image is from OpenCV (OpenCV uses BGR, matplotlib uses RGB)
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def pil_to_cv2(pil_image):
    """Convert a PIL Image to OpenCV format (BGR)"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert an OpenCV image (BGR) to PIL format (RGB)"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

class FPS:
    """A simple class to measure how many frames per second we're processing"""
    def __init__(self, avarageof=50):
        self.frametimes = []
        self.avarageof = avarageof
        self.timestart = time.time()

    def update(self):
        """Record the time for this frame"""
        self.frametimes.append(time.time() - self.timestart)
        self.timestart = time.time()

    def get(self):
        """Calculate the average FPS over the last few frames"""
        if len(self.frametimes) > self.avarageof:
            # Keep only the most recent frames
            self.frametimes = self.frametimes[-self.avarageof:]
        return (len(self.frametimes) / sum(self.frametimes))