#!/usr/bin/env python
# src/utils.py

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

def get_project_root():
    """Get the absolute path to the project root directory"""
    # This file is in src/, so go up one level
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_model_path(model_filename):
    """Get the absolute path to a model file"""
    return os.path.join(get_project_root(), "models", model_filename)

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize image while preserving aspect ratio"""
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)

def display_image(image, title="Image", figsize=(10, 6)):
    """Display an image using matplotlib"""
    plt.figure(figsize=figsize)
    
    # Convert BGR to RGB if the image is from OpenCV
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

class FPS:
    """Class to measure frames per second"""
    def __init__(self, avarageof=50):
        self.frametimes = []
        self.avarageof = avarageof
        self.timestart = time.time()

    def update(self):
        self.frametimes.append(time.time() - self.timestart)
        self.timestart = time.time()

    def get(self):
        if len(self.frametimes) > self.avarageof:
            self.frametimes = self.frametimes[-self.avarageof:]
        return (len(self.frametimes) / sum(self.frametimes))