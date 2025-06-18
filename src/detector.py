#!/usr/bin/env python
# src/detector.py

import cv2
import torch
import numpy as np
from PIL import Image
import os
import time
from transformers import ViTForImageClassification, ViTImageProcessor

from .utils import get_model_path, cv2_to_pil, FPS

class GenderDetector:
    def __init__(self, confidence_threshold=0.5, use_cuda=True):
        """
        Initialize the gender detector
        
        Args:
            confidence_threshold (float): Threshold for person detection confidence
            use_cuda (bool): Whether to use CUDA if available
        """
        self.confidence_threshold = confidence_threshold
        
        # Set device (GPU if available and requested)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO model for person detection
        self._load_yolo()
        
        # Load Vision Transformer for gender classification
        self._load_vit()
        
        # Initialize FPS counter
        self.fps_counter = FPS(avarageof=10)
        
    def _load_yolo(self):
        """Load YOLOv4 model for person detection"""
        print("Loading YOLOv4 model...")
        
        # Get paths to model files
        weights_path = get_model_path("yolov4.weights")
        config_path = get_model_path("yolov4.cfg")
        
        # Check if files exist
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            raise FileNotFoundError(
                "YOLOv4 model files not found. Please run the download script first: "
                "python scripts/download_models.py --yolo"
            )
        
        # Load the network
        self.yolo = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Use CUDA if available
        if self.device.type == 'cuda':
            print("Using CUDA for YOLO inference")
            self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        # Get output layer names
        self.layer_names = self.yolo.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.yolo.getUnconnectedOutLayers()]
        
        # In COCO dataset, person is class 0
        self.person_class_id = 0
        
        print("YOLOv4 model loaded successfully")
    
    def _load_vit(self):
        """Load Vision Transformer model for gender classification"""
        print("Loading Vision Transformer model...")
        
        # Load processor and model from Hugging Face
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        
        # Modify the classification head for binary gender classification
        num_labels = 2  # Male and Female
        self.vit_model.classifier = torch.nn.Linear(self.vit_model.config.hidden_size, num_labels)
        
        # Set up class names
        self.id2label = {0: "Female", 1: "Male"}
        self.label2id = {"Female": 0, "Male": 1}
        
        # Move model to device
        self.vit_model.to(self.device)
        self.vit_model.eval()
        
        print("Vision Transformer model loaded successfully")
        print("NOTE: This model needs to be fine-tuned for gender classification to work properly.")
        print("      Currently it will produce random results since it's been initialized with random weights.")
        
    def detect_persons(self, frame):
        """
        Detect persons in the frame using YOLO
        
        Args:
            frame (numpy.ndarray): Input image in BGR format
            
        Returns:
            list: List of (x, y, w, h, confidence) tuples for each detected person
        """
        height, width = frame.shape[:2]
        
        # Create a blob from the image
        # Parameters: image, scale factor, size, mean subtraction, BGR2RGB conversion, crop
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            (416, 416), 
            swapRB=True, 
            crop=False
        )
        
        # Set the input to the network
        self.yolo.setInput(blob)
        
        # Forward pass
        outputs = self.yolo.forward(self.output_layers)
        
        # Process detections
        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for persons with confidence above threshold
                if class_id == self.person_class_id and confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate corner coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append((x, y, w, h, confidence))
        
        return boxes
    
    def classify_gender(self, frame, boxes):
        """
        Classify gender for each detected person
        
        Args:
            frame (numpy.ndarray): Input image in BGR format
            boxes (list): List of (x, y, w, h, confidence) tuples for detected persons
            
        Returns:
            list: List of dictionaries with detection results
        """
        results = []
        
        if not boxes:
            return results
        
        # Prepare batch of cropped person images
        person_images = []
        valid_boxes = []
        
        for box in boxes:
            x, y, w, h, conf = box
            
            # Ensure coordinates are within frame boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
                
            # Crop person from frame
            person_crop = frame[y:y+h, x:x+w]
            
            # Convert to PIL Image
            pil_image = cv2_to_pil(person_crop)
            
            person_images.append(pil_image)
            valid_boxes.append(box)
        
        if not person_images:
            return results
            
        # Process all images in batch
        inputs = self.vit_processor(person_images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        # Process results
        for i, (box, prediction, prob) in enumerate(zip(valid_boxes, predictions, probs)):
            x, y, w, h, conf = box
            gender = self.id2label[prediction.item()]
            confidence = prob[prediction].item()
            
            results.append({
                "box": (x, y, w, h),
                "gender": gender,
                "confidence": confidence,
                "detection_confidence": conf
            })
        
        return results
    
    def process_frame(self, frame):
        """
        Process a single frame for gender detection
        
        Args:
            frame (numpy.ndarray): Input image in BGR format
            
        Returns:
            numpy.ndarray: Processed frame with annotations
        """
        # Update FPS counter
        self.fps_counter.update()
        
        # Make a copy of the frame to avoid modifying the original
        processed_frame = frame.copy()
        
        # Detect persons
        person_boxes = self.detect_persons(frame)
        
        # Classify gender for each person
        results = self.classify_gender(frame, person_boxes)
        
        # Draw results on frame
        for result in results:
            x, y, w, h = result["box"]
            gender = result["gender"]
            confidence = result["confidence"]
            
            # Draw bounding box
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label background
            label = f"{gender}: {confidence:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                processed_frame, 
                (x, y - text_size[1] - 10), 
                (x + text_size[0], y), 
                (0, 255, 0), 
                cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                processed_frame, 
                label, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
        
        # Display FPS
        fps = self.fps_counter.get()
        cv2.putText(
            processed_frame, 
            f"FPS: {fps:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2
        )
        
        return processed_frame
    
    def run_on_video(self, video_source=0):
        """
        Run gender detection on video stream
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Display information
        print(f"Video size: {frame_width}x{frame_height}, FPS: {fps}")
        print("Press 'q' to quit")
        
        # Process video frames
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display the frame
                cv2.imshow('Gender Detection', processed_frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            print("Video stream closed")