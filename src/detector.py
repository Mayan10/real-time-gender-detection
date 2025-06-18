#!/usr/bin/env python
# Our main gender detection class that combines YOLO and Vision Transformer

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
        Let's set up our gender detector with all the AI models we need
        
        Args:
            confidence_threshold (float): How confident we need to be to detect a person
            use_cuda (bool): Whether to use GPU acceleration if available
        """
        self.confidence_threshold = confidence_threshold
        
        # Figure out whether to use GPU or CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load up our AI models
        self._load_yolo()
        self._load_vit()
        
        # Set up our FPS counter to see how fast we're running
        self.fps_counter = FPS(avarageof=10)
        
    def _load_yolo(self):
        """Load up YOLOv4 to detect people in images"""
        print("Loading YOLOv4 model...")
        
        # Get the paths to our model files
        weights_path = get_model_path("yolov4.weights")
        config_path = get_model_path("yolov4.cfg")
        
        # Make sure the files are actually there
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            raise FileNotFoundError(
                "YOLOv4 model files not found. Please run the download script first: "
                "python scripts/download_models.py --yolo"
            )
        
        # Load the YOLO network into OpenCV
        self.yolo = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # If we have a GPU, let's use it for faster processing
        if self.device.type == 'cuda':
            print("Using CUDA for YOLO inference")
            self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        # Get the names of the output layers we need
        self.layer_names = self.yolo.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.yolo.getUnconnectedOutLayers()]
        
        # In the COCO dataset, people are class 0
        self.person_class_id = 0
        
        print("YOLOv4 model loaded successfully")
    
    def _load_vit(self):
        """Load up the Vision Transformer to classify gender"""
        print("Loading Vision Transformer model...")
        
        # Load the processor and model from Hugging Face
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        
        # We need to modify the model to classify just two things: male and female
        num_labels = 2  # Just two options: male and female
        self.vit_model.classifier = torch.nn.Linear(self.vit_model.config.hidden_size, num_labels)
        
        # Set up our class names so we know what the model is predicting
        self.id2label = {0: "Female", 1: "Male"}
        self.label2id = {"Female": 0, "Male": 1}
        
        # Move the model to GPU if available, otherwise use CPU
        self.vit_model.to(self.device)
        self.vit_model.eval()
        
        print("Vision Transformer model loaded successfully")
        print("NOTE: This model needs to be fine-tuned for gender classification to work properly.")
        print("      Currently it will produce random results since it's been initialized with random weights.")
        
    def detect_persons(self, frame):
        """
        Find all the people in an image using YOLOv4
        
        Args:
            frame (numpy.ndarray): Input image in BGR format
            
        Returns:
            list: List of (x, y, w, h, confidence) tuples for each detected person
        """
        height, width = frame.shape[:2]
        
        # Prepare the image for YOLO by creating a blob (this is how YOLO likes its input)
        # Parameters: image, scale factor, size, mean subtraction, BGR2RGB conversion, crop
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0,  # Scale the pixel values down
            (416, 416),  # Resize to YOLO's preferred size
            swapRB=True,  # Convert BGR to RGB
            crop=False
        )
        
        # Feed the image through the YOLO network
        self.yolo.setInput(blob)
        
        # Get the detection results
        outputs = self.yolo.forward(self.output_layers)
        
        # Process what YOLO found
        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only keep detections of people with good confidence
                if class_id == self.person_class_id and confidence > self.confidence_threshold:
                    # Convert YOLO's center coordinates to corner coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate the top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append((x, y, w, h, confidence))
        
        return boxes
    
    def classify_gender(self, frame, boxes):
        """
        Figure out the gender of each person we detected
        
        Args:
            frame (numpy.ndarray): Input image in BGR format
            boxes (list): List of (x, y, w, h, confidence) tuples for detected persons
            
        Returns:
            list: List of dictionaries with detection results
        """
        results = []
        
        if not boxes:
            return results
        
        # Get ready to process each person we found
        person_images = []
        valid_boxes = []
        
        for box in boxes:
            x, y, w, h, conf = box
            
            # Make sure we don't go outside the image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
                
            # Cut out just the person from the image
            person_crop = frame[y:y+h, x:x+w]
            
            # Convert to the format the Vision Transformer expects
            pil_image = cv2_to_pil(person_crop)
            
            person_images.append(pil_image)
            valid_boxes.append(box)
        
        if not person_images:
            return results
            
        # Process all the people at once (this is faster than one by one)
        inputs = self.vit_processor(person_images, return_tensors="pt").to(self.device)
        
        # Get the gender predictions
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        # Package up the results nicely
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
        # Update our FPS counter
        self.fps_counter.update()
        
        # Make a copy of the frame so we don't modify the original
        processed_frame = frame.copy()
        
        # Find all the people in the frame
        person_boxes = self.detect_persons(frame)
        
        # Figure out everyone's gender
        results = self.classify_gender(frame, person_boxes)
        
        # Draw the results on the frame
        for result in results:
            x, y, w, h = result["box"]
            gender = result["gender"]
            confidence = result["confidence"]
            
            # Draw a box around the person
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw a background for the label
            label = f"{gender}: {confidence:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                processed_frame, 
                (x, y - text_size[1] - 10), 
                (x + text_size[0], y), 
                (0, 255, 0), 
                cv2.FILLED
            )
            
            # Draw the gender label
            cv2.putText(
                processed_frame, 
                label, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
        
        # Show the current FPS
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
        Run gender detection on a video stream
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Oops! Could not open video source {video_source}")
            return
        
        # Get some info about the video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Show some helpful information
        print(f"Video size: {frame_width}x{frame_height}, FPS: {fps}")
        print("Press 'q' to quit")
        
        # Process each frame of the video
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Process this frame
                processed_frame = self.process_frame(frame)
                
                # Show the frame
                cv2.imshow('Gender Detection', processed_frame)
                
                # Exit if they press 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Clean up when we're done
            cap.release()
            cv2.destroyAllWindows()
            print("Video stream closed")