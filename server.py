#!/usr/bin/env python
# server.py

from flask import Flask, request, jsonify, Response, send_from_directory, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import logging
import traceback
import time
import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS to allow all origins
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# Initialize models
try:
    # Load YOLOv4 model
    weights_path = 'models/yolov4.weights'
    config_path = 'models/yolov4.cfg'
    
    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        logger.info("Downloading YOLOv4 model files...")
        import urllib.request
        # Download YOLOv4 weights
        url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
        urllib.request.urlretrieve(url, weights_path)
        # Download YOLOv4 config
        url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
        urllib.request.urlretrieve(url, config_path)
    
    yolo = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layer_names = yolo.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]
    person_class_id = 0  # In COCO dataset, person is class 0
    logger.info("YOLOv4 model loaded successfully")
    
    # Load Vision Transformer model
    vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    
    # Modify the classification head for binary gender classification
    num_labels = 2  # Male and Female
    vit_model.classifier = torch.nn.Linear(vit_model.config.hidden_size, num_labels)
    
    # Set up class names
    id2label = {0: "Female", 1: "Male"}
    label2id = {"Female": 0, "Male": 1}
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model.to(device)
    vit_model.eval()
    logger.info("Vision Transformer model loaded successfully")
    logger.warning("NOTE: This model needs to be fine-tuned for gender classification to work properly.")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    logger.error(traceback.format_exc())
    yolo = None
    vit_model = None

def detect_persons(img):
    try:
        height, width = img.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            img, 
            1/255.0, 
            (416, 416), 
            swapRB=True, 
            crop=False
        )
        
        # Set the input to the network
        yolo.setInput(blob)
        
        # Forward pass
        outputs = yolo.forward(output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for persons with confidence above threshold
                if class_id == person_class_id and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate corner coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append((x, y, w, h))
                    confidences.append(float(confidence))
                    
        return boxes, confidences
    except Exception as e:
        logger.error(f"Error in detect_persons: {str(e)}")
        logger.error(traceback.format_exc())
        return [], []

def classify_gender(img, boxes):
    try:
        results = []
        
        if not boxes:
            return results
        
        # Prepare batch of cropped person images
        person_images = []
        valid_boxes = []
        
        for box in boxes:
            x, y, w, h = box
            
            # Ensure coordinates are within frame boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
                
            # Crop person from frame
            person_crop = img[y:y+h, x:x+w]
            
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            
            person_images.append(pil_image)
            valid_boxes.append(box)
        
        if not person_images:
            return results
            
        # Process all images in batch
        inputs = vit_processor(person_images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = vit_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        # Process results
        for i, (box, prediction, prob) in enumerate(zip(valid_boxes, predictions, probs)):
            x, y, w, h = box
            gender = id2label[prediction.item()]
            confidence = prob[prediction].item()
            
            results.append({
                "box": (x, y, w, h),
                "gender": gender,
                "confidence": confidence
            })
        
        return results
    except Exception as e:
        logger.error(f"Error in classify_gender: {str(e)}")
        logger.error(traceback.format_exc())
        return []

@app.route('/')
def index():
    try:
        with open('camera.html', 'r') as file:
            html_content = file.read()
        return html_content
    except Exception as e:
        logger.error(f"Error serving camera.html: {str(e)}")
        return "Error loading camera interface. Please check server logs."

@app.route('/health')
def health():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "models_loaded": yolo is not None and vit_model is not None,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }), 200

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "ok", "message": "API is working"}), 200

@app.route('/api/detect', methods=['POST', 'OPTIONS'])
def detect_gender():
    if request.method == 'OPTIONS':
        return '', 204
        
    request_time = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"[{request_time}] New detection request received")
    
    if 'image' not in request.files:
        logger.warning("No image file in request")
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        file = request.files['image']
        logger.info(f"Received image file: {file.filename}")
        
        # Convert image file to OpenCV format
        img = Image.open(file.stream)
        img = np.array(img)
        logger.info(f"Image converted to array, shape: {img.shape}")
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        logger.info("Image converted to BGR format")
        
        if yolo is None or vit_model is None:
            logger.error("Models not loaded")
            return jsonify({'error': 'Models not available'}), 500
        
        # Detect persons
        logger.info("Starting person detection...")
        boxes, confidences = detect_persons(img)
        logger.info(f"Person detection completed. Found {len(boxes)} persons")
        
        if len(boxes) > 0:
            # Classify gender for each person
            logger.info("Starting gender classification...")
            results = classify_gender(img, boxes)
            logger.info(f"Gender classification completed. Results: {results}")
            
            # Format results
            formatted_results = []
            for result in results:
                x, y, w, h = result["box"]
                formatted_results.append({
                    "box": [x, y, w, h],
                    "gender": result["gender"],
                    "confidence": float(result["confidence"]),
                    "timestamp": request_time
                })
            
            return jsonify({
                "results": formatted_results,
                "total_persons": len(formatted_results)
            })
        else:
            return jsonify({
                "results": [],
                "total_persons": 0,
                "timestamp": request_time
            })
            
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({
            'error': error_msg,
            'timestamp': request_time
        }), 500

@app.route('/api/camera')
def camera_stream():
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)  # Try default camera
        if not cap.isOpened():
            # If default camera fails, try alternative camera index
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                return jsonify({'error': 'Could not open camera'}), 500

        def generate():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect persons
                boxes, confidences = detect_persons(frame)
                
                # Draw boxes and labels
                for box in boxes:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Classify gender for each person
                results = classify_gender(frame, boxes)
                
                # Draw gender labels
                for result in results:
                    x, y, w, h = result["box"]
                    gender = result["gender"]
                    confidence = result["confidence"]
                    label = f"{gender} ({confidence:.2f})"
                    cv2.putText(frame, label, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Encode the frame in JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                # Convert to bytes and yield
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Small delay to control frame rate
                time.sleep(0.03)  # Approximately 30 FPS

        return Response(generate(),
                      mimetype='multipart/x-mixed-replace; boundary=frame')

    except Exception as e:
        logger.error(f"Error in camera stream: {str(e)}")
        logger.error(traceback.format_exc())
        if 'cap' in locals():
            cap.release()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Gender Detection API server...")
    
    # Get port from environment variable (for production deployment)
    port = int(os.environ.get('PORT', 5001))
    
    # Set debug mode based on environment
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Server starting on port {port}, debug={debug_mode}")
    app.run(port=port, debug=debug_mode, threaded=True, host='0.0.0.0') 