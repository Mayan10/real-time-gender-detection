#!/usr/bin/env python
# Our main Flask server that handles gender detection requests

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

# Let's set up logging so we can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Allow any frontend to talk to our API (for development and deployment flexibility)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# Load up our AI models - this is where the magic happens!
try:
    # First, let's load YOLOv4 to detect people in images
    weights_path = 'models/yolov4.weights'
    config_path = 'models/yolov4.cfg'
    
    # If the model files aren't here, let's download them automatically
    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        logger.info("Hey! I need to download the YOLOv4 model files first...")
        import urllib.request
        # Grab the YOLOv4 weights (this is the big file with all the learned knowledge)
        url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
        urllib.request.urlretrieve(url, weights_path)
        # And the configuration file that tells us how to use it
        url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
        urllib.request.urlretrieve(url, config_path)
    
    # Now let's load the YOLO network into OpenCV
    yolo = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layer_names = yolo.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]
    person_class_id = 0  # In the COCO dataset, people are class 0
    logger.info("Great! YOLOv4 model is loaded and ready to detect people")
    
    # Now let's load the Vision Transformer for gender classification
    vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    
    # We need to modify the model to classify just two things: male and female
    num_labels = 2  # Just two options: male and female
    vit_model.classifier = torch.nn.Linear(vit_model.config.hidden_size, num_labels)
    
    # Set up our class names so we know what the model is predicting
    id2label = {0: "Female", 1: "Male"}
    label2id = {"Female": 0, "Male": 1}
    
    # Move the model to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model.to(device)
    vit_model.eval()
    logger.info("Awesome! Vision Transformer model is loaded and ready to classify gender")
    logger.warning("Heads up: This model needs to be fine-tuned for gender classification to work properly.")
    
except Exception as e:
    logger.error(f"Oops! Something went wrong loading the models: {str(e)}")
    logger.error(traceback.format_exc())
    yolo = None
    vit_model = None

def detect_persons(img):
    """Find all the people in an image using YOLOv4"""
    try:
        height, width = img.shape[:2]
        
        # Prepare the image for YOLO by creating a blob (this is how YOLO likes its input)
        blob = cv2.dnn.blobFromImage(
            img, 
            1/255.0,  # Scale the pixel values down
            (416, 416),  # Resize to YOLO's preferred size
            swapRB=True,  # Convert BGR to RGB
            crop=False
        )
        
        # Feed the image through the YOLO network
        yolo.setInput(blob)
        
        # Get the detection results
        outputs = yolo.forward(output_layers)
        
        # Process what YOLO found
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only keep detections of people with good confidence
                if class_id == person_class_id and confidence > 0.5:
                    # Convert YOLO's center coordinates to corner coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate the top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append((x, y, w, h))
                    confidences.append(float(confidence))
                    
        return boxes, confidences
    except Exception as e:
        logger.error(f"Something went wrong detecting people: {str(e)}")
        logger.error(traceback.format_exc())
        return [], []

def classify_gender(img, boxes):
    """Figure out the gender of each person we detected"""
    try:
        results = []
        
        if not boxes:
            return results
        
        # Get ready to process each person we found
        person_images = []
        valid_boxes = []
        
        for box in boxes:
            x, y, w, h = box
            
            # Make sure we don't go outside the image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
                
            # Cut out just the person from the image
            person_crop = img[y:y+h, x:x+w]
            
            # Convert to the format the Vision Transformer expects
            pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            
            person_images.append(pil_image)
            valid_boxes.append(box)
        
        if not person_images:
            return results
            
        # Process all the people at once (this is faster than one by one)
        inputs = vit_processor(person_images, return_tensors="pt").to(device)
        
        # Get the gender predictions
        with torch.no_grad():
            outputs = vit_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        # Package up the results nicely
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
        logger.error(f"Something went wrong classifying gender: {str(e)}")
        logger.error(traceback.format_exc())
        return []

@app.route('/')
def index():
    """Serve our simple HTML interface"""
    try:
        with open('camera.html', 'r') as file:
            html_content = file.read()
        return html_content
    except Exception as e:
        logger.error(f"Couldn't load the camera interface: {str(e)}")
        return "Error loading camera interface. Please check server logs."

@app.route('/health')
def health():
    """Let's check if everything is working properly"""
    return jsonify({
        "status": "healthy",
        "models_loaded": yolo is not None and vit_model is not None,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }), 200

@app.route('/test', methods=['GET'])
def test():
    """Simple test to make sure the API is responding"""
    return jsonify({"status": "ok", "message": "API is working"}), 200

@app.route('/api/detect', methods=['POST', 'OPTIONS'])
def detect_gender():
    """The main endpoint that processes images and detects gender"""
    if request.method == 'OPTIONS':
        return '', 204
        
    request_time = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"[{request_time}] Someone sent us an image to analyze!")
    
    if 'image' not in request.files:
        logger.warning("They forgot to include an image!")
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        file = request.files['image']
        logger.info(f"Got an image file: {file.filename}")
        
        # Convert the uploaded image to a format we can work with
        img = Image.open(file.stream)
        img = np.array(img)
        logger.info(f"Image converted to array, shape: {img.shape}")
        
        # OpenCV likes BGR format, so let's convert it
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        logger.info("Image converted to BGR format")
        
        if yolo is None or vit_model is None:
            logger.error("Our models aren't loaded yet!")
            return jsonify({'error': 'Models not available'}), 500
        
        # Step 1: Find all the people in the image
        logger.info("Looking for people in the image...")
        boxes, confidences = detect_persons(img)
        logger.info(f"Found {len(boxes)} people in the image")
        
        if len(boxes) > 0:
            # Step 2: Figure out the gender of each person
            logger.info("Now let's figure out everyone's gender...")
            results = classify_gender(img, boxes)
            logger.info(f"Gender classification done! Results: {results}")
            
            # Format the results nicely for the frontend
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
        error_msg = f"Something went wrong processing the image: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({
            'error': error_msg,
            'timestamp': request_time
        }), 500

@app.route('/api/camera')
def camera_stream():
    """Stream live camera feed with detection overlays"""
    try:
        # Try to open the camera
        cap = cv2.VideoCapture(0)  # Start with the default camera
        if not cap.isOpened():
            # If that doesn't work, try the next camera
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                return jsonify({'error': 'Could not open camera'}), 500

        def generate():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Find people in this frame
                boxes, confidences = detect_persons(frame)
                
                # Draw boxes around the people we found
                for box in boxes:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Figure out everyone's gender
                results = classify_gender(frame, boxes)
                
                # Add gender labels to the image
                for result in results:
                    x, y, w, h = result["box"]
                    gender = result["gender"]
                    confidence = result["confidence"]
                    label = f"{gender} ({confidence:.2f})"
                    cv2.putText(frame, label, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Convert the frame to JPEG format for streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                # Send the frame to the browser
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Take a small break to control the frame rate
                time.sleep(0.03)  # About 30 frames per second

        return Response(generate(),
                      mimetype='multipart/x-mixed-replace; boundary=frame')

    except Exception as e:
        logger.error(f"Something went wrong with the camera stream: {str(e)}")
        logger.error(traceback.format_exc())
        if 'cap' in locals():
            cap.release()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting up our Gender Detection API server...")
    
    # Get the port from environment variable (useful for deployment)
    port = int(os.environ.get('PORT', 5001))
    
    # Set debug mode based on environment
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Server starting on port {port}, debug={debug_mode}")
    app.run(port=port, debug=debug_mode, threaded=True, host='0.0.0.0') 