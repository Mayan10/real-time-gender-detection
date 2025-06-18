# Real-time Gender Detection Web Application

A modern web application that performs real-time gender detection using computer vision and deep learning. The application uses YOLOv4 for person detection and a Vision Transformer (ViT) for gender classification.

## Features

- **Real-time Video Processing**: Live camera feed with real-time gender detection
- **Modern Web Interface**: React-based frontend with responsive design
- **RESTful API**: Flask backend with CORS support for easy integration
- **Multiple Detection**: Can detect and classify multiple persons simultaneously
- **Confidence Scores**: Provides confidence levels for each detection
- **Cross-platform**: Works on desktop and mobile browsers

## Architecture

### Backend (Flask API)
- **Person Detection**: YOLOv4 model for detecting persons in images
- **Gender Classification**: Vision Transformer (ViT) for gender classification
- **RESTful Endpoints**: `/api/detect` for image processing, `/health` for monitoring
- **CORS Enabled**: Supports cross-origin requests from any frontend

### Frontend (React)
- **Live Video Feed**: Real-time camera access and processing
- **Detection Overlay**: Visual bounding boxes and labels on detected persons
- **Results Panel**: Detailed information about each detected person
- **Responsive Design**: Works on desktop and mobile devices

## Models Used

1. **YOLOv4** (`models/yolov4.weights`, `models/yolov4.cfg`)
   - Purpose: Person detection
   - Input: Images/video frames
   - Output: Bounding boxes around detected persons

2. **Vision Transformer (ViT)** (`google/vit-base-patch16-224`)
   - Purpose: Gender classification
   - Input: Cropped person images
   - Output: Gender prediction (Male/Female) with confidence scores

**Note**: The ViT model currently uses random weights for the gender classification head. For accurate results, you need to fine-tune it on a gender classification dataset.

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+
- Webcam access
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mayan10/real-time-gender-detection.git
   cd real-time-gender-detection
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Download required model files**:
   
   **Important**: The large model files are not included in this repository due to GitHub's file size limits. You need to download them separately:
   
   ```bash
   # Download YOLOv4 weights (245MB)
   python scripts/download_models.py --yolo
   
   # Or download manually:
   # YOLOv4 weights: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
   # Face detection model: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
   ```
   
   **Alternative manual download**:
   - Download `yolov4.weights` (245MB) from [YOLOv4 releases](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
   - Download `face_detection_yunet_2023mar.onnx` from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)
   - Place both files in the `models/` directory

### Running Locally

1. **Start the Flask backend**:
   ```bash
   python server.py
   ```
   The server will start on `http://localhost:5001`

2. **Start the React frontend** (in a new terminal):
   ```bash
   cd frontend
   npm start
   ```
   The frontend will start on `http://localhost:5173`

3. **Test the API**:
   ```bash
   python test_api.py
   ```

4. **Access the application**:
   - **Simple interface**: Open `http://localhost:5001` in your browser
   - **React interface**: Open `http://localhost:5173` in your browser

### Usage

1. **Allow camera access** when prompted by your browser
2. **Position yourself** in front of the camera
3. **View real-time detection** with bounding boxes and gender labels
4. **Check the results panel** for detailed information about detected persons

## API Reference

### Endpoints

#### `GET /test`
Test if the API is running.
```bash
curl http://localhost:5001/test
```

#### `GET /health`
Health check with model status.
```bash
curl http://localhost:5001/health
```

#### `POST /api/detect`
Detect persons and classify gender in an image.

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `image` field containing the image file

**Response**:
```json
{
  "results": [
    {
      "box": [x, y, width, height],
      "gender": "Male",
      "confidence": 0.85,
      "timestamp": "2024-01-01 12:00:00"
    }
  ],
  "total_persons": 1
}
```

**Example**:
```bash
curl -X POST -F "image=@test-image.jpg" http://localhost:5001/api/detect
```

#### `GET /api/camera`
Stream camera feed with detection overlays (for simple HTML interface).

## Deployment

### Option 1: Render (Recommended)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

**Quick deployment to Render**:

1. **Backend**:
   - Connect your GitHub repo to Render
   - Create a new Web Service
   - Build command: `pip install -r requirements.txt`
   - Start command: `python server.py`

2. **Frontend**:
   - Build: `cd frontend && npm run build`
   - Deploy to Netlify or Vercel
   - Update API endpoint in `App.jsx`

### Option 2: Heroku

1. **Install Heroku CLI**
2. **Deploy backend**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```
3. **Deploy frontend** to Netlify or similar

### Option 3: AWS

1. **Backend**: Use Elastic Beanstalk
2. **Frontend**: Use S3 + CloudFront

## Development

### Project Structure

```
gender-detection-project/
├── server.py              # Flask backend server
├── main.py                # Command-line interface
├── requirements.txt       # Python dependencies
├── frontend/              # React frontend
│   ├── src/
│   │   └── App.jsx       # Main React component
│   ├── package.json
│   └── vite.config.js
├── models/                # Model files
│   ├── yolov4.weights
│   ├── yolov4.cfg
│   └── coco.names
├── scripts/               # Utility scripts
├── data/                  # Training data (if any)
└── camera.html           # Simple HTML interface
```

### Adding New Features

1. **Backend**: Add new routes in `server.py`
2. **Frontend**: Modify `frontend/src/App.jsx`
3. **Models**: Add new model files to `models/` directory

### Fine-tuning the ViT Model

To improve gender classification accuracy:

1. **Prepare dataset** in `data/` directory
2. **Run training script** (see `scripts/train.py`)
3. **Update model loading** in `server.py`

## Troubleshooting

### Common Issues

1. **Camera not working**:
   - Check browser permissions
   - Try different camera index in `server.py`
   - Ensure no other app is using the camera

2. **Models not loading**:
   - Run `python scripts/download_models.py --yolo`
   - Check file permissions in `models/` directory
   - Verify model file sizes

3. **CORS errors**:
   - Backend has CORS enabled for all origins
   - Check if frontend URL is correct
   - Verify API endpoint in React app

4. **Slow performance**:
   - Reduce detection frequency in React app
   - Use smaller input image sizes
   - Consider GPU acceleration

### Debug Mode

Enable debug mode for development:
```bash
export FLASK_ENV=development
python server.py
```

### Logs

Check server logs for detailed error information:
```bash
python server.py 2>&1 | tee server.log
```

## Performance

### Optimization Tips

1. **Reduce detection frequency** (currently 1 second)
2. **Use smaller input images** for faster processing
3. **Enable GPU acceleration** if available
4. **Implement caching** for repeated detections

### Benchmarks

- **Person Detection**: ~100ms per frame (CPU)
- **Gender Classification**: ~50ms per person (CPU)
- **Total Pipeline**: ~150-200ms per frame

## Security Considerations

1. **Input Validation**: Images are validated before processing
2. **Rate Limiting**: Consider implementing rate limiting for production
3. **HTTPS**: Always use HTTPS in production
4. **API Keys**: Consider adding authentication for public deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv4 model from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- Vision Transformer from [Hugging Face](https://huggingface.co/google/vit-base-patch16-224)
- React and Flask communities for excellent documentation

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review server logs
3. Open an issue on GitHub
4. Check the deployment guide for hosting issues
