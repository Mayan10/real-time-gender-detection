# Deployment Guide for Gender Detection Web Service

This guide will help you deploy your gender detection application to make it publicly accessible as a web service.

## Overview

The application consists of two parts:
1. **Backend**: Flask API server (`server.py`) that handles gender detection
2. **Frontend**: React web app (`frontend/`) that provides the user interface

## Prerequisites

- Python 3.9+
- Node.js 16+
- Git
- A cloud hosting account (Render, Heroku, AWS, etc.)

## Option 1: Deploy to Render (Recommended)

### Backend Deployment (Flask API)

1. **Create a Render account** at [render.com](https://render.com)

2. **Create a new Web Service**:
   - Connect your GitHub repository
   - Choose "Python" as the runtime
   - Set the build command: `pip install -r requirements.txt`
   - Set the start command: `python server.py`
   - Set environment variables:
     - `PORT`: 10000 (Render will set this automatically)

3. **Update server.py for production**:
   ```python
   if __name__ == '__main__':
       port = int(os.environ.get('PORT', 5001))
       app.run(host='0.0.0.0', port=port, debug=False)
   ```

4. **Deploy**: Render will automatically deploy your app

### Frontend Deployment (React App)

1. **Build the React app**:
   ```bash
   cd frontend
   npm install
   npm run build
   ```

2. **Deploy to Netlify**:
   - Go to [netlify.com](https://netlify.com)
   - Drag and drop the `frontend/dist` folder
   - Or connect your GitHub repo and set build command: `npm run build`

3. **Update API endpoint**:
   - In `frontend/src/App.jsx`, change `http://localhost:5001` to your Render backend URL
   - Example: `https://your-app-name.onrender.com`

## Option 2: Deploy to Heroku

### Backend Deployment

1. **Install Heroku CLI** and login:
   ```bash
   heroku login
   ```

2. **Create Heroku app**:
   ```bash
   heroku create your-gender-detection-app
   ```

3. **Deploy**:
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

4. **Scale the app**:
   ```bash
   heroku ps:scale web=1
   ```

### Frontend Deployment

Same as Render option, but deploy to Heroku or Netlify.

## Option 3: Deploy to AWS

### Backend (AWS Elastic Beanstalk)

1. **Install AWS CLI and EB CLI**

2. **Initialize EB application**:
   ```bash
   eb init
   eb create gender-detection-api
   ```

3. **Deploy**:
   ```bash
   eb deploy
   ```

### Frontend (AWS S3 + CloudFront)

1. **Build React app**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Upload to S3** and configure CloudFront for CDN

## Environment Variables

Set these environment variables in your deployment platform:

- `PORT`: Port number (usually set automatically)
- `FLASK_ENV`: `production`
- `PYTHONPATH`: `.`

## CORS Configuration

The Flask backend already has CORS enabled for all origins. If you need to restrict it:

```python
# In server.py
CORS(app, resources={r"/*": {"origins": ["https://your-frontend-domain.com"]}})
```

## Model Files

Make sure your model files are included in the deployment:
- `models/yolov4.weights`
- `models/yolov4.cfg`
- `models/coco.names`

## Testing the Deployment

1. **Test the backend API**:
   ```bash
   curl -X POST -F "image=@test-image.jpg" https://your-backend-url/api/detect
   ```

2. **Test the frontend**:
   - Open your frontend URL
   - Allow camera access
   - Check if detection works

## Troubleshooting

### Common Issues

1. **Model files not found**:
   - Ensure all model files are in the `models/` directory
   - Check file permissions

2. **CORS errors**:
   - Verify CORS configuration in `server.py`
   - Check if frontend URL is allowed

3. **Memory issues**:
   - YOLOv4 model requires significant memory
   - Consider using a larger instance type

4. **Timeout issues**:
   - Detection can take time
   - Increase timeout settings in your hosting platform

### Logs

Check application logs:
- **Render**: Dashboard → Your app → Logs
- **Heroku**: `heroku logs --tail`
- **AWS**: EB Console → Environment → Logs

## Security Considerations

1. **Rate limiting**: Implement rate limiting to prevent abuse
2. **Input validation**: Validate uploaded images
3. **HTTPS**: Always use HTTPS in production
4. **API keys**: Consider adding API key authentication for production use

## Cost Optimization

1. **Use appropriate instance sizes**
2. **Implement auto-scaling**
3. **Use CDN for static assets**
4. **Consider serverless options** (AWS Lambda, Vercel Functions)

## Monitoring

1. **Set up health checks**:
   ```python
   @app.route('/health')
   def health():
       return jsonify({"status": "healthy"}), 200
   ```

2. **Add logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

3. **Monitor performance** and set up alerts

## Next Steps

1. **Fine-tune the ViT model** for better gender classification accuracy
2. **Add authentication** for production use
3. **Implement caching** for better performance
4. **Add analytics** to track usage
5. **Set up CI/CD** for automatic deployments 