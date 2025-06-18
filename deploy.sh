#!/bin/bash

# Gender Detection App Deployment Script
# This script helps deploy the backend to Render and frontend to Netlify

echo "🚀 Starting deployment process..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "server.py" ] || [ ! -d "frontend" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_status "Building frontend for production..."

# Build the React frontend
cd frontend
if npm run build; then
    print_status "Frontend build successful!"
else
    print_error "Frontend build failed!"
    exit 1
fi
cd ..

print_status "Deployment preparation complete!"

echo ""
echo "📋 Next steps for deployment:"
echo ""
echo "1. 🎯 BACKEND DEPLOYMENT (Render):"
echo "   - Go to https://render.com"
echo "   - Create a new Web Service"
echo "   - Connect your GitHub repository"
echo "   - Set build command: pip install -r requirements.txt"
echo "   - Set start command: python server.py"
echo "   - Deploy and note the URL (e.g., https://your-app.onrender.com)"
echo ""
echo "2. 🎨 FRONTEND DEPLOYMENT (Netlify):"
echo "   - Go to https://netlify.com"
echo "   - Drag and drop the 'frontend/dist' folder"
echo "   - Or connect your GitHub repo and set build command: npm run build"
echo "   - Update the API URL in frontend/env.production with your Render URL"
echo ""
echo "3. 🔧 CONFIGURATION:"
echo "   - Update REACT_APP_API_URL in frontend/env.production"
echo "   - Redeploy frontend after updating the API URL"
echo ""
echo "4. ✅ TESTING:"
echo "   - Test the deployed application"
echo "   - Check that camera access works"
echo "   - Verify gender detection is working"
echo ""

print_status "Deployment script completed!" 