#!/usr/bin/env python
# test_api.py

import requests
import json

def test_backend_connection():
    """Test if the backend is running and responding"""
    try:
        response = requests.get('http://localhost:5001/test')
        if response.status_code == 200:
            print("✅ Backend connection successful")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Backend connection failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend connection failed: Could not connect to server")
        return False
    except Exception as e:
        print(f"❌ Backend connection failed: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get('http://localhost:5001/health')
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check successful")
            print(f"Status: {data['status']}")
            print(f"Models loaded: {data['models_loaded']}")
            return data['models_loaded']
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Gender Detection API...")
    print("=" * 40)
    
    # Test backend connection
    if test_backend_connection():
        # Test health endpoint
        models_loaded = test_health_endpoint()
        
        if models_loaded:
            print("\n🎉 All tests passed! The API is ready to use.")
            print("\nYou can now:")
            print("1. Open http://localhost:5001 in your browser for the simple interface")
            print("2. Start the React frontend with: cd frontend && npm start")
            print("3. Use the API endpoint: POST http://localhost:5001/api/detect")
        else:
            print("\n⚠️  Backend is running but models are not loaded properly.")
            print("Check the server logs for model loading errors.")
    else:
        print("\n❌ Backend is not running.")
        print("Start the server with: python server.py") 