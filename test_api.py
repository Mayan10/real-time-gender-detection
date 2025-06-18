#!/usr/bin/env python
# A simple test script to make sure our API is working properly

import requests
import json

def test_backend_connection():
    """Let's see if our backend is running and responding to requests"""
    try:
        response = requests.get('http://localhost:5001/test')
        if response.status_code == 200:
            print("✅ Great! Backend connection is working")
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
    """Check if our models are loaded and everything is healthy"""
    try:
        response = requests.get('http://localhost:5001/health')
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
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
    print("Testing our Gender Detection API...")
    print("=" * 40)
    
    # First, let's test if the backend is running
    if test_backend_connection():
        # Then check if the models are loaded properly
        models_loaded = test_health_endpoint()
        
        if models_loaded:
            print("\n🎉 Awesome! All tests passed! The API is ready to use.")
            print("\nYou can now:")
            print("1. Open http://localhost:5001 in your browser for the simple interface")
            print("2. Start the React frontend with: cd frontend && npm start")
            print("3. Use the API endpoint: POST http://localhost:5001/api/detect")
        else:
            print("\n⚠️  The backend is running but the models aren't loaded properly.")
            print("Check the server logs for model loading errors.")
    else:
        print("\n❌ The backend isn't running.")
        print("Start the server with: python server.py") 