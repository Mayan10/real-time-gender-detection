import React, { useState, useRef, useEffect } from 'react'

// Where our backend API lives - change this when you deploy to production
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001'

export default function App() {
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isDetecting, setIsDetecting] = useState(false)
  const [backendStatus, setBackendStatus] = useState('Checking...')
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const canvasRef = useRef(document.createElement('canvas'))
  const contextRef = useRef(null)
  const detectingRef = useRef(false)

  // Set up our canvas for capturing video frames
  useEffect(() => {
    canvasRef.current.width = 640
    canvasRef.current.height = 480
    contextRef.current = canvasRef.current.getContext('2d')
  }, [])

  // Let's check if our backend is running
  useEffect(() => {
    fetch(`${API_BASE_URL}/test`)
      .then(response => response.json())
      .then(data => {
        console.log('Backend connection test successful:', data)
        setBackendStatus('Connected')
      })
      .catch(err => {
        console.error('Backend connection test failed:', err)
        setBackendStatus('Connection Failed')
      })
  }, [])

  const captureFrame = async () => {
    if (!videoRef.current || !contextRef.current) return null

    try {
      // Draw the current video frame onto our canvas
      contextRef.current.drawImage(videoRef.current, 0, 0, 640, 480)
      return await new Promise((resolve) => {
        canvasRef.current.toBlob(resolve, 'image/jpeg', 0.8)
      })
    } catch (err) {
      console.error('Frame capture error:', err)
      return null
    }
  }

  const detectGender = async () => {
    if (!videoRef.current || !isStreaming || !detectingRef.current) {
      return
    }

    try {
      // Make sure the video is actually playing
      if (videoRef.current.readyState === 4 && !videoRef.current.paused) {
        const blob = await captureFrame()
        if (!blob) {
          throw new Error('Failed to capture frame')
        }

        // Prepare the image data to send to our backend
        const formData = new FormData()
        formData.append('image', blob)

        // Send the image to our backend for analysis
        const response = await fetch(`${API_BASE_URL}/api/detect`, {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          throw new Error(`Detection failed: ${response.status} ${response.statusText}`)
        }

        const data = await response.json()
        console.log('Detection result:', data)
        
        // Update our results with what the backend found
        setResults(data.results || [])
      }
    } catch (err) {
      console.error('Detection error:', err)
      setError(err.message)
    }

    // Schedule the next detection (we do this every second)
    if (detectingRef.current) {
      setTimeout(() => {
        if (detectingRef.current) {
          detectGender()
        }
      }, 1000) // Detect every second
    }
  }

  const startGenderDetection = () => {
    if (detectingRef.current) {
      return
    }

    console.log('Starting gender detection')
    detectingRef.current = true
    setIsDetecting(true)
    setError(null)
    detectGender()
  }

  const stopGenderDetection = () => {
    console.log('Stopping gender detection')
    detectingRef.current = false
    setIsDetecting(false)
  }

  const startCamera = async () => {
    try {
      console.log('Requesting camera access...')

      // Ask the browser for permission to use the camera
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640,
          height: 480,
          facingMode: "user"  // Use the front-facing camera
        } 
      })

      console.log('Camera access granted')
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        
        // Wait for the video to be ready to play
        videoRef.current.onloadedmetadata = () => {
          console.log('Video metadata loaded')
          videoRef.current.play()
            .then(() => {
              console.log('Video playback started')
              setIsStreaming(true)
              // Start detection after making sure the video is playing
              setTimeout(() => {
                if (videoRef.current && !videoRef.current.paused) {
                  startGenderDetection()
                }
              }, 1000)
            })
            .catch(err => {
              console.error('Video playback error:', err)
              setError("Video playback failed: " + err.message)
            })
        }

        videoRef.current.onerror = (err) => {
          console.error('Video error:', err)
          setError("Video error: " + (err.message || 'Unknown error'))
        }
      } else {
        throw new Error('Video element not initialized')
      }
    } catch (err) {
      console.error('Camera access error:', err)
      setError("Could not access camera: " + err.message)
    }
  }

  const stopCamera = () => {
    stopGenderDetection()
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsStreaming(false)
    setResults([])
  }

  // Start the camera when our component first loads
  useEffect(() => {
    console.log('Component mounted, starting camera...')
    startCamera()
    return () => {
      console.log('Component unmounting, cleaning up...')
      detectingRef.current = false
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  return (
    <div style={{ 
      maxWidth: '1200px', 
      margin: '0 auto', 
      padding: '20px',
      fontFamily: 'Arial, sans-serif'
    }}>
      <h1 style={{ textAlign: 'center', color: '#333', marginBottom: '30px' }}>
        Real-time Gender Detection
      </h1>
      
      <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
        {/* Video Feed */}
        <div style={{ flex: '1' }}>
          <div style={{ 
            border: '2px solid #333', 
            borderRadius: '8px', 
            overflow: 'hidden',
            backgroundColor: '#000',
            position: 'relative'
          }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{
                width: '100%',
                height: 'auto',
                display: 'block'
              }}
            />
            
            {/* Overlay for detection results */}
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none'
            }}>
              {results.map((result, index) => {
                const [x, y, w, h] = result.box
                const scaleX = videoRef.current ? videoRef.current.offsetWidth / 640 : 1
                const scaleY = videoRef.current ? videoRef.current.offsetHeight / 480 : 1
                
                return (
                  <div key={index}>
                    {/* Bounding box */}
                    <div style={{
                      position: 'absolute',
                      left: x * scaleX,
                      top: y * scaleY,
                      width: w * scaleX,
                      height: h * scaleY,
                      border: '2px solid #00ff00',
                      backgroundColor: 'rgba(0, 255, 0, 0.1)'
                    }} />
                    
                    {/* Label */}
                    <div style={{
                      position: 'absolute',
                      left: x * scaleX,
                      top: (y * scaleY) - 25,
                      backgroundColor: result.gender === 'Male' ? '#4169e1' : '#ff69b4',
                      color: 'white',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      fontSize: '12px',
                      fontWeight: 'bold'
                    }}>
                      {result.gender} ({result.confidence.toFixed(2)})
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
          
          {/* Status and Controls */}
          <div style={{ marginTop: '15px', textAlign: 'center' }}>
            <div style={{ marginBottom: '10px' }}>
              <span style={{ 
                padding: '5px 10px', 
                borderRadius: '4px', 
                backgroundColor: backendStatus === 'Connected' ? '#4CAF50' : '#f44336',
                color: 'white',
                fontSize: '14px'
              }}>
                Backend: {backendStatus}
              </span>
              <span style={{ 
                marginLeft: '10px',
                padding: '5px 10px', 
                borderRadius: '4px', 
                backgroundColor: isStreaming ? '#4CAF50' : '#ff9800',
                color: 'white',
                fontSize: '14px'
              }}>
                Camera: {isStreaming ? 'Active' : 'Starting...'}
              </span>
              <span style={{ 
                marginLeft: '10px',
                padding: '5px 10px', 
                borderRadius: '4px', 
                backgroundColor: isDetecting ? '#4CAF50' : '#9e9e9e',
                color: 'white',
                fontSize: '14px'
              }}>
                Detection: {isDetecting ? 'Running' : 'Stopped'}
              </span>
            </div>
            
            <div>
              <button 
                onClick={isDetecting ? stopGenderDetection : startGenderDetection}
                style={{
                  padding: '10px 20px',
                  fontSize: '16px',
                  backgroundColor: isDetecting ? '#f44336' : '#4CAF50',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  marginRight: '10px'
                }}
              >
                {isDetecting ? 'Stop Detection' : 'Start Detection'}
              </button>
              
              <button 
                onClick={stopCamera}
                style={{
                  padding: '10px 20px',
                  fontSize: '16px',
                  backgroundColor: '#ff9800',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Stop Camera
              </button>
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div style={{ 
          flex: '1', 
          backgroundColor: '#f5f5f5', 
          padding: '20px', 
          borderRadius: '8px',
          minHeight: '400px'
        }}>
          <h3 style={{ marginTop: 0, color: '#333' }}>Detection Results</h3>
          
          {error && (
            <div style={{ 
              backgroundColor: '#ffebee', 
              color: '#c62828', 
              padding: '10px', 
              borderRadius: '4px',
              marginBottom: '15px'
            }}>
              Error: {error}
            </div>
          )}
          
          {results.length === 0 ? (
            <p style={{ color: '#666', fontStyle: 'italic' }}>
              No persons detected yet. Make sure you're visible in the camera.
            </p>
          ) : (
            <div>
              <p style={{ marginBottom: '15px' }}>
                <strong>Total persons detected: {results.length}</strong>
              </p>
              
              {results.map((result, index) => (
                <div key={index} style={{
                  backgroundColor: 'white',
                  padding: '15px',
                  borderRadius: '4px',
                  marginBottom: '10px',
                  borderLeft: `4px solid ${result.gender === 'Male' ? '#4169e1' : '#ff69b4'}`
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <h4 style={{ 
                        margin: '0 0 5px 0', 
                        color: result.gender === 'Male' ? '#4169e1' : '#ff69b4'
                      }}>
                        Person {index + 1}
                      </h4>
                      <p style={{ margin: '0', fontSize: '14px', color: '#666' }}>
                        Position: ({result.box[0]}, {result.box[1]}) - {result.box[2]}x{result.box[3]}
                      </p>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ 
                        fontSize: '18px', 
                        fontWeight: 'bold',
                        color: result.gender === 'Male' ? '#4169e1' : '#ff69b4'
                      }}>
                        {result.gender}
                      </div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        Confidence: {(result.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 