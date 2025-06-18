import React, { useState, useRef } from 'react'
import { Box, Text, Html, useGLTF } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'

function Model({ result }) {
  const headModel = useRef()
  
  useFrame((state) => {
    if (headModel.current) {
      headModel.current.rotation.y += 0.01
    }
  })

  return (
    <Box
      ref={headModel}
      args={[2, 2, 2]}
      scale={1}
    >
      <meshStandardMaterial
        color={result ? (result === 'male' ? '#4169e1' : '#ff69b4') : '#808080'}
        metalness={0.5}
        roughness={0.5}
      />
    </Box>
  )
}

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    const formData = new FormData()
    formData.append('image', file)

    setLoading(true)
    setError(null)

    try {
      const response = await fetch('http://localhost:5000/api/detect', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Detection failed')

      const data = await response.json()
      setResult(data.gender)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Html position={[-2, 3, 0]}>
        <div style={{ 
          background: 'rgba(0,0,0,0.7)', 
          padding: '20px', 
          borderRadius: '10px',
          color: 'white',
          width: '200px'
        }}>
          <h2>Gender Detection</h2>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            style={{ color: 'white' }}
          />
          {loading && <p>Processing...</p>}
          {error && <p style={{ color: 'red' }}>{error}</p>}
          {result && <p>Detected: {result}</p>}
        </div>
      </Html>

      <Model result={result} />
      
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
    </>
  )
} 