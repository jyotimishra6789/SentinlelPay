// src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const simulateBehavior = () => {
    setLoading(true);
    setResult(null);

    const behaviorData = {
      keystroke_delay: Math.random() * 100 + 50,
      mouse_speed: Math.random() * 300 + 100,
      swipe_velocity: Math.random() * 100 + 30,
      erratic_moves: Math.floor(Math.random() * 10) + 1,
    };

    axios.post('http://localhost:5000/predict', behaviorData)
      .then((res) => {
        setResult(res.data.trust_score);
        setLoading(false);
      })
      .catch((err) => {
        console.error("API Error:", err);
        setResult("API Error");
        setLoading(false);
      });
  };

  return (
    <div className="App" style={{ padding: '2rem', fontFamily: 'Arial' }}>
      <h1>🛡️ SentinelPay - AI Trust Detector</h1>
      <p>Simulate user biometric + behavioral data to detect trustworthiness in real-time.</p>
      
      <button onClick={simulateBehavior} style={{ padding: '10px 20px', fontSize: '16px' }}>
        Simulate User
      </button>

      {loading && <p>⏳ Predicting trust score...</p>}

      {result !== null && !loading && (
        <h2 style={{ marginTop: '20px' }}>
          {result === "API Error" ? "❌ API Error" : `✅ Trust Score: ${Math.round(result * 100)}%`}
        </h2>
      )}
    </div>
  );
}

export default App;
