import React, { useState, useRef, useEffect } from 'react';
import './App.css';

const API_URL = '';

const MODEL_INFO = {
  main:     { color: '#4dd4ac', label: 'Main',     method: 'LoRA + MLP head',         iou: 0.386, rmse: 0.172, mae: 0.119 },
  ablation: { color: '#6ea8fe', label: 'Ablation', method: 'Frozen LLaVA + MLP head', iou: 0.284, rmse: 0.224, mae: 0.177 },
  baseline: { color: '#f08080', label: 'Baseline', method: 'Vanilla LLaVA + regex',   iou: 0.097, rmse: 0.288, mae: 0.238 },
};

function App() {
  const [image, setImage] = useState(null);
  const [imageB64, setImageB64] = useState('');
  const [text, setText] = useState('');
  const [model, setModel] = useState('main');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const lastText = useRef('');

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setImage(e.target.result);
      setImageB64(e.target.result.split(',')[1]);
      setResult(null);
      setHistory([]);
      setError('');
    };
    reader.readAsDataURL(file);
  };

  // Draw bboxes on canvas
  useEffect(() => {
    const allResults = [...history, ...(result ? [result] : [])];
    if (!image || !canvasRef.current || allResults.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new window.Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      allResults.forEach((r) => {
        const [xc, yc, w, h] = r.bbox;
        const x1 = (xc - w / 2) * img.width;
        const y1 = (yc - h / 2) * img.height;
        const bw = w * img.width;
        const bh = h * img.height;
        const info = MODEL_INFO[r.model];

        // Fill
        ctx.fillStyle = info.color + '40';
        ctx.fillRect(x1, y1, bw, bh);

        // Border
        ctx.strokeStyle = info.color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, bw, bh);

        // Corner accents
        const cl = Math.min(20, bw * 0.18, bh * 0.18);
        ctx.lineWidth = 5;
        ctx.strokeStyle = info.color;
        ctx.beginPath(); ctx.moveTo(x1, y1 + cl); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cl, y1); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x1 + bw - cl, y1); ctx.lineTo(x1 + bw, y1); ctx.lineTo(x1 + bw, y1 + cl); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x1, y1 + bh - cl); ctx.lineTo(x1, y1 + bh); ctx.lineTo(x1 + cl, y1 + bh); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x1 + bw - cl, y1 + bh); ctx.lineTo(x1 + bw, y1 + bh); ctx.lineTo(x1 + bw, y1 + bh - cl); ctx.stroke();

        // Rounded label badge
        ctx.font = '500 13px DM Sans, sans-serif';
        const label = `${info.label}  ${r.inference_time_ms}ms`;
        const tw = ctx.measureText(label).width;
        const ly = Math.max(0, y1 - 28);
        ctx.fillStyle = info.color;
        roundRect(ctx, x1, ly, tw + 14, 22, 4);
        ctx.fill();
        ctx.fillStyle = '#111';
        ctx.fillText(label, x1 + 7, ly + 16);
      });
    };
    img.src = image;
  }, [result, image, history]);

  const allResults = [...history, ...(result ? [result] : [])];

  const handlePredict = async () => {
    if (!imageB64) { setError('Upload an image first'); return; }
    if (!text.trim()) { setError('Enter a referring expression'); return; }

    const textChanged = text.trim() !== lastText.current;
    const allModelsDone = allResults.length >= 3;
    if (textChanged || allModelsDone) {
      setHistory([]);
      setResult(null);
    } else if (result) {
      setHistory(prev => [...prev, result]);
      setResult(null);
    }
    lastText.current = text.trim();

    setLoading(true);
    setError('');

    try {
      const resp = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageB64, text: text.trim(), model }),
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || 'Prediction failed');
      }
      const data = await resp.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const clearResults = () => {
    setResult(null);
    setHistory([]);
  };

  const latencyColor = (ms) => ms < 200 ? '#6ec87a' : ms < 500 ? '#ccc' : '#f08080';

  return (
    <div className="app">
      <div className="header">
        <h1>Spatial-LLaVA</h1>
        <p>Visual grounding — locate objects from natural language descriptions</p>
      </div>

      <div className="main-layout">
        {/* Left panel */}
        <div className="controls">
          <div
            className={`dropzone ${dragOver ? 'active' : ''} ${image ? 'has-image' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
            onClick={() => fileInputRef.current?.click()}
          >
            {image ? (
              <img src={image} alt="uploaded" className="preview-img" />
            ) : (
              <div className="drop-text">
                <p>+</p>
                <p>Drop image here or click to upload</p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>

          <input
            className="text-input"
            type="text"
            placeholder='Describe the object, e.g. "the person on the left"'
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !loading && handlePredict()}
          />

          <div className="model-selector">
            {Object.entries(MODEL_INFO).map(([key, info]) => (
              <button
                key={key}
                className={`model-btn ${model === key ? `selected-${key}` : ''}`}
                onClick={() => setModel(key)}
              >
                {info.label}
              </button>
            ))}
          </div>

          <div className="btn-row">
            <button
              className="predict-btn"
              onClick={handlePredict}
              disabled={loading}
              style={{ opacity: loading ? 0.5 : 1 }}
            >
              {loading ? 'Predicting...' : 'Predict'}
            </button>
            {allResults.length > 0 && (
              <button className="clear-btn" onClick={clearResults}>Clear</button>
            )}
          </div>

          {error && <div className="error-msg">{error}</div>}

          {allResults.map((r, i) => (
            <div className="result-box" key={i}>
              <div className="result-row">
                <span className="result-label">Model</span>
                <span className="result-value" style={{ color: MODEL_INFO[r.model].color, fontWeight: 600 }}>
                  {MODEL_INFO[r.model].label}
                </span>
              </div>
              <div className="result-row">
                <span className="result-label">BBox</span>
                <span className="result-value">[{r.bbox.map(v => v.toFixed(4)).join(', ')}]</span>
              </div>
              <div className="result-row">
                <span className="result-label">Latency</span>
                <span className="result-value" style={{ color: latencyColor(r.inference_time_ms), fontWeight: 600 }}>
                  {r.inference_time_ms} ms
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Right panel */}
        <div>
          <div className="canvas-container">
            <canvas
              ref={canvasRef}
              className="result-canvas"
              style={{ display: allResults.length > 0 ? 'block' : 'none' }}
            />
            {allResults.length === 0 && (
              <div className="placeholder">
                <p>Upload an image and run a prediction</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Comparison table */}
      <div className="comparison">
        <h3>Model Comparison — RefCOCO Test (1,975 samples)</h3>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Method</th>
              <th>Test IoU</th>
              <th>RMSE</th>
              <th>MAE</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(MODEL_INFO).map(([key, info]) => (
              <tr key={key}>
                <td>
                  <span className="model-dot" style={{ backgroundColor: info.color }} />
                  {info.label}
                </td>
                <td style={{ color: '#555' }}>{info.method}</td>
                <td>
                  {info.iou.toFixed(3)}
                  <span className="iou-bar" style={{ width: info.iou * 180, backgroundColor: info.color }} />
                </td>
                <td>{info.rmse.toFixed(3)}</td>
                <td>{info.mae.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

export default App;
