import React, { useState, useRef, useEffect } from 'react';

const API_URL = 'http://localhost:8000';

function App() {
  const [image, setImage] = useState(null);
  const [imageB64, setImageB64] = useState('');
  const [text, setText] = useState('');
  const [model, setModel] = useState('main');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setImage(e.target.result);
      setImageB64(e.target.result.split(',')[1]);
      setResult(null);
      setError('');
    };
    reader.readAsDataURL(file);
  };

  // Draw bbox on canvas whenever result changes
  useEffect(() => {
    if (!result || !image || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new window.Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      const [xc, yc, w, h] = result.bbox;
      const x1 = (xc - w / 2) * img.width;
      const y1 = (yc - h / 2) * img.height;
      const bw = w * img.width;
      const bh = h * img.height;

      const color = result.model === 'main' ? '#00CC66' : result.model === 'ablation' ? '#FF9900' : '#FF4444';

      // Draw bbox
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, bw, bh);

      // Draw label
      const label = `${result.model} [${result.bbox.map(v => v.toFixed(3)).join(', ')}]`;
      ctx.font = 'bold 16px monospace';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - 26), tw + 10, 26);
      ctx.fillStyle = 'white';
      ctx.fillText(label, x1 + 5, Math.max(16, y1 - 8));
    };
    img.src = image;
  }, [result, image]);

  const handlePredict = async () => {
    if (!imageB64) { setError('Upload an image first'); return; }
    if (!text.trim()) { setError('Enter a referring expression'); return; }

    setLoading(true);
    setError('');
    setResult(null);

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

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Spatial-LLaVA Visual Grounding</h1>
      <p style={styles.subtitle}>Upload an image and describe an object to locate it</p>

      <div style={styles.main}>
        {/* Left panel */}
        <div style={styles.panel}>
          {/* Image upload */}
          <div
            style={{ ...styles.dropzone, ...(dragOver ? styles.dropzoneActive : {}) }}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
            onClick={() => fileInputRef.current?.click()}
          >
            {image ? (
              <img src={image} alt="uploaded" style={styles.preview} />
            ) : (
              <div style={styles.dropText}>
                <p style={{ fontSize: 32, margin: 0 }}>+</p>
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

          {/* Text input */}
          <input
            style={styles.input}
            type="text"
            placeholder='e.g. "the person on the left"'
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handlePredict()}
          />

          {/* Model selector */}
          <div style={styles.modelRow}>
            {['main', 'ablation', 'baseline'].map((m) => (
              <button
                key={m}
                style={{
                  ...styles.modelBtn,
                  backgroundColor: model === m ? (m === 'main' ? '#00CC66' : m === 'ablation' ? '#FF9900' : '#FF4444') : '#2a2a2a',
                  color: model === m ? '#fff' : '#888',
                }}
                onClick={() => setModel(m)}
              >
                {m}
              </button>
            ))}
          </div>

          {/* Predict button */}
          <button
            style={{ ...styles.predictBtn, opacity: loading ? 0.6 : 1 }}
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? 'Predicting...' : 'Predict'}
          </button>

          {error && <p style={styles.error}>{error}</p>}

          {/* Result info */}
          {result && (
            <div style={styles.resultBox}>
              <p><strong>Model:</strong> {result.model}</p>
              <p><strong>BBox:</strong> [{result.bbox.map(v => v.toFixed(4)).join(', ')}]</p>
              <p><strong>Latency:</strong> {result.inference_time_ms} ms</p>
            </div>
          )}
        </div>

        {/* Right panel — canvas with bbox overlay */}
        <div style={styles.canvasPanel}>
          <canvas
            ref={canvasRef}
            style={{
              ...styles.canvas,
              display: result ? 'block' : 'none',
            }}
          />
          {!result && (
            <div style={styles.placeholder}>
              <p style={{ color: '#555' }}>Prediction result will appear here</p>
            </div>
          )}
        </div>
      </div>

      {/* Model comparison table */}
      <div style={styles.tableWrap}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Model</th>
              <th style={styles.th}>Method</th>
              <th style={styles.th}>Test IoU</th>
              <th style={styles.th}>RMSE</th>
              <th style={styles.th}>MAE</th>
            </tr>
          </thead>
          <tbody>
            <tr><td style={{...styles.td, color:'#FF4444'}}>Baseline</td><td style={styles.td}>Vanilla LLaVA + regex</td><td style={styles.td}>0.097</td><td style={styles.td}>0.288</td><td style={styles.td}>0.238</td></tr>
            <tr><td style={{...styles.td, color:'#FF9900'}}>Ablation</td><td style={styles.td}>Frozen LLaVA + MLP head</td><td style={styles.td}>0.284</td><td style={styles.td}>0.224</td><td style={styles.td}>0.177</td></tr>
            <tr><td style={{...styles.td, color:'#00CC66'}}>Main</td><td style={styles.td}>LoRA + MLP head</td><td style={styles.td}>0.386</td><td style={styles.td}>0.172</td><td style={styles.td}>0.119</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

const styles = {
  container: { maxWidth: 1100, margin: '0 auto', padding: 24, fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif', color: '#e0e0e0', backgroundColor: '#1a1a1a', minHeight: '100vh' },
  title: { margin: 0, fontSize: 28, fontWeight: 700, color: '#fff' },
  subtitle: { color: '#888', marginTop: 4, marginBottom: 24 },
  main: { display: 'flex', gap: 24, alignItems: 'flex-start' },
  panel: { flex: '0 0 380px', display: 'flex', flexDirection: 'column', gap: 12 },
  dropzone: { border: '2px dashed #444', borderRadius: 8, padding: 8, cursor: 'pointer', minHeight: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'border-color 0.2s' },
  dropzoneActive: { borderColor: '#00CC66' },
  dropText: { textAlign: 'center', color: '#666' },
  preview: { maxWidth: '100%', maxHeight: 300, borderRadius: 6 },
  input: { padding: '10px 14px', borderRadius: 6, border: '1px solid #444', backgroundColor: '#2a2a2a', color: '#e0e0e0', fontSize: 14, outline: 'none' },
  modelRow: { display: 'flex', gap: 8 },
  modelBtn: { flex: 1, padding: '8px 0', borderRadius: 6, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13, transition: 'all 0.2s' },
  predictBtn: { padding: '12px 0', borderRadius: 6, border: 'none', backgroundColor: '#00CC66', color: '#fff', fontWeight: 700, fontSize: 16, cursor: 'pointer' },
  error: { color: '#FF4444', fontSize: 13, margin: 0 },
  resultBox: { backgroundColor: '#2a2a2a', borderRadius: 8, padding: 14, fontSize: 14, lineHeight: 1.8 },
  canvasPanel: { flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400 },
  canvas: { maxWidth: '100%', borderRadius: 8, border: '1px solid #333' },
  placeholder: { width: '100%', height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed #333', borderRadius: 8 },
  tableWrap: { marginTop: 32 },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 14 },
  th: { textAlign: 'left', padding: '10px 14px', borderBottom: '1px solid #333', color: '#888' },
  td: { padding: '10px 14px', borderBottom: '1px solid #222' },
};

export default App;
