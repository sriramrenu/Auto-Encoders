import React, { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { ActivitySquare, UploadCloud, Activity } from 'lucide-react';

export default function ImagePage() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!file) return alert('Please select an image file first.');
    setLoading(true);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('domain', 'image');

      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await axios.post(`${apiUrl}/api/analyze/file`, formData);
      setResult(res.data);
    } catch (e) {
      setResult({ error: e.message || 'Failed to analyze image' });
    }
    setLoading(false);
  };

  return (
    <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.4 }} className="glass-panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '20px' }}>
        <ActivitySquare size={40} className="glow-text-cyan" />
        <div>
          <h1 style={{ margin: 0, fontSize: '2.2rem' }}>Medical Image Analysis</h1>
          <p style={{ margin: 0, color: 'var(--text-muted)' }}>Convolutional AutoEncoder Sequence</p>
        </div>
      </div>
      
      <div style={{ marginTop: '30px' }}>
        <p style={{ color: 'var(--text-muted)', marginBottom: '10px' }}>IMAGE_TENSOR [HxWxC]:</p>
        <div style={{ position: 'relative' }}>
          <input 
            type="file" 
            accept="image/*"
            onChange={(e) => setFile(e.target.files[0])}
            style={{ opacity: 0, position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', cursor: 'pointer', zIndex: 10 }}
          />
          <div className={`upload-zone ${file ? 'active' : ''}`}>
             <UploadCloud size={48} style={{ color: file ? 'var(--magenta-glow)' : 'var(--text-muted)', marginBottom: '15px' }} />
             <h3 style={{ margin: 0 }}>
               {file ? file.name : 'DROP IMAGE TENSOR HERE'}
             </h3>
             <p style={{ margin: '5px 0 0', color: 'var(--text-muted)' }}>or click to browse local databanks</p>
          </div>
        </div>
        
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '20px' }}>
          <button className="cyber-button" onClick={handleAnalyze} disabled={loading || !file}>
            {loading ? <><Activity size={18} style={{ animation: 'spin 2s linear infinite' }} /> PROCESSING...</> : 'INITIATE SCAN'}
          </button>
        </div>
      </div>

      {result && (
        <div className={`glass-panel result-card ${result.is_anomaly ? 'result-anomaly' : ''}`}>
          <h3 style={{ margin: '0 0 15px 0', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '10px' }}>INFERENCE_RESULT</h3>
          {result.error ? (
            <p style={{ color: 'var(--danger)' }}>ERR: {result.error}</p>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
              <div><span style={{color: 'var(--text-muted)'}}>&gt; DOMAIN:</span> {result.domain}</div>
              <div><span style={{color: 'var(--text-muted)'}}>&gt; ANOMALY:</span> {result.is_anomaly ? <span style={{color:'var(--danger)', fontWeight:'bold'}}>[ DETECTED ]</span> : <span style={{color:'var(--success)'}}>[ NEGATIVE ]</span>}</div>
              <div><span style={{color: 'var(--text-muted)'}}>&gt; CONFIDENCE:</span> {result.confidence}</div>
              <div><span style={{color: 'var(--text-muted)'}}>&gt; RECON ERROR:</span> {result.reconstruction_error}</div>
              <div style={{ gridColumn: '1 / -1', marginTop: '10px', padding: '10px', background: 'rgba(0,0,0,0.3)', borderRadius: '8px' }}>
                 <span style={{color: 'var(--text-muted)'}}>&gt; INSIGHT:</span> {result.insights}
              </div>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}
