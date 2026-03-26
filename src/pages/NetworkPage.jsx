import React, { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { Network, Activity, Terminal } from 'lucide-react';

export default function NetworkPage() {
  const [data, setData] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!data.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const parsedData = data.split(',').map(n => parseFloat(n.trim())).filter(n => !isNaN(n));
      const res = await axios.post('http://localhost:8000/api/analyze', {
        domain: 'network',
        data: parsedData
      });
      setResult(res.data);
    } catch (e) {
      setResult({ error: e.message || 'Failed to analyze network logs' });
    }
    setLoading(false);
  };

  return (
    <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.4 }} className="glass-panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '20px' }}>
        <Network size={40} className="glow-text-cyan" />
        <div>
          <h1 style={{ margin: 0, fontSize: '2.2rem' }}>IT Systems Intrusion</h1>
          <p style={{ margin: 0, color: 'var(--text-muted)' }}>Dense AutoEncoder Sequence</p>
        </div>
      </div>
      
      <div style={{ marginTop: '30px' }}>
        <p style={{ color: 'var(--text-muted)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
           <Terminal size={14} /> PACKET_LOG_TENSOR [n_features]:
        </p>
        <textarea 
          className="cyber-input"
          value={data}
          onChange={(e) => setData(e.target.value)}
          placeholder="0.5, 12, 0, 1.4, 300... (Enter numerical array)"
          rows={4}
          style={{ fontFamily: 'monospace' }}
        />
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '20px' }}>
          <button className="cyber-button" onClick={handleAnalyze} disabled={loading || !data.trim()}>
            {loading ? <><Activity size={18} style={{ animation: 'spin 2s linear infinite' }} /> INTERCEPTING...</> : 'EVALUATE LOGS'}
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
              <div><span style={{color: 'var(--text-muted)'}}>&gt; ANOMALY:</span> {result.is_anomaly ? <span style={{color:'var(--danger)', fontWeight:'bold'}}>[ INTRUSION DETECTED ]</span> : <span style={{color:'var(--success)'}}>[ SECURE ]</span>}</div>
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
