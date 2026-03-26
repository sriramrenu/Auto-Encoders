import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ShieldAlert, ActivitySquare, Network } from 'lucide-react';

export default function Home() {
  const navigate = useNavigate();

  const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.15 } }
  };
  const item = { hidden: { opacity: 0, y: 30 }, show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } } };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
      <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
        <motion.h1 
           initial={{ scale: 0.9, opacity: 0 }} 
           animate={{ scale: 1, opacity: 1 }} 
           transition={{ duration: 0.5 }}
           style={{ fontSize: '3.5rem', fontWeight: 800, marginBottom: '1rem', letterSpacing: '-1px' }}
        >
          Welcome to the <span className="gradient-text">Neural Hub</span>
        </motion.h1>
        <motion.p 
          initial={{ y: 20, opacity: 0 }} 
          animate={{ y: 0, opacity: 1 }} 
          transition={{ delay: 0.2 }}
          style={{ fontSize: '1.2rem', color: 'var(--text-muted)', maxWidth: '600px', margin: '0 auto' }}
        >
          Multi-domain autoencoder platform detecting anomalies across isolated data ecosystems. Select a simulation below.
        </motion.p>
      </div>

      <motion.div className="domain-grid" variants={container} initial="hidden" animate="show">
        
        <motion.div className="glass-panel domain-card" variants={item} onClick={() => navigate('/fraud')}>
          <ShieldAlert size={48} strokeWidth={1.5} />
          <h2>Banking / Fraud</h2>
          <p style={{ color: 'var(--text-muted)' }}>Analyze transaction tensors against normal spending clusters to detect high-deviation financial anomalies.</p>
        </motion.div>

        <motion.div className="glass-panel domain-card" variants={item} onClick={() => navigate('/image')}>
          <ActivitySquare size={48} strokeWidth={1.5} />
          <h2>Healthcare / Imaging</h2>
          <p style={{ color: 'var(--text-muted)' }}>Perform deep convolution across medical image arrays to identify non-standard cellular structures.</p>
        </motion.div>

        <motion.div className="glass-panel domain-card" variants={item} onClick={() => navigate('/network')}>
          <Network size={48} strokeWidth={1.5} />
          <h2>IT / Networks</h2>
          <p style={{ color: 'var(--text-muted)' }}>Monitor packet sequences to intercept unauthorized intrusion vectors based on structural deviations.</p>
        </motion.div>

      </motion.div>
    </motion.div>
  );
}
