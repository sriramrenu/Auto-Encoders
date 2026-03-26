import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity } from 'lucide-react';

export default function Navbar() {
  const loc = useLocation();
  return (
    <nav className="nav-header">
      <div style={{display: 'flex', alignItems: 'center', gap: '15px', fontWeight: '800', fontSize: '1.4rem', letterSpacing: '2px'}} className="gradient-text">
        <Activity color="#00f0ff" size={28} /> AE_NEURAL_NET
      </div>
      <div className="nav-links">
        <Link to="/" className={`nav-link ${loc.pathname === '/' ? 'active':''}`}>Home</Link>
        <Link to="/fraud" className={`nav-link ${loc.pathname === '/fraud' ? 'active':''}`}>Banking</Link>
        <Link to="/image" className={`nav-link ${loc.pathname === '/image' ? 'active':''}`}>Healthcare</Link>
        <Link to="/network" className={`nav-link ${loc.pathname === '/network' ? 'active':''}`}>IT Systems</Link>
      </div>
    </nav>
  );
}