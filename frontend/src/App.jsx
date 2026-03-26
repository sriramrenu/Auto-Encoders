import React from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import Navbar from './components/Navbar.jsx';
import Home from './pages/Home.jsx';
import FraudPage from './pages/FraudPage.jsx';
import ImagePage from './pages/ImagePage.jsx';
import NetworkPage from './pages/NetworkPage.jsx';

function App() {
  const location = useLocation();

  return (
    <>
      <div className="cyber-bg">
        <div className="cyber-grid"></div>
      </div>
      <Navbar />
      <div className="app-container">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<Home />} />
            <Route path="/fraud" element={<FraudPage />} />
            <Route path="/image" element={<ImagePage />} />
            <Route path="/network" element={<NetworkPage />} />
          </Routes>
        </AnimatePresence>
      </div>
    </>
  );
}

export default App;