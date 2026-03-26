---
title: Autoencoders
emoji: 🛡️
colorFrom: cyan
colorTo: pink
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
---

# Multi-Domain Anomaly Detection Platform 🏆
**An Intelligent Suite for Finance, Healthcare, and Cybersecurity**

This platform is a production-grade anomaly detection system that leverages **Deep Learning Autoencoders** to identify structural and behavioral deviations in real-time. It features a unique "Cyber-Neural" interface and a high-performance hybrid cloud architecture.

---

## 🏛️ Architecture: The Hybrid Proxy Model
To ensure maximum scalability and bypass memory limits, this platform uses a dual-cloud strategy:
1.  **Hugging Face Spaces (The Intelligence):** Hosts the heavy PyTorch models and provides a high-speed inference API via Gradio.
2.  **Vercel (The Interface):** Hosts the React frontend and a lightweight FastAPI "Proxy" backend.

---

## 🧠 Integrated AI Engines
1.  **Financial Fraud:** Dense Autoencoder for 30-feature transaction PCA data (**~95% Accuracy**).
2.  **Medical Diagnostics:** Conv2D Spatial Autoencoder for 64x64 Chest X-Rays (**Pathology Detection**).
3.  **Network Intrusion:** 77-Dimensional Deep Dense Autoencoder for NSL-KDD packet analysis (**94.27% Accuracy**).

---

## 🚀 Deployment & Local Setup

### Cloud Deployment (Recommended)
1.  **AI Models:** Create a Gradio Space on Hugging Face and upload the `models/` directory + `app.py`.
2.  **Web App:** Connect this repository to Vercel. 
3.  **Config:** Add the `HF_SPACE_URL` environment variable in Vercel.

### Local Development
1.  **Backend:**
    ```bash
    cd backend
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    pip install -r requirements.txt
    python main.py
    ```
2.  **Frontend:**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

---

## 📑 Project Documentation
*   **[Technical Proposal](.gemini/antigravity/brain/f91ad3c9-8d8f-4976-83e1-21fda0a61165/project_proposal.md):** Architectural design and business value.
*   **[Hosting Guide](.gemini/antigravity/brain/f91ad3c9-8d8f-4976-83e1-21fda0a61165/hosting_guide.md):** Step-by-step cloud assembly instructions.

---
**Developed by:** Antigravity (Google DeepMind)
**Date:** March 2026
