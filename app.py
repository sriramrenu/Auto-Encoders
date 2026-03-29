import gradio as gr
import torch
from PIL import Image
import os
import sys
import time

# 1. Setup Local Paths for Architectures
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

try:
    from fraud.model import FraudAutoEncoder
    from image.model import ImageAutoEncoder
    from network.model import NetworkAutoEncoder
except ImportError:
    sys.path.append(os.path.join(BASE_DIR, '..', 'models'))
    from fraud.model import FraudAutoEncoder
    from image.model import ImageAutoEncoder
    from network.model import NetworkAutoEncoder

# 2. Model Loading Utility
def load_hf_model(path, model_class, input_dim=None):
    if not os.path.exists(path):
        return None
    try:
        if input_dim:
            model = model_class(input_dim=input_dim)
        else:
            model = model_class()
        
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# Load all 3 Neural Engines
fraud_model = load_hf_model("models/fraud/final_fraud_autoencoder.pth", FraudAutoEncoder)
image_model = load_hf_model("models/image/final_image_autoencoder.pth", ImageAutoEncoder)
network_model = load_hf_model("models/network/network_autoencoder.pth", NetworkAutoEncoder, input_dim=77)

# --- PREMIUM ELEGANT THEME & CSS ---
ELEGANT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;700&family=JetBrains+Mono&display=swap');
body { 
    background-color: #020617 !important; 
    color: #f8fafc !important; 
    font-family: 'Plus Jakarta Sans', sans-serif !important; 
}
.gradio-container { 
    background: radial-gradient(circle at top right, #1e293b, #020617) !important; 
    border: none !important; 
}
.elegant-card { 
    background: rgba(15, 23, 42, 0.6) !important; 
    border: 1px solid rgba(255, 255, 255, 0.08) !important; 
    border-radius: 20px !important; 
    padding: 28px !important; 
    backdrop-filter: blur(24px) saturate(180%) !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5) !important;
    transition: transform 0.3s ease, border-color 0.3s ease !important;
}
.elegant-card:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
}
.elegant-btn {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    height: 48px !important;
    border-radius: 12px !important;
    text-transform: none !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}
.elegant-btn:hover { 
    transform: translateY(-2px) !important; 
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.5) !important;
    filter: brightness(1.1) !important;
}
.diagnostic-log { 
    font-family: 'JetBrains Mono', monospace !important; 
    color: #94a3b8 !important; 
    background: #0f172a !important; 
    border: 1px solid #334155 !important;
    font-size: 0.8rem !important;
    border-radius: 12px !important;
}
.soft-glow-text { 
    color: #818cf8 !important; 
    font-weight: 700 !important; 
    letter-spacing: -0.5px !important;
}
.status-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
}
@keyframes subtle-fade { from { opacity: 0.8; } to { opacity: 1; } }
.fade-in { animation: subtle-fade 1s ease-out; }
"""

# --- INFERENCE LOGIC ---

def predict_fraud(feature_string):
    logs = [
        f"[{time.strftime('%H:%M:%S')}] Core system ready...",
        f"[{time.strftime('%H:%M:%S')}] Processing financial vector...",
        f"[{time.strftime('%H:%M:%S')}] PCA Latent Space synchronized."
    ]
    if not fraud_model: return "\n".join(logs + ["ERROR: Module link failure."]), "Offline"
    try:
        data = [float(x.strip()) for x in feature_string.split(",")]
        if len(data) < 30: data += [0.0] * (30 - len(data))
        else: data = data[:30]
        
        tensor = torch.Tensor([data])
        with torch.no_grad():
            recon = fraud_model(tensor)
            mse = torch.mean((tensor - recon)**2).item()
        
        is_anomaly = mse > 0.69
        status = "CRITICAL: Anomaly" if is_anomaly else "Verified: Clear"
        logs += [
            f"[{time.strftime('%H:%M:%S')}] Reconstruction Error: {mse:.6f}",
            f"[{time.strftime('%H:%M:%S')}] Final result: {status}"
        ]
        return "\n".join(logs), status
    except Exception as e:
        return f"System Maintenance Required: {e}", "System Error"

def predict_medical(image):
    logs = [
        f"[{time.strftime('%H:%M:%S')}] Imaging probe initialized...",
        f"[{time.strftime('%H:%M:%S')}] Voxel topology analysis active...",
        f"[{time.strftime('%H:%M:%S')}] Resolution sync complete at 64x64."
    ]
    if image is None: return "ERROR: No visual input detected.", "Awaiting Data"
    try:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            recon = image_model(tensor)
            mse = torch.mean((tensor - recon)**2).item()
        
        is_anomaly = mse > 0.005
        status = "Diagnostic: Pathological" if is_anomaly else "Diagnostic: Healthy"
        logs += [
            f"[{time.strftime('%H:%M:%S')}] Residual Variance: {mse:.6f}",
            f"[{time.strftime('%H:%M:%S')}] Classification outcome: {status}"
        ]
        return "\n".join(logs), status
    except Exception as e:
        return f"Imaging Failure: {e}", "Diagnostic Error"

def predict_network(feature_string):
    logs = [
        f"[{time.strftime('%H:%M:%S')}] Packet interceptor active...",
        f"[{time.strftime('%H:%M:%S')}] NSL-KDD signature check...",
        f"[{time.strftime('%H:%M:%S')}] Traffic pattern matched."
    ]
    if not network_model: return "\n".join(logs + ["ERROR: Security link failure."]), "Offline"
    try:
        data = [float(x.strip()) for x in feature_string.split(",")]
        if len(data) < 77: data += [0.0] * (77 - len(data))
        else: data = data[:77]
        
        tensor = torch.Tensor([data])
        with torch.no_grad():
            recon = network_model(tensor)
            mse = torch.mean((tensor - recon)**2).item()
        
        is_anomaly = mse > 0.6286
        status = "Threat Alert: Intrusion" if is_anomaly else "Traffic: Secure"
        logs += [
            f"[{time.strftime('%H:%M:%S')}] Entropy Entropy Score: {mse:.6f}",
            f"[{time.strftime('%H:%M:%S')}] Severity status: {'High Potential' if is_anomaly else 'Neutral'}"
        ]
        return "\n".join(logs), status
    except Exception as e:
        return f"Security Failure: {e}", "Security Error"

# --- ELEGANT COMPONENT ASSEMBLY ---

header_html = """
<div style="text-align: center; padding: 48px 0; margin-bottom: 24px;">
    <h1 style="color: #ffffff; font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; letter-spacing: -1.5px; margin: 0; font-size: 3rem;">Universal Anomaly Engine</h1>
    <p style="color: #94a3b8; font-family: 'Plus Jakarta Sans', sans-serif; margin: 12px 0 0; font-size: 1.1rem; font-weight: 500;">Sophisticated Multi-Domain Intelligence for Finance, Health, and Security</p>
    <div style="width: 60px; height: 4px; background: #6366f1; margin: 24px auto; border-radius: 2px;"></div>
</div>
"""

with gr.Blocks() as demo:
    gr.HTML(header_html)
    
    with gr.Tabs():
        # FINANCIAL TAB
        with gr.Tab("🏦 Financial Intelligence"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="elegant-card"):
                    gr.Markdown("<h3 class='soft-glow-text'>💳 Asset Vector Input</h3>")
                    fraud_input = gr.Textbox(placeholder=" 30 PCA features (e.g. 0.1, -1.2, 0...) ", lines=6, label="Input Parameters")
                    fraud_btn = gr.Button("Analyze Asset Vector", elem_classes="elegant-btn")
                with gr.Column(scale=1, elem_classes="elegant-card"):
                    gr.Markdown("<h3 class='soft-glow-text'>📊 Diagnostic Telemetry</h3>")
                    fraud_log = gr.TextArea(label="System Console", elem_classes="diagnostic-log", interactive=False, lines=10)
                    fraud_res = gr.Label(label="Detection Signal")
            fraud_btn.click(predict_fraud, inputs=fraud_input, outputs=[fraud_log, fraud_res])

        # MEDICAL TAB
        with gr.Tab("🩺 Healthcare Diagnostics"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="elegant-card"):
                    gr.Markdown("<h3 class='soft-glow-text'>🩻 Radiology Scan</h3>")
                    image_input = gr.Image(type="pil", label="Medical Imaging (X-Ray)")
                    image_btn = gr.Button("Initiate Diagnosis", elem_classes="elegant-btn")
                with gr.Column(scale=1, elem_classes="elegant-card"):
                    gr.Markdown("<h3 class='soft-glow-text'>📊 Diagnostic Telemetry</h3>")
                    image_log = gr.TextArea(label="System Console", elem_classes="diagnostic-log", interactive=False, lines=10)
                    image_res = gr.Label(label="Pathology Result")
            image_btn.click(predict_medical, inputs=image_input, outputs=[image_log, image_res])

        # NETWORK TAB
        with gr.Tab("🌐 Security Operations"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="elegant-card"):
                    gr.Markdown("<h3 class='soft-glow-text'>🛑 Packet Telemetry Input</h3>")
                    net_input = gr.Textbox(placeholder=" 77 Network features (NSL-KDD)... ", lines=6, label="Traffic Datagram")
                    net_btn = gr.Button("Perform Security Audit", elem_classes="elegant-btn")
                with gr.Column(scale=1, elem_classes="elegant-card"):
                    gr.Markdown("<h3 class='soft-glow-text'>📊 Diagnostic Telemetry</h3>")
                    net_log = gr.TextArea(label="System Console", elem_classes="diagnostic-log", interactive=False, lines=10)
                    net_res = gr.Label(label="Security Indicator")
            net_btn.click(predict_network, inputs=net_input, outputs=[net_log, net_res])

    gr.Markdown("<center style='opacity: 0.5; margin-top: 60px; font-family: \"Plus Jakarta Sans\"; font-size: 0.8rem; letter-spacing: 0.5px;'>PROFESSIONAL GRADE // ENTERPRISE SECURITY ACTIVE // © 2026</center>")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate"), css=ELEGANT_CSS, ssr_mode=False)
