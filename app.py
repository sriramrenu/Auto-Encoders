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

# Load all 3 Engines
fraud_model = load_hf_model("models/fraud/final_fraud_autoencoder.pth", FraudAutoEncoder)
image_model = load_hf_model("models/image/final_image_autoencoder.pth", ImageAutoEncoder)
network_model = load_hf_model("models/network/network_autoencoder.pth", NetworkAutoEncoder, input_dim=77)

# --- THEME & CSS ---
CYBER_CSS = """
body { background-color: #050508 !important; color: #e0e0e0 !important; font-family: 'Inter', sans-serif; }
.gradio-container { background: radial-gradient(circle at top, #101020 0%, #050508 100%) !important; border: none !important; }
.cyber-card { 
    background: rgba(15, 15, 25, 0.7) !important; 
    border: 1px solid rgba(0, 242, 255, 0.2) !important; 
    border-radius: 12px !important; 
    padding: 20px !important; 
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8) !important;
}
.glow-cyan { border-color: #00f2ff !important; box-shadow: 0 0 15px rgba(0, 242, 255, 0.3) !important; }
.glow-magenta { border-color: #ff00e5 !important; box-shadow: 0 0 15px rgba(255, 0, 229, 0.3) !important; }
.cyber-btn {
    background: linear-gradient(90deg, #00f2ff, #0080ff) !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
}
.cyber-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 0 20px rgba(0, 242, 255, 0.6) !important; }
.neural-log { font-family: 'Courier New', monospace !important; color: #00f2ff !important; background: #000 !important; }
"""

# --- INFERENCE FUNCTIONS ---

def predict_fraud(feature_string):
    logs = [f"[{time.strftime('%H:%M:%S')}] INITIATING_TENSOR_SCAN...", f"[{time.strftime('%H:%M:%S')}] LOADING_PCA_WEIGHTS..."]
    if not fraud_model: return "\n".join(logs + ["ERROR: Model data-link failed."]), "OFFLINE"
    try:
        data = [float(x.strip()) for x in feature_string.split(",")]
        if len(data) < 30: data += [0.0] * (30 - len(data))
        else: data = data[:30]
        
        tensor = torch.Tensor([data])
        with torch.no_grad():
            recon = fraud_model(tensor)
            mse = torch.mean((tensor - recon)**2).item()
        
        is_anomaly = mse > 0.69
        logs.append(f"[{time.strftime('%H:%M:%S')}] RECONSTRUCTION_COMPLETE.")
        logs.append(f"[{time.strftime('%H:%M:%S')}] LOSS_VALUE: {mse:.6f}")
        
        res_text = f"🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
        return "\n".join(logs), res_text
    except Exception as e:
        return f"CRITICAL_EXCEPTION: {e}", "ERROR"

def predict_medical(image):
    logs = [f"[{time.strftime('%H:%M:%S')}] RADIOLOGY_TENSOR_SYNC...", f"[{time.strftime('%H:%M:%S')}] ANALYZING_SPATIAL_TOPOLOGY..."]
    if image is None: return "ERROR: NO_INPUT", "WAITING"
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
        logs.append(f"[{time.strftime('%H:%M:%S')}] VOXEL_RECON_COMPLETE.")
        logs.append(f"[{time.strftime('%H:%M:%S')}] RESIDUAL_VARIANCE: {mse:.6f}")
        
        res_text = f"🚨 PATHOLOGY" if is_anomaly else "✅ HEALTHY"
        return "\n".join(logs), res_text
    except Exception as e:
        return f"CRITICAL_EXCEPTION: {e}", "ERROR"

def predict_network(feature_string):
    logs = [f"[{time.strftime('%H:%M:%S')}] INTERCEPTING_PACKET_STREAM...", f"[{time.strftime('%H:%M:%S')}] DECODING_NSL_KDD_FEATURES..."]
    if not network_model: return "\n".join(logs + ["ERROR: Link failed."]), "OFFLINE"
    try:
        data = [float(x.strip()) for x in feature_string.split(",")]
        if len(data) < 77: data += [0.0] * (77 - len(data))
        else: data = data[:77]
        
        tensor = torch.Tensor([data])
        with torch.no_grad():
            recon = network_model(tensor)
            mse = torch.mean((tensor - recon)**2).item()
        
        is_anomaly = mse > 0.6286
        logs.append(f"[{time.strftime('%H:%M:%S')}] VECTOR_ALIGNED.")
        logs.append(f"[{time.strftime('%H:%M:%S')}] ENTROPY_SCORE: {mse:.6f}")
        
        res_text = f"🚨 INTRUSION" if is_anomaly else "✅ SECURE"
        return "\n".join(logs), res_text
    except Exception as e:
        return f"CRITICAL_EXCEPTION: {e}", "ERROR"

# --- GRADIO INTERFACE ---

header_html = """
<div style="text-align: center; padding: 30px; border-bottom: 1px solid rgba(0, 242, 255, 0.3);">
    <h1 style="color: #00f2ff; font-family: 'Inter', sans-serif; letter-spacing: 5px; margin: 0; font-size: 2.5rem;">NEURAL ANOMALY PLATFORM</h1>
    <p style="color: #ff00e5; font-family: 'Courier New', monospace; margin: 5px 0 0; text-transform: uppercase;">Engine v2.0 // Hybrid AutoEncoder Cluster</p>
</div>
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=CYBER_CSS) as demo:
    gr.HTML(header_html)
    
    with gr.Tabs():
        with gr.Tab("🏦 BANKING_CORE"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🛡️ PAYLOAD_INJECTION")
                    fraud_input = gr.Textbox(placeholder=" PCA features (comma-separated)...", lines=5, label="Input Tensor")
                    fraud_btn = gr.Button("INITIALIZE ANALYSIS", elem_classes="cyber-btn")
                with gr.Column(scale=1):
                    gr.Markdown("### 📡 NEURAL_MONITOR")
                    fraud_log = gr.TextArea(label="System Logs", elem_classes="neural-log", interactive=False)
                    fraud_res = gr.Label(label="Classification")
            fraud_btn.click(predict_fraud, inputs=fraud_input, outputs=[fraud_log, fraud_res])

        with gr.Tab("🩻 HEALTHCARE_CORE"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🩻 X-RAY_SCANNER")
                    image_input = gr.Image(type="pil", label="Radiographic Scan")
                    image_btn = gr.Button("INITIATE DIAGNOSTIC", elem_classes="cyber-btn")
                with gr.Column(scale=1):
                    gr.Markdown("### 📡 NEURAL_MONITOR")
                    image_log = gr.TextArea(label="Diagnostic Logs", elem_classes="neural-log", interactive=False)
                    image_res = gr.Label(label="Pathology Status")
            image_btn.click(predict_medical, inputs=image_input, outputs=[image_log, image_res])

        with gr.Tab("🛡️ NETWORK_CORE"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🌐 PACKET_LOG_TENSOR")
                    net_input = gr.Textbox(placeholder="Packet features (NSL-KDD)...", lines=5, label="Network Payload")
                    net_btn = gr.Button("EVALUATE SECURITY", elem_classes="cyber-btn")
                with gr.Column(scale=1):
                    gr.Markdown("### 📡 NEURAL_MONITOR")
                    net_log = gr.TextArea(label="Traffic Logs", elem_classes="neural-log", interactive=False)
                    net_res = gr.Label(label="Security Alert")
            net_btn.click(predict_network, inputs=net_input, outputs=[net_log, net_res])

    gr.Markdown("<center style='opacity: 0.5; margin-top: 50px;'>UNAUTHORIZED ACCESS TO NEURAL CORE IS STRICTLY PROHIBITED. © 2026</center>")

if __name__ == "__main__":
    demo.launch()

