import gradio as gr
import torch
from PIL import Image
import os
import sys

# 1. Setup Local Paths for Architectures
# We'll append the models dir to sys.path so we can import the classes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

try:
    from fraud.model import FraudAutoEncoder
    from image.model import ImageAutoEncoder
    from network.model import NetworkAutoEncoder
except ImportError:
    # Fallback if directories are nested differently on HF
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

# --- INFERENCE FUNCTIONS ---

def predict_fraud(feature_string):
    if not fraud_model: return "Model not found."
    try:
        # Convert comma-separated string to list of floats
        data = [float(x.strip()) for x in feature_string.split(",")]
        if len(data) < 30: data += [0.0] * (30 - len(data))
        else: data = data[:30]
        
        tensor = torch.Tensor([data])
        with torch.no_grad():
            recon = fraud_model(tensor)
            mse = torch.mean((tensor - recon)**2).item()
        
        is_anomaly = mse > 0.69
        return f"MSE Reconstruction Error: {mse:.4f}\nResult: {'🚨 ANOMALY DETECTED' if is_anomaly else '✅ NORMAL'}"
    except Exception as e:
        return f"Error: {e}"

def predict_medical(image):
    if not image_model: return "Model not found."
    if image is None: return "Please upload an image."
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
        
        is_anomaly = mse > 0.005 # Using 64x64 threshold
        return f"MSE Reconstruction Error: {mse:.4f}\nResult: {'🚨 PATHOLOGY DETECTED (Anomaly)' if is_anomaly else '✅ HEALTHY SCAN (Normal)'}"
    except Exception as e:
        return f"Error: {e}"

def predict_network(feature_string):
    if not network_model: return "Model not found."
    try:
        data = [float(x.strip()) for x in feature_string.split(",")]
        if len(data) < 77: data += [0.0] * (77 - len(data))
        else: data = data[:77]
        
        tensor = torch.Tensor([data])
        with torch.no_grad():
            recon = network_model(tensor)
            mse = torch.mean((tensor - recon)**2).item()
        
        is_anomaly = mse > 0.6286
        return f"MSE Reconstruction Error: {mse:.4f}\nResult: {'🚨 INTRUSION DETECTED' if is_anomaly else '✅ SECURE CONNECTION'}"
    except Exception as e:
        return f"Error: {e}"

# --- GRADIO INTERFACE ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏆 Multi-Domain Anomaly Detection Hub")
    gr.Markdown("Toggle between Finance, Medical, and Network Security engines powered by PyTorch Autoencoders.")
    
    with gr.Tab("🏦 Financial Fraud"):
        gr.Markdown("Input 30 PCA-encoded transaction features (comma-separated):")
        fraud_input = gr.Textbox(placeholder="0.1, -1.2, 0.45...", label="Transaction Vector")
        fraud_btn = gr.Button("Analyze Transaction")
        fraud_out = gr.Textbox(label="Detection Result")
        fraud_btn.click(predict_fraud, inputs=fraud_input, outputs=fraud_out)

    with gr.Tab("🩻 Medical Imaging"):
        gr.Markdown("Upload a Chest X-Ray to detect Pediatric Pneumonia:")
        image_input = gr.Image(type="pil", label="Chest X-Ray")
        image_btn = gr.Button("Initiate Scan")
        image_out = gr.Textbox(label="Diagnostic Result")
        image_btn.click(predict_medical, inputs=image_input, outputs=image_out)

    with gr.Tab("🛡️ Network Intrusion"):
        gr.Markdown("Input 77 packet features (comma-separated NSL-KDD mapping):")
        network_input = gr.Textbox(placeholder="0.0, 1.0, 0.0, 255.0...", label="Packet Vector")
        network_btn = gr.Button("Analyze Packet")
        network_out = gr.Textbox(label="Security Alert")
        network_btn.click(predict_network, inputs=network_input, outputs=network_out)

if __name__ == "__main__":
    demo.launch()
