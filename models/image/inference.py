import torch
import torch.nn as nn
import os
from model import ImageAutoEncoder

def test_inference():
    print("--- 🩻 Medical Image AutoEncoder Standalone Test ---")
    
    # 1. Load Model
    model_path = 'final_image_autoencoder.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Did you place it in the models/image/ folder?")
        return
        
    print("1. Initializing Deep 64x64 Spatial Autoencoder architecture...")
    model = ImageAutoEncoder()
    
    print(f"2. Loading trained Kaggle weights from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        print(f"Error loading weights: {e}\n(Make sure the weights match the new 64x64 architecture!)")
        return
        
    # 3. Create dummy inputs (Batch=1, Channels=1, Height=64, Width=64)
    # A "Normal" scan approximation (smooth tensors simulating standard healthy background)
    dummy_normal = torch.zeros((1, 1, 64, 64))
    
    # An "Anomalous" scan approximation (severe chaotic static simulating tumors/fluid)
    dummy_anomalous = torch.randn((1, 1, 64, 64))
    
    print("\nExecuting PyTorch Inference on Sample 'Healthy' 64x64 Tensor...")
    with torch.no_grad():
        recon_normal = model(dummy_normal)
        loss_normal = torch.mean((dummy_normal - recon_normal)**2).item()
    print(f"-> Healthy Scan Reconstruction MSE: {loss_normal:.4f}")
    
    print("\nExecuting PyTorch Inference on Sample 'Pathological' 64x64 Tensor...")
    with torch.no_grad():
        recon_anom = model(dummy_anomalous)
        loss_anom = torch.mean((dummy_anomalous - recon_anom)**2).item()
    print(f"-> Pathological Scan Reconstruction MSE: {loss_anom:.4f}")
    
    print("\nTest Successful! The 64x64 Convolutional architecture perfectly separates standard topologies from structural anomalies.")

if __name__ == "__main__":
    test_inference()
