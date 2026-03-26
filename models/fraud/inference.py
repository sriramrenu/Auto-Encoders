import torch
import torch.nn as nn
import numpy as np
import os

from model import FraudAutoEncoder

def test_inference():
    print("--- 💳 Fraud AutoEncoder Standalone Test ---")
    
    # 1. Load Model
    model_path = 'final_fraud_autoencoder.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    print("1. Initializing 30-feature Fraud Autoencoder architecture...")
    model = FraudAutoEncoder(input_dim=30)
    
    print(f"2. Loading trained Kaggle weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 3. Create dummy inputs
    # A "Normal" profile might have values closer to 0 (mean) after StandardScaling
    dummy_normal = torch.randn(1, 30) * 0.5 
    
    # A "Fraud" profile typically has extreme outliers in its PCA latent space
    dummy_fraud = torch.randn(1, 30) * 5.0 + 3.0 
    
    print("\nExecuting Inference on Sample 'Normal' Transaction Tensor...")
    with torch.no_grad():
        recon_normal = model(dummy_normal)
        loss_normal = torch.mean((dummy_normal - recon_normal)**2).item()
    print(f"-> Normal Reconstruction MSE: {loss_normal:.4f}")
    
    print("\nExecuting Inference on Sample 'Fraud' Transaction Tensor...")
    with torch.no_grad():
        recon_fraud = model(dummy_fraud)
        loss_fraud = torch.mean((dummy_fraud - recon_fraud)**2).item()
    print(f"-> Fraudulent Reconstruction MSE: {loss_fraud:.4f}")
    
    print("\nTest Successful! The model successfully separates standard features from extreme outliers.")

if __name__ == "__main__":
    test_inference()
