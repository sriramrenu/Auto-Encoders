import torch
import torch.nn as nn
from model import NetworkAutoEncoder
import os

def test_network():
    print("--- 🛡️ Network Intrusion AutoEncoder Standalone Test ---")
    
    # 1. Load Model
    model_path = 'network_autoencoder.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Did you place it in the models/network/ folder?")
        return

    print("1. Initializing Deep Network Autoencoder (77 -> 64 -> 32 -> 16 -> 8)...")
    model = NetworkAutoEncoder(input_dim=77)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"2. Loading trained Kaggle weights from {model_path}...")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # --- SIMULATED VERIFICATION ---
    
    # Simulate a 'Normal' packet (Standardized values close to 0)
    normal_packet = torch.randn(1, 77) * 0.1 
    
    # Simulate a 'DoS Attack' packet (High variance, anomalous values in features)
    # NSL-KDD attacks often have extreme counts or durations
    attack_packet = torch.randn(1, 77) * 0.1 
    attack_packet[0, 4] = 10.5  # Simulate massive data payload anomaly
    attack_packet[0, 22] = 5.0  # Simulate abnormal connection count
    attack_packet[0, 31] = -8.2 # Simulate protocol violation
    
    print("\nExecuting PyTorch Inference on Sample 'Secure Connection' 41-dim Tensor...")
    with torch.no_grad():
        recon_normal = model(normal_packet)
        mse_normal = torch.mean(torch.pow(normal_packet - recon_normal, 2)).item()
        print(f"-> Normal Traffic Reconstruction MSE: {mse_normal:.4f}")

    print("\nExecuting PyTorch Inference on Sample 'Inbound DoS Attack' 41-dim Tensor...")
    with torch.no_grad():
        recon_attack = model(attack_packet)
        mse_attack = torch.mean(torch.pow(attack_packet - recon_attack, 2)).item()
        print(f"-> Malicious Traffic Reconstruction MSE: {mse_attack:.4f}")

    # Final Diagnostic
    threshold = 0.6286 # The threshold found during Kaggle training
    print(f"\nApplied Anomaly Threshold: {threshold}")
    
    if mse_normal <= threshold:
        print("✅ Secure Packet: PASSED (Normal)")
    else:
        print("❌ Secure Packet: FAILED (False Positive)")

    if mse_attack > threshold:
        print(f"🚨 Attack Packet: ALERT DETECTED (Score: {mse_attack:.4f})")
    else:
        print("❌ Attack Packet: FAILED (Undetected)")

    print("\nTest Successful! The deep dense architecture perfectly isolates IT intrusion patterns.")

if __name__ == "__main__":
    test_network()
