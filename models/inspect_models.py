import torch
import sys

def inspect_model(path):
    print(f"Inspecting {path}")
    try:
        model = torch.load(path, map_location=torch.device('cpu'))
        if isinstance(model, dict):
            print("Keys in state_dict:")
            for k, v in model.items():
                print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
        else:
            print("Model is a full object:", type(model))
            print(model)
    except Exception as e:
        print("Error loading model:", e)
    print("-" * 40)

if __name__ == "__main__":
    inspect_model("fraud/final_fraud_autoencoder.pth")
    inspect_model("image/final_mnist_autoencoder.pth")
    inspect_model("network/network_autoencoder.pth")
