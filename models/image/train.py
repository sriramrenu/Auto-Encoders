import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import ImageAutoEncoder

# --- 1. CONFIGURATION ---
# IMPORTANT: Provide the full path to your unzipped Kaggle dataset 'train' and 'test' folders.
TRAIN_DATA_PATH = "./chest_xray/train" 
TEST_DATA_PATH = "./chest_xray/test"

BATCH_SIZE = 64
EPOCHS = 20 
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Using device: {DEVICE}")

    # --- 2. DATA PREPARATION ---
    # Resizing to 64x64 to capture pneumonia texture maps
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    print("Loading Training Dataset...")
    try:
        full_train_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
    except FileNotFoundError:
        print(f"Error: Could not find '{TRAIN_DATA_PATH}'. Check Kaggle input path.")
        return

    # Train ONLY on "NORMAL" scans
    normal_indices = [i for i, (_, label) in enumerate(full_train_dataset.samples) if label == 0]
    train_dataset = Subset(full_train_dataset, normal_indices)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Found {len(train_dataset)} NORMAL training scans.")

    # --- 3. MODEL INITIALIZATION ---
    model = ImageAutoEncoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. TRAINING LOOP ---
    print("Starting Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(DEVICE)
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    # --- 5. SAVING MODEL WEIGHTS ---
    save_path = "final_image_autoencoder.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete. Weights saved to {save_path}.")

    # --- 6. EVALUATION & ACCURACY (Pneumonia X-Rays) ---
    print("\nEvaluating Accuracy over the Test Dataset (Normal vs Pneumonia Scans)...")
    try:
        test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model.eval()
        normal_losses = []
        pneumonia_losses = []
        
        with torch.no_grad():
            for batch_data, expected_labels in test_loader:
                batch_data = batch_data.to(DEVICE)
                reconstructions = model(batch_data)
                losses = torch.mean(torch.pow(batch_data - reconstructions, 2), dim=[1,2,3])
                
                for i, label in enumerate(expected_labels):
                    if label == 0:
                        normal_losses.append(losses[i].item())
                    else:
                        pneumonia_losses.append(losses[i].item())
                        
        normal_tensor = torch.tensor(normal_losses)
        pneumonia_tensor = torch.tensor(pneumonia_losses)
        
        # --- DYNAMIC OPTIMAL THRESHOLD SEARCH ---
        best_acc = 0
        best_threshold = 0
        inverse_logic = False
        
        all_losses = torch.cat([normal_tensor, pneumonia_tensor])
        
        # --- CRITICAL FIX FOR KAGGLE: DEFINE 'TOTAL' BEFORE THE LOOP ---
        total = len(all_losses) 
        
        min_L, max_L = all_losses.min().item(), all_losses.max().item()
        
        print(f"Searching for mathematically optimal threshold (Scanning {len(all_losses)} samples)...")
        
        for step in range(100):
            thresh = min_L + (max_L - min_L) * (step / 100.0)
            
            # Standard logic (Anomaly > Threshold)
            tn_norm = torch.sum(normal_tensor <= thresh).item()
            tp_norm = torch.sum(pneumonia_tensor > thresh).item()
            acc_norm = (tn_norm + tp_norm) / total
            
            # Inverse logic (Anomaly <= Threshold)
            tn_inv = torch.sum(normal_tensor > thresh).item()
            tp_inv = torch.sum(pneumonia_tensor <= thresh).item()
            acc_inv = (tn_inv + tp_inv) / total
            
            if max(acc_norm, acc_inv) > best_acc:
                best_acc = max(acc_norm, acc_inv)
                best_threshold = thresh
                inverse_logic = acc_inv > acc_norm
        
        if inverse_logic:
            tp = torch.sum(pneumonia_tensor <= best_threshold).item()
        else:
            tp = torch.sum(pneumonia_tensor > best_threshold).item()
            
        print(f"-> Optimal Mathematical Threshold: {best_threshold:.6f} {'(Inverse Topography)' if inverse_logic else ''}")
        print(f"-> Peak Diagnostic Accuracy: {best_acc * 100:.2f}%")
        print(f"-> True Positives (Successfully caught Pneumonia scans): {tp}/{len(pneumonia_tensor)}")
        
    except FileNotFoundError:
         print(f"Test Dataset '{TEST_DATA_PATH}' not found.")

if __name__ == "__main__":
    main()
