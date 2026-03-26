import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import NetworkAutoEncoder

# --- 1. CONFIGURATION ---
# IMPORTANT: Provide the full path to your NSL-KDD KDDTrain+.txt file downloaded from Kaggle.
DATA_PATH = "KDDTrain+.txt" 
BATCH_SIZE = 256
EPOCHS = 20 # Increased for deeper network convergence
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Using device: {DEVICE}")

    # --- 2. DATA PREPARATION ---
    print("Loading Dataset...")
    
    # NSL-KDD typically doesn't have headers in the pure txt. 
    # Usually, column 41 is the class label, and col 1,2,3 are categorical.
    # Below is a standard generic preprocessing block for KDD data.
    try:
        df = pd.read_csv(DATA_PATH, header=None)
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}. Please download NSL-KDD from Kaggle.")
        return

    # Assuming standard NSL-KDD txt: 
    # Col 41 is the attack type ('normal', 'neptune', etc.)
    label_col = 41
    
    # Filter for ONLY 'normal' packets.
    normal_df = df[df[label_col] == 'normal'].copy()
    
    # Drop the label and the score (last column 42 if it exists)
    drop_cols = [label_col]
    if 42 in normal_df.columns:
        drop_cols.append(42)
    normal_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # One-Hot Encode categorical columns 1, 2, 3 (protocol_type, service, flag)
    # Note: On Kaggle, you'd fit this OneHotEncoder on BOTH train/test together to ensure feature shapes match!
    # Using get_dummies for simplicity in this script.
    normal_df = pd.get_dummies(normal_df, columns=[1, 2, 3])

    X_train = normal_df.astype(float).values

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Convert to PyTorch Tensors
    tensor_x = torch.Tensor(X_train_scaled)
    dataset = TensorDataset(tensor_x, tensor_x) 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 3. MODEL INITIALIZATION ---
    input_dim = X_train_scaled.shape[1] 
    
    model = NetworkAutoEncoder(input_dim=input_dim).to(DEVICE)
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
    save_path = "network_autoencoder.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete. Weights saved to {save_path}.")

    # --- 6. EVALUATION & ACCURACY (NSL-KDD) ---
    print("\nEvaluating Accuracy over the entire dataset (Normal vs Intrusions)...")
    model.eval()
    with torch.no_grad():
        # Get losses for normal data
        normal_tensor = torch.Tensor(X_train_scaled).to(DEVICE)
        normal_reconstructions = model(normal_tensor)
        normal_loss = torch.mean(torch.pow(normal_tensor - normal_reconstructions, 2), dim=1)
        
        # Set threshold at the 95th percentile of normal data loss
        threshold = torch.quantile(normal_loss, 0.95).item()
        
        # Get losses for INTRUSION/ATTACK data
        attack_df = df[df[label_col] != 'normal'].copy()
        attack_cols = [label_col, 42] if 42 in attack_df.columns else [label_col]
        attack_df.drop(columns=attack_cols, inplace=True, errors='ignore')
        
        # Standardize matching OHE columns
        attack_df = pd.get_dummies(attack_df, columns=[1, 2, 3])
        # Realign columns to exactly match the training layout
        attack_df = attack_df.reindex(columns=normal_df.columns, fill_value=0)
        
        X_attack = attack_df.astype(float).values
        X_attack_scaled = scaler.transform(X_attack)
        attack_tensor = torch.Tensor(X_attack_scaled).to(DEVICE)
        
        attack_reconstructions = model(attack_tensor)
        attack_loss = torch.mean(torch.pow(attack_tensor - attack_reconstructions, 2), dim=1)
        
        # Calculate Accuracy
        tn = torch.sum(normal_loss <= threshold).item()
        fp = torch.sum(normal_loss > threshold).item()
        tp = torch.sum(attack_loss > threshold).item()
        fn = torch.sum(attack_loss <= threshold).item()
        
        total = tn + fp + tp + fn
        accuracy = (tn + tp) / total if total > 0 else 0
        
        print(f"-> Selected Anomaly Threshold: {threshold:.4f}")
        print(f"-> Accuracy: {accuracy * 100:.2f}% (Goal: >90%)")
        print(f"-> True Positives (Caught Intrusions): {tp}/{len(attack_loss)}")

if __name__ == "__main__":
    main()
