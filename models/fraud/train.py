import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import FraudAutoEncoder

# --- 1. CONFIGURATION ---
DATA_PATH = "creditcard.csv" # Download this from Kaggle
BATCH_SIZE = 256
EPOCHS = 20 # Increased to allow loss to drop below 0.1
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Using device: {DEVICE}")

    # --- 2. DATA PREPARATION ---
    print("Loading Dataset...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}. Please download the 'Credit Card Fraud Detection' dataset from Kaggle.")
        return

    # Autoencoders train ONLY on normal data (Class == 0)
    normal_df = df[df['Class'] == 0].copy()
    
    # Drop the target label so the model only sees the features
    X_train = normal_df.drop('Class', axis=1).values

    # Scale the data using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Convert to PyTorch Tensors
    tensor_x = torch.Tensor(X_train_scaled)
    dataset = TensorDataset(tensor_x, tensor_x) # Input and Target are the same
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 3. MODEL INITIALIZATION ---
    # Input dim must match exactly the number of features in your scaled data
    input_dim = X_train_scaled.shape[1] 
    
    model = FraudAutoEncoder(input_dim=input_dim).to(DEVICE)
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
    save_path = "final_fraud_autoencoder.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete. Weights saved to {save_path}.")

    # --- 6. EVALUATION & ACCURACY ---
    print("\nEvaluating Accuracy over the entire dataset...")
    model.eval()
    with torch.no_grad():
        # Get losses for normal data
        normal_tensor = torch.Tensor(X_train_scaled).to(DEVICE)
        normal_reconstructions = model(normal_tensor)
        normal_loss = torch.mean(torch.pow(normal_tensor - normal_reconstructions, 2), dim=1)
        
        # Set threshold at the 95th percentile of normal data loss
        threshold = torch.quantile(normal_loss, 0.95).item()
        
        # Get losses for fraud data
        fraud_df = df[df['Class'] == 1].copy()
        X_fraud = fraud_df.drop('Class', axis=1).values
        X_fraud_scaled = scaler.transform(X_fraud) # Ensure we use the same scaler
        fraud_tensor = torch.Tensor(X_fraud_scaled).to(DEVICE)
        
        fraud_reconstructions = model(fraud_tensor)
        fraud_loss = torch.mean(torch.pow(fraud_tensor - fraud_reconstructions, 2), dim=1)
        
        # Calculate Accuracy
        # True Negatives (Normal predicted as Normal)
        tn = torch.sum(normal_loss <= threshold).item()
        # False Positives (Normal predicted as Fraud)
        fp = torch.sum(normal_loss > threshold).item()
        
        # True Positives (Fraud predicted as Fraud)
        tp = torch.sum(fraud_loss > threshold).item()
        # False Negatives (Fraud predicted as Normal)
        fn = torch.sum(fraud_loss <= threshold).item()
        
        total = tn + fp + tp + fn
        accuracy = (tn + tp) / total
        
        print(f"-> Selected Anomaly Threshold: {threshold:.4f}")
        print(f"-> Accuracy: {accuracy * 100:.2f}% (Goal: >90%)")
        print(f"-> True Positives (Caught Fraud): {tp}/{len(fraud_loss)}")

if __name__ == "__main__":
    main()
