import torch
import torch.nn as nn

class FraudAutoEncoder(nn.Module):
    """
    Standard Dense AutoEncoder for Tabular Fraud Data.
    Expected Input: (Batch_Size, 30)
    """
    def __init__(self, input_dim=30):
        super(FraudAutoEncoder, self).__init__()
        
        # Deeper Encoder for tighter learning
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True)
        )
        
        # Deeper Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
