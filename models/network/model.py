import torch
import torch.nn as nn

class NetworkAutoEncoder(nn.Module):
    """
    Dense AutoEncoder for Network Intrusion Logs.
    Expected Input: (Batch_Size, 41) 
    (Typical for datasets like NSL-KDD after one-hot encoding categorical variables)
    """
    def __init__(self, input_dim=77):
        super(NetworkAutoEncoder, self).__init__()
        
        # Deeper Encoder for precise IT intrusion detection
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
            # No Sigmoid; StandardScaler requires unbounded predictions
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
