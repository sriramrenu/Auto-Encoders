import torch
import torch.nn as nn

class ImageAutoEncoder(nn.Module):
    def __init__(self):
        super(ImageAutoEncoder, self).__init__()
        
        # Deep High-Resolution Encoder (Input: 64x64 Grayscale)
        self.encoder = nn.Sequential(
            # 1 -> 32 channels. Resolution splits to 32x32
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(True),
            
            # 32 -> 64 channels. Resolution splits to 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(True),
            
            # 64 -> 128 channels. Resolution splits to 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            
            # Bottleneck: 128 -> 256 channels. Resolution splits to 4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        # Deep Spatial Decoder (Rebuilding the 64x64 structural footprint)
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   
            nn.ReLU(True),
            
            # 32x32 -> 64x64 Original Form
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),    
            nn.Sigmoid() # Re-bounding the pixel intensities to [0.0, 1.0]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
