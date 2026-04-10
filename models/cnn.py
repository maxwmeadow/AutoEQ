import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.perceptron = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
        )

    def forward(self, x):
        average = x.mean(dim=[2, 3])
        max = x.amax(dim=[2, 3])
        scale = torch.sigmoid(self.perceptron(average) + self.perceptron(max))
        return x * scale.unsqueeze(2).unsqueeze(3)
        
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        average = x.mean(dim=1, keepdim=True)
        max = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([average, max], dim=1)))
    
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))

class AutoEQ(nn.Module):
    def __init__(self):
        super(AutoEQ, self).__init__()

        self.feature_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        
        self.feature_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0)), 
            nn.BatchNorm2d(64),  
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.feature_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2)
        )
        self.feature_block_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2)
        )

        self.cbam_1 = CBAM(32)
        self.cbam_2 = CBAM(64)
        self.cbam_3 = CBAM(128)
        self.cbam_4 = CBAM(256)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head_low  = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3))
        self.head_mid  = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
        self.head_high = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, spectrogram):
        x = self.cbam_1(self.feature_block_1(spectrogram))
        x = self.cbam_2(self.feature_block_2(x))
        x = self.cbam_3(self.feature_block_3(x))
        x = self.cbam_4(self.feature_block_4(x))
        pooled_features = self.global_pool(x).view(x.size(0), -1)
        return torch.cat([self.head_low(pooled_features), self.head_mid(pooled_features), self.head_high(pooled_features)], dim=1)