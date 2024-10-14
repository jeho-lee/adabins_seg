"""
2024-09-13
- Remove softmax and add conv2d as final layer
- Fetch backbone from timm
- Simplified decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, n_classes):
        super(Decoder, self).__init__()
        self.up_blocks = nn.ModuleList()

        in_channels = encoder_channels[-1]  # Start from the bottleneck

        for i in range(len(decoder_channels)):
            skip_channels = encoder_channels[-(i + 2)]  # Corresponding skip connection channels
            out_channels = decoder_channels[i]
            self.up_blocks.append(UpSampleBlock(in_channels + skip_channels, out_channels))
            in_channels = out_channels  # Update for next block

        self.final_conv = nn.Conv2d(decoder_channels[-1], n_classes, kernel_size=1)

    def forward(self, features):
        x = features[-1]  # Bottleneck features
        for i, up_block in enumerate(self.up_blocks):
            skip = features[-(i + 2)]  # Corresponding encoder features
            x = up_block(x, skip)
        x = self.final_conv(x)
        return x

class UNetV1(nn.Module):
    def __init__(self, encoder_name='efficientnet_b5', n_classes=19):
        super(UNetV1, self).__init__()
        self.encoder = create_model(encoder_name, pretrained=True, features_only=True)
        encoder_channels = self.encoder.feature_info.channels()
        encoder_channels = encoder_channels[::-1]  # Reverse to match decoder order
        decoder_channels = [256, 128, 64, 32]  # Define decoder channels
        self.decoder = Decoder(encoder_channels, decoder_channels, n_classes)

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder]
        for m in modules:
            yield from m.parameters()
            
    def forward(self, x):
        features = self.encoder(x)
        features = features[::-1]  # Reverse to match decoder order
        x = self.decoder(features)
        return x
    
    
if __name__ == '__main__':
    model = UNetV1(n_classes=21)
    x = torch.rand(2, 3, 480, 640)
    seg_out = model(x)
    print(seg_out.shape)
