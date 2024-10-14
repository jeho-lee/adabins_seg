import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=16, embedding_dim=None, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        self.patch_size = patch_size
        if embedding_dim is None:
            embedding_dim = in_channels
        self.embedding_conv = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encodings = nn.Parameter(torch.rand(10000, embedding_dim), requires_grad=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4) # Adabins uses 4 layers

    def forward(self, x):
        embeddings = self.embedding_conv(x)  # [N, E, H', W']
        H_prime, W_prime = embeddings.shape[2], embeddings.shape[3]
        S = H_prime * W_prime  # Total number of patches
        embeddings = embeddings.flatten(2)  # [N, E, S]
        embeddings = embeddings + self.positional_encodings[:S, :].T.unsqueeze(0)  # [N, E, S]
        embeddings = embeddings.permute(2, 0, 1)  # [S, N, E]
        x = self.transformer_encoder(embeddings)  # [S, N, E]
        x = x.permute(1, 2, 0)  # [N, E, S]
        x = x.view(x.size(0), x.size(1), H_prime, W_prime)  # [N, E, H', W']
        
        # Upsample to match the size before the embedding_conv
        x = F.interpolate(x, size=(H_prime * self.patch_size, W_prime * self.patch_size), mode='bilinear', align_corners=False)
        return x

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
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, n_classes):
        super(Decoder, self).__init__()
        self.up_blocks = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()

        in_channels = encoder_channels[0]  # Start from bottleneck

        for i in range(len(decoder_channels)):
            skip_channels = encoder_channels[i + 1]  # Corresponding skip connection channels
            out_channels = decoder_channels[i]
            self.up_blocks.append(UpSampleBlock(in_channels + skip_channels, out_channels))
            self.transformer_blocks.append(
                PatchTransformerEncoder(
                    in_channels=out_channels,
                    embedding_dim=out_channels  # Ensure embedding_dim matches out_channels
                )
            )
            in_channels = out_channels  # Update for next block

        self.final_conv = nn.Conv2d(out_channels, n_classes, kernel_size=1)

    def forward(self, features):
        x = features[0]  # Bottleneck features
        for i in range(len(self.up_blocks)):
            skip = features[i + 1]  # Corresponding encoder features
            x = self.up_blocks[i](x, skip)
            x = self.transformer_blocks[i](x)
        x = self.final_conv(x)
        return x

class UNet_Transformer(nn.Module):
    def __init__(self, encoder_name='efficientnet_b5', n_classes=19):
        super(UNet_Transformer, self).__init__()
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
    model = UNet_Transformer(n_classes=19)
    x = torch.rand(4, 3, 512, 1024)
    seg_out = model(x)
    print(seg_out.shape)