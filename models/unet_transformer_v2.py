import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

def disable_inplace_relu(model):
    """
    Recursively disable in-place ReLU activations in a given model.
    """
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
        elif isinstance(module, nn.ReLU6):
            module.inplace = False
        # Add other activation types if necessary
    return model

class AdaBinsTransformerSegmentation(nn.Module):
    def __init__(self, in_channels, n_classes, embedding_dim=256, num_heads=8, num_layers=4, patch_size=16):
        super(AdaBinsTransformerSegmentation, self).__init__()
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Learnable Class Embeddings
        self.class_embeddings = nn.Parameter(torch.randn(n_classes, embedding_dim))
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1 + n_classes, embedding_dim))
        
        # Projection Layers
        self.global_proj = nn.Linear(in_channels, embedding_dim)
        self.feature_proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, padding=1)
        
        # Activation (Disabled In-Place)
        self.activation = nn.ReLU(inplace=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_normal_(self.global_proj.weight, mode='fan_out', nonlinearity='relu')
        if self.global_proj.bias is not None:
            nn.init.constant_(self.global_proj.bias, 0)
        nn.init.kaiming_normal_(self.feature_proj.weight, mode='fan_out', nonlinearity='relu')
        if self.feature_proj.bias is not None:
            nn.init.constant_(self.feature_proj.bias, 0)
        nn.init.normal_(self.class_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)
        
    def forward(self, decoder_features):
        # decoder_features: [N, C, H, W]
        N, C, H, W = decoder_features.shape

        # Step 1: Extract Global Context Vector
        global_context = F.adaptive_avg_pool2d(decoder_features, 1).view(N, C)  # [N, C]
        global_context = self.global_proj(global_context)  # [N, embedding_dim]
        global_context = self.activation(global_context)
        
        # Step 2: Prepare Transformer Input
        # Expand global context for each class
        global_context = global_context.unsqueeze(0)  # [1, N, embedding_dim]
        
        # Expand class embeddings for the batch
        class_embeddings = self.class_embeddings.unsqueeze(1).expand(-1, N, -1)  # [n_classes, N, embedding_dim]
        
        # Concatenate global context with class embeddings
        transformer_input = torch.cat([global_context, class_embeddings], dim=0)  # [1 + n_classes, N, embedding_dim]
        
        # Add Positional Encodings
        transformer_input = transformer_input + self.positional_encoding[:, None, :]  # Broadcasting over batch
        
        # Step 3: Transformer Encoding
        transformer_output = self.transformer_encoder(transformer_input)  # [1 + n_classes, N, embedding_dim]
        
        # Step 4: Generate Class-Specific Kernels
        class_kernels = transformer_output[1:]  # [n_classes, N, embedding_dim]
        class_kernels = class_kernels.permute(1, 0, 2)  # [N, n_classes, embedding_dim]
        
        # Step 5: Project Decoder Features
        projected_features = self.feature_proj(decoder_features)  # [N, embedding_dim, H, W]
        
        # Step 6: Reshape for Batch Matrix Multiplication
        projected_features = projected_features.view(N, self.embedding_dim, H * W)  # [N, embedding_dim, H*W]
        class_kernels = class_kernels  # [N, n_classes, embedding_dim]
        
        # Step 7: Perform Batch Matrix Multiplication
        attention_maps = torch.bmm(class_kernels, projected_features)  # [N, n_classes, H*W]
        
        # Step 8: Reshape to [N, n_classes, H, W]
        attention_maps = attention_maps.view(N, self.n_classes, H, W)  # [N, n_classes, H, W]
        
        return attention_maps

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)  # Disabled In-Place
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        residual = self.residual(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out += residual
        return out

class Decoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super(Decoder, self).__init__()
        self.up_blocks = nn.ModuleList()
        in_channels = encoder_channels[0]
        for i in range(len(decoder_channels)):
            skip_channels = encoder_channels[i + 1]
            out_channels = decoder_channels[i]
            self.up_blocks.append(UpSampleBlock(in_channels + skip_channels, out_channels))
            in_channels = out_channels

    def forward(self, features):
        x = features[0]
        for i in range(len(self.up_blocks)):
            skip = features[i + 1]
            x = self.up_blocks[i](x, skip)
        return x

class UNet_Transformer_V2(nn.Module):
    def __init__(self, encoder_name='efficientnet_b5', n_classes=19):
        super(UNet_Transformer_V2, self).__init__()
        self.encoder = create_model(encoder_name, pretrained=True, features_only=True)
        
        # Disable in-place ReLU activations in the encoder
        self.encoder = disable_inplace_relu(self.encoder)
        
        encoder_channels = self.encoder.feature_info.channels()
        encoder_channels = encoder_channels[::-1]  # Reverse to match decoder order
        decoder_channels = [256, 128, 64, 32]
        self.decoder = Decoder(encoder_channels, decoder_channels)
        
        self.transformer_module = AdaBinsTransformerSegmentation(
            in_channels=decoder_channels[-1],
            n_classes=n_classes,
            embedding_dim=256,  # Adjust as needed
            num_heads=8,
            num_layers=4,
            patch_size=16  # If needed for positional encoding
        )
        
        # Final Segmentation Map
        self.final_conv = nn.Conv2d(n_classes, n_classes, kernel_size=1)
        # Optionally include activation here if needed
        # self.final_activation = nn.Softmax(dim=1)

        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.final_conv.bias is not None:
            nn.init.constant_(self.final_conv.bias, 0)
        
    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()
    
    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.transformer_module]
        for m in modules:
            yield from m.parameters()
            
    def forward(self, x):
        features = self.encoder(x)
        features = features[::-1]  # Reverse to match decoder order
        x = self.decoder(features)
        attention_maps = self.transformer_module(x)
        
        # Final Segmentation Map
        output = self.final_conv(attention_maps)  # [N, n_classes, H, W]
        # Optionally apply activation
        # output = self.final_activation(output)
        return output
