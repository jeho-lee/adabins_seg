import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class PatchTransformerEncoder(nn.Module):
#     def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4):
#         super(PatchTransformerEncoder, self).__init__()
#         encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # S, N, E
#         self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
#         self.positional_encodings = nn.Parameter(torch.rand(5000, embedding_dim), requires_grad=True)
        
#     def forward(self, x):
#         embeddings = self.embedding_convPxP(x).flatten(2)  # [N, E, S]
#         embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)
#         embeddings = embeddings.permute(2, 0, 1)
#         x = self.transformer_encoder(embeddings)  # [S, N, E]
#         return x
class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # S, N, E
        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True) # TODO: num patches
        
    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2)  # [N, E, S]
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0) # TODO: num patches
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # [S, N, E]
        return x

class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size() # [N, S, E]
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # [N, HW, C_out]
        return y.permute(0, 2, 1).view(n, cout, h, w)  # [N, C_out, H, W]


# class mViT(nn.Module):
#     def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
#                  embedding_dim=128, num_heads=4, norm='linear'):
#         super(mViT, self).__init__()
#         self.norm = norm
#         self.n_query_channels = n_query_channels
#         self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
#         self.dot_product_layer = PixelWiseDotProduct()
#         self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         # x: [N, C, H, W]
#         tgt = self.patch_transformer(x.clone())  # [S, N, E]
#         x = self.conv3x3(x)
#         # Assume dim_out corresponds to the number of classes
#         queries = tgt[1:self.n_query_channels + 1, ...]  # [n_query_channels, N, E]
#         # Change from S, N, E to N, S, E
#         queries = queries.permute(1, 0, 2)  # [N, S, E]
#         attention_maps = self.dot_product_layer(x, queries)  # [N, S, H, W]
#         return attention_maps  # [N, S, H, W]
class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size=patch_size, embedding_dim=embedding_dim, num_heads=num_heads)
        self.dot_product_layer = PixelWiseDotProduct()
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: [N, C, H, W] -> decoder output
        
        # Get global information for each patch
        transformer_output = self.patch_transformer(x)  # [S, N, E], S: Number of patches, N: Batch size, E: Embedding dimension
        
        x = self.conv3x3(x)
        
        queries = transformer_output[1:self.n_query_channels + 1, ...]  # [n_query_channels, N, E]
        queries = queries.permute(1, 0, 2)  # [N, S, E]
        attention_maps = self.dot_product_layer(x, queries)  # [N, S, H, W]
        return attention_maps  # [N, S, H, W]


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU()
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=256, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSampleBN(skip_input=features + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=(features // 2) + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=(features // 4) + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=(features // 8) + 16 + 8, output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)

        out = self.conv3(x_d4)
        return out


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UNet_Transformer_Adabins(nn.Module):
    def __init__(self, backend, n_classes=19, norm='linear'):
        super(UNet_Transformer_Adabins, self).__init__()
        self.num_classes = n_classes
        self.encoder = Encoder(backend)
        self.decoder = DecoderBN(num_classes=128)  # Assuming embedding_dim=128 for Transformer

        # Transformer Module adapted for segmentation
        self.transformer_module = mViT(
            in_channels=128,  # Should match decoder's output channels
            n_query_channels=n_classes,  # One query per class
            patch_size=16,
            dim_out=n_classes,
            embedding_dim=128,
            norm=norm
        )

        # Segmentation Head
        self.segmentation_head = nn.Conv2d(n_classes, n_classes, kernel_size=1, stride=1, padding=0)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, **kwargs):
        encoder_features = self.encoder(x)
        decoder_output = self.decoder(encoder_features)
        attention_maps = self.transformer_module(decoder_output)  # [N, n_classes, H, W]
        segmentation_maps = self.segmentation_head(attention_maps)  # [N, n_classes, H, W]

        return segmentation_maps

    @classmethod
    def build(cls, backbone_name, n_classes, **kwargs):
        print('Loading base model ({})...'.format(backbone_name), end='')
        backbone = torch.hub.load('rwightman/gen-efficientnet-pytorch', backbone_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        backbone.global_pool = nn.Identity()
        backbone.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(backend=backbone, n_classes=n_classes, **kwargs)
        print('Done.')
        return m
