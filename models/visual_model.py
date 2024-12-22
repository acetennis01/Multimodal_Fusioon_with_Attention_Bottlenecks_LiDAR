# visual_model.py

import torch
import torch.nn as nn
import timm
from models.pet_modules import AdaptFormer
import torch.nn.functional as F

class PointCloudEncoder(nn.Module):
    def __init__(self, dim):
        super(PointCloudEncoder, self).__init__()

        # Convolutional layers to process the pseudo-image
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Input: 3 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        # Update the fully connected layer to match the flattened size
        self.fc = nn.Linear(512 * 14 * 14, dim)  # Changed from 512*16*16 to 512*14*14

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # print(f"Input to PointCloudEncoder: {x.shape}")

        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"After conv1: {x.shape}")
        x = F.max_pool2d(x, 2)
        # print(f"After pool1: {x.shape}")
        
        x = F.relu(self.bn2(self.conv2(x)))
        # print(f"After conv2: {x.shape}")
        x = F.max_pool2d(x, 2)
        # print(f"After pool2: {x.shape}")

        x = F.relu(self.bn3(self.conv3(x)))
        # print(f"After conv3: {x.shape}")
        x = F.max_pool2d(x, 2)
        # print(f"After pool3: {x.shape}")

        x = F.relu(self.bn4(self.conv4(x)))
        # print(f"After conv4: {x.shape}")
        x = F.max_pool2d(x, 2)
        # print(f"After pool4: {x.shape}")

        # Now we adaptively pool to 14x14, regardless of input size.
        x = F.adaptive_avg_pool2d(x, (14, 14))
        # print(f"After adaptive_avg_pool2d: {x.shape}")  # [B, 512, 14, 14]

        # Flatten once
        x = x.flatten(start_dim=1)  # [B, 512 * 14 * 14]
        # print(f"After flatten: {x.shape}")

        self.fc = nn.Linear(512 * 14 * 14, 768)

        # Pass through the fully connected layer
        x = self.fc(x)  # [B, dim]
        # print(f"After fc: {x.shape}")

        # Apply dropout for regularization
        x = self.dropout(x)

        return x

class AVmodel(nn.Module):
    def __init__(self, num_classes, num_latents, dim):
        super(AVmodel, self).__init__()

        # RGB Vision Transformer (ViT)
        self.v2 = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Remove unnecessary layers for RGB processing
        self.v2.pre_logits = nn.Identity()
        self.v2.head = nn.Identity()

        # Initialize Point Cloud Encoder
        self.pc_encoder = PointCloudEncoder(dim=dim)

        # Define Transformer blocks (AdaptFormer without encoders)
        encoder_layers = []
        for i in range(12):
            encoder_layers.append(
                AdaptFormer(
                    num_latents=num_latents,
                    dim=768
                )
            )
        self.pointcloud_rgb_blocks = nn.Sequential(*encoder_layers)

        # Final normalization layers for both RGB and Point Cloud Encoders
        self.rgb_post_norm = self.v2.norm

        # Classifier head for classification
        self.classifier = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward_pc_features(self, pc):
        # Process pseudo-image LiDAR data through the PointCloudEncoder
        pc = self.pc_encoder(pc)  # [B, dim]
        # # print(f"AVmodel - PC Features after Encoder: {pc.shape}")
        return pc

    def forward_rgb_features(self, x):
        # Ensure input is 5D: (batch_size, no_of_frames, channels, height, width)
        if len(x.shape) != 5:
            raise ValueError(f"Expected input of shape (B, no_of_frames, C, H, W), but got {x.shape}")
        
        B, no_of_frames, C, H, W = x.shape

        # Flatten batch and frames for patch embedding
        x = x.reshape(B * no_of_frames, C, H, W)

        # Pass through patch embedding
        x = self.v2.patch_embed(x)  # Shape: (batch_size * no_of_frames, num_patches, embed_dim)

        if len(x.shape) != 3:  # Check for 3D output
            raise ValueError(f"Expected 3D output from patch embedding, but got {x.shape}")
        
        # Unpack the output shape
        _, num_patches, dim = x.shape

        # Reshape back to include frames
        x = x.reshape(B, no_of_frames, num_patches, dim)

        # Flatten spatial and temporal dimensions
        x = x.permute(0, 3, 1, 2).reshape(B, dim, -1).permute(0, 2, 1)  # Shape: (B, no_of_tokens, dim)
        # print(f"AVmodel - RGB Features after Patch Embed and Reshape: {x.shape}")

        # Add class token
        cls_token = self.v2.cls_token.expand(B, -1, -1)  # [B,1,dim]
        x = torch.cat((cls_token, x), dim=1)  # [B, 1 + no_of_tokens, dim]
        # print(f"AVmodel - RGB Features after Adding Class Token: {x.shape}")

        # Add positional embeddings
        pos_embed = self.v2.pos_embed.permute(0, 2, 1)  # [B, dim, seq_len]
        if pos_embed.shape[1] != x.shape[1]:
            pos_embed = nn.functional.interpolate(pos_embed, size=x.shape[1], mode="linear")
        x = x + pos_embed.permute(0, 2, 1)  # [B, 1 + no_of_tokens, dim]
        # print(f"AVmodel - RGB Features after Adding Positional Embeddings: {x.shape}")

        return x  # [B, 1 + num_tokens, dim]

    def forward_encoder(self, pc, rgb):
        # Ensure pc has shape [B, 1, dim]
        pc = pc.unsqueeze(1)  # [B, 1, dim]
        # print(f"AVmodel - PC Features after Unsqueeze: {pc.shape}")

        for idx, blk in enumerate(self.pointcloud_rgb_blocks):
            # print(f"AVmodel - Processing AdaptFormer Block {idx + 1}")
            pc, rgb = blk(pc, rgb)
            # print(f"AVmodel - After Block {idx + 1}: PC: {pc.shape}, RGB: {rgb.shape}")

        # Post-processing (norm) for both modalities
        pc = self.rgb_post_norm(pc)  # [B, 1, dim]
        rgb = self.rgb_post_norm(rgb)  # [B, 1 + num_tokens, dim]
        # print(f"AVmodel - After Normalization: PC: {pc.shape}, RGB: {rgb.shape}")

        # Extract class tokens
        pc = pc[:, 0]  # [B, dim]
        rgb = rgb[:, 0]  # [B, dim]
        # print(f"AVmodel - Extracted Class Tokens: PC: {pc.shape}, RGB: {rgb.shape}")

        return pc, rgb

    def forward(self, pc, rgb):
        # Process point cloud (pseudo-image) and RGB features
        pc = self.forward_pc_features(pc)  # [B, dim]
        rgb = self.forward_rgb_features(rgb)  # [B, 1 + num_tokens, dim]

        # Process through the encoder (fusion of modalities)
        pc, rgb = self.forward_encoder(pc, rgb)  # [B, dim], [B, dim]

        # Combine features from both modalities and classify
        logits = (pc + rgb) * 0.5  # [B, dim]
        logits = self.classifier(logits)  # [B, num_classes]
        # print(f"AVmodel - Logits Shape: {logits.shape}")

        return logits
