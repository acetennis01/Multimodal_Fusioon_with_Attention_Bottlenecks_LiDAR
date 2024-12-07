import torch
import torch.nn as nn
import timm
from models.pet_modules import AdaptFormer
import torch.nn.functional as F

'''
class PointCloudEncoder(nn.Module):
    def __init__(self, dim):
        super(PointCloudEncoder, self).__init__()

        # Convolutional layers to process the pseudo-image
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Input: 3 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        # Optionally, a fully connected layer at the end to reduce the output to the desired `dim`
        self.fc = nn.Linear(512 * 16 * 16, dim)  # Assuming the image is 256x256 and pooling reduces size

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply the convolutional layers with activations and batch normalization

        x = F.relu(self.bn1(self.conv1(x)))  # Shape: (B, 64, 256, 256)
        x = F.max_pool2d(x, 2)  # Pooling: (B, 64, 128, 128)
        
        x = F.relu(self.bn2(self.conv2(x)))  # Shape: (B, 128, 128, 128)
        x = F.max_pool2d(x, 2)  # Pooling: (B, 128, 64, 64)

        x = F.relu(self.bn3(self.conv3(x)))  # Shape: (B, 256, 64, 64)
        x = F.max_pool2d(x, 2)  # Pooling: (B, 256, 32, 32)

        x = F.relu(self.bn4(self.conv4(x)))  # Shape: (B, 512, 32, 32)
        x = F.max_pool2d(x, 2)  # Pooling: (B, 512, 16, 16)

        # Flatten the tensor before passing through the fully connected layer
        x = x.flatten(start_dim=1)  # Shape: (B, 512*16*16)

        # Pass through the fully connected layer to reduce to the desired embedding size `dim`
        x = self.fc(x)  # Shape: (B, dim)

        # Apply dropout for regularization
        x = self.dropout(x)

        return x

'''

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
        print(f"Input to PointCloudEncoder: {x.shape}")  # [B, 3, 224, 224]

        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 224, 224]
        print(f"After conv1: {x.shape}")
        x = F.max_pool2d(x, 2)  # [B, 64, 112, 112]
        print(f"After pool1: {x.shape}")
        
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, 112, 112]
        print(f"After conv2: {x.shape}")
        x = F.max_pool2d(x, 2)  # [B, 128, 56, 56]
        print(f"After pool2: {x.shape}")

        x = F.relu(self.bn3(self.conv3(x)))  # [B, 256, 56, 56]
        print(f"After conv3: {x.shape}")
        x = F.max_pool2d(x, 2)  # [B, 256, 28, 28]
        print(f"After pool3: {x.shape}")

        x = F.relu(self.bn4(self.conv4(x)))  # [B, 512, 28, 28]
        print(f"After conv4: {x.shape}")
        x = F.max_pool2d(x, 2)  # [B, 512, 14, 14]
        print(f"After pool4: {x.shape}")

        # Flatten the tensor before passing through the fully connected layer
        x = x.flatten(start_dim=1)  # [B, 512 *14 *14=100352]
        print(f"After flatten: {x.shape}")

        # Pass through the fully connected layer to reduce to the desired embedding size `dim`
        x = self.fc(x)  # [B, dim=768]
        print(f"After fc: {x.shape}")

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

        # Learnable projection layer for Point Clouds (now for pseudo-image input)
        # self.pc_proj = nn.Conv2d(3, 768, kernel_size=1)  # Convolution for pseudo-image to match embedding size

        # Initialize Point Cloud Encoder
        self.pc_encoder = PointCloudEncoder(dim=768)

        # Define Transformer blocks (AdaptFormer should be similar to a ViT block)
        encoder_layers = []
        for i in range(12):
            encoder_layers.append(
                AdaptFormer(
                    num_latents=num_latents,
                    dim=dim,
                    pc_enc=self.pc_encoder,  # Point cloud encoder to handle pseudo-image input
                    rgb_enc=self.v2.blocks[i],  # RGB Vision Transformer encoder block
                )
            )
        self.pointcloud_rgb_blocks = nn.Sequential(*encoder_layers)

        # Final normalization layers for both RGB and Point Cloud Encoders
        self.rgb_post_norm = self.v2.norm

        # Classifier head for classification
        self.classifier = nn.Sequential(
            nn.Linear(768, num_classes)
        )

    def forward_pc_features(self, pc):
        # Process pseudo-image LiDAR data through convolutional projection layer
        # pc = self.pc_proj(pc)  # Project the 3-channel pseudo-image to 768 dimensions
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

        # Add class token
        cls_token = self.v2.cls_token.expand(B, -1, -1)  # Ensure correct batch size
        x = torch.cat((cls_token, x), dim=1)  # Shape: (B, 1 + no_of_tokens, dim)

        # Add positional embeddings
        pos_embed = self.v2.pos_embed.permute(0, 2, 1)
        if pos_embed.shape[1] != x.shape[1]:
            pos_embed = nn.functional.interpolate(pos_embed, size=x.shape[1], mode="linear")
        x = x + pos_embed.permute(0, 2, 1)

        return x

    def forward_encoder(self, pc, rgb):
        # Process the pseudo-image and RGB features with the transformer encoder blocks
        for blk in self.pointcloud_rgb_blocks:
            pc, rgb = blk(pc, rgb)
        
        # Post-processing (norm) for both modalities
        pc = self.rgb_post_norm(pc)
        rgb = self.rgb_post_norm(rgb)

        # Extract class tokens
        pc = pc[:, 0]  # Shape: (B, dim)
        rgb = rgb[:, 0]  # Shape: (B, dim)
        
        return pc, rgb

    def forward(self, pc, rgb):
        # Process point cloud (pseudo-image) and RGB features
        pc = self.forward_pc_features(pc)  # Process pseudo-image LiDAR data
        rgb = self.forward_rgb_features(rgb)  # Process RGB frames

        # Process through the encoder (fusion of modalities)
        pc, rgb = self.forward_encoder(pc, rgb)

        # Combine features from both modalities and classify
        logits = (pc + rgb) * 0.5  # Average the features (sensor fusion)
        logits = self.classifier(logits)  # Final classification layer

        return logits