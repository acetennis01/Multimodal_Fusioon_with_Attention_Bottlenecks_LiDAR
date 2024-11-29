import torch
import torch.nn as nn
import timm
from models.pet_modules import AdaptFormer

class PointCloudEncoder(nn.Module):
    def __init__(self, dim=512):
        super(PointCloudEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(x, x, x)[0]
        x = x + self.mlp(self.norm2(x))
        return self.norm1(x)

class AVmodel(nn.Module):
    def __init__(self, num_classes, num_latents, dim):
        super(AVmodel, self).__init__()

        # RGB Vision Transformer
        self.v2 = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Remove unnecessary layers for RGB processing
        self.v2.pre_logits = nn.Identity()
        self.v2.head = nn.Identity()

        # Learnable projection layer for Point Clouds
        self.pc_proj = nn.Linear(3, 512)

        # Initialize Point Cloud Encoder
        pc_encoder = PointCloudEncoder(dim=512)

        # Encoders
        encoder_layers = []
        for i in range(12):
            encoder_layers.append(
                AdaptFormer(
                    num_latents=num_latents,
                    dim=dim,
                    pc_enc=pc_encoder,  # Use the PointCloudEncoder here
                    rgb_enc=self.v2.blocks[i],
                )
            )
        self.pointcloud_rgb_blocks = nn.Sequential(*encoder_layers)

        # Final normalization layers
        self.rgb_post_norm = self.v2.norm

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward_pc_features(self, pc):
        pc = self.pc_proj(pc)  # Linear projection
        return pc

    # def forward_rgb_features(self, x):
    #     B, no_of_frames, C, H, W = x.shape
    #     x = x.reshape(B * no_of_frames, C, H, W)  # Flatten batch and frames
    #     x = self.v2.patch_embed(x)  # Pass through patch embedding
    #     _, dim, h, w = x.shape
    #     x = x.reshape(B, no_of_frames, dim, h, w)
    #     x = x.permute(0, 2, 1, 3, 4).reshape(B, dim, -1).permute(0, 2, 1)  # Flatten spatial and temporal dimensions
    #     x = torch.cat((self.v2.cls_token.expand(B, -1, -1), x), dim=1)  # Add class token
    #     x = x + nn.functional.interpolate(
    #         self.v2.pos_embed.permute(0, 2, 1), x.shape[1], mode="linear"
    #     ).permute(0, 2, 1)  # Add positional embeddings
    #     return x

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
        for blk in self.pointcloud_rgb_blocks:
            pc, rgb = blk(pc, rgb)
        pc = self.rgb_post_norm(pc)
        rgb = self.rgb_post_norm(rgb)
        pc = pc[:, 0]  # Extract class token
        rgb = rgb[:, 0]  # Extract class token
        return pc, rgb

    def forward(self, pc, rgb):
        pc = self.forward_pc_features(pc)  # Process point clouds
        rgb = self.forward_rgb_features(rgb)  # Process RGB frames
        pc, rgb = self.forward_encoder(pc, rgb)  # Encode features
        logits = (pc + rgb) * 0.5  # Combine modalities
        logits = self.classifier(logits)  # Classify
        return logits
