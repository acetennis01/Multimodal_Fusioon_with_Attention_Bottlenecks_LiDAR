import torch
import torch.nn as nn

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AdaptFormer(nn.Module):
    def __init__(self, num_latents, dim, pc_enc, rgb_enc):
        super(AdaptFormer, self).__init__()

        # Point Cloud Encoder Components (from the PointCloudEncoder)
        self.pc_enc = pc_enc  # Expecting PointCloudEncoder that produces a 768-dim feature vector

        # RGB Encoder Components (from Vision Transformer block)
        self.rgb_enc = rgb_enc

        # Adapter parameters
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

        # Down and Up Projection for Point Cloud (pseudo-image)
        self.pc_down = nn.Linear(768, dim)  # Assuming point cloud input has shape (B, N_points, 768)
        self.pc_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.pc_down.weight)
        nn.init.zeros_(self.pc_down.bias)
        nn.init.zeros_(self.pc_up.weight)
        nn.init.zeros_(self.pc_up.bias)
        self.pc_scale = nn.Parameter(torch.ones(1))

        # Down and Up Projection for RGB
        self.rgb_down = nn.Linear(768, dim)
        self.rgb_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.rgb_down.weight)
        nn.init.zeros_(self.rgb_down.bias)
        nn.init.zeros_(self.rgb_up.weight)
        nn.init.zeros_(self.rgb_up.bias)
        self.rgb_scale = nn.Parameter(torch.ones(1))

        # Latents
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1, num_latents, 768).normal_(std=0.02))
        self.scale_pc = nn.Parameter(torch.zeros(1))
        self.scale_v = nn.Parameter(torch.zeros(1))

    def attention(self, q, k, v):  # requires q, k, v to have same dim
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)  # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x

    def fusion(self, pc_tokens, visual_tokens):
        # shapes
        BS = pc_tokens.shape[0]
        # Concatenate point cloud and RGB tokens
        concat_ = torch.cat((pc_tokens, visual_tokens), dim=1)
        # Cross-attention (PC+RGB → latents)
        fused_latents = self.attention(q=self.latents.expand(BS, -1, -1), k=concat_, v=concat_)
        # Cross-attention (latents → PC+RGB)
        pc_tokens = pc_tokens + self.scale_pc * self.attention(q=pc_tokens, k=fused_latents, v=fused_latents)
        visual_tokens = visual_tokens + self.scale_v * self.attention(q=visual_tokens, k=fused_latents, v=fused_latents)
        return pc_tokens, visual_tokens

    def forward_pc_AF(self, x):
        B, N, C = x.shape  # B: batch size, N: sequence length, C: feature dimension
        x = x.reshape(B * N, C)  # Flatten batch and sequence dimensions
        x_down = self.pc_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.pc_up(x_down)
        x_up = x_up.reshape(B, N, -1)  # Restore original batch and sequence dimensions
        return x_up

    def forward_visual_AF(self, x):
        x_down = self.rgb_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.rgb_up(x_down)
        return x_up

    def forward(self, pc, rgb):
        # Process Point Cloud (LiDAR data via PointCloudEncoder)
        pc_features = self.pc_enc(pc)  # Output shape: (B, 768) (flattened pseudo-image features)

        # Process RGB Frames
        rgb_features = self.rgb_enc(rgb)  # Output shape: (B, 768)

        # Bottleneck Fusion (Cross-Attention between Point Cloud and RGB)
        pc_features, rgb_features = self.fusion(pc_features, rgb_features)

        # Point Cloud Attention (skip connections)
        pc_features = pc_features + self.pc_enc.attn(pc_features, pc_features, pc_features)[0]
        rgb_features = rgb_features + self.rgb_enc.attn(rgb_features, rgb_features, rgb_features)[0]

        # Feed-forward Network + Skip Connections
        pc_features = pc_features + self.pc_enc.mlp(pc_features) + self.forward_pc_AF(pc_features) * self.pc_scale
        rgb_features = rgb_features + self.rgb_enc.mlp(rgb_features) + self.forward_visual_AF(rgb_features) * self.rgb_scale

        return pc_features, rgb_features