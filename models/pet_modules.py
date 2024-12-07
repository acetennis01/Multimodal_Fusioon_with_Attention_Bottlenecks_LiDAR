# pet_modules.py

import torch
import torch.nn as nn

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AdaptFormer(nn.Module):
    def __init__(self, num_latents, dim):
        super(AdaptFormer, self).__init__()

        # Adapter parameters
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

        # Down and Up Projection for Point Cloud (pseudo-image)
        self.pc_down = nn.Linear(dim, dim)
        self.pc_up = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.pc_down.weight)
        nn.init.zeros_(self.pc_down.bias)
        nn.init.xavier_uniform_(self.pc_up.weight)
        nn.init.zeros_(self.pc_up.bias)
        self.pc_scale = nn.Parameter(torch.ones(1))

        # Down and Up Projection for RGB
        self.rgb_down = nn.Linear(dim, dim)
        self.rgb_up = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.rgb_down.weight)
        nn.init.zeros_(self.rgb_down.bias)
        nn.init.xavier_uniform_(self.rgb_up.weight)
        nn.init.zeros_(self.rgb_up.bias)
        self.rgb_scale = nn.Parameter(torch.ones(1))

        # Latents
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1, num_latents, dim).normal_(std=0.02))
        self.scale_pc = nn.Parameter(torch.zeros(1))
        self.scale_v = nn.Parameter(torch.zeros(1))

        # Define attention layers
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8)

    def attention(self, q, k, v):
        # q, k, v: [seq_len, batch, dim]
        attn_output, attn_weights = self.attn(q, k, v)
        return attn_output

    def fusion(self, pc_tokens, visual_tokens):
        print(f"Fusion Input - PC Tokens: {pc_tokens.shape}, Visual Tokens: {visual_tokens.shape}")

        BS, pc_len, dim = pc_tokens.shape
        _, vis_len, _ = visual_tokens.shape

        # Concatenate along the sequence length
        concat_ = torch.cat((pc_tokens, visual_tokens), dim=1)  # [B, 1 + num_tokens, dim]
        print(f"Concatenated Tokens: {concat_.shape}")

        # Permute for MultiheadAttention: [seq_len, batch, dim]
        concat_ = concat_.permute(1, 0, 2)  # [1 + num_tokens, B, dim]
        print(f"Permuted Concatenated Tokens: {concat_.shape}")

        # Initialize latents for cross-attention
        latents = self.latents.expand(BS, -1, -1).permute(1, 0, 2)  # [num_latents, B, dim]
        print(f"Latents: {latents.shape}")

        # Cross-attention: latents attend to concat_
        fused_latents = self.attention(latents, concat_, concat_)  # [num_latents, B, dim]
        print(f"Fused Latents: {fused_latents.shape}")

        # Permute back to [B, num_latents, dim]
        fused_latents = fused_latents.permute(1, 0, 2)  # [B, num_latents, dim]
        print(f"Permuted Fused Latents: {fused_latents.shape}")

        # Cross-attention: pc_tokens attend to fused_latents
        pc_attn = self.attention(
            pc_tokens.permute(1, 0, 2),  # [1, B, dim]
            fused_latents.permute(1, 0, 2),  # [num_latents, B, dim]
            fused_latents.permute(1, 0, 2)   # [num_latents, B, dim]
        )  # [1, B, dim]
        print(f"PC Attention Output: {pc_attn.shape}")

        pc_attn = pc_attn.permute(1, 0, 2)  # [B, 1, dim]
        print(f"PC Attention Permuted: {pc_attn.shape}")

        pc_tokens = pc_tokens + self.pc_scale * pc_attn  # [B,1,dim]
        print(f"PC Tokens after Attention and Scaling: {pc_tokens.shape}")

        # Cross-attention: visual_tokens attend to fused_latents
        visual_attn = self.attention(
            visual_tokens.permute(1, 0, 2),  # [num_tokens, B, dim]
            fused_latents.permute(1, 0, 2),  # [num_latents, B, dim]
            fused_latents.permute(1, 0, 2)   # [num_latents, B, dim]
        )  # [num_tokens, B, dim]
        print(f"Visual Attention Output: {visual_attn.shape}")

        visual_attn = visual_attn.permute(1, 0, 2)  # [B, num_tokens, dim]
        print(f"Visual Attention Permuted: {visual_attn.shape}")

        visual_tokens = visual_tokens + self.rgb_scale * visual_attn  # [B, num_tokens, dim]
        print(f"Visual Tokens after Attention and Scaling: {visual_tokens.shape}")

        return pc_tokens, visual_tokens

    def forward_pc_AF(self, x):
        x_down = self.pc_down(x)  # [B,1,dim] -> [B,1,dim]
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.pc_up(x_down)  # [B,1,dim]
        return x_up * self.pc_scale  # [B,1,dim]

    def forward_visual_AF(self, x):
        x_down = self.rgb_down(x)  # [B,num_tokens,dim] -> [B,num_tokens,dim]
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.rgb_up(x_down)  # [B,num_tokens,dim]
        return x_up * self.rgb_scale  # [B,num_tokens,dim]

    def forward(self, pc, rgb):
        # pc: [B, 1, dim]
        # rgb: [B, num_tokens, dim]

        print(f"AdaptFormer Forward - PC Features: {pc.shape}")
        print(f"AdaptFormer Forward - RGB Features: {rgb.shape}")

        # Fusion: [B,1,dim], [B, num_tokens, dim]
        pc_features, rgb_features = self.fusion(pc, rgb)  # [B,1,dim], [B,num_tokens,dim]

        # Feed-forward Network + Skip Connections
        pc_features = pc_features + self.forward_pc_AF(pc_features)  # [B,1,dim]
        rgb_features = rgb_features + self.forward_visual_AF(rgb_features)  # [B,num_tokens,dim]

        print(f"AdaptFormer Forward - After FFN: PC Features: {pc_features.shape}, RGB Features: {rgb_features.shape}")

        return pc_features, rgb_features
