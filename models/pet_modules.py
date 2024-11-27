import torch
import torch.nn as nn

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AdaptFormer(nn.Module):
    def __init__(self, num_latents, dim, pc_enc, rgb_enc):
        super(AdaptFormer, self).__init__()

        # Point Cloud
        # Attention Layer
        self.pc_norm1 = pc_enc.norm1
        self.pc_attn = pc_enc.attn
        # Feed Forward Layers
        self.pc_norm2 = pc_enc.norm2
        self.pc_mlp = pc_enc.mlp

        # RGB
        # Attention Layer
        self.rgb_norm1 = rgb_enc.norm1
        self.rgb_attn = rgb_enc.attn
        # Feed Forward Layers
        self.rgb_norm2 = rgb_enc.norm2
        self.rgb_mlp = rgb_enc.mlp

        # Adapter parameters
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

        # Point Cloud
        self.pc_down = nn.Linear(768, dim)  # Assuming point cloud input has shape (B, N_points, 3)
        self.pc_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.pc_down.weight)
        nn.init.zeros_(self.pc_down.bias)
        nn.init.zeros_(self.pc_up.weight)
        nn.init.zeros_(self.pc_up.bias)
        self.pc_scale = nn.Parameter(torch.ones(1))

        # RGB images
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

    # Latent Fusion
    def fusion(self, pc_tokens, visual_tokens):
        # shapes
        BS = pc_tokens.shape[0]
        # concat all the tokens
        concat_ = torch.cat((pc_tokens, visual_tokens), dim=1)
        # cross attention (PC+RGB -->> latents)
        fused_latents = self.attention(q=self.latents.expand(BS, -1, -1), k=concat_, v=concat_)
        # cross attention (latents -->> PC+RGB)
        pc_tokens = pc_tokens + self.scale_pc * self.attention(q=pc_tokens, k=fused_latents, v=fused_latents)
        visual_tokens = visual_tokens + self.scale_v * self.attention(q=visual_tokens, k=fused_latents, v=fused_latents)
        return pc_tokens, visual_tokens

    # def forward_pc_AF(self, x):
    #     x_down = self.pc_down(x)
    #     x_down = self.act(x_down)
    #     x_down = self.dropout(x_down)
    #     x_up = self.pc_up(x_down)
    #     return x_up

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

    def forward(self, pc, y):
        # Bottleneck Fusion
        pc, y = self.fusion(pc, y)

        # Attn skip connections
        # pc = pc + self.pc_attn(self.pc_norm1(pc))
        pc = pc + self.pc_attn(self.pc_norm1(pc), self.pc_norm1(pc), self.pc_norm1(pc))[0]
        y = y + self.rgb_attn(self.rgb_norm1(y))

        # FFN + skip connections
        pc = pc + self.pc_mlp(self.pc_norm2(pc)) + self.forward_pc_AF(self.pc_norm2(pc)) * self.pc_scale
        y = y + self.rgb_mlp(self.rgb_norm2(y)) + self.forward_visual_AF(self.rgb_norm2(y)) * self.rgb_scale

        return pc, y
