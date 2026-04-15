import torch
import torch.nn as nn
import math


def build_sincos_pos_embed(num_patches, embed_dim):
    h = w = int(num_patches ** 0.5)
    assert h * w == num_patches, "num_patches doit être un carré parfait"

    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=0).float()  # (2, h, w)
    grid = grid.reshape(2, -1).T                          # (num_patches, 2)

    assert embed_dim % 4 == 0, "embed_dim doit être divisible par 4 pour le pos_embed sincos"
    omega = torch.arange(embed_dim // 4, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 4)))  # fréquences

    x_enc = torch.outer(grid[:, 0], omega)  # (num_patches, embed_dim//4)
    y_enc = torch.outer(grid[:, 1], omega)

    pos = torch.cat([x_enc.sin(), x_enc.cos(), y_enc.sin(), y_enc.cos()], dim=1)  # (num_patches, embed_dim)

    # Ajout du CLS token (position zéro)
    cls_pos = torch.zeros(1, embed_dim)
    pos = torch.cat([cls_pos, pos], dim=0)  # (num_patches + 1, embed_dim)
    return pos.unsqueeze(0)  # (1, num_patches + 1, embed_dim)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.proj = nn.Linear(self.patch_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        pos_embed = build_sincos_pos_embed(self.num_patches, embed_dim)
        self.register_buffer("pos_embed", pos_embed)

    def forward(self, patches):
        B = patches.shape[0]

        x = self.proj(patches)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=768, heads=12, dropout=0.1, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(emb_size * mlp_ratio, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.dropout(self.attn(normed, normed, normed)[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, emb_size=512, heads=8, dropout=0.1, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(emb_size * mlp_ratio, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.dropout(self.attn(normed, normed, normed)[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x