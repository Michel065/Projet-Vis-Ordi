import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    #changement par rapport https://medium.com/@saadasif78656/building-a-masked-autoencoder-mae-from-scratch-in-pytorch-a-complete-guide-7e2a8fcf632e
    # pour faire en sorte de prendre directement dans forwrd la version en patchs, du coup ca change la methdoe de train c'est mieux normalement plus claire
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
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

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
    self.attn = nn.MultiheadAttention(emb_size,heads,dropout=dropout,batch_first=True)
    self.norm2 = nn.LayerNorm(emb_size)
    self.mlp = nn.Sequential(
    nn.Linear(emb_size, emb_size * mlp_ratio),
    nn.GELU(),
    nn.Linear(emb_size * mlp_ratio, emb_size))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x),self.norm1(x))[0])
    x = x + self.dropout(self.mlp(self.norm2(x)))
    return x