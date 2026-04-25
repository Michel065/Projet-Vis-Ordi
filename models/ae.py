import torch.nn as nn
from . import blocks as bl

class AE(nn.Module):
  """
  Un AE simple qui utilie nos TransformerEncoderBlock qui sont conçue sur un model Vit. pour encoder et decoder l'image entiere
  """
  def __init__(self, img_size=96, patch_size=16, in_channels=3, embed_dim=768, decoder_embed_dim=576):
    super().__init__()

    self.patch_embedder = bl.PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

    self.encoder_blocks = nn.Sequential(*[bl.TransformerEncoderBlock(emb_size=embed_dim) for i in range(4)])

    self.encoder_to_decoder = nn.Linear(embed_dim, decoder_embed_dim)

    self.decoder_blocks = nn.Sequential(*[bl.TransformerEncoderBlock(emb_size=decoder_embed_dim) for i in range(2)])

    pixels_per_patch = in_channels * patch_size * patch_size
    self.decoder_pred = nn.Linear(decoder_embed_dim, pixels_per_patch)

  def forward(self, images_patchs):
    x = self.patch_embedder(images_patchs)

    for blk in self.encoder_blocks:
      x = blk(x)

    x = self.encoder_to_decoder(x)

    for blk in self.decoder_blocks:
      x = blk(x)

    patches = x[:, 1:, :]
    x = self.decoder_pred(patches)

    return x
  
  def get_encoder_output(self, images_patchs):
    #methode bonus pour simplifier l'utilisation apres dans nos tests
    x = self.patch_embedder(images_patchs)
    for blk in self.encoder_blocks:
        x = blk(x)
    return x