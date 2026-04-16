import torch
import torch.nn as nn
from . import blocks as bl

def random_masking(x, mask_ratio=0.75):
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    bruit = torch.rand(B, L, device=x.device)

    ids_shuffle = torch.argsort(bruit, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]

    indices_keep = ids_keep.unsqueeze(-1).repeat(1, 1, D)
    x_masked = torch.gather(x, dim=1, index=indices_keep)

    indice_restore = torch.argsort(ids_shuffle, dim=1)

    mask = torch.ones(B, L, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=indice_restore)

    return x_masked, indice_restore, mask


class MAE(nn.Module):
    def __init__(self, img_size=96, patch_size=16, in_channels=3, embed_dim=768, decoder_embed_dim=512):
        super().__init__()

        # Embedding de l'image
        self.patch_embedder = bl.PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embedder.num_patches

        # Encodeur
        self.encoder_blocks = nn.Sequential(*[bl.TransformerEncoderBlock(emb_size=embed_dim) for _ in range(4)])

        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_embed_dim)

        # Token de remplacement pour les 75% masqués
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim))

        # Positional embedding du décodeur
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, decoder_embed_dim))

        # Décodeur
        self.decoder_blocks = nn.Sequential(*[bl.TransformerEncoderBlock(emb_size=decoder_embed_dim) for _ in range(2)])

        # Projection finale vers les pixels
        pixels_per_patch = in_channels * patch_size * patch_size
        self.decoder_pred = nn.Linear(decoder_embed_dim, pixels_per_patch)

    def forward(self, images_patchs):
        # --- ENCODEUR ---
        x = self.patch_embedder(images_patchs)

        cls_token = x[:, :1, :]
        patches = x[:, 1:, :]

        patches_mask, restore, mask = random_masking(patches, mask_ratio=0.75)
        x = torch.cat((cls_token, patches_mask), dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)

        x = self.encoder_to_decoder(x)

        # --- DECODEUR ---
        cls_token = x[:, :1, :]
        patches = x[:, 1:, :]

        B = x.shape[0]
        num_mask = restore.shape[1] - patches.shape[1]
        mask_tokens = self.mask_token.repeat(B, num_mask, 1)

        x_ = torch.cat([patches, mask_tokens], dim=1)

        restore_expand = restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        x_ = torch.gather(x_, dim=1, index=restore_expand)

        x = torch.cat([cls_token, x_], dim=1)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        patches_out = x[:, 1:, :]
        pred = self.decoder_pred(patches_out)

        return pred, mask