import torch
import torch.nn as nn
from . import blocks as bl
from .blocks import build_sincos_pos_embed


def random_masking(x, mask_ratio=0.75):
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    bruit = torch.rand(B, L, device=x.device)

    ids_shuffle = torch.argsort(bruit, dim=1)
    ids_keep    = ids_shuffle[:, :len_keep]

    indices_keep = ids_keep.unsqueeze(-1).repeat(1, 1, D)
    x_masked     = torch.gather(x, dim=1, index=indices_keep)

    indice_restore = torch.argsort(ids_shuffle, dim=1)

    mask = torch.ones(B, L, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=indice_restore)

    return x_masked, indice_restore, mask


def block_masking(x, mask_ratio=0.75, grid_size=6):
    B, L, D = x.shape
    assert L == grid_size * grid_size, f"L={L} ne correspond pas à grid_size={grid_size}"

    n_mask   = int(L * mask_ratio)
    n_keep   = L - n_mask

    all_ids_keep    = []
    all_ids_restore = []
    all_masks       = []

    for b in range(B):
        mask_2d = torch.zeros(grid_size, grid_size, device=x.device)

        block_h = max(1, int((n_mask ** 0.5) * 0.9))
        block_w = max(1, int(n_mask / block_h))
        block_h = min(block_h, grid_size)
        block_w = min(block_w, grid_size)

        # Point de départ aléatoire (le bloc reste dans la grille)
        r0 = torch.randint(0, grid_size - block_h + 1, (1,)).item()
        c0 = torch.randint(0, grid_size - block_w + 1, (1,)).item()
        mask_2d[r0:r0 + block_h, c0:c0 + block_w] = 1.0

        # Complète si le bloc ne couvre pas assez (bords, grilles non carrées)
        current = int(mask_2d.sum().item())
        if current < n_mask:
            remaining = n_mask - current
            flat = mask_2d.view(-1)
            unmasked_ids = (flat == 0).nonzero(as_tuple=True)[0]
            perm = unmasked_ids[torch.randperm(len(unmasked_ids))[:remaining]]
            flat[perm] = 1.0
            mask_2d = flat.view(grid_size, grid_size)

        mask_flat = mask_2d.view(-1)  # (L,) - 1=masqué, 0=visible

        # Construit ids_keep (patches visibles en premier) + ids_restore
        visible_ids = (mask_flat == 0).nonzero(as_tuple=True)[0]
        masked_ids  = (mask_flat == 1).nonzero(as_tuple=True)[0]

        # ids_shuffle : visible d'abord, masqué ensuite
        ids_shuffle = torch.cat([visible_ids, masked_ids])
        ids_restore = torch.argsort(ids_shuffle)

        all_ids_keep.append(visible_ids)
        all_ids_restore.append(ids_restore)
        all_masks.append(mask_flat)

    # Regroupe en tenseurs batch
    ids_keep_batch    = torch.stack(all_ids_keep,    dim=0)  # (B, n_keep)
    ids_restore_batch = torch.stack(all_ids_restore, dim=0)  # (B, L)
    mask_batch        = torch.stack(all_masks,       dim=0)  # (B, L)

    indices_keep = ids_keep_batch.unsqueeze(-1).repeat(1, 1, D)
    x_masked     = torch.gather(x, dim=1, index=indices_keep)

    return x_masked, ids_restore_batch, mask_batch


class MAE(nn.Module):
    def __init__(self, img_size=96, patch_size=16, in_channels=3,
                 embed_dim=768, decoder_embed_dim=512,
                 mask_ratio=0.75, use_block_masking=True):
        super().__init__()

        self.mask_ratio        = mask_ratio
        self.use_block_masking = use_block_masking

        # Embedding de l'image (pos_embed sinusoïdal fixe)
        self.patch_embedder = bl.PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embedder.num_patches
        self.grid_size = img_size // patch_size  # 6 pour 96px / 16px

        # Encodeur - blocs lourds (heads=12, mlp_ratio=4)
        self.encoder_blocks = nn.ModuleList([
            bl.TransformerEncoderBlock(emb_size=embed_dim) for _ in range(4)
        ])
        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_embed_dim)

        # Token masque et pos_embed décodeur
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim))
        decoder_pos_embed = build_sincos_pos_embed(num_patches, decoder_embed_dim)
        self.register_buffer("decoder_pos_embed", decoder_pos_embed)

        # Décodeur allégé (heads=8, mlp_ratio=2)
        self.decoder_blocks = nn.ModuleList([
            bl.TransformerDecoderBlock(emb_size=decoder_embed_dim) for _ in range(2)
        ])

        pixels_per_patch = in_channels * patch_size * patch_size
        self.decoder_pred = nn.Linear(decoder_embed_dim, pixels_per_patch)

    def forward(self, images_patchs):
        # --- ENCODEUR ---
        x = self.patch_embedder(images_patchs)

        cls_token = x[:, :1, :]
        patches   = x[:, 1:, :]

        # Choix de la stratégie de masquage
        if self.use_block_masking:
            patches_mask, restore, mask = block_masking(
                patches, mask_ratio=self.mask_ratio, grid_size=self.grid_size
            )
        else:
            patches_mask, restore, mask = random_masking(
                patches, mask_ratio=self.mask_ratio
            )

        x = torch.cat((cls_token, patches_mask), dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)

        x = self.encoder_to_decoder(x)

        # --- DECODEUR ---
        cls_token = x[:, :1, :]
        patches   = x[:, 1:, :]

        B        = x.shape[0]
        num_mask = restore.shape[1] - patches.shape[1]
        mask_tokens = self.mask_token.repeat(B, num_mask, 1)

        x_ = torch.cat([patches, mask_tokens], dim=1)
        restore_expand = restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        x_ = torch.gather(x_, dim=1, index=restore_expand)

        x = torch.cat([cls_token, x_], dim=1)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        pred = self.decoder_pred(x[:, 1:, :])
        return pred, mask