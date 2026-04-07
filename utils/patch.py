import torch
import torch.nn as nn

def patchify(x, patch_size=16):
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Patch size non valide, division non entière"

    h = H // patch_size
    w = W // patch_size

    x = x.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, h * w, C * patch_size * patch_size)
    return x

def unpatchify(x, patch_size=16, img_size=96, in_channels=3):
    (B, N, patch_dim) = x.shape
    h = img_size // patch_size
    w = img_size // patch_size
    assert N == h * w, "Nombre de patchs non valide"
    assert patch_dim == in_channels * patch_size * patch_size, "Dimension patch non valide"

    x = x.reshape(B, h, w, in_channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(B, in_channels, img_size, img_size)
    return x