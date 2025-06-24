import numpy as np
import torch
import torch.nn as nn

class SpectrumRangeProjection(nn.Module):
    """Patch Embedding of a sensor without patchify"""
    def __init__(
            self,
            spectral_range,
            spectrum_spec,
            patch_size,
            embed_dim,
            bias=True
    ):
        super().__init__()
        self.spectral_range = spectral_range
        self.name = spectrum_spec['name']
        self.min_wavelength = spectrum_spec['min_wavelength']
        self.max_wavelength = spectrum_spec['max_wavelength']
        self.sensors = spectrum_spec['sensors']
        self.nb_pixels = patch_size**2
        self.proj = nn.Linear(self.nb_pixels, embed_dim, bias=bias)
    
    def forward(self, x):
        return self.proj(x.view(-1, self.nb_pixels)) 

class SpectrumRangeProjectionAvg(nn.Module):
    """Patch Embedding of a sensor without patchify"""
    def __init__(
            self,
            spectrum_projections,
            spectrum_spec,
            embed_dim
    ):
        super().__init__()
        self.min_wavelength = spectrum_spec['min_wavelength']
        self.max_wavelength = spectrum_spec['max_wavelength']
        self.central_lambda = 0.5*(float(self.min_wavelength) + float(self.max_wavelength))
        self.spectrum_projections = spectrum_projections
        self.weights = []
        for spectrum_proj in self.spectrum_projections:
            central_lambda = 0.5*(float(spectrum_proj.min_wavelength) + float(spectrum_proj.max_wavelength))
            self.weights.append(abs(self.central_lambda-central_lambda))
        self.weights = np.array(self.weights) / sum(self.weights)
        self.embed_dim = embed_dim
    
    def forward(self, x):
        out = 0. #torch.zeros((len(x),self.embed_dim))
        for i, spectrum_proj in enumerate(self.spectrum_projections):
            out += spectrum_proj(x) * self.weights[i]
        return out
    

class SpectrumAwareProjection(nn.Module):
    """Patch Embedding of a sensor without patchify"""
    def __init__(
            self,
            spectrum_specs,
            patch_size,
            embed_dim,
            bias=True
    ):
        super().__init__()
        self.nb_pixels = patch_size**2
    
        self.spectrum_embeds = torch.nn.ModuleList()
        for spectral_range in sorted(spectrum_specs,key=lambda key:spectrum_specs[key]['projection_idx']):
            if ((spectrum_specs[spectral_range]['projection_idx'] != -1) and (len(spectrum_specs[spectral_range]['agg_projections']) == 0)) :
                self.spectrum_embeds.append(SpectrumRangeProjection(
                    spectral_range, spectrum_specs[spectral_range], patch_size, embed_dim
                ))

        for spectral_range in sorted(spectrum_specs,key=lambda key:spectrum_specs[key]['projection_idx']): 
            if ((spectrum_specs[spectral_range]['projection_idx'] != -1) and (len(spectrum_specs[spectral_range]['agg_projections']) > 0)):
                self.spectrum_embeds.append(
                    SpectrumRangeProjectionAvg(
                        [self.spectrum_embeds[agg_proj_idx] for agg_proj_idx in spectrum_specs[spectral_range]['agg_projections']], 
                        spectrum_specs[spectral_range],
                        embed_dim))
                
    def forward(self, x, projection_idx):
        return self.spectrum_embeds[projection_idx](x)

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model, num_extra_tokens=1):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.pos_embed.shape[-2] - num_extra_tokens

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed

def tensor_patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h, w, p, p, imgs.shape[1])).permute(0,1,2,5,3,4)
    return x

def apply_label_mixup_fn(batch, mixup_fn, patch_size):
    imgs, img_projection_indices, targets = batch
    imgs, targets = mixup_fn(imgs, targets)
    img_patches = tensor_patchify(imgs, patch_size)
    return (img_patches, img_projection_indices, targets)