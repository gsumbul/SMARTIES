from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer
from utils.model_utils import get_2d_sincos_pos_embed, SpectrumAwareProjection, tensor_patchify
from utils.utils import get_dtype

class SmartiesVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(
        self, global_pool=False, all_tokens=False, spectrum_specs=None, multi_modal=False, num_sources=1, mixed_precision='no', **kwargs
    ):
        super(SmartiesVisionTransformer, self).__init__(**kwargs)
        del self.patch_embed

        self.dtype = get_dtype(mixed_precision)
        self.patch_size = kwargs['patch_size']
        self.spectrum_projection = SpectrumAwareProjection(
            spectrum_specs=spectrum_specs,
            patch_size=self.patch_size,
            embed_dim=kwargs["embed_dim"]
        )

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(kwargs['img_size']/self.patch_size),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.projection_scaler = 12
        self.all_tokens = all_tokens
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            self.fc_norm = norm_layer(kwargs["embed_dim"])
            del self.norm
        
        if multi_modal:
            del self.head
            self.num_sources = num_sources
            self.head = nn.Linear(self.embed_dim*self.num_sources, kwargs['num_classes'])
        self.multi_modal = multi_modal

    def forward_encoder(self, batch, is_patchify):
        imgs, proj_indices = batch
        if is_patchify:
            img_patches = tensor_patchify(imgs, self.patch_size)
        else:
            img_patches = imgs
        B, nb_patch_h, nb_patch_w, nb_bands, _, _ = img_patches.shape
        device = img_patches.device

        img_spectrum_embeds = torch.zeros((B, nb_patch_h, nb_patch_w, nb_bands, self.embed_dim), device=device, dtype=self.dtype)

        for projection_idx in torch.unbind(torch.unique(proj_indices)):
            mask = (proj_indices==projection_idx)
            img_spectrum_embeds[mask] = self.spectrum_projection(img_patches[mask], projection_idx) 

        img_embeddings = self.projection_scaler*img_spectrum_embeds.mean(dim=3)
        img_embeddings = img_embeddings.reshape(-1,nb_patch_h*nb_patch_w,self.embed_dim)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )
        x = torch.cat((cls_tokens, img_embeddings), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.all_tokens:
            return x[:, 1:, :].permute(0,2,1).reshape(-1, self.embed_dim, nb_patch_h, nb_patch_w)
        
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, is_patchify=False):
        if not self.multi_modal:
            x = self.forward_encoder(x, is_patchify)
        else:
            feats = []
            for i in range(self.num_sources):
                feats.append(self.forward_encoder(x[i], is_patchify))
            if self.all_tokens:
                x = torch.cat(feats, dim=-3)
            else:
                x = torch.cat(feats, dim=-1)
        x = self.head(x)
        return x

def vit_base_patch16(**kwargs):
    model = SmartiesVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = SmartiesVisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    model = SmartiesVisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
