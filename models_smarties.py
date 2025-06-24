from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

from utils.model_utils import get_2d_sincos_pos_embed, SpectrumAwareProjection
from utils.utils import get_dtype

class SMARTIES(nn.Module):
    def __init__(self, spectrum_specs, mixed_precision, img_size=224, patch_size=16, in_chans=12,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, **kwargs):
        super().__init__()

        self.in_c = in_chans
        self.dtype = get_dtype(mixed_precision)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.spectrum_projection = SpectrumAwareProjection(
            spectrum_specs=spectrum_specs,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        self.num_patches = int(img_size / patch_size)**2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.projection_scaler = 12

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_preds = torch.nn.ModuleList()
        for band_idx in sorted(spectrum_specs,key=lambda key:spectrum_specs[key]['projection_idx']):
            if ((spectrum_specs[band_idx]['projection_idx'] != -1) and (len(spectrum_specs[band_idx]['agg_projections']) == 0)) :
                self.decoder_preds.append(nn.Linear(decoder_embed_dim, patch_size**2, bias=True))

        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x1, x2, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """
        N, L, D = x1.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x1.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        x1_masked = torch.gather(x1, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x2_masked = torch.gather(x2, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x1.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x1_masked, x2_masked, mask, ids_restore

    def mixup_pairs(self, x1, x2, target1, target2, proj_indices1, proj_indices2, mixup_ratio=0.5):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """
        N, L, D = x1.shape  # batch, length, dim
        len_x1 = int(L * (1 - mixup_ratio))
        noise = torch.rand(N, L, device=x1.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # keep the first subset
        ids_keep_x1 = ids_shuffle[:, :len_x1]
        ids_keep_x2 = ids_shuffle[:, len_x1:]

        ids_keep_x1_full = ids_keep_x1.unsqueeze(-1).repeat(1, 1, D)
        ids_keep_x2_full = ids_keep_x2.unsqueeze(-1).repeat(1, 1, D)
        x1 = torch.zeros_like(x1).scatter_(
            1, ids_keep_x1_full, torch.gather(x1, dim=1, index=ids_keep_x1_full)).scatter_(
                1, ids_keep_x2_full, torch.gather(x2, dim=1, index=ids_keep_x2_full)
            )
        x2 = torch.zeros_like(x2).scatter_(
            1, ids_keep_x2_full, torch.gather(x1, dim=1, index=ids_keep_x2_full)).scatter_(
                1, ids_keep_x1_full, torch.gather(x2, dim=1, index=ids_keep_x1_full)
            )

        ids_keep_x1_full = ids_keep_x1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, target1.shape[-2], target1.shape[-1])
        ids_keep_x2_full = ids_keep_x2.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, target2.shape[-2], target2.shape[-1])
        target1 = torch.zeros_like(target1).scatter_(
            1, ids_keep_x1_full, torch.gather(target1, dim=1, index=ids_keep_x1_full)).scatter_(
                1, ids_keep_x2_full, torch.gather(target2, dim=1, index=ids_keep_x2_full)
            )
        target2 = torch.zeros_like(target2).scatter_(
            1, ids_keep_x2_full, torch.gather(target1, dim=1, index=ids_keep_x2_full)).scatter_(
                1, ids_keep_x1_full, torch.gather(target2, dim=1, index=ids_keep_x1_full)
            )
        
        ids_keep_x1_full = ids_keep_x1.unsqueeze(-1).repeat(1, 1, proj_indices1.shape[-1])
        ids_keep_x2_full = ids_keep_x2.unsqueeze(-1).repeat(1, 1, proj_indices2.shape[-1])
        proj_indices1 = torch.zeros_like(proj_indices1).scatter_(
            1, ids_keep_x1_full, torch.gather(proj_indices1, dim=1, index=ids_keep_x1_full)).scatter_(
                1, ids_keep_x2_full, torch.gather(proj_indices2, dim=1, index=ids_keep_x2_full)
            )
        proj_indices2 = torch.zeros_like(proj_indices2).scatter_(
            1, ids_keep_x2_full, torch.gather(proj_indices1, dim=1, index=ids_keep_x2_full)).scatter_(
                1, ids_keep_x1_full, torch.gather(proj_indices2, dim=1, index=ids_keep_x1_full)
            )
        
        return x1, x2, target1, target2, proj_indices1, proj_indices2
    
    def forward_loss(self, target, pred, mask):
        """
        target: [N, L, nb_bands*nb_pixels]
        pred: [N, L, nb_bands*nb_pixels]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / (mask.sum()+1e-6)  # mean loss on removed patches
        return loss
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            
            x = blk(x)
        x = self.decoder_norm(x)
        return x

    def forward_features(self, batch):
        img_patches, img_projection_indices = batch
        B, nb_patch_h, nb_patch_w, nb_bands, _, _ = img_patches.shape
        device = img_patches.device

        img_spectrum_embeds = torch.zeros((B, nb_patch_h, nb_patch_w, nb_bands, self.embed_dim), device=device, dtype=self.dtype)

        for projection_idx in torch.unbind(torch.unique(img_projection_indices)):
            mask = (img_projection_indices==projection_idx)
            img_spectrum_embeds[mask] = self.spectrum_projection(img_patches[mask], projection_idx) 

        img_embeddings = self.projection_scaler*img_spectrum_embeds.mean(dim=3)
        img_embeddings = img_embeddings.reshape(-1,nb_patch_h*nb_patch_w,self.embed_dim)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )
        x = torch.cat((cls_tokens, img_embeddings), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward_encoder(self, batch, mask_ratio):
        first_img_patches, second_img_patches, first_proj_indices, second_proj_indices = batch
        B, nb_patch_h, nb_patch_w, nb_bands, _, _ = first_img_patches.shape
        device = first_img_patches.device

        first_img_spectrum_embeds = torch.zeros((B, nb_patch_h, nb_patch_w, nb_bands, self.embed_dim), device=device, dtype=self.dtype)
        second_img_spectrum_embeds = torch.zeros((B, nb_patch_h, nb_patch_w, nb_bands, self.embed_dim), device=device, dtype=self.dtype)

        for projection_idx in torch.unbind(torch.unique(first_proj_indices)):
            mask = (first_proj_indices==projection_idx)
            first_img_spectrum_embeds[mask] = self.spectrum_projection(first_img_patches[mask], projection_idx) 

        for projection_idx in torch.unbind(torch.unique(second_proj_indices)):
            mask = (second_proj_indices==projection_idx) 
            second_img_spectrum_embeds[mask] = self.spectrum_projection(second_img_patches[mask], projection_idx)

        first_img_embeds = self.projection_scaler*first_img_spectrum_embeds.mean(dim=3)
        second_img_embeds = self.projection_scaler*second_img_spectrum_embeds.mean(dim=3)
        
        first_img_embeds = first_img_embeds.reshape(-1,nb_patch_h*nb_patch_w,self.embed_dim)
        second_img_embeds = second_img_embeds.reshape(-1,nb_patch_h*nb_patch_w,self.embed_dim)

        first_img_patches = first_img_patches.reshape(B, nb_patch_h*nb_patch_w, nb_bands, -1)
        second_img_patches = second_img_patches.reshape(B, nb_patch_h*nb_patch_w, nb_bands, -1)
        first_proj_indices = first_proj_indices.reshape(B, -1, nb_bands)
        second_proj_indices = second_proj_indices.reshape(B, -1, nb_bands)

        x1, x2, target1, target2, proj_indices1, proj_indices2 = self.mixup_pairs(
            first_img_embeds, 
            second_img_embeds, 
            first_img_patches, 
            second_img_patches, 
            first_proj_indices,
            second_proj_indices,
            mixup_ratio=0.5)
        
        # add pos embed w/o cls token
        x1 = x1 + self.pos_embed[:, 1:, :]
        x2 = x2 + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x1, x2, mask, ids_restore = self.random_masking(x1, x2, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x1 = torch.cat((cls_tokens, x1), dim=1)
        x2 = torch.cat((cls_tokens, x2), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x1 = blk(x1)
            x2 = blk(x2)
        latent1 = self.norm(x1)
        latent2 = self.norm(x2)
        
        return latent1, latent2, target1, target2, proj_indices1, proj_indices2, mask, ids_restore
    
    def forward(self, batch, mask_ratio=0.75, knn_feats=False):
        if knn_feats:
            return self.forward_features(batch)
        latent1, latent2, target1, target2, proj_indices1, proj_indices2, mask, ids_restore = self.forward_encoder(batch, mask_ratio)

        decoder_out1 = self.forward_decoder(latent1, ids_restore)
        decoder_out2 = self.forward_decoder(latent2, ids_restore)

        # predictor projection
        (B, nb_patches, nb_bands, nb_pixels) = target1.shape
        device = target1.device

        pred1 = torch.zeros((B, nb_patches+1, nb_bands, nb_pixels), device=device, dtype=self.dtype)
        pred2 = torch.zeros((B, nb_patches+1, nb_bands, nb_pixels), device=device, dtype=self.dtype)

        for projection_idx in torch.unbind(torch.unique(proj_indices1)):
            band_mask = proj_indices1==projection_idx
            band_mask = torch.cat((torch.ones((band_mask.shape[0],1,band_mask.shape[-1]), dtype=torch.bool, device=device), band_mask), dim=1)
            pred1[band_mask] = self.decoder_preds[projection_idx](decoder_out1.unsqueeze(-2).repeat(1,1,nb_bands,1)[band_mask])
        
        for projection_idx in torch.unbind(torch.unique(proj_indices2)):
            band_mask = proj_indices2==projection_idx
            band_mask = torch.cat((torch.ones((band_mask.shape[0],1,band_mask.shape[-1]), dtype=torch.bool, device=device), band_mask), dim=1)
            pred2[band_mask] = self.decoder_preds[projection_idx](decoder_out2.unsqueeze(-2).repeat(1,1,nb_bands,1)[band_mask])

        # remove cls token
        pred1 = pred1[:, 1:, :, :].reshape(B, nb_patches, -1)
        pred2 = pred2[:, 1:, :, :].reshape(B, nb_patches, -1)

        target1 = target1.reshape(B, nb_patches, -1)
        target2 = target2.reshape(B, nb_patches, -1)

        loss = 0.5*(self.forward_loss(target1, pred1, mask) + self.forward_loss(target2, pred2, mask))

        return loss, (pred1, pred2), mask

def smarties_vit_base_patch16_dec512d8b(**kwargs):
    model = SMARTIES(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def smarties_vit_large_patch16_dec512d8b(**kwargs):
    model = SMARTIES(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def smarties_vit_huge_patch14_dec512d8b(**kwargs):
    model = SMARTIES(
        embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


smarties_vit_base_patch16 = smarties_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
smarties_vit_large_patch16 = smarties_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
smarties_vit_huge_patch14 = smarties_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
