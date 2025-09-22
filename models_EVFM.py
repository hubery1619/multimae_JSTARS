# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
import math
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from crossattention_utils import CrossAttentionBlock

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class EncoderMap(nn.Module):
    def __init__(self, k, embed_dim, *, norm_layer=nn.LayerNorm, decoder_depth):
        super(EncoderMap, self).__init__()
        self.linear = nn.Linear(k, decoder_depth, bias=False)
        
        std_dev = 1. / math.sqrt(k)
        nn.init.normal_(self.linear.weight, mean=0., std=std_dev)

    def forward(self, feature_maps):
        # Ensure the input is a list
        assert isinstance(feature_maps, list), "Input should be a list of feature maps"
        # Ensure the list has the same length as the number of weights
        assert len(feature_maps) == (self.linear.weight.shape[1]), "Number of feature maps and weights should match"
        stacked_feature_maps = torch.stack(feature_maps, dim=-1)  # shape: (B, L, C, k)
        # compute a weighted average of the feature maps
        # decoder_depth is denoted as j
        output = self.linear(stacked_feature_maps)
        return output


class EfficientMaskedAutoencoderGroupChannelViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, spatial_mask=False,
                 channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),
                 channel_embed=256, embed_dim=1024, depth=24, num_heads=16,
                 decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, fusion_weight=False, 
                 use_encoder_feature=[-1], use_ori_img=False, self_attn=False,
                 ):
        super().__init__()

        self.in_c = in_chans
        self.patch_size = patch_size
        self.channel_groups = channel_groups
        self.spatial_mask = spatial_mask  # Whether to mask all channels of same spatial location
        num_groups = len(channel_groups)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = nn.ModuleList([PatchEmbed(img_size, patch_size, len(group), embed_dim)
                                          for group in channel_groups])
        # self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed[0].num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - channel_embed),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.fusion_weight = fusion_weight
        self.use_ori_img = use_ori_img # use input as one of the feature maps
        if len(use_encoder_feature) == 1 and use_encoder_feature[0] == -1:
            self.use_encoder_feature = list(range(depth))
        else:
            self.use_encoder_feature = [i if i >= 0 else depth + i for i in use_encoder_feature]
        if self.fusion_weight:

            dec_norms = []
            for i in range(decoder_depth):
                norm_layer_i = norm_layer(embed_dim)
                dec_norms.append(norm_layer_i)
            self.dec_norms = nn.ModuleList(dec_norms)

            # feature weighting
            self.wfm = EncoderMap(len(self.use_encoder_feature) + (1 if self.use_ori_img else 0), embed_dim, norm_layer=norm_layer, decoder_depth=decoder_depth)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim - decoder_channel_embed),
            requires_grad=False)  # fixed sin-cos embedding
        # Extra channel for decoder to represent special place for cls token
        self.decoder_channel_embed = nn.Parameter(torch.zeros(1, num_groups + 1, decoder_channel_embed),
                                                  requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, self_attn=self_attn)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.ModuleList([nn.Linear(decoder_embed_dim, len(group) * patch_size**2)
                                           for group in channel_groups])
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed[0].num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1],
                                                          torch.arange(len(self.channel_groups)).numpy())
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed[0].num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        dec_channel_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_channel_embed.shape[-1],
                                                              torch.arange(len(self.channel_groups) + 1).numpy())
        self.decoder_channel_embed.data.copy_(torch.from_numpy(dec_channel_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        for patch_embed in self.patch_embed:
            w = patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

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

    def patchify(self, imgs, p, c):
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, C*patch_size**2)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, p, c):
        """
        x: (N, L, C*patch_size**2)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, c, p, p))
        x = torch.einsum('nhwcpq->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, kept_mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        len_masked = int(L * (mask_ratio - kept_mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # partial mask token
        # generate the binary mask: 0 is keep, 1 is remove
        mask_partial = torch.ones([N, L], device=x.device)
        mask_partial[:, :(len_keep + len_masked)] = 0
        # unshuffle to get the binary mask
        mask_partial = torch.gather(mask_partial, dim=1, index=ids_restore)

        return x_masked, mask, mask_partial, ids_restore

    def forward_encoder(self, x, mask_ratio, kept_mask_ratio):
        # x is (N, C, H, W)
        b, c, h, w = x.shape
        # print("The shape of x image:", x.shape)

        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(self.patch_embed[i](x_c))  # (N, L, D)

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
        _, G, L, D = x.shape
        # print("The encoder embedding of x image:", x.shape)

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, G, 1, cD)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        # Channel embed same across (x,y) position, and pos embed same across channel (c)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, G, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, G, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)

        # add pos embed w/o cls token
        x = x + pos_channel  # (N, G, L, D)

        if self.spatial_mask:
            # Mask spatial location across all channels (i.e. spatial location as either all/no channels)
            x = x.permute(0, 2, 1, 3).reshape(b, L, -1)  # (N, L, G*D)
            x, mask, mask_partial, ids_restore = self.random_masking(x, mask_ratio, kept_mask_ratio)  # (N, 0.25*L, G*D)
            x = x.view(b, x.shape[1], G, D).permute(0, 2, 1, 3).reshape(b, -1, D)  # (N, 0.25*G*L, D)
            mask = mask.repeat(1, G)  # (N, G*L)
            mask = mask.view(b, G, L)
        else:
            # Independently mask each channel (i.e. spatial location has subset of channels visible)
            x, mask, mask_partial, ids_restore = self.random_masking(x.view(b, -1, D), mask_ratio, kept_mask_ratio)  # (N, 0.25*G*L, D)
            mask = mask.view(b, G, L)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, G*L + 1, D)

        # apply Transformer blocks
        x_feats = []
        if self.use_ori_img:
            x_feats.append(x)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if self.fusion_weight and idx in self.use_encoder_feature:
                x_feats.append(x)

        if self.fusion_weight:
            return x_feats, x, mask, mask_partial, ids_restore
        else:
            x = self.norm(x)
            return x, x, mask, mask_partial, ids_restore



    def forward_decoder(self, x_feat, x_encoder, mask, mask_partial, ids_restore):
        # embed tokens
        x = self.decoder_embed(x_encoder)  # (N, 1 + G*0.25*L, D)

        # append mask tokens to sequence
        G = len(self.channel_groups)
        if self.spatial_mask:
            N, L = ids_restore.shape

            x_ = x[:, 1:, :].view(N, G, -1, x.shape[2]).permute(0, 2, 1, 3)  # (N, 0.25*L, G, D)
            _, ml, _, D = x_.shape
            x_ = x_.reshape(N, ml, G * D)  # (N, 0.25*L, G*D)

            mask_tokens = self.mask_token.repeat(N, L - ml, G)
            x_ = torch.cat((x_, mask_tokens), dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))  # (N, L, G*D)
            x_ = x_.view(N, L, G, D).permute(0, 2, 1, 3).reshape(N, -1, D)  # (N, G*L, D)
            x = torch.cat((x[:, :1, :], x_), dim=1)  # append cls token  (N, 1 + G*L, D)
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token  (N, 1 + c*L, D)

        # add pos and channel embed
        channel_embed = self.decoder_channel_embed[:, :-1, :].unsqueeze(2)  # (1, G, 1, cD)
        pos_embed = self.decoder_pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, G, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, G, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)
        pos_channel = pos_channel.view(1, -1, pos_channel.shape[-1])  # (1, G*L, D)

        extra = torch.cat((self.decoder_pos_embed[:, :1, :],
                           self.decoder_channel_embed[:, -1:, :]), dim=-1)  # (1, 1, D)

        pos_channel = torch.cat((extra, pos_channel), dim=1)  # (1, 1+G*L, D)
        x = x + pos_channel  # (N, 1+G*L, D)


        # group operation to extract the masked patches with extra_embedding
        x_no_class = x[:, 1:, :] # (N, G*L, D)
        N, GL, D = x_no_class.shape
        mask_partial_reshape = mask_partial.view(N, -1).bool()
        num_masked_partial_per_sample = int(mask_partial[0].sum().item())
        masked_partial_elements = torch.masked_select(x_no_class, mask_partial_reshape.unsqueeze(-1))
        x_mask_partial_reshaped = masked_partial_elements.view(-1, num_masked_partial_per_sample, x_no_class.size(-1))

        if self.fusion_weight:
            x_encoder_w_class = x_feat
        else:
            x_encoder_w_class = x_encoder

        if self.fusion_weight:
            y = self.wfm(x_encoder_w_class)

        for i, blk in enumerate(self.decoder_blocks):
            if self.fusion_weight:
                x_reconstruct = blk(x_mask_partial_reshaped, self.dec_norms[i](y[..., i]))
            else:
                x_reconstruct = blk(x_mask_partial_reshaped, x_encoder_w_class)

        x_decoder = self.decoder_norm(x_reconstruct)
        N_decoder, L_decoder, D_decoder = x_decoder.shape

        # predictor projection
        mask_expanded = mask_partial_reshape.unsqueeze(-1).expand(-1, -1, D)
        x_decoder_flat = x_decoder.reshape(-1)
        x_no_class_updated = torch.masked_scatter(x_no_class.reshape(-1), mask_expanded.reshape(-1), x_decoder_flat)
        x_no_class_updated = x_no_class_updated.reshape(N, GL, D)

        # to verify if the concate is right in each location for the x_decoder
        masked_partial_decoder = torch.masked_select(x_no_class_updated, mask_partial_reshape.unsqueeze(-1))
        x_mask_partial_decoder = masked_partial_decoder.view(-1, num_masked_partial_per_sample, x_no_class_updated.size(-1))
        are_equal = torch.equal(x_decoder, x_mask_partial_decoder)

        # Separate channel axis
        N, GL, D = x_no_class_updated.shape
        x_no_class_updated = x_no_class_updated.view(N, G, GL//G, D)

        # predictor projection
        x_c_patch = []
        for i, group in enumerate(self.channel_groups):
            x_c = x_no_class_updated[:, i]  # (N, L, D)
            dec = self.decoder_pred[i](x_c)  # (N, L, g_c * p^2)
            dec = dec.view(N, x_c.shape[1], -1, int(self.patch_size**2))  # (N, L, g_c, p^2)
            dec = torch.einsum('nlcp->nclp', dec)  # (N, g_c, L, p^2)
            x_c_patch.append(dec)

        x = torch.cat(x_c_patch, dim=1)  # (N, c, L, p**2)
        return x


    def forward_loss(self, imgs, pred, mask, mask_partial):
        """
        imgs: [N, c, H, W]
        pred: [N, L, c*p*p]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        device = imgs.device
        inverse_order = []
        for i, group in enumerate(self.channel_groups):
            for idx in group:
                inverse_order.append(idx)
    
        # Create the inverse index tensor
        inverse_order_tensor = torch.argsort(torch.tensor(inverse_order, device=device))
        # reorder the channels of predicted tensor
        pred_reordered = torch.index_select(pred, 1, inverse_order_tensor)
        target = self.patchify(imgs, self.patch_embed[0].patch_size[0], self.in_c)  # (N, L, C*P*P)
        b, _, _, _ = imgs.shape
        G = len(self.channel_groups)
        mask_partial = mask_partial.view(b, G, -1)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        N, L, _ = target.shape
        target = target.view(N, L, self.in_c, -1)  # (N, L, C, p^2)
        target = torch.einsum('nlcp->nclp', target)  # (N, C, L, p^2)

        loss = (pred_reordered - target) ** 2
        loss = loss.mean(dim=-1)  # [N, C, L], mean loss per patch

        total_loss, num_removed = 0., 0.
        for i, group in enumerate(self.channel_groups):
            group_loss = loss[:, group, :].mean(dim=1)  # (N, L)
            total_loss += (group_loss * mask_partial[:, i]).sum()
            num_removed += mask_partial[:, i].sum()  # mean loss on removed patches
        return total_loss/num_removed


    def forward(self, imgs, mask_ratio=0.75, kept_mask_ratio=0.75):
        x_feat, latent, mask, mask_partial, ids_restore = self.forward_encoder(imgs, mask_ratio, kept_mask_ratio)
        pred = self.forward_decoder(x_feat, latent, mask, mask_partial, ids_restore)  # [N, C, L, p*p]            

        loss = self.forward_loss(imgs, pred, mask, mask_partial)
        return loss, pred, mask


def emae_vit_base_patch16_dec512d8b(**kwargs):
    model = EfficientMaskedAutoencoderGroupChannelViT(
        channel_embed=256, embed_dim=768, depth=12, num_heads=12,
        decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def emae_vit_large_patch16_dec512d8b(**kwargs):
    model = EfficientMaskedAutoencoderGroupChannelViT(
        channel_embed=256, embed_dim=1024, depth=24, num_heads=16,
        decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def emae_vit_huge_patch14_dec512d8b(**kwargs):
    model = EfficientMaskedAutoencoderGroupChannelViT(
        channel_embed=256, embed_dim=1280, depth=32, num_heads=16,
        decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
emae_vit_base_patch16 = emae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
emae_vit_large_patch16 = emae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
emae_vit_huge_patch14 = emae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
 