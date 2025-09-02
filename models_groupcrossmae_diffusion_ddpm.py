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
from ddpm.diffusion_ddpm import DiffusionDDPM

def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class WeightedFeatureMaps(nn.Module):
    def __init__(self, k, embed_dim, *, norm_layer=nn.LayerNorm, decoder_depth):
        super(WeightedFeatureMaps, self).__init__()
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


class CrossMaskedAutoencoderGroupChannelViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, spatial_mask=False,
                 channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),
                 channel_embed=256, embed_dim=1024, depth=24, num_heads=16,
                 decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, weight_fm=False, 
                 use_fm=[-1], use_input=False, self_attn=False, mlp_time_embed=False
                 ):
        super().__init__()

        self.in_c = in_chans
        self.patch_size = patch_size
        self.channel_groups = channel_groups
        self.spatial_mask = spatial_mask  # Whether to mask all channels of same spatial location
        self.decoder_embed_dim = decoder_embed_dim
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
        # self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # weighted feature maps for cross attention
        self.weight_fm = weight_fm
        self.use_input = use_input # use input as one of the feature maps
        if len(use_fm) == 1 and use_fm[0] == -1:
            self.use_fm = list(range(depth))
        else:
            self.use_fm = [i if i >= 0 else depth + i for i in use_fm]
        if self.weight_fm:
            # print("Weighting feature maps!")
            # print("using feature maps: ", self.use_fm)
            dec_norms = []
            for i in range(decoder_depth):
                norm_layer_i = norm_layer(embed_dim)
                dec_norms.append(norm_layer_i)
            self.dec_norms = nn.ModuleList(dec_norms)

            # feature weighting
            self.wfm = WeightedFeatureMaps(len(self.use_fm) + (1 if self.use_input else 0), embed_dim, norm_layer=norm_layer, decoder_depth=decoder_depth)

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

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])

        # original mae decoder module
        self.decoder_mae_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, self_attn=self_attn)
            for i in range(decoder_depth)])

        self.decoder_mae_norm = norm_layer(decoder_embed_dim)

        self.decoder_mae_pred = nn.ModuleList([nn.Linear(decoder_embed_dim, len(group) * patch_size**2)
                                           for group in channel_groups])
        
        # diffusion decoder module 
        self.diff_noising = DiffusionDDPM()
        self.decoder_diff_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, self_attn=self_attn)
            for i in range(decoder_depth)])

        self.decoder_diff_norm = norm_layer(decoder_embed_dim)

        self.decoder_diff_pred = nn.ModuleList([nn.Linear(decoder_embed_dim, len(group) * patch_size**2)
                                           for group in channel_groups])

        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        ### diffusion decoder specifics
        self.decoder_diffusion = None # as we only need the diffusion process of sde.py we don't need to specify it.
        self.time_embed = nn.Sequential(
            nn.Linear(decoder_embed_dim, 4 * decoder_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * decoder_embed_dim, decoder_embed_dim),
        ) if mlp_time_embed else nn.Identity()
        self.decoder_patch_embed = nn.ModuleList([PatchEmbed(img_size, patch_size, len(group), decoder_embed_dim)
                                          for group in channel_groups])        
        
        self.norm_pix_loss = norm_pix_loss
        self.w_loss = torch.nn.Parameter(torch.tensor(0.9, requires_grad=True))

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
            w0 = patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w0.view([w0.shape[0], -1]))

        # initialize decoder_patch_embed like nn.Linear (instead of nn.Conv2d)
        for decoder_patch_embed in self.decoder_patch_embed:
            w1 = decoder_patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))

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

        # print("The shape of weight_fm:", self.weight_fm)
        # print("The shape of use_input:", self.use_input)
        x_feats = []
        if self.use_input:
            x_feats.append(x)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if self.weight_fm and idx in self.use_fm:
                x_feats.append(x)

        if self.weight_fm:
            return x_feats, x, mask, mask_partial, ids_restore
        else:
            x = self.norm(x)
            return x, x, mask, mask_partial, ids_restore

        # for blk in self.blocks:
        #     x = blk(x)
        # x = self.norm(x)

        # return x, mask, mask_partial, ids_restore



    # def forward_decoder_mae(self, x_feat, x_encoder, mask, mask_partial, ids_restore):
    #     # embed tokens
    #     # x = self.decoder_embed(x_encoder)  # (N, 1 + G*0.25*L, D)
    #     x = x_encoder

    #     # append mask tokens to sequence
    #     G = len(self.channel_groups)
    #     if self.spatial_mask:
    #         mask_partial = mask_partial.repeat(1, G)
    #         N, L = ids_restore.shape

    #         x_ = x[:, 1:, :].view(N, G, -1, x.shape[2]).permute(0, 2, 1, 3)  # (N, 0.25*L, G, D)
    #         _, ml, _, D = x_.shape
    #         x_ = x_.reshape(N, ml, G * D)  # (N, 0.25*L, G*D)

    #         mask_tokens = self.mask_token.repeat(N, L - ml, G)
    #         x_ = torch.cat((x_, mask_tokens), dim=1)  # no cls token
    #         x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))  # (N, L, G*D)
    #         x_ = x_.view(N, L, G, D).permute(0, 2, 1, 3).reshape(N, -1, D)  # (N, G*L, D)
    #         x = torch.cat((x[:, :1, :], x_), dim=1)  # append cls token  (N, 1 + G*L, D)
    #     else:
    #         mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    #         x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    #         x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    #         x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token  (N, 1 + c*L, D)

    #     # add pos and channel embed
    #     channel_embed = self.decoder_channel_embed[:, :-1, :].unsqueeze(2)  # (1, G, 1, cD)
    #     pos_embed = self.decoder_pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

    #     channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, G, L, cD)
    #     pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, G, L, pD)
    #     pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)
    #     pos_channel = pos_channel.view(1, -1, pos_channel.shape[-1])  # (1, G*L, D)

    #     extra = torch.cat((self.decoder_pos_embed[:, :1, :],
    #                        self.decoder_channel_embed[:, -1:, :]), dim=-1)  # (1, 1, D)

    #     pos_channel = torch.cat((extra, pos_channel), dim=1)  # (1, 1+G*L, D)
    #     x = x + pos_channel  # (N, 1+G*L, D)


    #     # group operation to extract the masked patches with extra_embedding
    #     x_no_class = x[:, 1:, :] # (N, G*L, D)
    #     # print("The shape of x_no_class:", x_no_class.shape)
    #     # print("The shape of mask:", mask.shape)
    #     N, GL, D = x_no_class.shape
    #     mask_partial_reshape = mask_partial.view(N, -1).bool()
    #     # print("The shape of mask_partial_reshape:", mask_partial_reshape.shape)
    #     num_masked_partial_per_sample = int(mask_partial[0].sum().item())
    #     # print("The shape of num_masked_partial_per_sample:", num_masked_partial_per_sample)
    #     masked_partial_elements = torch.masked_select(x_no_class, mask_partial_reshape.unsqueeze(-1))
    #     x_mask_partial_reshaped = masked_partial_elements.view(-1, num_masked_partial_per_sample, x_no_class.size(-1))
    #     # print("The shape of x_mask_partial_reshaped:", x_mask_partial_reshaped.shape)

    #     if self.weight_fm:
    #         # extract unmask patches (before decoder embedding)
    #         x_encoder_w_class = x_feat
    #         # print("The shape of x_encoder_no_class:", x_encoder_w_class.shape)
    #     else:
    #         x_encoder_w_class = x_encoder
    #         # print("The shape of x_encoder_no_class:", x_encoder_w_class.shape)           

    #     # for blk in self.decoder_blocks:
    #     #     x_reconstruct = blk(x_mask_partial_reshaped, x_encoder_no_class)

    #     if self.weight_fm:
    #         y = self.wfm(x_encoder_w_class)

    #     for i, blk in enumerate(self.decoder_mae_blocks):
    #         if self.weight_fm:
    #             x_reconstruct = blk(x_mask_partial_reshaped, self.dec_norms[i](y[..., i]))
    #         else:
    #             x_reconstruct = blk(x_mask_partial_reshaped, x_encoder_w_class)



    #     x_decoder = self.decoder_mae_norm(x_reconstruct)
    #     # x_decoder = x_reconstruct
    #     # print("The shape of x_decoder:", x_decoder.shape)
    #     N_decoder, L_decoder, D_decoder = x_decoder.shape

    #     # predictor projection
    #     mask_expanded = mask_partial_reshape.unsqueeze(-1).expand(-1, -1, D)
    #     x_decoder_flat = x_decoder.reshape(-1)
    #     x_no_class_updated = torch.masked_scatter(x_no_class.reshape(-1), mask_expanded.reshape(-1), x_decoder_flat)
    #     x_no_class_updated = x_no_class_updated.reshape(N, GL, D)
    #     # print("The shape of x_no_class_updated:", x_no_class_updated.shape)

    #     # to verify if the concate is right in each location for the x_decoder
    #     masked_partial_decoder = torch.masked_select(x_no_class_updated, mask_partial_reshape.unsqueeze(-1))
    #     x_mask_partial_decoder = masked_partial_decoder.view(-1, num_masked_partial_per_sample, x_no_class_updated.size(-1))
    #     are_equal = torch.equal(x_decoder, x_mask_partial_decoder)
    #     # print(f"The two tensors are equal: {are_equal}")
    #     # print("The x_decoder:", x_decoder)
    #     # print("The x_mask_partial_decoder:", x_mask_partial_decoder)

    #     # Separate channel axis
    #     N, GL, D = x_no_class_updated.shape
    #     x_no_class_updated = x_no_class_updated.view(N, G, GL//G, D)

    #     # predictor projection
    #     x_c_patch = []
    #     for i, group in enumerate(self.channel_groups):
    #         x_c = x_no_class_updated[:, i]  # (N, L, D)
    #         dec = self.decoder_mae_pred[i](x_c)  # (N, L, g_c * p^2)
    #         dec = dec.view(N, x_c.shape[1], -1, int(self.patch_size**2))  # (N, L, g_c, p^2)
    #         dec = torch.einsum('nlcp->nclp', dec)  # (N, g_c, L, p^2)
    #         x_c_patch.append(dec)

    #     x = torch.cat(x_c_patch, dim=1)  # (N, c, L, p**2)
    #     return x

    def forward_decoder_diffusion(self, x_noise, timesteps, x_feat, x_encoder, mask, mask_partial, ids_restore, y=None):
        ### x_noise decoder setting
        # x is (N, C, H, W)
        b, c, h, w = x_noise.shape
        # print("The shape of x image:", x.shape)

        x_c_decoder_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c_decoder = x_noise[:, group, :, :]
            x_c_decoder_embed.append(self.decoder_patch_embed[i](x_c_decoder))  # (N, L, D)

        x_decoder = torch.stack(x_c_decoder_embed, dim=1)  # (N, G, L, D) 
        ### x_noise decoder setting       
        
        
        x = x_encoder
        # append mask tokens to sequence
        G = len(self.channel_groups)
        # N, L = ids_restore.shape
        if self.spatial_mask:
            # N, L = ids_restore.shape

            # x_ = x[:, 1:, :].view(N, G, -1, x.shape[2]).permute(0, 2, 1, 3)  # (N, 0.25*L, G, D)
            # _, ml, _, D = x_.shape
            # x_ = x_.reshape(N, ml, G * D)  # (N, 0.25*L, G*D)

            # mask_tokens = self.mask_token.repeat(N, L - ml, G)
            # x_ = torch.cat((x_, mask_tokens), dim=1)  # no cls token
            # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))  # (N, L, G*D)
            # x_ = x_.view(N, L, G, D).permute(0, 2, 1, 3).reshape(N, -1, D)  # (N, G*L, D)
            # x = torch.cat((x[:, :1, :], x_), dim=1)  # append cls token  (N, 1 + G*L, D)

            # mask_partial = mask_partial.repeat(1, G)
            N_noise, G_noise, L_noise, D_noise = x_decoder.shape
            x_encoder_noclass = x[:, 1:, :].reshape(N_noise, G_noise, -1, D_noise)
            x_en_de_merge = []
            for i, group in enumerate(self.channel_groups):
                x_c_decoder = x_decoder[:, i, :, :]
                mask_partial_reshape = mask_partial.view(N_noise, -1).bool()
                num_masked_partial_per_sample = int(mask_partial[0].sum().item())
                masked_partial_elements = torch.masked_select(x_c_decoder, mask_partial_reshape.unsqueeze(-1))
                mask_tokens = masked_partial_elements.view(-1, num_masked_partial_per_sample, x_c_decoder.size(-1))
                x_merge = torch.cat([x_encoder_noclass[:, i, :, :], mask_tokens], dim=1)
                x_en_de_merge.append(x_merge)
            x_ = torch.stack(x_c_decoder_embed, dim=1)  # (N, G, L, D) 
            x_ = x_.reshape(N_noise, -1, D_noise)
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token  (N, 1 + c*L, D)
            # x_decoder_reshape = x_decoder.reshape(N_noise, G_noise * L_noise, D_noise)
            # mask_partial_reshape = mask_partial.view(N_noise, -1).bool()
            # num_masked_partial_per_sample = int(mask_partial[0].sum().item())
            # masked_partial_elements = torch.masked_select(x_decoder_reshape, mask_partial_reshape.unsqueeze(-1))
            # mask_tokens = masked_partial_elements.view(-1, num_masked_partial_per_sample, x_decoder_reshape.size(-1))
            # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token  (N, 1 + c*L, D)

        else:
            N_noise, G_noise, L_noise, D_noise = x_decoder.shape
            x_decoder_reshape = x_decoder.reshape(N_noise, G_noise * L_noise, D_noise)
            mask_partial_reshape = mask_partial.view(N_noise, -1).bool()
            num_masked_partial_per_sample = int(mask_partial[0].sum().item())
            masked_partial_elements = torch.masked_select(x_decoder_reshape, mask_partial_reshape.unsqueeze(-1))
            mask_tokens = masked_partial_elements.view(-1, num_masked_partial_per_sample, x_decoder_reshape.size(-1))
            # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
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


        # # group operation to extract the masked patches with extra_embedding
        # x_no_class = x[:, 1:, :] # (N, G*L, D)
        # # print("The shape of x_no_class:", x_no_class.shape)
        # # print("The shape of mask:", mask.shape)
        # N, GL, D = x_no_class.shape
        # mask_partial_reshape = mask_partial.view(N, -1).bool()
        # # print("The shape of mask_partial_reshape:", mask_partial_reshape.shape)
        # num_masked_partial_per_sample = int(mask_partial[0].sum().item())
        # # print("The shape of num_masked_partial_per_sample:", num_masked_partial_per_sample)
        # masked_partial_elements = torch.masked_select(x_no_class, mask_partial_reshape.unsqueeze(-1))
        # x_mask_partial_reshaped = masked_partial_elements.view(-1, num_masked_partial_per_sample, x_no_class.size(-1))
        # # print("The shape of x_mask_partial_reshaped:", x_mask_partial_reshaped.shape)

        ### obtain the x_noise masked patches
        # group operation to extract the masked patches with extra_embedding
        # x_no_class = x[:, 1:, :] # (N, G*L, D)
        # print("The shape of x_no_class:", x_no_class.shape)
        # print("The shape of mask:", mask.shape)

        N, G, L, D = x_decoder.shape  # (N, G, L, D)
        x_no_class = x_decoder.reshape(N, G*L, D)  # x_decoder shape: (N, G, L, D)
        N, GL, D = x_no_class.shape
        if self.spatial_mask:
            mask_partial = mask_partial.repeat(1, G)
        mask_partial_reshape = mask_partial.view(N, -1).bool()
        # print("The shape of mask_partial_reshape:", mask_partial_reshape.shape)
        num_masked_partial_per_sample = int(mask_partial[0].sum().item())
        # print("The shape of num_masked_partial_per_sample:", num_masked_partial_per_sample)
        masked_partial_elements = torch.masked_select(x_no_class, mask_partial_reshape.unsqueeze(-1))
        x_mask_partial_reshaped = masked_partial_elements.view(-1, num_masked_partial_per_sample, x_no_class.size(-1))
        # print("The shape of x_mask_partial_reshaped:", x_mask_partial_reshaped.shape)
        time_token = self.time_embed(timestep_embedding(timesteps, self.decoder_embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        time_token_expanded = time_token.expand(x_mask_partial_reshaped.shape[0], -1, -1)
        x_mask_time = torch.cat((time_token_expanded, x_mask_partial_reshaped), dim=1)

        if self.weight_fm:
            # extract unmask patches (before decoder embedding)
            x_encoder_w_class = x_feat
            # print("The shape of x_encoder_no_class:", x_encoder_w_class.shape)
        else:
            x_encoder_w_class = x_encoder
            # print("The shape of x_encoder_no_class:", x_encoder_w_class.shape)           

        # for blk in self.decoder_blocks:
        #     x_reconstruct = blk(x_mask_partial_reshaped, x_encoder_no_class)

        if self.weight_fm:
            y = self.wfm(x_encoder_w_class)

        for i, blk in enumerate(self.decoder_diff_blocks):
            if self.weight_fm:
                x_reconstruct = blk(x_mask_time, self.dec_norms[i](y[..., i]))
            else:
                x_reconstruct = blk(x_mask_time, x_encoder_w_class)


        x_reconstruct_notime = x_reconstruct[:, 1:, :]
        x_decoder = self.decoder_diff_norm(x_reconstruct_notime)
        # print("The shape of x_decoder:", x_decoder.shape)
        N_decoder, L_decoder, D_decoder = x_decoder.shape

        # predictor projection
        mask_expanded = mask_partial_reshape.unsqueeze(-1).expand(-1, -1, D)
        x_decoder_flat = x_decoder.reshape(-1)
        x_no_class_updated = torch.masked_scatter(x_no_class.to(torch.float32).reshape(-1), mask_expanded.reshape(-1), x_decoder_flat)
        x_no_class_updated = x_no_class_updated.reshape(N, GL, D)
        # print("The shape of x_no_class_updated:", x_no_class_updated.shape)

        # to verify if the concate is right in each location for the x_decoder
        masked_partial_decoder = torch.masked_select(x_no_class_updated, mask_partial_reshape.unsqueeze(-1))
        x_mask_partial_decoder = masked_partial_decoder.view(-1, num_masked_partial_per_sample, x_no_class_updated.size(-1))
        are_equal = torch.equal(x_decoder, x_mask_partial_decoder)
        # print(f"The two tensors are equal: {are_equal}")
        # print("The x_decoder:", x_decoder)
        # print("The x_mask_partial_decoder:", x_mask_partial_decoder)

        # Separate channel axis
        N, GL, D = x_no_class_updated.shape
        x_no_class_updated = x_no_class_updated.view(N, G, GL//G, D)

        # predictor projection
        x_c_patch = []
        for i, group in enumerate(self.channel_groups):
            x_c = x_no_class_updated[:, i]  # (N, L, D)
            dec = self.decoder_diff_pred[i](x_c)  # (N, L, g_c * p^2)
            dec = dec.view(N, x_c.shape[1], -1, int(self.patch_size**2))  # (N, L, g_c, p^2)
            dec = torch.einsum('nlcp->nclp', dec)  # (N, g_c, L, p^2)
            x_c_patch.append(dec)

        x = torch.cat(x_c_patch, dim=1)  # (N, c, L, p**2)
        # x = x.permute(0, 2, 1, 3)  # (N, L, c, p**2)
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # (N, L, c*p*p)

        return x


    def forward_loss(self, imgs, pred, mask, mask_partial):
        """
        imgs: [N, c, H, W]
        pred: [N, L, c*p*p]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs, self.patch_embed[0].patch_size[0], self.in_c)  # (N, L, C*P*P)
        # # print("The shape of target:", target.shape)

        device = imgs.device
        inverse_order = []
        for i, group in enumerate(self.channel_groups):
            for idx in group:
                inverse_order.append(idx)
        inverse_order_tensor = torch.argsort(torch.tensor(inverse_order, device=device))
        pred_reordered = torch.index_select(pred, 1, inverse_order_tensor)
        target = self.patchify(imgs, self.patch_embed[0].patch_size[0], self.in_c)  # (N, L, C*P*P)
        # print("The shape of target:", target.shape)

        
        b, _, _, _ = imgs.shape
        G = len(self.channel_groups)
        if self.spatial_mask:
            mask_partial = mask_partial.repeat(1, G)  # (N, G*L)
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

        ### mae decoder
        x_feat, latent, mask, mask_partial, ids_restore = self.forward_encoder(imgs, mask_ratio, kept_mask_ratio)
        encoder_x_feat = self.decoder_embed(x_feat)
        encoder_latent = self.decoder_embed(latent)
        
        

        # pred_mae = self.forward_decoder_mae(encoder_x_feat, encoder_latent, mask, mask_partial, ids_restore)  # [N, C, L, p*p]            

        # loss_mae = self.forward_loss(imgs, pred_mae, mask, mask_partial)
        ### mae decoder

        ### diffusion decoder
        # if not isinstance(t, torch.Tensor):
        #     t = torch.tensor(t)
        # t = t.to(xt.device)
        # if t.dim() == 0:
        #     t = duplicate(t, xt.size(0))
        
        # x_pred = self.forward_decoder_diffusion(xt, t * 999, encoder_x_feat, encoder_latent, mask, mask_partial, ids_restore, y=None)  # [N, L, p*p*3]
        x_noise, timestamp = self.diff_noising.ddpm_noising(imgs)
        x_pred = self.forward_decoder_diffusion(x_noise, timestamp, encoder_x_feat, encoder_latent, mask, mask_partial, ids_restore, y=None)  # [N, L, p*p*3]
        loss_diffusion = self.forward_loss(imgs, x_pred, mask, mask_partial)
        # w_clamped = torch.sigmoid(self.w_loss)
        loss = loss_diffusion


        return loss


def crossmaediffusion_vit_base_patch16_dec512d8b(**kwargs):
    model = CrossMaskedAutoencoderGroupChannelViT(
        channel_embed=256, embed_dim=768, depth=12, num_heads=12,
        decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def crossmaediffusion_vit_large_patch16_dec512d8b(**kwargs):
    model = CrossMaskedAutoencoderGroupChannelViT(
        channel_embed=256, embed_dim=1024, depth=24, num_heads=16,
        decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def crossmaediffusion_vit_huge_patch14_dec512d8b(**kwargs):
    model = CrossMaskedAutoencoderGroupChannelViT(
        channel_embed=256, embed_dim=1280, depth=32, num_heads=16,
        decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
crossmaediffddpm_vit_base_patch16 = crossmaediffusion_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
crossmaediffddpm_vit_large_patch16 = crossmaediffusion_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
crossmaediffddpm_vit_huge_patch14 = crossmaediffusion_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks 