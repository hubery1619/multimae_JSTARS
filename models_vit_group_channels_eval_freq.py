# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from timm.models.vision_transformer import PatchEmbed, Block, Attention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        attn_ft = attn @ v
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, attn_ft


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
            # Applying norm and attention
            x1 = self.norm1(x)
            x_attn, attn, attn_ft = self.attn(x1)  # attn is the attention weights
            x = x + self.drop_path(x_attn)
            x_mid = x
            
            # Applying MLP
            x2 = self.norm2(x)
            x = x + self.drop_path(self.mlp(x2))
            
            return x_mid, x, attn, attn_ft  # Return the output and the attention weights


class GroupChannelsVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, channel_embed=256,
                 channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)), **kwargs):
        super().__init__(**kwargs)
        img_size = kwargs['img_size']
        patch_size = kwargs['patch_size']
        in_c = kwargs['in_chans']
        embed_dim = kwargs['embed_dim']
        num_heads = kwargs['num_heads']
        mlp_ratio = kwargs['mlp_ratio']
        depth = kwargs['depth']

        self.channel_groups = channel_groups

        self.patch_embed = nn.ModuleList([PatchEmbed(img_size, patch_size, len(group), embed_dim)
                                          for group in channel_groups])
        # self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed[0].num_patches

        # Positional and channel embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - channel_embed))
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        num_groups = len(channel_groups)
        self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed))
        chan_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], torch.arange(num_groups).numpy())
        self.channel_embed.data.copy_(torch.from_numpy(chan_embed).float().unsqueeze(0))

        # Extra embedding for cls to fill embed_dim
        self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))
        channel_cls_embed = torch.zeros((1, channel_embed))
        self.channel_cls_embed.data.copy_(channel_cls_embed.float().unsqueeze(0))

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
    
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])

    def forward_features(self, x):
        b, c, h, w = x.shape

        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(self.patch_embed[i](x_c))  # (N, L, D)

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
        _, G, L, D = x.shape

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, c, 1, cD)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        # Channel embed same across (x,y) position, and pos embed same across channel (c)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, c, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, c, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, c, L, D)

        # add pos embed w/o cls token
        x = x + pos_channel  # (N, G, L, D)
        x = x.view(b, -1, D)  # (N, G*L, D)

        cls_pos_channel = torch.cat((self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)  # (1, 1, D)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, 1 + c*L, D)
        x = self.pos_drop(x)

        zs = []
        attns = []
        attn_fts = []

        zs.append(x)

        for blk in self.blocks:
            x_mid, x, attn, attn_ft = blk(x)
            zs.append(x_mid)
            zs.append(x)
            attns.append(attn)
            attn_fts.append(attn_ft)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, zs, attns, attn_fts
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, zs, attns, attn_fts = self.forward_features(x)
        x = self.head(x)
        return x, zs, attns, attn_fts


def vit_base_patch16(**kwargs):
    model = GroupChannelsVisionTransformer(
        channel_embed=256, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = GroupChannelsVisionTransformer(
        channel_embed=256, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = GroupChannelsVisionTransformer(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model