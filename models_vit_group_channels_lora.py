# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


import torch.nn as nn
import torch.nn.init as init
import math

# class LoRAAttention(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., rank=32):
#         super(LoRAAttention, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads  
#         if qk_scale is not None:
#             self.scale = qk_scale
#         else:
#             self.scale = 1.0 / (head_dim ** 0.5)
        
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop_ratio)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop_ratio)
#         self.lora_q = nn.Parameter(torch.empty(head_dim, rank))
#         self.lora_k = nn.Parameter(torch.empty(head_dim, rank))
#         self.lora_qa = nn.Parameter(torch.empty(rank, head_dim))
#         self.lora_ka = nn.Parameter(torch.empty(rank, head_dim))
  
#         init.kaiming_uniform_(self.lora_q, a=math.sqrt(5))  
#         init.kaiming_uniform_(self.lora_k, a=math.sqrt(5))
#         init.kaiming_uniform_(self.lora_qa, a=math.sqrt(5))
#         init.kaiming_uniform_(self.lora_ka, a=math.sqrt(5))

    # def forward(self, x):
    #     B, N, _ = x.shape
    #     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
    #     q = q + torch.einsum('b h n d, d r -> b h n r', q, self.lora_q) @ self.lora_qa
    #     k = k + torch.einsum('b h n d, d r -> b h n r', k, self.lora_k) @ self.lora_ka
    #     attn = (q @ k.transpose(-2, -1)) * self.scale
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)
    #     x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x


class GroupChannelsVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool=False, channel_embed=256, channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)), rank=32, **kwargs):
        super().__init__(**kwargs)
        img_size = kwargs['img_size']
        patch_size = kwargs['patch_size']
        in_c = kwargs['in_chans']
        embed_dim = kwargs['embed_dim']

        self.channel_groups = channel_groups

        self.patch_embed = nn.ModuleList([PatchEmbed(img_size, patch_size, len(group), embed_dim)
                                          for group in channel_groups])
        num_patches = self.patch_embed[0].num_patches

        # Positional and channel embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - channel_embed))
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        num_groups = len(channel_groups)
        self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed))
        chan_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], torch.arange(num_groups).numpy())
        self.channel_embed.data.copy_(torch.from_numpy(chan_embed).float().unsqueeze(0))

        self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))
        channel_cls_embed = torch.zeros((1, channel_embed))
        self.channel_cls_embed.data.copy_(channel_cls_embed.float().unsqueeze(0))

        # # Replacing the attention modules in self.blocks
        # for i, blk in enumerate(self.blocks):
        #     # Assuming the attention's qkv linear layer exists and has the shape [embed_dim * 3, embed_dim]
        #     qkv_linear = blk.attn.qkv
        #     embed_dim = qkv_linear.weight.shape[0] // 3
        #     num_heads = blk.attn.num_heads

        #     # Replace with LoRAAttention
        #     blk.attn = LoRAAttention(embed_dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop_ratio=blk.attn.attn_drop.p, proj_drop_ratio=blk.attn.proj_drop.p, rank=rank)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

    def freeze_except_lora(self):
        # Freeze all parameters except those related to LoRA
        for name, param in self.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

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

        x = x + pos_channel  # (N, G, L, D)
        x = x.view(b, -1, D)  # (N, G*L, D)

        cls_pos_channel = torch.cat((self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)  # (1, 1, D)
        cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, 1 + c*L, D)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    

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