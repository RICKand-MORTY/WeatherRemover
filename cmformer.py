import torch
import torch.nn as nn
import math
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Attention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os
import torchvision.utils as tvu
from torchvision.ops import DeformConv2d
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange
import numbers


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, frequency_embedding_size=256):
        super().__init__()
        hidden_size = 192
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.output_shape = (3, 8, 8)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, rH, rW):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        B = t.shape[0]
        C, H, W = self.output_shape
        t_emb = t_emb.view(B, C, H, W)
        # Bilinear interpolation to target shape (B, C, rH, rW)
        t_emb_resized = F.interpolate(t_emb, size=(rH, rW), mode='bilinear', align_corners=False)
        return t_emb_resized


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc_branch = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_branch = DWConv(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x2 = x.clone()
        x = self.fc1(x)
        x2 = self.fc_branch(x2)
        x = self.dwconv(x)
        x2 = self.dwconv_branch(x2)
        x = self.act(x)
        x = x * x2
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PVTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 learnable_scale=True, sra_size=7):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.sra_size = sra_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.scale = head_dim ** -0.5
        if self.sra_size != 0:
            self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
            self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=qkv_bias)
            self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=qkv_bias)
            self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=qkv_bias)

            # SRA
            self.pool = nn.AdaptiveAvgPool2d(sra_size)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = LayerNorm(dim, 'WithBias')
            self.act = nn.GELU()
        else:
            #NO SRA
            self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.sra_size != 0:
            q = self.q_dwconv(self.q(x))
            # Note that c need to be the last dim to ensure metric multiplication can run
            q = rearrange(q, 'b (head c) h w -> b head (h w) c ', head=self.num_heads)
            q = torch.nn.functional.normalize(q, dim=-1)
            # SRA
            x_ = self.sr(self.pool(x))
            x_ = self.norm(x_)
            x_ = self.act(x_)
            #Generate kv
            kv = self.kv_dwconv(self.kv(x_))
            k, v = kv.chunk(2, dim=1)
            k = rearrange(k, 'b (head c) h w -> b head (h w) c ', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head (h w) c ', head=self.num_heads)
            # print(f"k={k.shape}, v={v.shape}, q={q.shape}")
        else:
            #NO SRA
            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        if self.sra_size != 0:
            x = rearrange(x, 'b head (h w) c  -> b (head c) h w', head=self.num_heads, h=H, w=W)
        else:
            x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PVTBlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., bias=False,
                 drop_path=0., act_layer=nn.GELU, LayerNorm_type='WithBias', sra_size=7):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = PVTAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, sra_size=sra_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 2, act_layer=act_layer, drop=drop, bias=bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CMFormer(nn.Module):

    def __init__(
            self,
            input_size=256,
            in_channels=3,
            dim=48,
            depth=[4, 6, 6, 8, 4],
            num_heads=[4, 8, 12, 16, 4],
            sra_size=[7, 7, 7, 7, 7],
            qkv_bias=True,
            bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        print(
            f"depth={depth}, input_size={input_size}, dim={dim}, in_chann={in_channels}, num_heads={num_heads}")
        self.x_embedder = OverlapPatchEmbed(in_channels, dim)

        # branch 1
        # self.branch_x_emb1 = OverlapPatchEmbed(in_channels, dim)
        self.patch_dit1 = PVTBlock(
            dim=dim, num_heads=num_heads[0], qkv_bias=qkv_bias, drop=drop_rate,
            attn_drop=attn_drop_rate, sra_size=sra_size[0]
        )
        self.branch_gelu1 = nn.GELU()

        # branch 2
        # self.branch_x_emb2 = OverlapPatchEmbed(dim, dim*2*1)
        self.patch_dit2 = PVTBlock(
            dim=dim * 2 ** 1, num_heads=num_heads[1], qkv_bias=qkv_bias, drop=drop_rate,
            attn_drop=attn_drop_rate, sra_size=sra_size[1]
        )
        self.branch_gelu2 = nn.GELU()
        
        # branch 3
        # self.branch_x_emb3 = OverlapPatchEmbed(dim*2*1, dim * 2 * 2)
        self.patch_dit3 = PVTBlock(
            dim=dim * 2 ** 2, num_heads=num_heads[2], qkv_bias=qkv_bias, drop=drop_rate,
            attn_drop=attn_drop_rate, sra_size=sra_size[2]
        )
        self.branch_gelu3 = nn.GELU()
        """
        self.t_initial = TimestepEmbedder()
        self.t_embedder = OverlapPatchEmbed(in_channels, dim)
        self.t_linear1 = nn.Conv2d(in_channels=dim, out_channels=dim * 2 ** 1, kernel_size=3, stride=2, padding=1)
        self.t_linear2 = nn.Conv2d(in_channels=dim, out_channels=dim * 2 ** 2, kernel_size=3, stride=4, padding=1)
        self.t_linear3 = nn.Conv2d(in_channels=dim, out_channels=dim * 2 ** 3, kernel_size=3, stride=8, padding=1)
        """
        
        # inital feature
        """
        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([3, input_size, input_size]),
            nn.GELU()
        )
        """
        self.end_layer = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, stride=1, padding=1),
        )

        self.blocks1 = nn.ModuleList([
            PVTBlock(
                dim=dim, num_heads=num_heads[0], qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, sra_size=sra_size[0]
            )
            for j in range(depth[0])
        ])

        self.blocks2 = nn.ModuleList([
            PVTBlock(
                dim=dim * 2 ** 1, num_heads=num_heads[1], qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, sra_size=sra_size[1]
            )
            for j in range(depth[1])
        ])

        self.blocks3 = nn.ModuleList([
            PVTBlock(
                dim=dim * 2 ** 2, num_heads=num_heads[2], qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, sra_size=sra_size[2]
            )
            for j in range(depth[2])
        ])

        self.blocks4 = nn.ModuleList([
            PVTBlock(
                dim=dim * 2 ** 3, num_heads=num_heads[3], qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, sra_size=sra_size[3]
            )
            for j in range(depth[3])
        ])

        # decode level
        self.dec_blocks1 = nn.ModuleList([
            PVTBlock(
                dim=dim, num_heads=num_heads[0], qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, sra_size=sra_size[0]
            )
            for j in range(depth[0])
        ])

        self.dec_blocks2 = nn.ModuleList([
            PVTBlock(
                dim=dim * 2 ** 1, num_heads=num_heads[1], qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, sra_size=sra_size[1]
            )
            for j in range(depth[1])
        ])

        self.dec_blocks3 = nn.ModuleList([
            PVTBlock(
                dim=dim * 2 ** 2, num_heads=num_heads[2], qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, sra_size=sra_size[2]
            )
            for j in range(depth[2])
        ])
        norm_type = "BiasFree"
        # Downsample
        self.down1 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )  # [B, dim*2*1, H/2, W/2]

        self.down2 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 1, dim * 2 ** 1 // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )  # [B, dim*2*2, H/4, W/4]
        self.down3 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 2 // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )  # [B, dim*2*3, H/8, W/8]

        # Upsample
        self.up1 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 3, dim * 2 ** 3 * 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(in_channels=(dim * 2 ** 2) * 2, out_channels=(dim * 2 * 2), kernel_size=1, bias=bias),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 2), (dim * 2 ** 2) * 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )
        self.fusion3 = nn.Sequential(
            nn.Conv2d(in_channels=(dim * 2 ** 1) * 2, out_channels=(dim * 2 ** 1), kernel_size=1, bias=bias),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 1, dim * 2 * 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )
        self.fusion4 = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, bias=bias),
        )

        if depth[4] != 0:
            self.refine = True
            self.refine_layer = nn.ModuleList([
                PVTBlock(
                    dim=dim, num_heads=num_heads[4], qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, sra_size=sra_size[4]
                )
                for j in range(depth[4])
            ])
            self.t_linear_ref = OverlapPatchEmbed(dim, dim)
        else:
            print("No refineblock")
            self.refine = False
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def pad_to_multiple_of_eight(self, x):
        _, _, h, w = x.size()
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        x_padded = F.pad(x, (0, w_pad, 0, h_pad), 'constant', 0)
        return x_padded

    def forward(self, x):
        x_ori = x.clone()
        _, _, ori_h, ori_w = x.shape
        x = self.pad_to_multiple_of_eight(x)
        # print(f"t.shape={t.shape}")
        # print(f"x.shape={x.shape}")
        _, _, H, W = x.shape
        x = self.x_embedder(x)
        # prin(f"x.shape={x.shape}")

        # Stage 1
        # print("Stage 1")
        x1 = x.clone()
        # print(f"x.shape={x.shape}")
        # print(f"before block x.shape={x.shape}")
        for block in self.blocks1:
            x = block(x)  # (N, T, D)
        # print(f"after block x.shape={x.shape}")
        x = self.branch_gelu1(x) * self.patch_dit1(x1)
        # print(f"add x.shape={x.shape}")

        # Down1
        x_to_s1 = x.clone()
        # print(f"bef down1 x.shape={x.shape}")
        x = self.down1(x)
        # print(f"aft down1 x.shape={x.shape}")
        x2 = x.clone()

        # Stage 2
        # print("Stage 2")
        # print(f"x.shape={x.shape}")
        # print(f"before block x.shape={x.shape}")
        for block in self.blocks2:
            x = block(x)
        # print(f"after block x.shape={x.shape}")
        x = self.branch_gelu2(x) * self.patch_dit2(x2)
        # print(f"x.shape={x.shape}")

        # Down 2
        x_to_s2 = x.clone()
        # print(f"bef down2 x.shape={x.shape}")
        x = self.down2(x)
        # print(f"aft down2 x.shape={x.shape}")
        x3 = x.clone()

        # Stage 3
        # print("Stage 3")
        # print(f"x.shape={x.shape}")
        # print(f"x.shape={x.shape}")
        for block in self.blocks3:
            x = block(x)
        x = self.branch_gelu1(x) * self.patch_dit3(x3)
        # print(f"after block and addd x.shape={x.shape}")

        # print(f"bef down3 x.shape={x.shape}")
        x4 = x.clone()  # [B, dim*2*2, H/4, H/4]
        # print(f"aft down3 x.shape={x.shape}")

        # Down 3
        x = self.down3(x)

        # Stage 4
        # print("Stage 4")
        # print(f"x.shape={x.shape}")
        for block in self.blocks4:
            x = block(x)
        # print(f"afte block x.shape={x.shape}")
        # [B, dim*2*3, H/8, H/8]

        # Up 1
        x = self.up1(x)
        # print(f"x.shape={x.shape}")
        x = torch.cat((x, x4), dim=1)
        # print(f"x.shape={x.shape}")
        x = self.fusion2(x)

        # Stage 3'
        # print("Stage 4")
        # print(f"x.shape={x.shape}")
        # print(f"x.shape={x.shape}")
        for block in self.dec_blocks3:
            x = block(x)
        # print(f"afte block x.shape={x.shape}")

        # Up 2
        x = self.up2(x)
        x = torch.cat((x, x_to_s2), dim=1)
        x = self.fusion3(x)

        # Stage 2'
        # print("Stage 3")
        for block in self.dec_blocks2:
            x = block(x)
        # print(f"after x.shape={x.shape}")

        # Up 3
        x = self.up3(x)
        x = torch.cat((x, x_to_s1), dim=1)
        x = self.fusion4(x)

        # Stage 1'
        # print("Stage 1")
        # print(f"x.shape={x.shape}")
        for block in self.dec_blocks1:
            x = block(x)
        # print(f"after block x.shape={x.shape}")

        # Refinement
        # print("Refinement")
        # print(f"x.shape={x.shape}")
        for block in self.refine_layer:
            x = block(x)
        # print(f"after block x.shape={x.shape}")
        x = self.end_layer(x)
        x = x[:, :, :ori_h, :ori_w] + x_ori
        return x


