import torch
from torchvision.transforms import Resize 
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from models.utils.arch_util import LayerNorm
from models.utils.transformerBCHW_util import Downsample, Upsample, MDTA_TransformerBlock, OverlapPatchEmbed_Keep
from models.utils.Trans4DFTB import *
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_


class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class Mlp(nn.Module):  # DehazeFormer Block中的MLP模块，Linear+ReLU+Linear
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def window_partition(x, window_size):  # 以x={Tensor:{4,56,56,96}},window_size=7为例
    B, H, W, C = x.shape  # B:4 H:56 W:56 C:96
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)  # x=(4,8,7,8,7,96)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2,
                                                            C)  # {Tensor:{4,8,8,7,7,96}}---->>>>{Tensor:{256,49,96}}
    return windows  # {Tensor:{256,49,96}}


def window_reverse(windows, window_size, H, W):  # 以windows={Tensor:{256,49,96}，window_size=7, H=56, W=56
    B = int(windows.shape[0] / (H * W / window_size / window_size))  # 计算batchsize维度的值，256/(56*56/7/7)=4
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)  # {Tensor:{4,8,8,7,7,96}}
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W,
                                                      -1)  # {Tensor:{4,8,7,8,7,96}}---->>>>{Tensor:{4,56,56,96}}
    return x  # x={Tensor:{4,56,56,96}}


def get_relative_positions(window_size):
    # 构建二维位置矩阵
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    # print(coords_h,coords_w)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
    # print(coords)
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    # print(coords_flatten)
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    # print(relative_positions)

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    # print(relative_positions)
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())
    # print(relative_positions_log)

    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # QK+B

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x


class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type
        # 不管conv_type是什么都是提供给W-MHSA-PC中的reflection padding使用
        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        # 两维扩充对于h,w二维空间，左侧扩充self.shift_size，右侧扩充(self.window_size-self.shift_size+mod_pad_w) % self.window_size，（前两个参数对最后一个维度w有效）
        # 上边扩充self.shift_size，下边扩充(self.window_size-self.shift_size+mod_pad_h) % self.window_size，（前两个参数对倒数第二个维度h有效）
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # 两维扩充对于h,w二维空间，左侧扩充0，右侧扩充mod_pad_w，（前两个参数对最后一个维度w有效）
        # 上边扩充0，下边扩充mod_pad_h，（前两个参数对倒数第二个维度h有效）
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)
            # 顺操作
            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)  # shifted_QKV.shape=(B,C,Ht,Wt)
            Ht, Wt = shifted_QKV.shape[2:]  # 获得reflection padding后的QKV的二维尺寸

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)  # shifted_QKV.shape=(B,Ht,Wt,C)
            qkv = window_partition(shifted_QKV,
                                   self.window_size)  # (nW*B, window_size**2, C),其中nW=(Ht/window_size)*(Wt/window_size)

            attn_windows = self.attn(qkv)  # Multihead-self-attention
            # 逆操作
            # merge windows
            shifted_out = window_reverse(attn_windows, self.window_size, Ht,
                                         Wt)  # 将attn_windows由(nW*B, window_size**2, C)变为(B,Ht,Wt,C)

            # reverse cyclic shift
            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W),
                  :]  # 消除reflection padding的部分
            # 从(B,Ht,Wt,C)提取(B,H,W,C)
            attn_out = out.permute(0, 3, 1, 2)  # 从(B,H,W,C)变换为(B,C,H,W)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)  # no attention and use conv, no projection
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out


class TransformerBlock(nn.Module):  # Dehazeformer Block主要结构
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        # attention
        identity = x
        if self.use_attn: x, rescale, rebias = self.norm1(x)
        x = self.attn(x)
        if self.use_attn: x = x * rescale + rebias
        x = identity + x
        # mlp
        identity = x
        if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm: x = x * rescale + rebias
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth
        # 设置用多少个包含attention的Dehazeformer block
        if attn_loc == 'last':  # 一直使用
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        # build blocks 用于构建不同位置的Dehazeformer Block
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):  # SK融合模块
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class Key_MLP(nn.Module):

    def __init__(self, dim, num_degra, keep_degra, ffn_expansion_factor=2.66):
        super(Key_MLP, self).__init__()
        self.dim = dim
        self.num_degra = num_degra
        self.keep_degra = keep_degra
        self.convkey1 = default_conv(num_degra, num_degra, 1)
        self.relu = nn.ReLU(inplace=True)
        self.convkey2 = default_conv(num_degra, num_degra, 1)

    def _init_weights(self, m):  # 初始化参数
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, key):
        key = self.convkey1(key.unsqueeze(dim=3))
        key = self.relu(key)
        key = self.convkey2(key).squeeze()
        return key

class F_ext(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(F_ext, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 2, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 2, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 2, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)

        return out

class InjectFusion(nn.Module):  # SK融合模块
    def __init__(self, dim, height=2, reduction=8):
        super(InjectFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats, injection):
        # print("in",in_feats[0].shape,in_feats[1].shape,injection.shape)
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = attn.view(B, self.height, C, 1, 1)
        attn0 = attn[:, 0, :, :, :] * injection
        attn1 = attn[:, 1, :, :, :]
        # attn1 = attn[:,0,:,:,:] * injection#.view(B, C, 1, 1)
        # attn[:, 0, :, :, :] = attn1
        attn_f = torch.cat((attn0, attn1), dim=1).view(B, self.height, C, 1, 1)
        attn = self.softmax(attn_f)

        out = torch.sum(in_feats * attn, dim=1)  # * injection
        # print(out.shape)
        return out

class Ada4DIR(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, window_size=8,  # in_chans=4, out_chans=5,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN],
                 LayerNorm_type='WithBias',
                 bias=False,
                 num_degra_queries=24,
                 keep_degra=48,
                 degra_type=4,
                 sam=True,
                 ops_type=4,
                 pred=True):
        super(Ada4DIR, self).__init__()

        self.de_dict = {'deblur': 0, 'denoise': 1, 'dehaze': 2, 'dedark': 3, 'clean': 4}

        # setting
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        self.embed_dims = embed_dims

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        self.F_ext_net = F_ext(in_nc=in_chans, nf=embed_dims[0])

        # backbone
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])
        self.injection1 = nn.Linear(embed_dims[0], embed_dims[0], bias=True)
        self.infusion1 = InjectFusion(embed_dims[0])
        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])
        self.injection2 = nn.Linear(embed_dims[0], embed_dims[1], bias=True)
        self.infusion2 = InjectFusion(embed_dims[1])
        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])
        self.injection3 = nn.Linear(embed_dims[0], embed_dims[2], bias=True)
        self.infusion3 = InjectFusion(embed_dims[2])
        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 norm_layer=norm_layer[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])
        self.injection4 = nn.Linear(embed_dims[0], embed_dims[3], bias=True)
        self.infusion4 = InjectFusion(embed_dims[3])
        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])
        self.injection5 = nn.Linear(embed_dims[0], embed_dims[4], bias=True)
        self.infusion5 = InjectFusion(embed_dims[4])
        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)
        self.degra_key = nn.Parameter(torch.randn(degra_type, num_degra_queries, embed_dims[2]*2, requires_grad=True))
        self.key_mlp = Key_MLP(dim=embed_dims[0], num_degra=num_degra_queries,
                                    keep_degra=keep_degra)
        self.Trans_4DFTB_level1 = TransformerBlock_4DFTB(dim=embed_dims[0], dimkey=embed_dims[2]*2, num_heads=num_heads[0],
                                               ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type,
                                               principle=True, sam=sam, ops_type=ops_type, pred=pred)
        self.Trans_4DFTB_level2 = TransformerBlock_4DFTB(dim=embed_dims[1], dimkey=embed_dims[2]*2, num_heads=num_heads[1],
                                               ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type,
                                               principle=True, sam=sam, ops_type=ops_type, pred=pred)
        self.Trans_4DFTB_level3 = TransformerBlock_4DFTB(dim=embed_dims[2], dimkey=embed_dims[2]*2, num_heads=num_heads[2],
                                               ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type,
                                               principle=True, sam=sam, ops_type=ops_type, pred=pred)
        self.prompt1 = nn.Linear(embed_dims[0], embed_dims[0], bias=True)
        self.prompt2 = nn.Linear(embed_dims[0], embed_dims[1], bias=True)
        self.prompt3 = nn.Linear(embed_dims[0], embed_dims[2], bias=True)
        self.cri_pix = nn.L1Loss().cuda()

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, inp_img, degra_type=None, gt=None, epoch=None):
        x = inp_img
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        feat = x
        prompt = self.F_ext_net(x)
        prompt1 = self.prompt1(prompt)
        prompt2 = self.prompt2(prompt)
        prompt3 = self.prompt3(prompt)

        """
        only input_image is required during inference
        """
        flag = 0
        batch_size, c, h, w = inp_img.shape
        if epoch == None:
            degra_key = self.degra_key.detach()
            degra_key = self.key_mlp(degra_key)
            de_type = None
        else:
            if epoch <= 350:
                de_type = degra_type  # [0]
                degra_id = self.de_dict[de_type]
                degra_key = self.degra_key[degra_id, :, :].unsqueeze(0).expand(batch_size, -1, -1)
            else:
                degra_key = self.degra_key.detach()
                degra_key = self.key_mlp(degra_key)
                de_type = None

        x = self.patch_embed(x)
        injection1 = self.injection1(prompt)
        x = self.infusion1([self.layer1(x), x], injection1.view(-1, self.embed_dims[0], 1, 1))
        torch_resize1 = Resize([x.shape[2], x.shape[3]])
        inp_img1 = torch_resize1(inp_img)
        x, output_img1, pred1 = self.Trans_4DFTB_level1(x, degra_key, prompt1, inp_img1, degra_type=de_type)
        skip1 = x

        x = self.patch_merge1(x)
        injection2 = self.injection2(prompt)
        x = self.infusion2([self.layer2(x), x], injection2.view(-1, self.embed_dims[1], 1, 1))
        torch_resize2 = Resize([x.shape[2], x.shape[3]])
        inp_img2 = torch_resize2(inp_img)
        x, output_img2, pred2 = self.Trans_4DFTB_level2(x, degra_key, prompt2, inp_img2, degra_type=de_type)
        skip2 = x

        x = self.patch_merge2(x)
        injection3 = self.injection3(prompt)
        x = self.infusion3([self.layer3(x), x], injection3.view(-1, self.embed_dims[2], 1, 1))
        torch_resize3 = Resize([x.shape[2], x.shape[3]])
        inp_img3 = torch_resize3(inp_img)
        x, output_img3, pred3 = self.Trans_4DFTB_level3(x, degra_key, prompt3, inp_img3, degra_type=de_type)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        injection4 = self.injection4(prompt)
        x = self.infusion4([self.layer4(x), x], injection4.view(-1, self.embed_dims[3], 1, 1))
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        injection5 = self.injection5(prompt)
        x = self.infusion5([self.layer5(x), x], injection5.view(-1, self.embed_dims[4], 1, 1))
        x = self.patch_unembed(x)
        x = x + feat
        x = x[:, :, :H, :W]

        if gt is not None:
            gt_img1 = torch_resize1(gt)
            gt_img2 = torch_resize2(gt)
            gt_img3 = torch_resize3(gt)
            output_img = [output_img1,output_img2,output_img3]
            gt_img = [gt_img1,gt_img2,gt_img3]
            loss = 0
            for j in range(len(output_img)):
                loss = loss + self.cri_pix(output_img[j], gt_img[j])
            loss = torch.sum(loss)

            return [x,loss,pred1,pred2,pred3]
        else:
            return x

def Ada4DIR_t():
    return Ada4DIR(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[4, 4, 4, 2, 2],
		num_heads=[2, 4, 6, 1, 1],
		attn_ratio=[0, 1/2, 1, 0, 0],
		conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        bias=False,
        LayerNorm_type='WithBias',  ## Other option 'BiasFree'
        num_degra_queries = 24,
        keep_degra=48,
        degra_type=4,
        sam=True,
        ops_type=4,
        pred=True)


def Ada4DIR_s():
    return Ada4DIR(
        embed_dims=[48, 96, 192, 96, 48],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[4, 4, 4, 2, 2],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        bias=False,
        LayerNorm_type='WithBias',  ## Other option 'BiasFree'
        num_degra_queries = 24,
        keep_degra=48,
        degra_type=4,
        sam=True,
        ops_type=4,
        pred=True)

def Ada4DIR_b():
    return Ada4DIR(
        embed_dims=[48, 96, 192, 96, 48],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        bias=False,
        LayerNorm_type='WithBias',  ## Other option 'BiasFree'
        num_degra_queries = 24,
        keep_degra=48,
        degra_type=4,
        sam=True,
        ops_type=4,
        pred=True)

def Ada4DIR_d():
    return Ada4DIR(
        embed_dims=[96, 192, 384, 192, 96],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        bias=False,
        LayerNorm_type='WithBias',  ## Other option 'BiasFree'
        num_degra_queries = 24,
        keep_degra=48,
        degra_type=4,
        sam=True,
        ops_type=4,
        pred=True)



