# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

import itertools
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import timm
from timm.models.layers import DropPath as TimmDropPath,\
    to_2tuple, trunc_normal_
from timm.models.registry import register_model
try:
    # timm.__version__ >= "0.6"
    from timm.models._builder import build_model_with_cfg
except (ImportError, ModuleNotFoundError):
    # timm.__version__ < "0.6"
    from timm.models.helpers import build_model_with_cfg

# 继承自 torch.nn.Sequential：它是一个按顺序容器，内部模块会按添加顺序被调用。
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,groups=1, bn_weight_init=1):
        super().__init__()
        # 添加卷积层，注意 bias=False，因为后面有 BN，会有偏置项
        # 卷积核的大小是 （in_channel,ks_w,ks_h）, 卷积和的个数是 out_channel，因此输入输出的通道数可以是任意的
        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)
    # 把 Conv + BN 合并成一个等效的 Conv（包含偏置）。常用推理优化技巧 TODO
    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(0), w.shape[2:], 
                            stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

"""
    DropPath（或 Stochastic Depth）是在训练时随机丢弃部分路径（block）的一种正则化。
    继承自 timm 的 DropPath 实现，仅用于封装并保存 drop_prob 以便调试/打印。
"""
class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None): # 这是DropPath的构造函数
        super().__init__(drop_prob=drop_prob)   # 这是父类TimmDropPath的构造函数，用DropPath的参数初始化父类
        self.drop_prob = drop_prob              # 将drop_prob参数保存为实例变量，以便在其他方法中使用
    def __repr__(self):
        # 复用父类的字符串表示，并附加 当前的drop_prob 以便查看
        msg = super().__repr__()
        msg += f'(drop_prob={self.drop_prob})'
        return msg
    
# 做两次下采样，将patch数量缩小为原来的1/16，并逐步将通道数从输入通道变为输出通道
# 输出是 [B, embedding, pathch_x, patch_y]
class PatchEmbed(nn.Module):
    '''
        resolution: 输入图像的尺寸大小
    '''
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        # to_2tuple让函数同时兼容单个整数或一个 (h, w) 元组作为输入尺寸。单个数会转为(x,x)
        img_size: Tuple[int, int] = to_2tuple(resolution) 
        # 切成num_patches块patch，每块大小为4*4
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        # 两次下采样，并逐步将通道数从输入通道变为输出通道
        # 结果：空间尺寸 /4, 通道变为 embed_dim
        self.seq = nn.Sequential(
            # 没指定的话就是前n个参数
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)

# MBConv 是一种轻量化卷积模块，可在较少计算量下提取强特征
# 先变高维通道，再逐通道卷积，最后逐点卷积，并在整体上做残差连接
class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio,
                 activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans
        # 通道扩展
        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()
        # 特征提取，stride=1不是下采样
        # 采用了逐通道卷积，可以减少运算量，但损失了通道间信息交互
        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans, ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()
        # 通道数量变换+通道信息融合 ks=1为逐点卷积，只在通道维度上进行卷积
        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()
        # nn.Identity() 的作用就是不做任何操作，它会原样返回输入
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)
        # 有概率将x置为0
        x = self.drop_path(x)
        # 残差网络的精髓是：让网络学习输入与输出之间的差值（残差），而不是直接学习从输入到输出的完整映射。
        # 传统网络：y = F(x)
        # 残差网络：y = F(x) + x
        x += shortcut
        x = self.act3(x)

        return x

# 下采样减少patch个数，将patch数量缩小为原来的1/4，扩大patch包含的信息，因为最后是每个元素作为一个patch
# 先通道调整，再下采样并逐通道卷积，然后逐点卷积，最后转变为Attention输入的形式
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)                       # 通道调整
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, 2, 1, groups=out_dim)   # 下采样+逐通道卷积
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)                   # 逐点卷积

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        # flatten(2) 维度2及之后的展平
        # transpose(1, 2) 交换第一个和第二个维度，维度编号从0开始
        x = x.flatten(2).transpose(1, 2)
        return x

# 构成 TinyViT 的一个 stage（卷积层组：多个MBConv + 1或者0个PatchMerging）
class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth,
                 activation,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 out_dim=None,
                 conv_expand_ratio=4.,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks # 构建深度为depth的卷积特征提取block，不改变通道数，也就是不改变embedding
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path, # 不同深度是否用不同的drop_path的判断
                   )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x) # 不保存激活，就是不保存每一层计算后的输出值
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

# feed-forward-network，两个全连接层，前置归一化（即先做归一化再进行运算）
class Mlp(nn.Module):
    # features指的都是特征维度，没有的都会默认使用输入的特征维度
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features # 返回第一个为真的值
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 加了相对位置编码的自注意力计算块
class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=(14, 14),):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads        # 多头拼接后的长度
        self.d = int(attn_ratio * key_dim)              # 每个 head 的 value 维度
        self.dh = int(attn_ratio * key_dim) * num_heads # 所有 head 的 value 维度总和 # 可以先投影到更大的维度 再由proj投影回dim
        self.attn_ratio = attn_ratio                    # value 向量维度相对于 key_dim 的比例
        h = self.dh + nh_kd * 2                         # 将多头的Wqpk权重合并到一块计算，提高并行

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)        # 计算num_heads组QKV的线性层
        self.proj = nn.Linear(self.dh, dim) # 多头合并后投影输出的线性层 
        # 生成patch的坐标list
        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)         # patch 总数量 = H×W
        attention_offsets = {}  # 字典：保存“相对位移”对应的索引，根据相对偏移查找索引
        idxs = []               # 保存每对 patch 之间对应的偏置索引
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) # offset 表示两个 patch 之间的距离差（竖直差，水平差）
                if offset not in attention_offsets:
                    # 为每种独特的偏移分配唯一索引（例如 (0,0)->0, (0,1)->1, (1,0)->2, ...）
                    attention_offsets[offset] = len(attention_offsets) 
                idxs.append(attention_offsets[offset]) # 记录 (p1,p2) 之间对应的偏置索引
        # 注意力偏置参数：每个 head 都有一组相对位置偏置
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        # attention_bias_idxs: 记录每个 (p1,p2) 对应的偏置索引矩阵
        # register_buffer: 在模型中注册一个“固定张量”，不会被优化器更新
        # persistent=False 表示这个 buffer 不会保存到模型权重文件中
        # 最终形状：(N, N)，即所有 patch 对之间的偏置索引表
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N),
                             persistent=False)

    @torch.no_grad() # 表示该函数不需要计算梯度（减少内存）
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab # 如果进入训练模式，删除缓存的 ab（eval 时才会用）
        else:
            # 预计算注意力偏置矩阵，用于 eval 模式（推理时固定）
            self.ab = self.attention_biases[:, self.attention_bias_idxs] # 形状：(num_heads, N, N)

    def forward(self, x):   # x (B,N,C)
        B, N, _ = x.shape   # 批量大小、patch 数量、通道维

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x) # （B,N,h）= (B,N,dim)*(dim,H), h=head_num * (dk + dk + dv)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # 结果形状：(B, num_heads, N, N)
        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]  # 可以学习的相对位置编码
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        # attn @ v => (B, num_heads, N, dv)
        # 转置 + reshape => (B, N, num_heads*dv) = (B, N, dh)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x) # (B, N, dim)
        return x


class TinyViTBlock(nn.Module):
    r""" TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 local_conv_size=3,
                 activation=nn.GELU,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads,
                              attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H %
                     self.window_size) % self.window_size
            pad_r = (self.window_size - W %
                     self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C
            )
            x = self.attn(x)
            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):
    """ A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        out_dim: the output dimension of the layer. Default: dim
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0.,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 local_conv_size=3,
                 activation=nn.GELU,
                 out_dim=None,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            TinyViTBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(
                             drop_path, list) else drop_path,
                         local_conv_size=local_conv_size,
                         activation=activation,
                         )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyViT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 14, 7],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=1.0,
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(dim=embed_dims[i_layer],
                          input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=PatchMerging if (
                              i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          activation=activation,
                          )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs)
            self.layers.append(layer)

        # Classifier head
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    def forward_features(self, x):
        # x: (N, C, H, W)
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)

        x = x.mean(1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm_head(x)
        x = self.head(x)
        return x


_checkpoint_url_format = \
    'https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/{}.pth'


def _create_tiny_vit(variant, pretrained=False, **kwargs):
    
    # pretrained_type: 22kto1k_distill, 1k, 22k_distill
    # 从 kwargs 中提取 'pretrained_type' 参数，默认值为 '22kto1k_distill'
    # 'pretrained_type' 表示预训练模型的类型，可以是以下三种之一：
    # - '22kto1k_distill': 从 ImageNet-22k 到 ImageNet-1k 的蒸馏模型
    # - '1k': 仅在 ImageNet-1k 上训练的模型
    # - '22k_distill': 仅在 ImageNet-22k 上训练的蒸馏模型，教师模型是ViT
    pretrained_type = kwargs.pop('pretrained_type', '22kto1k_distill')
    assert pretrained_type in ['22kto1k_distill', '1k', '22k_distill'], \
        'pretrained_type should be one of 22kto1k_distill, 1k, 22k_distill'

    # 从 kwargs 中获取 'img_size' 参数，默认值为 224
    # 如果图像大小不是 224，则在 'pretrained_type' 中添加图像大小信息
    img_size = kwargs.get('img_size', 224)
    if img_size != 224:
        pretrained_type = pretrained_type.replace('_', f'_{img_size}_')

    # 根据预训练类型设置类别数量
    # - 如果是 '22k_distill'，类别数量为 21841（ImageNet-22k 的类别数）
    # - 否则，类别数量为 1000（ImageNet-1k 的类别数）
    num_classes_pretrained = 21841 if \
        pretrained_type == '22k_distill' else 1000

    # 从模型变体名称中移除图像大小信息
    # 例如，将 'tiny_vit_11m_224' 转换为 'tiny_vit_11m'
    variant_without_img_size = '_'.join(variant.split('_')[:-1])

    # 配置预训练模型的相关信息
    # - url: 预训练权重的下载地址
    # - num_classes: 预训练模型的类别数量
    # - classifier: 分类头的名称，默认为 'head'
    cfg = dict(
        url=_checkpoint_url_format.format(
            f'{variant_without_img_size}_{pretrained_type}'),
        num_classes=num_classes_pretrained,
        classifier='head',
    )

    def _pretrained_filter_fn(state_dict):
        state_dict = state_dict['model']
        # filter out attention_bias_idxs
        # 过滤掉所有键以 attention_bias_idxs 结尾的参数s
        state_dict = {k: v for k, v in state_dict.items() if \
            not k.endswith('attention_bias_idxs')}
        return state_dict

    if timm.__version__ >= "0.6":
        return build_model_with_cfg(
            TinyViT, variant, pretrained,
            pretrained_cfg=cfg,
            pretrained_filter_fn=_pretrained_filter_fn,
            **kwargs)
    else:
        return build_model_with_cfg(
            TinyViT, variant, pretrained,
            default_cfg=cfg,
            pretrained_filter_fn=_pretrained_filter_fn,
            **kwargs)


@register_model
def tiny_vit_5m_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.0,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_5m_224', pretrained, **model_kwargs)


@register_model
def tiny_vit_11m_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_11m_224', pretrained, **model_kwargs)


@register_model
def tiny_vit_21m_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.2,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_21m_224', pretrained, **model_kwargs)


@register_model
def tiny_vit_21m_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[12, 12, 24, 12],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_21m_384', pretrained, **model_kwargs)


@register_model
def tiny_vit_21m_512(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=512,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_21m_512', pretrained, **model_kwargs)
