import torch
import torch.nn as nn
from timm.models.layers import DropPath as TimmDropPath
from typing import Tuple
from timm.data import resolve_data_config
# 假设 to_2tuple 是一个简单工具函数，通常来自 timm 或 timm 的 util：
# 它把单个整数 x -> (x, x)，若已经是 (h, w) 则返回原值。
# from timm.models.helpers import to_2tuple

# ----------------------------
# Conv2d_BN: Conv2d + BatchNorm 封装（并提供 fuse 方法用于推理时合并）
# ----------------------------
class Conv2d_BN(torch.nn.Sequential):
    """
    把 Conv2d + BatchNorm2d 串起来的一个小模块，便于构建常见的卷积块。
    - 继承自 torch.nn.Sequential：它是一个按顺序容器，内部模块会按添加顺序被调用。
    - 提供一个 fuse() 方法：在推理时，将 Conv 和 BN 合并为一个等效 Conv（加速推理，减少内存/算子数）。
    """

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        """
        参数说明（按位置）：
        - a: 输入通道数（in_channels）
        - b: 输出通道数（out_channels）
        - ks: kernel size（卷积核大小），默认 1
        - stride: 卷积步幅
        - pad: padding 大小
        - dilation: 空洞卷积的膨胀率
        - groups: 分组卷积参数（groups=1 表示普通卷积）
        - bn_weight_init: 用于初始化 BatchNorm 的 weight（gamma），通常设为1
        """
        super().__init__()  # 初始化父类 Sequential 的内部结构
        # 添加卷积层，注意 bias=False，因为后面有 BN，会有偏置项
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        # 添加 BatchNorm2d 层
        bn = torch.nn.BatchNorm2d(b)
        # 初始化 bn 的 scale (weight/gamma) 和 bias（beta）
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        """
        把 Conv + BN 合并成一个等效的 Conv（包含偏置）。
        这是推理优化常用技巧：将两步（卷积 -> BN）合并为一步卷积，减少运行时开销。

        数学背景（简化）：
          BN 在推理时实现的是： y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
          对于卷积 x = Conv(input) = W * input (不带 bias)
          将两个式子合并可以得到一个新的卷积 W' 和偏置 b' 使得：
          y = Conv'(input) = W' * input + b'
        """
        # self._modules 是 Sequential 内部保留子模块的 OrderedDict
        c, bn = self._modules.values()  # c = Conv2d 模块, bn = BatchNorm2d 模块

        # 下面逐步计算合并后的权重 w 和偏置 b
        # bn.weight 是 gamma，bn.running_var 是方差，bn.eps是数值稳定项
        # 计算缩放因子 w_scale: gamma / sqrt(running_var + eps)
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        # 把缩放因子应用到 conv 的权重上
        # c.weight 的形状是 (out_channels, in_channels/groups, kH, kW)
        # w 的形状是 (out_channels,), 我们用广播把它扩展到 (out, 1, 1, 1)
        w = c.weight * w[:, None, None, None]

        # 计算合并后偏置 b
        # bn.bias 是 beta, bn.running_mean 是均值
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5

        # 创建一个新的 Conv2d 层 m，包含 bias（因为合并后需要偏置）
        # 注意构造 Conv2d 参数时，输入通道数需要与 weight 的第二维一致：
        # w.size(1) * self.c.groups -> 实际的 in_channels（考虑 groups 场景）
        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups,  # in_channels (考虑 groups)
            w.size(0),                  # out_channels
            w.shape[2:],                # kernel_size (kH, kW)
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups
        )
        # 复制权重和偏置到新 conv 中
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m  # 返回融合后的 Conv2d（带 bias），可用于推理替换原模块


# ----------------------------
# DropPath: 随机深度（stochastic depth）包装（继承 timm 的实现）
# ----------------------------
class DropPath(TimmDropPath):
    """
    DropPath（或 Stochastic Depth）是在训练时随机丢弃部分路径（block）的一种正则化。
    继承自 timm 的 DropPath 实现，仅用于封装并保存 drop_prob 以便调试/打印。
    """

    def __init__(self, drop_prob=None): # 这是DropPath的构造函数
        super().__init__(drop_prob=drop_prob)   # 这是父类TimmDropPath的构造函数，用DropPath的参数初始化父类
        self.drop_prob = drop_prob              # 将drop_prob参数保存为实例变量，以便在其他方法中使用

    def __repr__(self):
        # 复用父类的字符串表示，并附加 drop_prob 以便查看
        msg = super().__repr__()
        msg += f'(drop_prob={self.drop_prob})' #保存丢弃的路径
        return msg


# ----------------------------
# PatchEmbed: 用两层步幅为2的卷积把图像 downsample（相当于把图片切成 patch 并映射到 embedding）
# ----------------------------
class PatchEmbed(nn.Module):
    """
    PatchEmbed 的作用：
    - 将原始图片（H x W）通过两次 stride=2 的卷积下采样为 (H/4 x W/4) 的特征图
    - 将每个空间位置映射为一个 embedding 向量，后续 transformer block 使用这些向量作为 token
    为什么用两层 conv 而不是一次大 stride：
      - 两层 3x3 stride=2 的组合在参数量与感受野上通常比一次大的 stride 更稳定，并能更好地保留局部信息
    """

    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        # 把 resolution（可能是单个数字）转换成 (H, W)
        img_size: Tuple[int, int] = to_2tuple(resolution)  # to_2tuple(224) -> (224, 224)
        # patches_resolution 表示下采样后特征图的空间分辨率（H/4, W/4）
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        # 总的 patch 数量（每个空间位置视为一个 "patch"）
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        n = embed_dim
        # seq 是实际执行下采样和通道映射的顺序容器：
        # 1) Conv2d_BN(in_chans -> n//2, kernel=3, stride=2, padding=1)
        # 2) activation (例如 GELU)
        # 3) Conv2d_BN(n//2 -> n, kernel=3, stride=2, padding=1)
        # 结果：空间尺寸 /4, 通道变为 embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        """
        前向函数
        输入 x: 形状 [N, C, H, W]（N=batch size）
        输出：通常为 shape [N, num_patches, embed_dim] 或者 [N, embed_dim, H', W'] 取决于后续 layer 的期望。
        注意：这里 seq 的输出实际是 [N, embed_dim, H/4, W/4]。
        在调用端（模型其它部分）会把它视为 token 序列（可能会 flatten 空间维度）。
        """
        return self.seq(x)
