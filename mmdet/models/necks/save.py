# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer, xavier_init
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ..builder import NECKS
import torch
import numpy as np
from einops import rearrange
import math
import einops
import torch.nn.functional as F
"""
old
"""

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w').contiguous()


class DAU(nn.Module):
    def __init__(self, in_c, ratio=4, nh=1, up_factor=2., compress_ks=3, upsample_ks=3, n_groups=2,
                 range_factor=11, rpb=True):
        super(DAU, self).__init__()
        self.up_ks = upsample_ks
        self.num_head = nh
        self.up_factor = up_factor
        self.n_groups = n_groups
        self.offset_range_factor = range_factor

        self.attn_dim = in_c // (ratio * self.num_head)
        self.scale = self.attn_dim ** -0.5
        self.rpb = rpb
        self.hidden_dim = in_c // ratio
        self.proj_q = nn.Conv2d(
            in_c, self.hidden_dim,
            kernel_size=1, stride=1, padding=0, bias=False
        )

        self.proj_k = nn.Conv2d(
            in_c, self.hidden_dim,
            kernel_size=1, stride=1, padding=0, bias=False
        )

        self.group_channel = in_c // (ratio * self.n_groups)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.group_channel, self.group_channel, 3, 1, 1,
                      groups=self.group_channel, bias=False),
            LayerNormProxy(self.group_channel),
            nn.GELU(),
            nn.Conv2d(self.group_channel, 2 * upsample_ks ** 2, compress_ks, 1, compress_ks // 2)
        )
        self.layer_norm = LayerNormProxy(in_c)

        self.pad = int((self.up_ks - 1) / 2)
        base = np.arange(-self.pad, self.pad + 1).astype(np.float32)
        base_y = np.repeat(base, self.up_ks)
        base_x = np.tile(base, self.up_ks)
        base_offset = np.stack([base_y, base_x], axis=1).flatten()
        base_offset = torch.tensor(base_offset).view(1, -1, 1, 1)
        self.register_buffer("base_offset", base_offset, persistent=False)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.num_head, 1, self.up_ks ** 2, self.hidden_dim // self.num_head))
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def extract_feats(self, x, Hout, Wout, offset, ks=3):
        B, C, Hin, Win = x.shape
        device = offset.device

        row_indices = torch.arange(Hout, device=device)
        col_indices = torch.arange(Wout, device=device)
        row_indices, col_indices = torch.meshgrid(row_indices, col_indices)
        index_tensor = torch.stack((row_indices, col_indices), dim=-1).view(1, Hout, Wout, 2)
        offset = rearrange(offset, "b (kh kw d) h w -> b kh h kw w d", kh=ks, kw=ks)
        offset = offset + index_tensor.view(1, 1, Hout, 1, Wout, 2)
        offset = offset.contiguous().view(B, ks * Hout, ks * Wout, 2)

        offset[..., 0] = (2 * offset[..., 0] / (Hout - 1) - 1)
        offset[..., 1] = (2 * offset[..., 1] / (Wout - 1) - 1)
        offset = offset.flip(-1)

        out = nn.functional.grid_sample(x, offset, mode="bilinear", padding_mode="zeros", align_corners=True)
        # out = out.reshape(B, -1, ks, Hout, ks, Wout).permute(0, 2, 4, 1, 3, 5).reshape(B, ks * ks, -1, Hout, Wout)
        out = rearrange(out, "b c (ksh h) (ksw w) -> b (ksh ksw) c h w", ksh=ks, ksw=ks)
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        out_H, out_W = int(H * self.up_factor), int(W * self.up_factor)
        v = x
        x = self.layer_norm(x).contiguous()
        q = self.proj_q(x)
        k = self.proj_k(x)

        q = torch.nn.functional.interpolate(q, (out_H, out_W), mode="bilinear", align_corners=True)
        # q = torch.nn.functional.interpolate(q, (out_H, out_W), mode="nearest")

        q_off = q.view(B * self.n_groups, -1, out_H, out_W)
        pred_offset = self.conv_offset(q_off).contiguous()
        offset = pred_offset.tanh().mul(self.offset_range_factor) + self.base_offset.to(x.dtype)

        k = k.view(B * self.n_groups, self.hidden_dim // self.n_groups, H, W)
        v = v.view(B * self.n_groups, C // self.n_groups, H, W)
        k = self.extract_feats(k, out_H, out_W, offset=offset, ks=self.up_ks)
        v = self.extract_feats(v, out_H, out_W, offset=offset, ks=self.up_ks)

        q = rearrange(q, "b (nh c) h w -> b nh (h w) () c", nh=self.num_head)
        k = rearrange(k, "(b g) n c h w -> b (h w) n (g c)", g=self.n_groups)
        v = rearrange(v, "(b g) n c h w -> b (h w) n (g c)", g=self.n_groups)
        k = rearrange(k, "b n1 n (nh c) -> b nh n1 n c", nh=self.num_head)
        v = rearrange(v, "b n1 n (nh c) -> b nh n1 n c", nh=self.num_head)

        if self.rpb:
            k = k + self.relative_position_bias_table
        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, "b nh (h w) t c -> b (nh c) (t h) w", h=out_H)
        return out


@NECKS.register_module()
class DeformAttnFPN(BaseModule):
    """FPN_CARAFE is a more flexible implementation of FPN. It allows more
    choice for upsample methods during the top-down pathway.

    It can reproduce the performance of ICCV 2019 paper
    CARAFE: Content-Aware ReAssembly of FEatures
    Please refer to https://arxiv.org/abs/1905.02188 for more details.

    Args:
        in_channels (list[int]): Number of channels for each input feature map.
        out_channels (int): Output channels of feature pyramids.
        num_outs (int): Number of output stages.
        start_level (int): Start level of feature pyramids.
            (Default: 0)
        end_level (int): End level of feature pyramids.
            (Default: -1 indicates the last level).
        norm_cfg (dict): Dictionary to construct and config norm layer.
        activate (str): Type of activation function in ConvModule
            (Default: None indicates w/o activation).
        order (dict): Order of components in ConvModule.
        upsample (str): Type of upsample layer.
        upsample_cfg (dict): Dictionary to construct and config upsample layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=None,
                 act_cfg=None,
                 order=('conv', 'norm', 'act'),
                 upsample_type="carafe",
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(DeformAttnFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_bias = norm_cfg is None
        self.relu = nn.ReLU(inplace=False)

        self.order = order
        assert order in [('conv', 'norm', 'act'), ('act', 'conv', 'norm')]

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        self.upsample_modules = ModuleList()
        self.upsample = True
        self.upsample_kernel = 5

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False,
                order=self.order)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False,
                order=self.order)
            if i != self.backbone_end_level - 1:
                print("##########################")
                print(f"{upsample_type}!")
                print("##########################")
                if upsample_type == "deform":
                    upsample_module = DAU(in_c=out_channels, ratio=4, upsample_ks=3)
                elif upsample_type == 'deconv':
                    upsample_cfg = dict(
                        type="deconv",
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=self.upsample_kernel,
                        stride=2,
                        padding=(self.upsample_kernel - 1) // 2,
                        output_padding=(self.upsample_kernel - 1) // 2)
                    upsample_module = build_upsample_layer(upsample_cfg)
                elif upsample_type == 'pixel_shuffle':
                    upsample_cfg = dict(
                        type="pixel_shuffle",
                        in_channels=out_channels,
                        out_channels=out_channels,
                        scale_factor=2,
                        upsample_kernel=self.upsample_kernel)
                    upsample_module = build_upsample_layer(upsample_cfg)
                elif upsample_type == 'carafe':
                    upsample_cfg = dict(
                        type='carafe',
                        channels=out_channels,
                        up_kernel=5,
                        up_group=1,
                        encoder_kernel=3,
                        encoder_dilation=1,
                        scale_factor=2,
                        compressed_channels=64)
                    upsample_module = build_upsample_layer(upsample_cfg)
                elif upsample_type == "nearest":
                    upsample_cfg = dict(
                        type="nearest",
                        scale_factor=2,
                        mode="nearest",
                        align_corners=None)
                    upsample_module = build_upsample_layer(upsample_cfg)
                self.upsample_modules.append(upsample_module)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_out_levels = (
                num_outs - self.backbone_end_level + self.start_level)
        if extra_out_levels >= 1:
            for i in range(extra_out_levels):
                in_channels = (
                    self.in_channels[self.backbone_end_level -
                                     1] if i == 0 else out_channels)
                extra_l_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=self.with_bias,
                    act_cfg=act_cfg,
                    inplace=False,
                    order=self.order)

                print("##########################")
                print(f"{upsample_type}!")
                print("##########################")
                if upsample_type == "deform":
                    upsample_module = DAU(in_c=out_channels, ratio=4, upsample_ks=3)
                elif upsample_type == 'deconv':
                    upsample_cfg = dict(
                        type="deconv",
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=self.upsample_kernel,
                        stride=2,
                        padding=(self.upsample_kernel - 1) // 2,
                        output_padding=(self.upsample_kernel - 1) // 2)
                    upsample_module = build_upsample_layer(upsample_cfg)
                elif upsample_type == 'pixel_shuffle':
                    upsample_cfg = dict(
                        type="pixel_shuffle",
                        in_channels=out_channels,
                        out_channels=out_channels,
                        scale_factor=2,
                        upsample_kernel=self.upsample_kernel)
                    upsample_module = build_upsample_layer(upsample_cfg)
                elif upsample_type == 'carafe':
                    upsample_cfg = dict(
                        type='carafe',
                        channels=out_channels,
                        up_kernel=5,
                        up_group=1,
                        encoder_kernel=3,
                        encoder_dilation=1,
                        scale_factor=2,
                        compressed_channels=64)
                    upsample_module = build_upsample_layer(upsample_cfg)
                elif upsample_type == "nearest":
                    upsample_cfg = dict(
                        type="nearest",
                        scale_factor=2,
                        mode="nearest",
                        align_corners=None)
                    upsample_module = build_upsample_layer(upsample_cfg)

                extra_fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.with_bias,
                    act_cfg=act_cfg,
                    inplace=False,
                    order=self.order)
                self.upsample_modules.append(upsample_module)
                self.fpn_convs.append(extra_fpn_conv)
                self.lateral_convs.append(extra_l_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of module."""
        super(DeformAttnFPN, self).init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()
        for m in self.modules():
            if isinstance(m, DAU):
                # nn.init.constant_(m.conv_offset[-1].weight, 0)
                m._init_weights()

    def slice_as(self, src, dst):
        """Slice ``src`` as ``dst``

        Note:
            ``src`` should have the same or larger size than ``dst``.

        Args:
            src (torch.Tensor): Tensors to be sliced.
            dst (torch.Tensor): ``src`` will be sliced to have the same
                size as ``dst``.

        Returns:
            torch.Tensor: Sliced tensor.
        """
        assert (src.size(2) >= dst.size(2)) and (src.size(3) >= dst.size(3))
        if src.size(2) == dst.size(2) and src.size(3) == dst.size(3):
            return src
        else:
            return src[:, :, :dst.size(2), :dst.size(3)]

    def tensor_add(self, a, b):
        """Add tensors ``a`` and ``b`` that might have different sizes."""
        if a.size() == b.size():
            c = a + b
        else:
            c = a + self.slice_as(b, a)
        return c

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i <= self.backbone_end_level - self.start_level:
                input = inputs[min(i + self.start_level, len(inputs) - 1)]
            else:
                input = laterals[-1]
            lateral = lateral_conv(input)
            laterals.append(lateral)

        # build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            if self.upsample is not None:
                upsample_feat = self.upsample_modules[i - 1](laterals[i])
            else:
                upsample_feat = laterals[i]
            laterals[i - 1] = self.tensor_add(laterals[i - 1], upsample_feat)
        # build outputs
        num_conv_outs = len(self.fpn_convs)
        outs = []
        for i in range(num_conv_outs):
            out = self.fpn_convs[i](laterals[i])
            outs.append(out)

        return tuple(outs)
