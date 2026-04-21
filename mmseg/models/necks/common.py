import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention

from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptConfigType

from mmseg.registry import MODELS

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=dict(type='ReLU', inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            ConvModule(channel, channel // reduction, kernel_size=1, act_cfg=activation),
            ConvModule(channel // reduction, channel, kernel_size=1, act_cfg=dict(type='Sigmoid'))
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y

class LearnableWeights(BaseModule):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out
    
class LocalityAwareFeedforward(BaseModule):
    """Locality-aware feedforward layer in SATRN, see `SATRN.

    <https://arxiv.org/abs/1910.04396>`_
    """

    def __init__(self,
                 d_in,
                 d_hid,
                 dropout=0.1,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Constant', layer='BatchNorm2d', val=1, bias=0)
                 ]):
        super().__init__(init_cfg=init_cfg)
        self.conv1 = ConvModule(
            d_in,
            d_hid,
            kernel_size=1,
            padding=0,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

        self.depthwise_conv = ConvModule(
            d_hid,
            d_hid,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=d_hid,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

        self.conv2 = ConvModule(
            d_hid,
            d_in,
            kernel_size=1,
            padding=0,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.conv2(x)

        return x

class DepthCrossattnLayer(BaseModule):
    """"""

    def __init__(self,
                 d_model=256,
                 d_inner=256,
                 n_head=8,
                 dropout=dict(type='Dropout', drop_prob=0.1),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.attn = MultiheadAttention(d_model, n_head, dropout_layer=dropout)
        self.cross_attn = MultiheadAttention(d_model, n_head, dropout_layer=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.feed_forward = LocalityAwareFeedforward(d_model, d_inner, dropout=dropout)
        self.feed_forward = FFN(d_model, d_inner, act_cfg=dict(type='ReLU', inplace=True))

    def forward(self, x, depth, h_anchor, w_anchor, q_pos, k_pos, mask=None):
        n, hw_anchor, c = x.size()
        residual = x
        # x = self.norm1(x)
        # x = residual + self.attn(x, x, x, attn_mask=mask) 
        # residual = x
        x = self.norm2(x)
        x = residual + self.cross_attn(depth, x, x, q_pos, k_pos, attn_mask=mask)
        residual = x
        x = self.norm3(x)
        # x = x.transpose(1, 2).contiguous().view(n, c, h_anchor, w_anchor)
        x = self.feed_forward(x)
        # x = x.view(n, c, hw_anchor).transpose(1, 2).contiguous()
        x = residual + x
        return x