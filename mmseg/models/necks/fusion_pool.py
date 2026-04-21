import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention

from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptConfigType
from mmdet.models.layers import SinePositionalEncoding

from mmseg.registry import MODELS
from .common import SqueezeAndExcitation, LearnableWeights, DepthCrossattnLayer


@MODELS.register_module()
class FusionPool(BaseModule):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 act=dict(type='ReLU', inplace=True),
                 anchor=(16, 16),
                 mode='add'):
        super().__init__()
        self.mode = mode
        if mode == 'se':
            self.se_rgbs = nn.ModuleList()
            self.se_depths = nn.ModuleList()
            self.conv1x1s = nn.ModuleList()
            for i in range(len(in_channels)):
                se_rgb = SqueezeAndExcitation(in_channels[i],
                                                activation=act)
                self.se_rgbs.append(se_rgb)
                se_depth = SqueezeAndExcitation(in_channels[i],
                                             activation=act)
                self.se_depths.append(se_depth)
                conv1x1 = ConvModule(in_channels[i]*2,
                                     in_channels[i],
                                     kernel_size=1,
                                     norm_cfg=dict(type='BN', requires_grad=True))
                self.conv1x1s.append(conv1x1)

        elif mode == 'concat':
            self.conv1x1s = nn.ModuleList()
            for i in range(len(in_channels)):
                conv1x1 = ConvModule(in_channels[i]*2,
                                     in_channels[i],
                                     kernel_size=1,
                                     norm_cfg=dict(type='BN', requires_grad=True))
                self.conv1x1s.append(conv1x1)

        elif mode == 'ca':
            self.h_anchor, self.w_anchor = anchor
            self.num_queries = self.h_anchor * self.w_anchor

            # self.avgpool = nn.AdaptiveAvgPool2d(self.num_queries)
            # self.maxpool = nn.AdaptiveMaxPool2d(self.num_queries)
            # self.img_feat_coefficient = LearnableWeights()
            # self.depth_feat_coefficient = LearnableWeights()

            self.avgpools = nn.ModuleList()
            self.img_pos_embed = nn.ModuleList()
            self.depth_pos_embed = nn.ModuleList()
            self.cross_attns = nn.ModuleList()
            self.conv1x1s = nn.ModuleList()
            for i in range(len(in_channels)):
                self.avgpools.append(nn.AdaptiveAvgPool2d((self.num_queries)))
                self.img_pos_embed.append(SinePositionalEncoding(num_feats=128, normalize=True))
                self.depth_pos_embed.append(SinePositionalEncoding(num_feats=128, normalize=True))
                self.cross_attns.append(DepthCrossattnLayer(d_model=in_channels[i],
                                                            d_inner=in_channels[i]//2,
                                                            n_head=8))
                self.conv1x1s.append(ConvModule(in_channels[i]*2,
                                     in_channels[i],
                                     kernel_size=1,
                                     norm_cfg=dict(type='BN', requires_grad=True)))

    def forward(self, rgb, depth, training=True):
        out = []
        for i, (_r, _d) in enumerate(zip(rgb, depth)):
            # import pdb; pdb.set_trace()
            if self.mode == 'add':
                _o = _r + _d
            elif self.mode == 'se':
                _r_feat = self.se_rgbs[i](_r)
                _d_feat = self.se_depths[i](_d)
                _o = torch.cat([_r_feat, _d_feat], dim=1)
                _o = self.conv1x1s[i](_o)
            elif self.mode == 'concat':
                _o = torch.cat([_r, _d], dim=1)
                _o = self.conv1x1s[i](_o)
            elif self.mode == 'mul':
                _o = torch.mul(_r, _d)
            
            elif self.mode == 'ca':
                bs, _, h, w = _r.shape
                # _r_feat = self.img_feat_coefficient(self.avgpool(_r), self.maxpool(_r))
                # _d_feat = self.depth_feat_coefficient(self.avgpool(_d), self.maxpool(_d))
                _r_feat = self.avgpools[i](_r)
                _d_feat = self.avgpools[i](_d)

                _r_feat = F.interpolate(_r, size=(self.h_anchor, self.w_anchor), mode='nearest' \
                                        if training else 'bilinear')
                _d_feat = F.interpolate(_d, size=(self.h_anchor, self.w_anchor), mode='nearest' \
                                        if training else 'bilinear')

                # img_pos_embed = self.img_pos_embed[i].weight.unsqueeze(0).repeat((bs, 1, 1))
                # depth_pos_embed = self.depth_pos_embed[i].weight.unsqueeze(0).repeat((bs, 1, 1))
                # img_pos_embed = self.img_pos_embed[i].repeat((bs, 1, 1))
                # depth_pos_embed = self.depth_pos_embed[i].repeat((bs, 1, 1))

                _r_feat_flat = _r_feat.view(bs, -1, self.num_queries).permute(0, 2, 1).contiguous()
                _d_feat_flat = _d_feat.view(bs, -1, self.num_queries).permute(0, 2, 1).contiguous()

                img_mask = _r_feat_flat.new_zeros((bs, ) + _r_feat_flat.shape[-2:], dtype=torch.bool)
                depth_mask = _d_feat_flat.new_zeros((bs, ) + _d_feat_flat.shape[-2:], dtype=torch.bool)
                img_pos_embed = self.img_pos_embed[i](img_mask).flatten(2).permute(0, 2, 1).contiguous()
                depth_pos_embed = self.img_pos_embed[i](depth_mask).flatten(2).permute(0, 2, 1).contiguous()

                fusion_feat_flat = self.cross_attns[i](_r_feat_flat,
                                                    _d_feat_flat,
                                                    h_anchor=self.h_anchor,
                                                    w_anchor=self.w_anchor,
                                                    q_pos=img_pos_embed,
                                                    k_pos=depth_pos_embed)
                fusion_feat = fusion_feat_flat.view(bs, self.h_anchor, self.w_anchor, -1).permute(0, 3, 1, 2).contiguous()
                
                _o = F.interpolate(fusion_feat, size=(h,w), mode='nearest' if training else 'bilinear')
                _o = torch.cat([_r, _o], dim=1)
                _o = self.conv1x1s[i](_o)

            out.append(_o)
        return tuple(out)