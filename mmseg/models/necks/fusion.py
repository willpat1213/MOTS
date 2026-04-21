import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention

from mmengine.model import BaseModule
from mmengine.model import ModuleList, caffe2_xavier_init
from mmdet.utils import ConfigType, OptConfigType
from mmdet.models.layers import SinePositionalEncoding

from mmseg.registry import MODELS
from .common import SqueezeAndExcitation, LearnableWeights, DepthCrossattnLayer


@MODELS.register_module()
class FusionAdd(BaseModule):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 act=dict(type='ReLU', inplace=True),
                 size=[128, 64, 32, 16],
                 anchor=(16, 16),
                 mode='add'):
        super().__init__()
        self.mode = mode
        if mode == 'se':
            self.se_rgbs = ModuleList()
            self.se_depths = ModuleList()
            self.conv1x1s = ModuleList()
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
            # self.h_anchor, self.w_anchor = anchor
            # self.num_queries = self.h_anchor * self.
            self.size = size
            # self.num_queries = [s*s for s in size]

            # self.avgpool = nn.AdaptiveAvgPool2d((size[i]*size[i]))
            # self.maxpool = nn.AdaptiveMaxPool2d((size[i]*size[i]))
            # self.img_feat_coefficient = LearnableWeights()
            # self.depth_feat_coefficient = LearnableWeights()

            # self.img_pos_embed = nn.ParameterList()
            # self.depth_pos_embed = nn.ParameterList() 
            # self.img_projs = nn.ModuleList()
            # self.depth_projs = nn.ModuleList()
            self.conv1x1s = ModuleList()
            self.img_pos_embed = nn.ModuleList()
            self.depth_pos_embed = nn.ModuleList()
            self.cross_attns = ModuleList()
            for i in range(len(in_channels)-1):
                i += 1
                # self.avgpools.append(nn.AdaptiveAvgPool2d((size[i]*size[i])))
                self.img_pos_embed.append(SinePositionalEncoding(num_feats=in_channels[i]/2, normalize=True))
                self.depth_pos_embed.append(SinePositionalEncoding(num_feats=in_channels[i]/2, normalize=True))
                self.cross_attns.append(DepthCrossattnLayer(d_model=in_channels[i],
                                                            d_inner=in_channels[i]*4,
                                                            n_head=8))
                self.conv1x1s.append(ConvModule(in_channels[i]*2,
                                     in_channels[i],
                                     kernel_size=1,
                                     norm_cfg=dict(type='BN', requires_grad=True)))
            self.conv1x1 = ConvModule(in_channels[0]*2,
                                        in_channels[0],
                                        kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
    
    # def init_weights(self):
    #     # Initialize weights for convolutional layers
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             caffe2_xavier_init(m)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)
    #     if self.mode == 'ca':
    #         # Initialize positional embeddings
    #         for pos_embed in self.img_pos_embed:
    #             nn.init.normal_(pos_embed, mean=0, std=0.02)
    #         for pos_embed in self.depth_pos_embed:
    #             nn.init.normal_(pos_embed, mean=0, std=0.02)
            
    #         # Initialize cross attention layers
    #         for cross_attn in self.cross_attns:
    #             for m in cross_attn.modules():
    #                 if isinstance(m, nn.Linear):
    #                     nn.init.xavier_normal_(m.weight)
    #                     if m.bias is not None:
    #                         nn.init.constant_(m.bias, 0)
        

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
                if i == 0:
                    _o = torch.cat([_r, _d], dim=1)
                    _o = self.conv1x1(_o)
                else:
                    bs, _, h, w = _r.shape
                    # num_queries = self.num_queries[i]
                    # _r_feat = self.img_feat_coefficient(self.avgpool(_r), self.maxpool(_r))
                    # _d_feat = self.depth_feat_coefficient(self.avgpool(_d), self.maxpool(_d))

                    # _r_feat = _r
                    _d_feat = _d
                    _f_feat = torch.cat([_r, _d], dim=1)
                    _f_feat = self.conv1x1s[i-1](_f_feat)

                    _f_feat_flat = _f_feat.view(bs, -1, h*w).permute(0, 2, 1).contiguous()
                    _d_feat_flat = _d_feat.view(bs, -1, h*w).permute(0, 2, 1).contiguous()

                    fusion_mask = _f_feat_flat.new_zeros((bs, h, w), dtype=torch.bool)
                    depth_mask = _d_feat_flat.new_zeros((bs, h, w), dtype=torch.bool)
                    img_pos_embed = self.img_pos_embed[i-1](fusion_mask).flatten(2).permute(0, 2, 1).contiguous()
                    depth_pos_embed = self.img_pos_embed[i-1](depth_mask).flatten(2).permute(0, 2, 1).contiguous()

                    # mask = _r_feat_flat.new_zeros((bs, ) + _r_feat_flat.shape[-2:], dtype=torch.bool)

                    fusion_feat_flat = self.cross_attns[i-1](_f_feat_flat,
                                                        _d_feat_flat,
                                                        h_anchor=h,
                                                        w_anchor=w,
                                                        q_pos=img_pos_embed,
                                                        k_pos=depth_pos_embed)
                    fusion_feat = fusion_feat_flat.view(bs, h, w, -1).permute(0, 3, 1, 2).contiguous()

                    # _o = F.interpolate(fusion_feat, size=(h,w), mode='nearest' if training else 'bilinear')
                    # _o = torch.cat([_d, fusion_feat], dim=1)
                    # _o = self.conv1x1s[i-1](_o)
                    # _o = fusion_feat + _d
                    _o = fusion_feat

            out.append(_o)
        return tuple(out)