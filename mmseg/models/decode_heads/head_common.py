import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmengine.model import BaseModule
from mmengine.model import ModuleList, caffe2_xavier_init
from mmdet.utils import ConfigType, OptConfigType
from mmdet.models.layers import SinePositionalEncoding

from mmseg.registry import MODELS
from ..necks.common import DepthCrossattnLayer


@MODELS.register_module()
class FusionCrossAttn(BaseModule):
    def __init__(self,
                 in_channels=256,
                 size=[16, 32, 64, 128]):
        super().__init__()
        self.size = size

        self.conv1x1s = ModuleList()
        self.img_pos_embed = nn.ModuleList()
        self.depth_pos_embed = nn.ModuleList()
        self.cross_attns = ModuleList()
        for _ in range(len(size)-1):
            self.img_pos_embed.append(SinePositionalEncoding(num_feats=in_channels/2, normalize=True))
            self.depth_pos_embed.append(SinePositionalEncoding(num_feats=in_channels/2, normalize=True))
            self.cross_attns.append(DepthCrossattnLayer(d_model=in_channels,
                                                        d_inner=in_channels*4,
                                                        n_head=8))
            self.conv1x1s.append(ConvModule(in_channels*2,
                                    in_channels,
                                    kernel_size=1,
                                    norm_cfg=dict(type='BN', requires_grad=True)))
        self.conv1x1s.append(ConvModule(in_channels*2,
                                in_channels,
                                kernel_size=1,
                                norm_cfg=dict(type='BN', requires_grad=True)))

    def forward(self, rgb, depth, training=True):
        out = []
        for i, (_r, _d) in enumerate(zip(rgb, depth)):
            if i == 3:
                _o = torch.cat([_r, _d], dim=1)
                _o = self.conv1x1s[i](_o)
            else:
                bs, _, h, w = _r.shape
                # _r_feat = _r
                _d_feat = _d

                _f_feat = torch.cat([_r, _d], dim=1)
                _f_feat = self.conv1x1s[i](_f_feat)

                # _r_feat_flat = _r_feat.view(bs, -1, h*w).permute(0, 2, 1).contiguous()
                _f_feat_flat = _f_feat.view(bs, -1, h*w).permute(0, 2, 1).contiguous()
                _d_feat_flat = _d_feat.view(bs, -1, h*w).permute(0, 2, 1).contiguous()

                # img_mask = _r_feat_flat.new_zeros((bs, h, w), dtype=torch.bool)
                fusion_mask = _f_feat_flat.new_zeros((bs, h, w), dtype=torch.bool)
                depth_mask = _d_feat_flat.new_zeros((bs, h, w), dtype=torch.bool)
                # img_pos_embed = self.img_pos_embed[i](img_mask).flatten(2).permute(0, 2, 1).contiguous()
                fusion_pos_embed = self.img_pos_embed[i](fusion_mask).flatten(2).permute(0, 2, 1).contiguous()
                depth_pos_embed = self.img_pos_embed[i](depth_mask).flatten(2).permute(0, 2, 1).contiguous()

                fusion_feat_flat = self.cross_attns[i](_f_feat_flat,
                                                    _d_feat_flat,
                                                    h_anchor=h,
                                                    w_anchor=w,
                                                    q_pos=fusion_pos_embed,
                                                    k_pos=depth_pos_embed)
                fusion_feat = fusion_feat_flat.view(bs, h, w, -1).permute(0, 3, 1, 2).contiguous()

                # _o = torch.cat([fusion_feat, _d], dim=1)
                # _o = self.conv1x1s[i](_o)
                _o = fusion_feat

            out.append(_o)
        return out[-1], out[:-1]