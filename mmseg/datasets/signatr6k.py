from mmseg.registry import DATASETS
from typing import Callable, Dict, List, Optional, Sequence, Union
from .basesegdataset import BaseSegDataset
import mmengine
import json
from mmengine.utils import is_abs
import mmengine.fileio as fileio
from mmengine.fileio import join_path, list_from_file, load
import os.path as osp


@DATASETS.register_module()
class SignaTR6K(BaseSegDataset):

    METAINFO = dict(
        classes=('BG', 'PT', 'HW', 'OL'),
        palette=[[0, 0, 0], [0, 192, 64], [0, 64, 96], [128, 50, 70]])

    def __init__(self,
                 ann_file='',
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super(SignaTR6K, self).__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        depth_dir = self.data_prefix.get('depth_map_path', None)
        _suffix_len = len(self.img_suffix)
        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args):
            data_info = dict(img_path=osp.join(img_dir, img))
            data_info['seg_map_path'] = osp.join(ann_dir, img)
            data_info['depth_map_path'] = osp.join(depth_dir, img)
            data_info['label_map'] = self.label_map
            data_info['overlap_label'] = None
            data_info['text'] = None
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)
        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list