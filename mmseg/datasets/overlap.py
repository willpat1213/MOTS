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
class OverlapTextDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('backgound', 'occlusion', 'occluded', 'overlapped'),
        palette=[[0, 0, 0], [0, 192, 64], [0, 64, 96], [128, 50, 70]])

    def __init__(self,
                 ann_file='',
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super(OverlapTextDataset, self).__init__(
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
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                line = json.loads(line)
                img_name = line['img_name']
                data_info = dict(
                    img_path=osp.join(img_dir, img_name))
                if ann_dir is not None:
                    seg_map = line['mask']
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                if depth_dir is not None:
                    depth_map = line['img_name']
                    data_info['depth_map_path'] = osp.join(depth_dir, depth_map)
                data_info['label_map'] = self.label_map
                data_info['overlap_label'] = line['overlap_label']
                data_info['text'] = [text['label'] for text in line['texts']]
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list

@DATASETS.register_module()
class SceneSynthOverlapTextDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('backgound', 'occlusion', 'occluded', 'overlapped'),
        palette=[[0, 0, 0], [0, 192, 64], [0, 64, 96], [128, 50, 70]])

    def __init__(self,
                 ann_file='',
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super(SceneSynthOverlapTextDataset, self).__init__(
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
        num = 40000
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        depth_dir = self.data_prefix.get('depth_map_path', None)
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines[:num]:
                line = json.loads(line)
                img_name = line['img_name']
                data_info = dict(
                    img_path=osp.join(img_dir, img_name))
                if ann_dir is not None:
                    seg_map = line['mask']
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                if depth_dir is not None:
                    depth_map = line['img_name']
                    data_info['depth_map_path'] = osp.join(depth_dir, depth_map)
                data_info['label_map'] = self.label_map
                data_info['overlap_label'] = 0
                data_info['text'] = line['texts']
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list

@DATASETS.register_module()
class SynthOverlapTextDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('backgound', 'occlusion', 'occluded', 'overlapped'),
        palette=[[0, 0, 0], [0, 192, 64], [0, 64, 96], [128, 50, 70]])

    def __init__(self,
                 ann_file='',
                 img_suffix='img.png',
                 seg_map_suffix='mask.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super(SynthOverlapTextDataset, self).__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def get_num_from_map(self, input_key):

        my_map = dict(
            printed_en = 40000,
            printed_hw_en = 40000,
            printed_hw_zh_en = 80000,
            printed_num = 40000,
            printed_zh_en_bill = 40000,
        )
        for key in my_map.keys():
            if key in input_key:
                num = my_map[key]
                return num
        return None
        
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dirs = self.data_prefix.get('img_path', None)
        ann_dirs = self.data_prefix.get('seg_map_path', None)
        depth_dirs = self.data_prefix.get('depth_map_path', None)
        ann_files = self.ann_file
        for img_dir, ann_dir, ann_file, depth_dir in zip(img_dirs, ann_dirs, ann_files, depth_dirs):
            num = self.get_num_from_map(img_dir)
            if not osp.isdir(ann_file) and ann_file:
                assert osp.isfile(ann_file), \
                    f'Failed to load `ann_file` {ann_file}'
                lines = mmengine.list_from_file(
                    ann_file, backend_args=self.backend_args)
                for line in lines[:num]:
                    line = json.loads(line)
                    img_name = line['img_name']
                    data_info = dict(
                        img_path=osp.join(img_dir, img_name))
                    if ann_dir is not None:
                        seg_map = line['mask']
                        data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                    if depth_dir is not None:
                        depth_map = line['img_name']
                        data_info['depth_map_path'] = osp.join(depth_dir, depth_map)
                    data_info['text'] = [text['label'] for text in line['texts']]
                    data_info['overlap_label'] = line['overlap_label']
                    data_info['label_map'] = self.label_map
                    data_info['reduce_zero_label'] = self.reduce_zero_label
                    data_info['seg_fields'] = []
                    data_list.append(data_info)
            else:
                _suffix_len = len(self.img_suffix)
                for img in fileio.list_dir_or_file(
                        dir_path=img_dir,
                        list_dir=False,
                        suffix=self.img_suffix,
                        recursive=True,
                        backend_args=self.backend_args):
                    data_info = dict(img_path=osp.join(img_dir, img))
                    if ann_dir is not None:
                        seg_map = img[:-_suffix_len] + self.seg_map_suffix
                        data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                    data_info['label_map'] = self.label_map
                    data_info['reduce_zero_label'] = self.reduce_zero_label
                    data_info['seg_fields'] = []
                    data_list.append(data_info)
                data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
    
    def _join_prefix(self):
        """Join ``self.data_root`` with ``self.data_prefix`` and
        ``self.ann_file``.

        Examples:
            >>> # self.data_prefix contains relative paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='a/b/c/d/e')
            >>> self.ann_file
            'a/b/c/f'
            >>> # self.data_prefix contains absolute paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='/d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='/d/e')
            >>> self.ann_file
            'a/b/c/f'
        """
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if self.ann_file[0] and not is_abs(self.ann_file[0]) and self.data_root:
            self.ann_file = [join_path(self.data_root, ann) for ann in self.ann_file]
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                if self.data_root:
                    self.data_prefix[data_key] = [join_path(self.data_root, pf) for pf in prefix]
                else:
                    self.data_prefix[data_key] = prefix
            else:
                if not is_abs(prefix) and self.data_root:
                    self.data_prefix[data_key] = join_path(self.data_root, prefix)
                else:
                    self.data_prefix[data_key] = prefix