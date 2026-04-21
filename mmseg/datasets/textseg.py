from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class TextSegDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('backgound', 'text'),
        palette=[[0, 0, 0], [0, 192, 64]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_mask.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super(TextSegDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)