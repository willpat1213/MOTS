dataset_type = 'SignaTR6K'
data_root = ""
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthAnnotation'),
    dict(type='OverlapLoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048
        ),
    # dict(
    #     type='RandomResize',
    #     scale=crop_size,
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs',
         meta_keys=('img_path', 'seg_map_path', 'depth_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label', 'overlap_label', 'text'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthAnnotation'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    # dict(type='ResizeToMultiple', size_divisor=16),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='OverlapLoadAnnotations'),
    dict(type='PackSegInputs',
         meta_keys=('img_path', 'seg_map_path', 'depth_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label', 'overlap_label', 'text'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/crop', seg_map_path='train/converted_label', depth_map_path='train/depth'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/crop', seg_map_path='test/converted_label', depth_map_path='test/depth'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='SignaIoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator