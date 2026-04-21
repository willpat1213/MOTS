dataset_type = 'SynthOverlapTextDataset'
scene_synth_dataset_type = 'SceneSynthOverlapTextDataset'
real_dataset_type = 'OverlapTextDataset'

data_root = ""
ann_file = ['printed_en/converted_gt.jsonl',
             'printed_hw_en/converted_gt.jsonl',
             'printed_hw_zh_en/converted_gt.jsonl',
             'printed_num/converted_gt.jsonl',
             'printed_zh_en_bill/converted_gt.jsonl']
scene_synth_data_root = ''
scene_synth_ann_file = 'gt.jsonl'
real_data_root = ""
real_ann_file = 'converted_train_gt_v2.jsonl'
test_ann_file = 'converted_test_gt_v2.jsonl'

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

synthtiger_dataset_train = dict(
        type=dataset_type,
        ann_file=ann_file,
        data_root=data_root,
        data_prefix=dict(
            img_path=['printed_en/img_dir/train',
                      'printed_hw_en/img_dir/train',
                      'printed_hw_zh_en/img_dir/train',
                      'printed_num/img_dir/train',
                      'printed_zh_en_bill/img_dir/train',
                      ],
            seg_map_path=['printed_en/ann_dir/train',
                          'printed_hw_en/ann_dir/train',
                          'printed_hw_zh_en/ann_dir/train',
                          'printed_num/ann_dir/train',
                          'printed_zh_en_bill/ann_dir/train',
                      ],
            depth_map_path=['printed_en/depth_dir/min_depth_gray_train',
                          'printed_hw_en/depth_dir/min_depth_gray_train',
                          'printed_hw_zh_en/depth_dir/min_depth_gray_train',
                          'printed_num/depth_dir/min_depth_gray_train',
                          'printed_zh_en_bill/depth_dir/min_depth_gray_train',
                      ],
            ),
        pipeline=train_pipeline)

scene_synth_dataset_train = dict(
        type=scene_synth_dataset_type,
        ann_file=scene_synth_ann_file,
        data_root=scene_synth_data_root,
        data_prefix=dict(
            img_path='target_processed', seg_map_path='ann_dir', depth_map_path='min_gray_train'),
        pipeline=train_pipeline)

real_dataset_train = dict(
        type=real_dataset_type,
        ann_file=real_ann_file,
        data_root=real_data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train', depth_map_path='depth_dir/vitb_gray_train'),
        pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            synthtiger_dataset_train,
            real_dataset_train,
            scene_synth_dataset_train,
    ]))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=real_dataset_type,
        ann_file=test_ann_file,
        data_root=real_data_root,
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val', depth_map_path='depth_dir/vitb_gray_val'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='OverlapIoUMetric', iou_metrics=['mIoU', 'mFscore', 'mTextIoU'])
test_evaluator = val_evaluator