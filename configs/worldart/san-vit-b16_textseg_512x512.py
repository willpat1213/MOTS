_base_ = [
    '../_base_/datasets/textseg.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
num_classes = 2

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type='MultimodalEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='',
    asymetric_input=True,
    encoder_resolution=0.5,
    image_encoder=dict(
        type='VisionTransformer',
        img_size=(224, 224),
        patch_size=16,
        patch_pad=0,
        in_channels=3,
        embed_dims=768,
        num_layers=9,
        num_heads=12,
        mlp_ratio=4,
        out_origin=True,
        out_indices=(2, 5, 8),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=True,
        patch_bias=False,
        pre_norm=True,
        norm_cfg=dict(type='LN', eps=1e-5),
        act_cfg=dict(type='QuickGELU'),
        norm_eval=False,
        interpolate_mode='bicubic',
        frozen_exclude=['pos_embed']),
    text_encoder=dict(
        type='CLIPTextEncoder',
        dataset_name='textseg',
        templates='vild',
        embed_dims=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4,
        output_dims=512,
        cache_feature=True,
        cat_bg=True,
        norm_cfg=dict(type='LN', eps=1e-5)
        ),
    decode_head=dict(
        type='SideAdapterCLIPHead',
        num_classes=num_classes,
        deep_supervision_idxs=[7],
        san_cfg=dict(
            in_channels=3,
            clip_channels=768,
            embed_dims=240,
            patch_size=16,
            patch_bias=True,
            num_queries=100,
            cfg_encoder=dict(
                num_encode_layer=8,
                num_heads=6,
                mlp_ratio=4
            ),
            fusion_index=[0, 1, 2, 3],
            cfg_decoder=dict(
                num_heads=12,
                num_layers=1,
                embed_channels=256,
                mlp_channels=256,
                num_mlp=3,
                rescale=True),
            norm_cfg=dict(type='LN', eps=1e-6),
        ),
        maskgen_cfg=dict(
            sos_token_format='cls_token',
            sos_token_num=100,
            cross_attn=False,
            num_layers=3,
            embed_dims=768,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            out_dims=512,
            final_norm=True,
            act_cfg=dict(type='QuickGELU'),
            norm_cfg=dict(type='LN', eps=1e-5),
            frozen_exclude=[]
        ),
        align_corners=False,
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='ClassificationCost', weight=2.0),
                    dict(
                        type='CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ])),
        loss_decode=[dict(type='CrossEntropyLoss',
                          reduction='mean',
                          use_sigmoid=False,
                          loss_name='loss_cls_ce',
                          loss_weight=2.0,
                          class_weight=[1.0] * (num_classes-1) + [0.1],
                          ),
                     dict(type='CrossEntropyLoss',
                          use_sigmoid=True,
                          loss_name='loss_mask_ce',
                          loss_weight=5.0),
                     dict(type='DiceLoss',
                          ignore_index=None,
                          naive_dice=True,
                          eps=1,
                          loss_name='loss_mask_dice',
                          loss_weight=5.0)
                     ]),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
