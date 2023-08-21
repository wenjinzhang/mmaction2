default_scope = 'mmaction'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
url = 'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth'
model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth'
    ),
    backbone=dict(
        type='mmaction.VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-06),
        drop_path_rate=0.2,
        use_mean_pooling=False,
        return_feat_map=True),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True,
            with_global=False),
        bbox_head=dict(
            type='BBoxHeadAVA',
            in_channels=1024,
            num_classes=6,
            focal_gamma=2.0,
            focal_alpha=0.25,
            topk=(1, 3),
            multilabel=False,
            dropout_ratio=0.5)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        _scope_='mmaction',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))
dataset_type = 'AVADataset'
data_root = 'data/trauma/rawframes'
anno_root = 'data/trauma/annotations'
ann_file_train = 'data/trauma/annotations/trauma_trainV2_N0.csv'
ann_file_val = 'data/trauma/annotations/trauma_valV2_N0.csv'
exclude_file_train = 'data/trauma/annotations/trauma_trainV2_excluded.csv'
exclude_file_val = 'data/trauma/annotations/trauma_valV2_excluded.csv'
label_file = 'data/trauma/annotations/action_list.pbtxt'
proposal_file_train = 'data/trauma/annotations/trauma_trainV2.csv.pkl'
proposal_file_val = 'data/trauma/annotations/trauma_valV2.csv.pkl'
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=16, frame_interval=4),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=16, frame_interval=4, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='AVADataset',
        ann_file='data/trauma/annotations/trauma_trainV2_N0.csv',
        exclude_file='data/trauma/annotations/trauma_trainV2_excluded.csv',
        pipeline=[
            dict(type='SampleAVAFrames', clip_len=16, frame_interval=4),
            dict(type='RawFrameDecode'),
            dict(type='RandomRescale', scale_range=(256, 320)),
            dict(type='RandomCrop', size=256),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='FormatShape', input_format='NCTHW', collapse=True),
            dict(type='PackActionInputs')
        ],
        label_file='data/trauma/annotations/action_list.pbtxt',
        timestamp_start=0,
        num_classes=6,
        custom_classes=[3, 5, 9, 11, 17],
        proposal_file='data/trauma/annotations/trauma_trainV2.csv.pkl',
        data_prefix=dict(img='data/trauma/rawframes')))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AVADataset',
        ann_file='data/trauma/annotations/trauma_valV2_N0.csv',
        exclude_file='data/trauma/annotations/trauma_valV2_excluded.csv',
        pipeline=[
            dict(
                type='SampleAVAFrames',
                clip_len=16,
                frame_interval=4,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='FormatShape', input_format='NCTHW', collapse=True),
            dict(type='PackActionInputs')
        ],
        label_file='data/trauma/annotations/action_list.pbtxt',
        timestamp_start=0,
        num_classes=6,
        custom_classes=[3, 5, 9, 11, 17],
        proposal_file='data/trauma/annotations/trauma_valV2.csv.pkl',
        data_prefix=dict(img='data/trauma/rawframes'),
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AVADataset',
        ann_file='data/trauma/annotations/trauma_valV2_N0.csv',
        exclude_file='data/trauma/annotations/trauma_valV2_excluded.csv',
        pipeline=[
            dict(
                type='SampleAVAFrames',
                clip_len=16,
                frame_interval=4,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='FormatShape', input_format='NCTHW', collapse=True),
            dict(type='PackActionInputs')
        ],
        label_file='data/trauma/annotations/action_list.pbtxt',
        timestamp_start=0,
        num_classes=6,
        custom_classes=[3, 5, 9, 11, 17],
        proposal_file='data/trauma/annotations/trauma_valV2.csv.pkl',
        data_prefix=dict(img='data/trauma/rawframes'),
        test_mode=True))
val_evaluator = dict(
    type='AVAMetric',
    ann_file='data/trauma/annotations/trauma_valV2_N0.csv',
    label_file='data/trauma/annotations/action_list.pbtxt',
    num_classes=6,
    custom_classes=[3, 5, 9, 11, 17],
    exclude_file='data/trauma/annotations/trauma_valV2_excluded.csv')
test_evaluator = dict(
    type='AVAMetric',
    ann_file='data/trauma/annotations/trauma_valV2_N0.csv',
    label_file='data/trauma/annotations/action_list.pbtxt',
    num_classes=6,
    custom_classes=[3, 5, 9, 11, 17],
    exclude_file='data/trauma/annotations/trauma_valV2_excluded.csv')
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=40, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=0,
        by_epoch=True,
        begin=20,
        end=40,
        convert_to_iter_based=True)
]
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.00025, weight_decay=0.05),
    accumulative_counts=32,
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg=dict(decay_rate=0.8, decay_type='layer_wise', num_layers=24),
    clip_grad=dict(max_norm=40, norm_type=2))
auto_scale_lr = dict(enable=False, base_batch_size=64)
launcher = 'pytorch'
work_dir = './work_dirs/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_trauma-rgb_subset_N0'
randomness = dict(seed=0, diff_rank_seed=False, deterministic=False)
