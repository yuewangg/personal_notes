_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='BN', requires_grad=True) 
crop_size = (256, 256) 

model = dict(
    data_preprocessor=dict(size=crop_size),
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes = 2 ,norm_cfg = norm_cfg
    ),
    auxiliary_head=dict(num_classes = 2,norm_cfg = norm_cfg
    )
)

dataset_type = 'StanfordBackgroundDataset'

data_root = '/HOME/scz5202/run/mmsegmentation/data/Glomeruli-dataset'
img_dir = '/HOME/scz5202/run/mmsegmentation/data/Glomeruli-dataset/images'
ann_dir = '/HOME/scz5202/run/mmsegmentation/data/Glomeruli-dataset/masks'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(320, 240), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(320, 240), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader =dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type = dataset_type,
        data_root = data_root,
        data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline = train_pipeline,
        ann_file = '/HOME/scz5202/run/mmsegmentation/data/Glomeruli-dataset/splits/train.txt')
)

val_dataloader =dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type = dataset_type,
        data_root = data_root,
        data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline = test_pipeline,
        ann_file = '/HOME/scz5202/run/mmsegmentation/data/Glomeruli-dataset/splits/val.txt')
)

test_dataloader =dict(
    batch_size=1,
    num_workers=4,
     dataset=dict(
        type = dataset_type,
        data_root = data_root,
        data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline = test_pipeline,
        ann_file = '/HOME/scz5202/run/mmsegmentation/data/Glomeruli-dataset/splits/val.txt')
)

load_from = '/HOME/scz5202/run/mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=1600, val_interval=400)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=400),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=0)