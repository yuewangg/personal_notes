_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='BN', requires_grad=True) # 只使用GPU时，BN取代SyncBN
crop_size = (256, 256)  # 训练时的裁剪大小

# 分割前景的种类数目。 通常情况下，cityscapes 为19，VOC为21，ADE20k 为150。
model = dict(
    data_preprocessor=dict(size=crop_size),
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes = 2 ,norm_cfg = norm_cfg
    ),
    auxiliary_head=dict(num_classes = 2,norm_cfg = norm_cfg
    )
)

# 修改数据集的 type 和 root
dataset_type = 'StanfordBackgroundDataset'

# 数据集图片和标注路径
data_root = '/HOME/scz5202/run/mmsegmentation/data/Glomeruli-dataset'
img_dir = '/HOME/scz5202/run/mmsegmentation/data/Glomeruli-dataset/images'
ann_dir = '/HOME/scz5202/run/mmsegmentation/data/Glomeruli-dataset/masks'

# 训练集的预处理方法
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(320, 240), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
# 测试集的预处理方法
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(320, 240), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
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

# 载入预训练模型权重
load_from = '/HOME/scz5202/run/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

optimizer = dict(type='SGD', lr=0.0015, momentum=0.9, weight_decay=0.0005)
# 训练迭代次数 ,对于 EpochBasedRunner 使用 `max_epochs` 
# 将使用的 runner 的类别 (例如 IterBasedRunner 或 EpochBasedRunner) 对于 EpochBasedRunner 使用 `max_epochs` 。
train_cfg = dict(type='IterBasedTrainLoop', max_iters=2400, val_interval=400)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=800),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=0)
