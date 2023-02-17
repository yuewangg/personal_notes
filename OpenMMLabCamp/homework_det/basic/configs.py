_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='./resnet50-0676ba61.pth')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1),
        mask_head=dict(
            num_classes=1)))




data_root = './data/coco'
CLASSES = ('balloon', )

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(240, 180), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(240, 180),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        ann_file='./data/coco/train.json',
        img_prefix='./data/coco/train/',
        pipeline=train_pipeline),
    val=dict(
        ann_file='./data/coco/val.json',
        img_prefix='./data/coco/val/',
        pipeline=test_pipeline),
    test=dict(
        ann_file='./data/coco/val.json',
        img_prefix='./data/coco/val/',
        pipeline=test_pipeline))

load_from = './mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

auto_scale_lr = dict(enable=False, base_batch_size=1)
