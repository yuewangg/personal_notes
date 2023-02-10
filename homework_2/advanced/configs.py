_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)),
             backbone=dict(        
                init_cfg=dict(type='Pretrained', checkpoint='./resnet50-0676ba61.pth')))


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            ann_file=['data/VOCdevkit/VOC2007/ImageSets/Layout/train.txt'],
            img_prefix=['data/VOCdevkit/VOC2007/'])),
    val=dict(
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Layout/val.txt',
        img_prefix='data/VOCdevkit/VOC2007/'),
    test=dict(
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Layout/val.txt',
        img_prefix='data/VOCdevkit/VOC2007/'))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

load_from = 'faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth'
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12


