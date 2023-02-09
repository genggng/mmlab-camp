_base_ = ["./retinanet_r50_fpn_1x_voc0712.py"]

model = dict(backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='/data/home/scv9611/run/mmdetection/checkpoint/resnet50-0676ba61.pth')))
data_root = '/data/public/PascalVOC/2012/VOC2012/'
data = dict(
    samples_per_gpu=36,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='VOCDataset',
            ann_file=data_root+'ImageSets/Main/train.txt',
            img_prefix=data_root,
)),
    val=dict(
        type='VOCDataset',
        ann_file=data_root+'ImageSets/Main/val.txt',
        img_prefix=data_root,
        ),
    test=dict(
        type='VOCDataset',
        ann_file=data_root+'ImageSets/Main/val.txt',
        img_prefix=data_root,
        ))

