_base_ = ["./mask_rcnn_r50_fpn_1x_coco.py"]
model = dict(
    backbone=dict(init_cfg=None),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    )
)

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='/data/home/scv9611/run/mmdetection/data/balloon/train/coco.json',
        img_prefix='/data/home/scv9611/run/mmdetection/data/balloon/train/',
        classes=("balloon",)),
    val=dict(
        type='CocoDataset',
        ann_file='/data/home/scv9611/run/mmdetection/data/balloon/val/coco.json',
        img_prefix='/data/home/scv9611/run/mmdetection/data/balloon/val/',
        classes=("balloon",)),
    test=dict(
        type='CocoDataset',
        ann_file='/data/home/scv9611/run/mmdetection/data/balloon/val/coco.json',
        img_prefix='/data/home/scv9611/run/mmdetection/data/balloon/val/',
        classes=("balloon",))
)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001) #原始学习率0.02
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
load_from = "/data/home/scv9611/run/mmdetection/checkpoint/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
