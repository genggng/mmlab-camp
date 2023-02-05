_base_ = ['./_base_/models/resnet18.py','./_base_/datasets/imagenet_bs32.py','./_base_/default_runtime.py']
model = dict(
    head=dict(
    num_classes=5,
    topk = (1, )
    ))

data = dict(
# 根据实验环境调整每个 batch_size 和 workers 数量
    samples_per_gpu = 32,
    workers_per_gpu=2,
    # 指定训练集路径
    train = dict(
        data_prefix = 'data',
        ann_file = 'data/flower_imagenet/train.txt',
        classes = 'data/flower_imagenet/classes.txt'
    ),
    # 指定验证集路径
    val = dict(
        data_prefix = 'data',
        ann_file = 'data/flower_imagenet/val.txt',
        classes = 'data/flower_imagenet/classes.txt'
    ),
)

evaluation = dict(metric_options={'topk': (1, )})

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# 学习率策略
lr_config = dict(policy='step',step=[8,10])
runner = dict(type='EpochBasedRunner', max_epochs=12)
load_from ="/data/home/scv9611/run/mmclassification/checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth"

