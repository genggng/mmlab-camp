_base_ = [
    '../_base_/models/resnet50_cifar_mixup.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]
dataset_type = 'CIFAR10'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type, data_prefix='/data/public/cifar'),
    val=dict(
        type=dataset_type,
        data_prefix='/data/public/cifar',
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='/data/public/cifar',
        test_mode=True))