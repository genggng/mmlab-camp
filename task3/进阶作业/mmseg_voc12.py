_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/pascal_voc12.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]


crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21))
data_root = '/data/public/PascalVOC/2012/VOC2012/'
train_dataloader = dict(
    batch_size=20,
    num_workers=4,
    dataset=dict(data_root=data_root))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(data_root=data_root))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(data_root=data_root))
