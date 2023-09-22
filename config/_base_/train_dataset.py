from mmpretrain import configs
from mmpretrain.datasets.custom import CustomDataset

from mmpretrain.datasets.transforms.processing import (RandomResizedCrop,
                                                       ColorJitter)
from mmpretrain.datasets.transforms.auto_augment import GaussianBlur


dataset_type=CustomDataset
data_root = "data"

dataset_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[128, 128, 128],
    std=[32, 32, 32],
    to_rgb=True)


# The difference between mocov2 and mocov1 is the transforms in the pipeline
view_pipeline = [
    dict(
        type=RandomResizedCrop,
        scale=224,
        crop_ratio_range=(0.2, 1.),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type=ColorJitter,
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type=GaussianBlur,
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.5),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='selfsup',
        pipeline=train_pipeline,
        with_label=False))

