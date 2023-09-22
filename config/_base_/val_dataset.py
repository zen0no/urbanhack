from mmpretrain import configs
from mmpretrain import CustomDataset

dataset_type = CustomDataset


val_pipeline = [
    dict(type='LoadImageFromFile')
]

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    sampler=dict(type="DefaultSampler"),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=val_pipeline))
