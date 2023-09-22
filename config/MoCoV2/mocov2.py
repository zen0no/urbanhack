from mmengine.config import read_base

from mmpretrain.models.selfsup.moco import MoCo
from mmpretrain.models.backbones import CSPDarkNet
from mmpretrain.models.necks import MoCoV2Neck
from mmpretrain.models.heads import ContrastiveHead

from mmpretrain.models.losses.cross_entropy_loss import CrossEntropyLoss

with read_base():
    from .._base_.train_dataset import *
    from .._base_.runtime import *
    from .._base_.scheduler import *

model = dict(
    type=MoCo,
    queue_len=65536,
    feat_dim=128,
    momentum=0.001,
    backbone=dict(
        type=CSPDarkNet,
        depth=53,
        norm_cfg=dict(type='BN'),
        ),
    neck=dict(
        type=MoCoV2Neck,
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type=ContrastiveHead,
        loss=dict(type=CrossEntropyLoss),
        temperature=0.2))


oks = dict(
        checkpoint=dict(interval=1, max_keep_ckpts=3, type='CheckpointHook'),
        logger=dict(interval=100, type='LoggerHook'),
        param_scheduler=dict(type='ParamSchedulerHook'),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        timer=dict(type='IterTimerHook'),
      )

auto_scale_lr = dict(base_batch_size=256)

