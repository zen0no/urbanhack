from mmengine.runner.loops import EpochBasedTrainLoop
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper


optim_wrapper = dict(
        type=OptimWrapper,
        optimizer=dict(type='Adam', lr=0.01))

# learning rate scheduler
param_scheduler = [
        dict(type=CosineAnnealingLR, T_max=200, by_epoch=True, begin=0,
             end=200)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=200)
