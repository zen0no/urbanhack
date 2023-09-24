_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# import mmselfsup.models to trigger register_module in mmselfsup
checkpoint_file = 'work_dirs/yolo/checkpoint/checkpoint.pth'
deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [256, 512, 1024]

model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='YOLOv8CSPDarknet',
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
)
