_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# import mmselfsup.models to trigger register_module in mmselfsup
checkpoint_file = 'work_dirs/yolo/checkpoint/checkpoint.pth'
deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [512, 1024, 2048]

model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='YOLOv8CSPDarknet',
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=channels, # Note: The 3 channels of ResNet-50 output are [512, 1024, 2048], which do not match the original yolov5-s neck and need to be changed.
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels, # input channels of head need to be changed accordingly
            widen_factor=widen_factor))
)
