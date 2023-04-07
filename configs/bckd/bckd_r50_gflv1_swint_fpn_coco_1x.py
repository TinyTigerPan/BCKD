_base_ = ['../ld/ld_r18_gflv1_r101_fpn_coco_1x.py']
teacher_ckpt = '/home/hpc/users/zhouxianpan/mmdetection/work_dirs/gfl_swint_fpn_mstrain_2x_coco/epoch_24.pth'  # noqa
model = dict(
    teacher_config='/home/hpc/users/zhouxianpan/mmdetection/configs/gfl/gfl_swint_fpn_mstrain_2x_coco.py',
    teacher_ckpt=teacher_ckpt,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='LDHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),

        loss_ld=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=0., T=10),
        loss_ld_vlr=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=0., T=10),
        loss_kd=dict(type='NovelKDLoss', loss_weight=1.0),
        loss_kd_vlr=dict(type='NovelKDLoss', loss_weight=0.),
        loss_score_kd=dict(type='GIoULoss', loss_weight=4.0),
        loss_score_kd_vlr=dict(type='GIoULoss', loss_weight=0.0),

        loss_im=dict(type='IMLoss', loss_weight=0.0),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        imitation_method='finegrained'  # gibox, finegrain, decouple, fitnet
    ))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=12)
