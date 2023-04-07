_base_ = ['../ld/ld_r18_gflv1_r101_fpn_coco_1x.py']
model = dict(
   backbone=dict(
        type='MobileNetV2',
        _delete_=True,
        out_indices=(1, 2, 4, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
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

find_unused_parameters=True