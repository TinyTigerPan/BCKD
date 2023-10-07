# Bridging Cross-task Protocol Inconsistency for Distillation in Dense Object Detection

## Introduction

This repository is the official implementation of ICCV2023: Bridging Cross-task Protocol Inconsistency for Distillation in Dense Object Detection.
* [arxiv](https://arxiv.org/abs/2308.14286)
* [ICCV 2023 open access](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Bridging_Cross-task_Protocol_Inconsistency_for_Distillation_in_Dense_Object_Detection_ICCV_2023_paper.html)

## News and ToDo List

- [x] [2023-09-01] Release checkpoints and logs
- [x] [2023-08-28] Release paper and code
- [x] [2023-07-14] Accepted by ICCV2023 🎉 
- [x] [2023-04-07] Publish initial code


## Install
This repo is build on mmdetection 2.28.2

Please refer [this link](https://github.com/open-mmlab/mmdetection/blob/2.x/docs/en/get_started.md/#Installation) to build the environment (mmcv...).

Then execute the following command to install.
```
git clone https://github.com/TinyTigerPan/BCKD.git
cd BCKD
pip install -v -e .
```

## Train

For single GPU
```bash
python tools/train.py configs/bckd/bckd_r50_gflv1_r101_fpn_coco_1x.py
```

For multi GPU
```bash
bash tools/dist_train.sh configs/bckd/bckd_r50_gflv1_r101_fpn_coco_1x.py 8
```

## Eval

For single GPU
```bash
python tools/test.py config_file ckpt_file --eval bbox
```

For multi GPU
```bash
bash tools/dist_test.sh configs_file ckpt_file 8
```

## Result

| Teacher     | Student    | Schedule | mAP   | download                                                                                                                                                                                     |
|-------------|------------|----------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|             | GFocal-R50 | 1x       | 40.1  |                                                                                                                                                                                              |
| GFocal-R101 | GFocal-R50 | 1x       | 43.2  | [log](https://drive.google.com/file/d/1bl7qbEYsrdXvzm0Ya8wMl7INBJXzaA9L/view?usp=drive_link) \| [ckpt](https://drive.google.com/file/d/1AeGpY4QbQ_PlanuauEogZWAbMRH59t-k/view?usp=drive_link)                                                                                      |
|             | GFocal-R34 | 1x       | 38.9  |                                                                                                                                                                                              |
| GFocal-R101 | GFocal-R34 | 1x       | 42.0  | [log](https://drive.google.com/file/d/1paU3nDKFNbZcBWXS1ralrDf8dWIIh2vF/view?usp=drive_link) \| [ckpt](https://drive.google.com/file/d/1hJo15YP71xgZdw262Urum89R6324u3dG/view?usp=drive_link) |
|             | GFocal-R18 | 1x       | 35.8  |                                                                                                                                                                                              |
| GFocal-R101 | GFocal-R18 | 1x       | 38.6  | [log](https://drive.google.com/file/d/1ijTPJX3hkjddl_GMLBWGnTe272NOTSrz/view?usp=drive_link) \| [ckpt](https://drive.google.com/file/d/1Oy6cSBeFKJx5tooHGRRFOv7QwPdScO6S/view?usp=drive_link) |


## Cite
```
@InProceedings{Yang_2023_ICCV,
    author    = {Yang, Longrong and Zhou, Xianpan and Li, Xuewei and Qiao, Liang and Li, Zheyang and Yang, Ziwei and Wang, Gaoang and Li, Xi},
    title     = {Bridging Cross-task Protocol Inconsistency for Distillation in Dense Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {17175-17184}
}
```
