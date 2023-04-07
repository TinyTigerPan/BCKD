# Bridging Cross-task Protocol Inconsistency for Distillation in Dense Object Detection

## Install
This repo is build on mmdetection 2.28.2

Please refer [this link](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation) to build the environment.

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
TODO
