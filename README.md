# Bridging Cross-task Protocol Inconsistency for Distillation in Dense Object Detection

## News and ToDo List

- [ ] Release checkpoints and logs
- [x] Release paper and code
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
TODO
