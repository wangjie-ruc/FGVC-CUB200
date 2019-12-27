# FGVC-CUB200

## training

具体设置可以按需修改

所有模型均未能复现原文结果，正在努力💪

baseline
```bash
python train/resnnet50.py configs/resnet50.py --gpu 0 2
```

STN
```bash
python train/stn.py configs/stn.py --data_root /data/CUB_200_2011
```

S3N
```bash
python train/s3n.py configs/s3n.py --train_transforms configs/transforms/train2.json
```

WS-DAN
```bash
python train/wd_dan.py configs/ws_dan.py --work_dir out/WSDAN
```


## models

1. STN
2. WS_DAN
3. S3N