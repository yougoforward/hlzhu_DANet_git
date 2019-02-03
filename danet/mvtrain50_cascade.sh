#!/usr/bin/env bash
python mvdanet_train.py --mviews 25 49 96 --dataset cityscapes --model  cascade_mvdanet --backbone resnet50 --checkname cascade_mvdanet50_254996  --base-size 1024 --crop-size 768 --epochs 240 --batch-size 8 --lr 0.006 --workers 2 --multi-grid --multi-dilation 4 8 16
