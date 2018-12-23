#!/usr/bin/env bash
python mvdanet_train.py --mviews 49 96 --dataset cityscapes --model  mvdanet --backbone resnet50 --checkname mvdanet50_4996  --base-size 1024 --crop-size 768 --epochs 240 --batch-size 8 --lr 0.003 --workers 2 --multi-grid --multi-dilation 4 8 16
