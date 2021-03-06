#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6,7 python train.py --dataset cityscapes --model  msdanet --backbone resnet101 --checkname msdanet101  --base-size 1024 --crop-size 768 --epochs 240 --batch-size 8 --lr 0.003 --workers 2 --multi-grid --multi-dilation 4 8 16
