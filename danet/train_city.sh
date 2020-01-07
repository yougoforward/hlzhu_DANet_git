#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_city.py --dataset cityscapes --model  new_psp3 --backbone resnet101 --checkname new_psp3  --base-size 1024 --crop-size 768 --epochs 240 --batch-size 16 --lr 0.015 --workers 2 --multi-grid --multi-dilation 4 8 16

CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset cityscapes --model new_psp3 --resume-dir cityscapes/model --base-size 2048 --crop-size 768 --workers 1 --backbone resnet101 --multi-grid --multi-dilation 4 8 16 --eval

CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset cityscapes --model new_psp3 --resume-dir cityscapes/model --base-size 2048 --crop-size 1024 --workers 1 --backbone resnet101 --multi-grid --multi-dilation 4 8 16 --eval --multi-scales
