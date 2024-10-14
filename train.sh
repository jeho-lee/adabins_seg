#!/bin/bash

source ~/.bashrc
conda activate omnicv

####################
GPU=6
####################

# NEW GPUs in AI Datacenter
# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"

# 82131 UNet v1, EfficientNet-B5, lr 0.01, wd 1e-3, epochs 200
# torchrun --nproc_per_node=$GPU train_seg_unet_v1.py --distributed --name unet_v1_b5_20240914_ep200 --backbone efficientnet_b5 --epochs 200 --batch_size 16 --lr 0.01 --wd 1e-3

# 82132 UNet v1, EfficientNet-B3 -> 16 batch size per GPU, lr 0.01, wd 1e-3, epochs 200
# torchrun --nproc_per_node=$GPU train_seg_unet_v1.py --distributed --name unet_v1_b3_20240914_ep200 --backbone efficientnet_b3 --img_width 600 --img_height 300 --epochs 200 --batch_size 64 --lr 0.01 --wd 1e-3

# 86168 UNet v1, EfficientNet-B5, lr 0.01, wd 1e-4, epochs 400
# torchrun --nproc_per_node=$GPU train_seg_unet_v1.py --distributed --name unet_v1_b5_20240918_ep400 --backbone efficientnet_b5 --epochs 400 --batch_size 32 --lr 0.01 --wd 1e-4

# 86169 UNet v1, EfficientNet-B3 -> 16 batch size per GPU, lr 0.01, wd 1e-4, epochs 400
# torchrun --nproc_per_node=$GPU train_seg_unet_v1.py --distributed --name unet_v1_b3_20240918_ep400 --backbone efficientnet_b3 --img_width 600 --img_height 300 --epochs 400 --batch_size 64 --lr 0.01 --wd 1e-4

# UNet Transformer, EfficientNet-B5, lr 0.01, wd 1e-4, epochs 400
# 88970 결과 안좋음
# torchrun --nproc_per_node=$GPU train_seg_unet_transformer.py --distributed --name unet_transformer_b5_20240919_ep400 --backbone efficientnet_b5 --epochs 400 --batch_size 16 --lr 0.01 --wd 1e-4

# UNet Transformer V2, EfficientNet-B5, lr 0.001, wd 1e-3, epochs 400
torchrun --nproc_per_node=$GPU train_seg_unet_transformer.py --distributed --name unet_transformer_v2_b5_20240921_ep400 --backbone efficientnet_b5 --epochs 400 --batch_size 24 --lr 0.001 --wd 0.01
