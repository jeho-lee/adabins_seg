#!/bin/bash

source ~/.bashrc
conda activate omnicv

####################
GPU=6
####################

# NEW GPUs in AI Datacenter
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# torchrun --nproc_per_node=$GPU train_seg_unet_transformer_adabins.py --distributed --name unet_transformer_adabins_b5_ep400_8gpu --backbone tf_efficientnet_b5_ap --epochs 400 --batch_size 32 --lr 0.001 --wd 0.01

# 95252 polylr
# torchrun --nproc_per_node=$GPU train_seg_unet_transformer_adabins.py --distributed --name unet_transformer_adabins_v2_b5_ep200_polylr --backbone tf_efficientnet_b5_ap --epochs 200 --batch_size 16 --lr 0.003 --wd 0.01

# 113567 class-specific tokens
# torchrun --nproc_per_node=$GPU train_seg_unet_transformer_adabins.py --distributed --name unet_transformer_adabins_v3 --backbone tf_efficientnet_b5_ap --epochs 200 --batch_size 16 --lr 0.003 --wd 0.01
