import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import shutil
import logging
import time
import timeit
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from tqdm import tqdm

# Distributed training
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

import model_io
import models

from utils.utils_seg import visualize_segmentation
from utils.utils_seg import get_confusion_matrix
from utils.utils_seg import AverageMeter

from dataloader import DepthDataLoader
import matplotlib

from datasets import Cityscapes
from loss import CrossEntropy

def is_distributed():
    return dist.is_initialized()

def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_sampler(dataset):
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None

def is_rank_zero(args):
    return args.rank == 0

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp.clone()
        dist.all_reduce(reduced_inp, op=dist.ReduceOp.SUM)
    return reduced_inp / world_size

######################################
seed = 304

# Cityscapes
data_root = '/home/dlwpgh1994/ARIA/datasets/Cityscapes/'
train_list = 'train.lst'
val_list = 'val.lst'
######################################

# Define the argument parser
parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                 conflict_handler='resolve')
parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
parser.add_argument('--n-bins', '--n_bins', default=80, type=int, help='number of bins/buckets to divide depth range into')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='max learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float, help='weight decay')
parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")
parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float, help="final div factor for lr")
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--validate-every', '--validate_every', default=200, type=int, help='validation period')
parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')

parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths", choices=['linear', 'softmax', 'sigmoid'])
parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true", help="Use same LR for all param groups")
parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")

parser.add_argument("--root", default="./experiments", type=str, help="Root folder to save data in")
parser.add_argument("--name", default="UnetAdaptiveBins")
parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")
parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")

# Depth Estimation
parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")
parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str, help="path to dataset")
parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str, help="path to dataset")
parser.add_argument('--filenames_file', default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt", type=str, help='path to the filenames text file')
parser.add_argument('--input_height', type=int, help='input height', default=416)
parser.add_argument('--input_width', type=int, help='input width', default=544)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
parser.add_argument('--do_random_rotate', default=True, help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI', action='store_true')
parser.add_argument('--data_path_eval', default="../dataset/nyu/official_splits/test/", type=str, help='path to the data for online evaluation')
parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/", type=str, help='path to the groundtruth data for online evaluation')
parser.add_argument('--filenames_file_eval', default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt", type=str, help='path to the filenames text file for online evaluation')
parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', help='if set, crops according to Garg ECCV16', action='store_true')

# Segmentation
parser.add_argument('--n_semantic_classes', help='Number of semantic classes', default=19, type=int)
parser.add_argument('--img_width', help='Width of input image', default=1024, type=int) 
parser.add_argument('--img_height', help='Height of input image', default=512, type=int)
parser.add_argument('--base_size', help='Base size of input image', default=2048, type=int) # Cityscapes original size: 2048x1024
parser.add_argument('--ignore_label', help='Label to ignore', default=255, type=int)
parser.add_argument('--backbone', help='Backbone network', default='efficientnet_b5', type=str)

args = parser.parse_args()

def main(args):
    if 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.batch_size = int(args.batch_size / args.world_size)
        args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = args.gpu if args.gpu is not None else 0
        torch.cuda.set_device(args.gpu)

    crop_size = (args.img_height, args.img_width)
    train_dataset = Cityscapes(root=data_root,
                            list_path=train_list,
                            num_samples=None,
                            num_classes=args.n_semantic_classes,
                            multi_scale=True,
                            flip=True,
                            ignore_label=args.ignore_label,
                            base_size=args.base_size,
                            crop_size=crop_size,
                            downsample_rate=1,
                            scale_factor=16)

    train_sampler = get_sampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler)
    
    val_dataset = Cityscapes(root=data_root,
                            list_path=val_list,
                            num_samples=None,
                            num_classes=args.n_semantic_classes,
                            multi_scale=False,
                            flip=False,
                            ignore_label=args.ignore_label,
                            base_size=args.base_size,
                            crop_size=crop_size,
                            downsample_rate=1)

    val_sampler = get_sampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler)

    criterion_entropy = CrossEntropy(ignore_label=args.ignore_label,
                            weight=train_dataset.class_weights)

    print("Load model")
    
    # model = models.UnetAdaptiveSegmentation.build(n_classes=args.n_semantic_classes)
    model = models.UNetV1(n_classes=args.n_semantic_classes)
    
    model = model.cuda(args.gpu)

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
    else:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1

    should_write = ((not args.distributed) or args.rank == 0)
    
    device = torch.device('cuda')

    model.train()
    
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if hasattr(model, 'module') else model
        params = [{"params": m.get_1x_lr_params(), "lr": args.lr / 10},
                    {"params": m.get_10x_lr_params(), "lr": args.lr}]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)

    iters = len(train_loader)
    step = args.epoch * iters
    best_mIoU = 0

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
    #                                         args.lr, 
    #                                         epochs=args.epochs, 
    #                                         steps_per_epoch=len(train_loader),
    #                                         cycle_momentum=True,
    #                                         base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
    #                                         div_factor=args.div_factor, final_div_factor=args.final_div_factor)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    
    print("Start training")
    for cur_epoch in range(args.epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(cur_epoch)
        loop = tqdm(enumerate(train_loader), desc=f"Epoch: {cur_epoch + 1}/{args.epochs}. Loop: Train",
                                    total=len(train_loader)) if is_rank_zero(args) else enumerate(train_loader)
        for i, batch in loop:
            optimizer.zero_grad()
            images, labels, _, _ = batch
            images = images.cuda()
            labels = labels.long().cuda()
            outputs = model(images)
            loss = criterion_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                reduced_loss = reduce_tensor(loss) if args.distributed else loss
                if is_rank_zero(args):
                    msg = 'Epoch: [{}/{}] Step:[{}], lr: {}, Loss: {:.6f}'.format(cur_epoch, args.epochs, step, [x['lr'] for x in optimizer.param_groups], reduced_loss.item())
                    print(msg)
            step += 1
            scheduler.step()
        
        # Validate at the end of each epoch
        model.eval()
        valid_loss, mean_IoU, IoU_array = validate(args, val_loader, model, criterion_entropy, cur_epoch, args.epochs, device)

        if should_write:
            model_io.save_checkpoint(model, optimizer, cur_epoch, f"{args.name}_latest.pt", root=os.path.join(args.root, args.name, "checkpoints"))
            if mean_IoU > best_mIoU:
                print(f"Best model found: {mean_IoU}, previous best: {best_mIoU}, saving model...")
                model_io.save_checkpoint(model, optimizer, cur_epoch, f"{args.name}_best.pt", root=os.path.join(args.root, args.name, "checkpoints"))
                best_mIoU = mean_IoU

        model.train()

def validate(args, test_loader, model, criterion, epoch, epochs, device='cuda'):
    """
    Validation function to compute loss, confusion matrix, and IoU metrics.
    Args:
        args: Command-line arguments including hyperparameters and paths.
        test_loader: DataLoader for validation data.
        model: Model to be evaluated.
        criterion: Loss function (e.g., CrossEntropy).
        epoch: Current epoch for logging.
        epochs: Total number of epochs.
        device: Device to use for computation ('cuda' or 'cpu').
    """
    model.eval()
    ave_loss = AverageMeter()
    nums = args.n_semantic_classes
    confusion_matrix = np.zeros((nums, nums))

    save_dir = os.path.join(args.root, args.name, 'segmentation_results')
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images, labels, _, _ = batch
            size = labels.size()
            images = images.to(device)
            labels = labels.long().to(device)

            # Forward pass
            outputs = model(images)

            # Resize the output to match the target size
            if isinstance(outputs, (list, tuple)):
                outputs = [F.interpolate(output, size=size[-2:], mode='bilinear', align_corners=True) for output in outputs]
            else:
                outputs = F.interpolate(outputs, size=size[-2:], mode='bilinear', align_corners=True)

            # Calculate loss
            loss = criterion(outputs, labels)
            reduced_loss = reduce_tensor(loss) if args.distributed else loss
            ave_loss.update(reduced_loss.item())

            # Compute confusion matrix
            if isinstance(outputs, (list, tuple)):
                for output in outputs:
                    confusion_matrix += get_confusion_matrix(
                        labels, output, size, args.n_semantic_classes, args.ignore_label
                    )
            else:
                confusion_matrix += get_confusion_matrix(
                    labels, outputs, size, args.n_semantic_classes, args.ignore_label
                )

            if idx % 10 == 0 and is_rank_zero(args):
                print(f"Validation: Iteration [{idx}/{len(test_loader)}], Loss: {reduced_loss.item()}")
            
            # Visualize segmentation results for the first batch
            if idx == 0 and is_rank_zero(args):
                predictions = torch.argmax(outputs, dim=1)  # Shape: (N, H, W)
                visualize_segmentation(images, labels, predictions, save_dir, epoch, ignore_label=args.ignore_label, num_vis=10)

    # Reduce confusion matrix across GPUs
    if args.distributed:
        confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
        dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
        confusion_matrix = confusion_matrix.cpu().numpy()

    # Compute mean IoU
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = tp / np.maximum(1.0, pos + res - tp)
    mean_IoU = IoU_array.mean()

    if is_rank_zero(args):
        print(f"Epoch: {epoch}/{epochs}, Validation Loss: {ave_loss.average():.6f}, Mean IoU: {mean_IoU:.6f}")
        logging.info(f"Epoch: {epoch}/{epochs}, IoU per class: {IoU_array}, Mean IoU: {mean_IoU:.6f}")

    return ave_loss.average(), mean_IoU, IoU_array

if __name__ == '__main__':
    if seed > 0:
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        
    args.num_threads = args.workers
    args.mode = 'train'

    # Check if torchrun environment variables are set
    if 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    main(args)
