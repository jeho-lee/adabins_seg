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

# Import PolyLR scheduler base class
from torch.optim.lr_scheduler import _LRScheduler

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
parser.add_argument("--workers", default=8, type=int, help="Number of workers for data loading")

# Segmentation
parser.add_argument('--n_semantic_classes', help='Number of semantic classes', default=19, type=int)
parser.add_argument('--img_width', help='Width of input image', default=1024, type=int) 
parser.add_argument('--img_height', help='Height of input image', default=512, type=int)
parser.add_argument('--base_size', help='Base size of input image', default=2048, type=int) # Cityscapes original size: 2048x1024
parser.add_argument('--ignore_label', help='Label to ignore', default=255, type=int)
parser.add_argument('--backbone', help='Backbone network', default='efficientnet_b5', type=str)

args = parser.parse_args()

# Define PolyLR Scheduler
class PolyLR(_LRScheduler):
    """
    Polynomial Learning Rate Decay
    """
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
        self.max_iters = max_iters
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        factor = (1 - min(self.last_epoch, self.max_iters) / self.max_iters) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]

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

    # Set random seeds for reproducibility
    if seed > 0:
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True

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
    
    from models import UNet_Transformer_Adabins
    
    # Initialize the model
    model = UNet_Transformer_Adabins.build(backbone_name=args.backbone, n_classes=args.n_semantic_classes)
    
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
    
    print("Using different LR for different parameter groups")
    m = model.module if hasattr(model, 'module') else model
    params = [
        {"params": m.encoder.parameters(), "lr": args.lr / 10},  # Backbone
        {"params": m.decoder.parameters(), "lr": args.lr},     # Decoder
        {"params": m.transformer_module.parameters(), "lr": args.lr},  # Transformer Module
        {"params": m.segmentation_head.parameters(), "lr": args.lr},  # Segmentation Head
    ]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)

    iters = len(train_loader)
    step = args.epoch * iters
    best_mIoU = 0

    # Calculate total number of iterations for PolyLR
    max_iters = args.epochs * iters

    # Initialize PolyLR scheduler
    scheduler = PolyLR(optimizer, max_iters=max_iters, power=0.9)
    
    # Check learning rate
    print("Initial LRs:", [group['lr'] for group in optimizer.param_groups])
    
    print("Start training")
    for cur_epoch in range(args.epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(cur_epoch)
        
        # Training loop
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            images, labels, _, _ = batch
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.long().cuda(args.gpu, non_blocking=True)
            outputs = model(images)
            loss = criterion_entropy(outputs, labels)
            loss.backward()

            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            if step % 10 == 0:
                reduced_loss = reduce_tensor(loss) if args.distributed else loss
                if is_rank_zero(args):
                    current_lr = scheduler.get_last_lr()[0]
                    msg = 'Epoch: [{}/{}] Step:[{}], lr: {:.6f}, Loss: {:.6f}'.format(
                        cur_epoch + 1, args.epochs, step, current_lr, reduced_loss.item()
                    )
                    print(msg)
            step += 1
        
        # Validate at the end of each epoch
        model.eval()
        valid_loss, mean_IoU, IoU_array = validate(
            args, val_loader, model, criterion_entropy, cur_epoch + 1, args.epochs, device
        )

        if should_write:
            if mean_IoU > best_mIoU:
                print(f"Best model found: mIoU={mean_IoU:.4f}, previous best: {best_mIoU:.4f}. Saving model...")
                # Save the best model
                checkpoint_path = os.path.join(args.root, args.name, "checkpoints")
                os.makedirs(checkpoint_path, exist_ok=True)
                best_model_filename = f"{args.name}_best_{mean_IoU:.2f}.pt"
                model_io.save_checkpoint(
                    model, optimizer, cur_epoch + 1, best_model_filename,
                    root=checkpoint_path
                )
                
                # Remove the previous best model if it exists
                if best_mIoU > 0:
                    previous_best_model = f"{args.name}_best_{best_mIoU:.2f}.pt"
                    previous_best_path = os.path.join(checkpoint_path, previous_best_model)
                    if os.path.exists(previous_best_path):
                        os.remove(previous_best_path)
                        print(f"Removed previous best model: {previous_best_model}")
                
                best_mIoU = mean_IoU

                # Visualize segmentation results for the best model
                visualize_segmentation_for_best_model(args, val_loader, model, cur_epoch + 1)
        
        model.train()

def visualize_segmentation_for_best_model(args, val_loader, model, epoch):
    """
    Visualize segmentation results for the best model.
    Processes only the first batch of the validation set.
    """
    save_dir = os.path.join(args.root, args.name, 'segmentation_results')
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            images, labels, _, _ = batch
            images = images.to(args.gpu, non_blocking=True)
            labels = labels.long().to(args.gpu, non_blocking=True)

            # Forward pass
            outputs = model(images)

            # Resize the output to match the target size
            if isinstance(outputs, (list, tuple)):
                outputs = [
                    F.interpolate(output, size=labels.size()[-2:], mode='bilinear', align_corners=True)
                    for output in outputs
                ]
            else:
                outputs = F.interpolate(
                    outputs, size=labels.size()[-2:], mode='bilinear', align_corners=True
                )

            predictions = torch.argmax(outputs, dim=1)  # Shape: (N, H, W)
            visualize_segmentation(
                images, labels, predictions, save_dir, epoch,
                ignore_label=args.ignore_label, num_vis=10
            )
            break  # Visualize only the first batch

def validate(args, test_loader, model, criterion, epoch, epochs, device='cuda'):
    """
    Validate the model on the validation set.
    Computes the average loss and mean IoU.
    """
    ave_loss = AverageMeter()
    nums = args.n_semantic_classes
    confusion_matrix = np.zeros((nums, nums))

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images, labels, _, _ = batch
            size = labels.size()
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)

            # Resize the output to match the target size
            if isinstance(outputs, (list, tuple)):
                outputs = [
                    F.interpolate(output, size=size[-2:], mode='bilinear', align_corners=True)
                    for output in outputs
                ]
            else:
                outputs = F.interpolate(
                    outputs, size=size[-2:], mode='bilinear', align_corners=True
                )

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
                print(f"Validation: Iteration [{idx}/{len(test_loader)}], Loss: {reduced_loss.item():.6f}")
    
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
    # Parse arguments and initialize
    args.num_threads = args.workers
    args.mode = 'train'
    main(args)
