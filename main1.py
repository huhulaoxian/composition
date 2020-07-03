# This code is built from the PyTorch examples repository: https://github.com/pytorch/examples/.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

import argparse
import logging
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models_lpf
import torchvision.models as models

from IPython import embed
from mydata import MyDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/home/yaoting/composition/KU_PCPdataset',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--freeze', dest='freeze', action='store_true',
                    help='freeze cnn ')
parser.add_argument('--multi', dest='multi', action='store_true',
                    help='use multi mothed')
parser.add_argument('--distortion', dest='distortion', action='store_true',
                    help='use distortion image')
parser.add_argument('-stn', dest='stn', action='store_true',
                    help='use stn structure')
parser.add_argument('-adam', dest='adam', action='store_true',
                    help='use adam lr s')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-erand', '--evaluate-random', dest='evaluate_random', action='store_true',
                    help='evaluate model on validation set random rotate')
parser.add_argument('--evaluate-save', dest='evaluate_save', action='store_true',
                    help='save validation images off')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# Added functionality from PyTorch codebase
parser.add_argument('--no-data-aug', dest='no_data_aug', action='store_true',
                    help='no shift-based data augmentation')
parser.add_argument('--out-dir', dest='out_dir', default='./', type=str,
                    help='output directory')
parser.add_argument('--log-name', dest='log_name', type=str, default='train.log',
                    help='log')
parser.add_argument('-f', '--filter_size', default=1, type=int,
                    help='anti-aliasing filter size')
parser.add_argument('-es', '--evaluate-shift', dest='evaluate_shift', action='store_true',
                    help='evaluate model on shift-invariance')
parser.add_argument('-ec', '--evaluate-consist', dest='evaluate_consist', action='store_true',
                    help='evaluate model on rotate-invariance')
parser.add_argument('--epochs-shift', default=5, type=int, metavar='N',
                    help='number of total epochs to run for shift-invariance test')
parser.add_argument('-ed', '--evaluate-diagonal', dest='evaluate_diagonal', action='store_true',
                    help='evaluate model on diagonal')
parser.add_argument('-er', '--evaluate-rotate', dest='evaluate_rotate', action='store_true',
                    help='evaluate model on rotate')
parser.add_argument('-ba', '--batch-accum', default=1, type=int,
                    metavar='N',
                    help='number of mini-batches to accumulate gradient over before updating (default: 1)')
parser.add_argument('--embed', dest='embed', action='store_true',
                    help='embed statement before anything is evaluated (for debugging)')
parser.add_argument('--val-debug', dest='val_debug', action='store_true',
                    help='debug by training on val set')
parser.add_argument('--weights', default=None, type=str, metavar='PATH',
                    help='path to pretrained model weights')
parser.add_argument('--save_weights', default=None, type=str, metavar='PATH',
                    help='path to save model weights')

best_acc1 = 0
args = parser.parse_args()

if (not os.path.exists(args.out_dir)):
    os.mkdir(args.out_dir)

logger = logging.getLogger('test')
logger.setLevel(level=logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelno)s: %(message)s')

file_handler = logging.FileHandler(os.path.join(args.out_dir, args.log_name), 'w')
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.pretrained:
        logger.info("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        if (args.arch == 'alexnet'):
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, 9)
        if (args.arch == 'vgg16'):
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, 9)
        if (args.arch == 'vgg16_bn'):
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, 9)
        if (args.arch == 'resnet50'):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 9)
        if (args.arch == 'resnet101'):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 9)
        if (args.arch == 'inception_v3'):
            model = models.__dict__[args.arch](pretrained=True, aux_logits=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 9)
            num_ftrs0 = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs0, 9)
    else:
        logger.info("=> creating model '{}'".format(args.arch))
        import models_lpf.alexnet
        import models_lpf.vgg
        import models_lpf.resnet
        import models_lpf.densenet
        import models_lpf.mobilenet
        import models_lpf.mymodel1

        if (args.arch == 'alexnet_lpf'):
            if args.stn:
                model = models_lpf.alexnet.AlexNet(filter_size=args.filter_size, use_stn=True)
            else:
                model = models_lpf.alexnet.AlexNet(filter_size=args.filter_size)
        elif (args.arch == 'vgg16_bn_lpf'):
            model = models_lpf.vgg.vgg16_bn(filter_size=args.filter_size)
        elif (args.arch == 'vgg16_lpf'):
            model = models_lpf.vgg.vgg16(filter_size=args.filter_size)

        elif (args.arch == 'resnet50_lpf'):
            model = models_lpf.resnet.resnet50(filter_size=args.filter_size)
        elif (args.arch == 'resnet101_lpf'):
            model = models_lpf.resnet.resnet101(filter_size=args.filter_size)

        elif (args.arch == 'mobilenet_v2_lpf'):
            model = models_lpf.mobilenet.mobilenet_v2(filter_size=args.filter_size)
        elif (args.arch == 'rotatenet50'):
            output_num = [4, 2, 1]
            model = models_lpf.mymodel1.rotatenet50(filter_size=args.filter_size, output_num=output_num,
                                                   pool_type='max_pool')
        elif (args.arch == 'alexrotatenet'):
            if args.stn:
                model = models_lpf.mymodel1.alexrotatenet(use_stn=True)
            else:
                model = models_lpf.mymodel1.alexrotatenet(use_stn=False)
        else:
            model = models.__dict__[args.arch]()
            if (args.arch == 'alexnet'):
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs, 9)
            elif (args.arch == 'vgg16'):
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs, 9)
            elif (args.arch == 'vgg16_bn'):
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs, 9)
            elif (args.arch == 'resnet50'):
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 9)
            elif (args.arch == 'resnet101'):
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 9)
            elif (args.arch == 'inception_v3'):
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 9)
                num_ftrs0 = model.AuxLogits.fc.in_features
                model.AuxLogits.fc = nn.Linear(num_ftrs0,9)

    if args.weights is not None:
        logger.info("=> using saved weights [%s]" % args.weights)
        weights = torch.load(args.weights)
        model.load_state_dict(weights['state_dict'])

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)


    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=args.weight_decay)
    else:
        if args.stn:
            optimizer = torch.optim.SGD([
                                         {'params': model.classifier.parameters()},
                                         {'params': model.features.parameters()},
                                         {'params':model.localization.parameters(),'lr':args.lr/1000},
                                         {'params':model.fc_loc.parameters(),'lr':args.lr/1000}
                                         ],
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if ('optimizer' in checkpoint.keys()):  # if no optimizer, then only load weights
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                logger.error('  No optimizer saved')
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.error("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train_list_multi.txt')
    valdir = os.path.join(args.data, 'test_list_multi.txt')

    if (args.arch == 'alexnet'):
        if args.distortion:
            size = (224, 224)
        else:
            size = 224
    elif (args.arch == 'inception_v3'):
        if args.distortion:
            size = (299, 299)
        else:
            size = 299
    else:
        if args.distortion:
            size = (256, 256)
        else:
            size = 256

    if (args.no_data_aug):
        train_dataset = MyDataset(
            traindir,
            transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]))
        train_dataset_multi = MyDataset(
            traindir,
            transforms.Compose([
                transforms.Resize((180, 180)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_dataset = MyDataset(
                traindir,
                transforms.Compose([
                    transforms.Resize(size),
                    transforms.Pad(20, padding_mode='symmetric'),
                    transforms.RandomRotation(8, resample=False, expand=False, center=None, fill=None),
                    transforms.CenterCrop(size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        train_dataset_multi = MyDataset(
            traindir,
            transforms.Compose([
                transforms.Resize((180, 180)),
                transforms.Pad(20, padding_mode='symmetric'),
                transforms.RandomRotation(8, resample=False, expand=False, center=None, fill=None),
                transforms.CenterCrop(180),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    train_loader_multi = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    args.batch_size = 1 if (args.evaluate_diagonal or args.evaluate_save or args.evaluate_rotate or args.evaluate_consist or args.evaluate or args.evaluate_random) else args.batch_size

    if args.evaluate_rotate or args.evaluate_consist or args.evaluate_random:
        val_loader = torch.utils.data.DataLoader(
            MyDataset(valdir, transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        val_loader = torch.utils.data.DataLoader(
            MyDataset(valdir, transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if (args.val_debug):  # debug mode - train on val set for faster epochs
        train_loader = val_loader

    if (args.embed):
        embed()

    if args.save_weights is not None:  # "deparallelize" saved weights
        logger.info("=> saving 'deparallelized' weights [%s]" % args.save_weights)
        # TO-DO: automatically save this during training
        if args.gpu is not None:
            torch.save({'state_dict': model.state_dict()}, args.save_weights)
        else:
            if (args.arch[:7] == 'alexnet' or args.arch[:3] == 'vgg'):
                model.features = model.features.module
                torch.save({'state_dict': model.state_dict()}, args.save_weights)
            else:
                torch.save({'state_dict': model.module.state_dict()}, args.save_weights)
        return

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.evaluate_random:
        validate_random(val_loader, model, criterion, args)
        return

    if (args.evaluate_shift):
        validate_shift(val_loader, model, args)
        return

    if (args.evaluate_consist):
        validate_consist(val_loader, model, args)
        return

    if (args.evaluate_rotate):
        validate_rotate(val_loader, model, args)
        return

    if (args.evaluate_diagonal):
        validate_diagonal(val_loader, model, args)
        return

    if (args.evaluate_save):
        validate_save(val_loader, mean, std, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if args.multi:
            if epoch % 2 == 0:
                train(train_loader, model, criterion, optimizer, epoch, args)
            else:
                train(train_loader_multi, model, criterion, optimizer, epoch, args)
        else:
            train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch, out_dir=args.out_dir)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    accum_track = 0
    optimizer.zero_grad()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target_train = target.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        new_label = []
        for i in range(input.size(0)):
            nonzero = torch.nonzero(torch.where(target_train[i] == 1, torch.tensor(1).cuda(args.gpu, non_blocking=True)
                                                , torch.tensor(0).cuda(args.gpu, non_blocking=True)))
            lable = nonzero.tolist()
            lable = random.choice(lable)
            new_label.append(lable)
        target_train = torch.tensor(new_label).reshape(input.size(0)).cuda(args.gpu, non_blocking=True)

        # compute output
        if (args.arch == 'inception_v3'):
            output, aux = model(input)
            loss0 = criterion(output, target_train)
            loss1 = criterion(aux,target_train)
            loss = loss0 + 0.3*loss1
        else:
            output = model(input)
            loss = criterion(output, target_train)


        # measure accuracy and record loss
        acc1, acc2 = accuracy(output, target, args, topk=(1, 2))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top2.update(acc2[0], input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        accum_track += 1
        if (accum_track == args.batch_accum):
            optimizer.step()
            accum_track = 0
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Acc@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top2=top2))
    logger.info(' Epoch: [{0}] Train Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f} Loss {loss.avg:.4f}'
                .format(epoch, top1=top1, top2=top2, loss=losses))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for index, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            new_label = []
            for i in range(input.size(0)):
                nonzero = torch.nonzero(
                    torch.where(target[i] == 1, torch.tensor(1).cuda(args.gpu), torch.tensor(0).cuda(args.gpu)))
                lable = nonzero.tolist()
                lable = random.choice(lable)
                new_label.append(lable)
            target_val = torch.tensor(new_label).reshape(input.size(0)).cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target_val)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, args, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top2.update(acc2[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Acc@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                    index, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1, top2=top2))

        logger.info(' Test Acc@1 {top1.avg:.4f} Acc@2 {top2.avg:.4f} Loss {loss.avg:.4f}'
                    .format(top1=top1, top2=top2, loss=losses))

    return top1.avg


def validate_random(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for index, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            new_label = []
            for i in range(input.size(0)):
                nonzero = torch.nonzero(
                    torch.where(target[i] == 1, torch.tensor(1).cuda(args.gpu), torch.tensor(0).cuda(args.gpu)))
                lable = nonzero.tolist()
                lable = random.choice(lable)
                new_label.append(lable)
            target_val = torch.tensor(new_label).reshape(input.size(0)).cuda(args.gpu, non_blocking=True)

            # compute output
            to_img = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            padding = transforms.Pad(20, padding_mode='symmetric')
            rotate = transforms.RandomRotation(8, resample=False, expand=False, center=None, fill=0)
            cropping = transforms.CenterCrop(256)
            img = to_img(input.squeeze(0).cpu())
            img = cropping(rotate(padding(img)))
            input = normalize(to_tensor(img)).unsqueeze(0).cuda(args.gpu, non_blocking=True)
            output = model(input)
            loss = criterion(output, target_val)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, args, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top2.update(acc2[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Acc@2 {top2.val:.3f} ({top2.avg:.3f})'.format(index, len(val_loader),
                                                                           batch_time=batch_time,
                                                                           loss=losses, top1=top1, top2=top2))

        logger.info(' Test Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f} Loss {loss.avg:.4f}'
                    .format(top1=top1, top2=top2, loss=losses))

    return top1.avg


def validate_shift(val_loader, model, args):
    batch_time = AverageMeter()
    consist = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for ep in range(args.epochs_shift):
            for i, (input, target) in enumerate(val_loader):
                if args.gpu is not None:
                    input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                off0 = np.random.randint(32, size=2)
                off1 = np.random.randint(32, size=2)

                output0 = model(input[:, :, off0[0]:off0[0] + 224, off0[1]:off0[1] + 224])
                output1 = model(input[:, :, off1[0]:off1[0] + 224, off1[1]:off1[1] + 224])

                cur_agree = agreement(output0, output1).type(torch.FloatTensor).to(output0.device)

                # measure agreement and record
                consist.update(cur_agree.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    logger.info('Ep [{0}/{1}]:\t'
                                'Test: [{2}/{3}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Consist {consist.val:.4f} ({consist.avg:.4f})\t'.format(
                        ep, args.epochs_shift, i, len(val_loader), batch_time=batch_time, consist=consist))

        logger.info(' * Consistency {consist.avg:.3f}'
                    .format(consist=consist))

    return consist.avg


def validate_consist(val_loader, model, args):
    batch_time = AverageMeter()
    consist = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for ep in range(args.epochs_shift):
            for i, (input, target) in enumerate(val_loader):
                if args.gpu is not None:
                    input = input.cuda(args.gpu, non_blocking=True)

                to_img = transforms.ToPILImage()
                to_tensor = transforms.ToTensor()
                padding = transforms.Pad(20, padding_mode='symmetric')
                rotate = transforms.RandomRotation(8, resample=False, expand=False, center=None, fill=0)
                cropping = transforms.CenterCrop(256)
                img = to_img(input.squeeze(0).cpu())
                img1 = cropping(rotate(padding(img)))
                img2 = cropping(rotate(padding(img)))

                output0 = model(input)
                output1 = model(normalize(to_tensor(img1)).unsqueeze(0).cuda(args.gpu, non_blocking=True))
                output2 = model(normalize(to_tensor(img2)).unsqueeze(0).cuda(args.gpu, non_blocking=True))

                cur_agree = agreement(output0, output1, output2).type(torch.FloatTensor).to(input.device)

                # measure agreement and record
                consist.update(cur_agree.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    logger.info('Ep [{0}/{1}]:\t'
                                'Test: [{2}/{3}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Consist {consist.val:.4f} ({consist.avg:.4f})\t'.format(
                        ep, args.epochs_shift, i, len(val_loader), batch_time=batch_time, consist=consist))

        logger.info(' * Consistency {consist.avg:.3f}'
                    .format(consist=consist))

    return consist.avg


def validate_diagonal(val_loader, model, args):
    batch_time = AverageMeter()
    prob = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    D = 33
    diag_probs = np.zeros((len(val_loader.dataset), D))
    diag_probs2 = np.zeros((len(val_loader.dataset), D))  # save highest probability, not including ground truth
    diag_corrs = np.zeros((len(val_loader.dataset), D))

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            inputs = []
            for off in range(D):
                inputs.append(input[:, :, off:off + 224, off:off + 224])
            inputs = torch.cat(inputs, dim=0)
            probs = torch.nn.Softmax(dim=1)(model(inputs))
            corrs = probs.argmax(dim=1).cpu().data.numpy() == target.item()
            outputs = 100. * probs[:, target.item()]

            acc1, acc2 = accuracy(probs, target.repeat(D), args, topk=(1, 2))

            probs[:, target.item()] = 0
            probs2 = 100. * probs.max(dim=1)[0].cpu().data.numpy()

            diag_probs[i, :] = outputs.cpu().data.numpy()
            diag_probs2[i, :] = probs2
            diag_corrs[i, :] = corrs

            # measure agreement and record
            prob.update(np.mean(diag_probs[i, :]), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top2.update(acc2.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Prob {prob.val:.4f} ({prob.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Acc@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, prob=prob, top1=top1, top2=top2))

    logger.info(' * Prob {prob.avg:.3f} Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f}'
                .format(prob=prob, top1=top1, top2=top2))

    np.save(os.path.join(args.out_dir, 'diag_probs'), diag_probs)
    np.save(os.path.join(args.out_dir, 'diag_probs2'), diag_probs2)
    np.save(os.path.join(args.out_dir, 'diag_corrs'), diag_corrs)


'''
def validate_rotate(val_loader, model, args):
    batch_time = AverageMeter()
    prob = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    ground = np.arange(-8, 8, 0.5).tolist()
    rotate_probs = np.zeros((len(val_loader.dataset), len(ground)))
    rotate_probs2 = np.zeros(
        (len(val_loader.dataset), len(ground)))  # save highest probability, not including ground truth
    rotate_corrs = np.zeros((len(val_loader.dataset), len(ground)))

    tensor2img = transforms.ToPILImage()
    img2tensor = transforms.ToTensor()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            inputs = []
            for degree in ground:
                img = tensor2img(input.squeeze(0).cpu())
                img = transforms.functional.resize(img, (256, 256), interpolation=2)
                img = transforms.functional.pad(img, 20, padding_mode='symmetric')
                img = transforms.functional.rotate(img, degree, fill=0)
                img = transforms.functional.center_crop(img, 256)
                input = img2tensor(img).unsqueeze(0).cuda(args.gpu)
                inputs.append(input)
            inputs = torch.cat(inputs, dim=0)
            probs = torch.nn.Softmax(dim=1)(model(inputs))
            target = torch.nonzero(target[0])[0]
            target_item = target.item()
            corrs = probs.argmax(dim=1).cpu().data.numpy() == target_item
            outputs = 100. * probs[:, target_item]
            acc1, acc2 = rotate_accuracy(probs, target.repeat(len(ground)), topk=(1, 2))

            probs[:, target.item()] = 0
            probs2 = 100. * probs.max(dim=1)[0].cpu().data.numpy()

            rotate_probs[i, :] = outputs.cpu().data.numpy()
            rotate_probs2[i, :] = probs2
            rotate_corrs[i, :] = corrs

            # measure agreement and record
            prob.update(np.mean(rotate_probs[i, :]), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top2.update(acc2.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Prob {prob.val:.4f} ({prob.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Acc@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, prob=prob, top1=top1, top2=top2))

    logger.info(' * Prob {prob.avg:.3f} Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f}'
                .format(prob=prob, top1=top1, top2=top2))

    np.save(os.path.join(args.out_dir, 'rotate_probs'), rotate_probs)
    np.save(os.path.join(args.out_dir, 'rotate_probs2'), rotate_probs2)
    np.save(os.path.join(args.out_dir, 'rotate_corrs'), rotate_corrs)
'''


def validate_rotate(val_loader, model, args):
    batch_time = AverageMeter()
    prob = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # ground = np.arange(-8, 8.5, 0.5).tolist()
    ground = [-3, 3.25, 0.25]
    rotate_probs = np.zeros((len(val_loader.dataset), len(ground)))
    rotate_probs2 = np.zeros(
        (len(val_loader.dataset), len(ground)))  # save highest probability, not including ground truth
    rotate_corrs = np.zeros((len(val_loader.dataset), len(ground)))

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            inputs = []
            to_img = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            for degree in ground:
                img = to_img(input.squeeze(0).cpu())
                img = transforms.functional.pad(img, 20, padding_mode='symmetric')
                img = transforms.functional.rotate(img, degree, fill=0)
                img = transforms.functional.center_crop(img, 256)
                input = normalize(to_tensor(img)).unsqueeze(0).cuda(args.gpu)
                inputs.append(input)
            inputs = torch.cat(inputs, dim=0)
            probs = torch.nn.Softmax(dim=1)(model(inputs))
            target_index = torch.nonzero(target[0])
            corrs = np.zeros_like(probs.argmax(dim=1).cpu().data.numpy())
            outputs = 0
            for target_item in target_index:
                corrs = (probs.argmax(dim=1).cpu().data.numpy() == target_item.item()) | corrs
                outputs = 100. * probs[:, target_item.item()] + outputs
            acc1, acc2 = accuracy(probs, target.repeat(len(ground), 1), args, topk=(1, 2))

            for target_item in target_index:
                probs[:, target_item.item()] = 0
            probs2 = 100. * probs.max(dim=1)[0].cpu().data.numpy()

            rotate_probs[i, :] = outputs.cpu().data.numpy().reshape(len(ground))
            rotate_probs2[i, :] = probs2
            rotate_corrs[i, :] = corrs

            # measure agreement and record
            prob.update(np.mean(rotate_probs[i, :]), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top2.update(acc2.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Prob {prob.val:.4f} ({prob.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Acc@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, prob=prob, top1=top1, top2=top2))

    logger.info(' * Prob {prob.avg:.3f} Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f}'
                .format(prob=prob, top1=top1, top2=top2))

    np.save(os.path.join(args.out_dir, 'rotate_probs'), rotate_probs)
    np.save(os.path.join(args.out_dir, 'rotate_probs2'), rotate_probs2)
    np.save(os.path.join(args.out_dir, 'rotate_corrs'), rotate_corrs)


def validate_save(val_loader, mean, std, args):
    import matplotlib.pyplot as plt
    import os
    for i, (input, target) in enumerate(val_loader):
        img = (255 * np.clip(input[0, ...].data.cpu().numpy() * np.array(std)[:, None, None] + mean[:, None, None], 0,
                             1)).astype('uint8').transpose((1, 2, 0))
        plt.imsave(os.path.join(args.out_dir, '%05d.png' % i), img)


# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
def save_checkpoint(state, is_best, epoch, out_dir='./'):
    torch.save(state, os.path.join(out_dir, 'checkpoint.pth.tar'))
    # if (epoch % 10 == 0):
    #     torch.save(state, os.path.join(out_dir, 'checkpoint_%03d.pth.tar' % epoch))
    if is_best:
        shutil.copyfile(os.path.join(out_dir, 'checkpoint.pth.tar'), os.path.join(out_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def rotate_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy(output, target, args, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        res = []
        for k in topk:
            _, pred = output.topk(k, 1, True, True)
            onehot = torch.zeros(batch_size, 9).cuda(args.gpu).scatter(1, pred, 1)
            correct = onehot.mul_(target)
            sum = correct.sum(1)
            correct_k = torch.where(sum > 0, torch.tensor(1).cuda(args.gpu), torch.tensor(0).cuda(args.gpu)) \
                .float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def agreement(output0, output1, output2):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)
    pred2 = output2.argmax(dim=1, keepdim=False)
    agree1 = pred0.eq(pred1)
    agree2 = pred1.eq(pred2)
    agree = agree1.eq(agree2)
    agree = 100. * torch.mean(agree.type(torch.FloatTensor).to(output0.device))
    return agree


if __name__ == '__main__':
    main()
