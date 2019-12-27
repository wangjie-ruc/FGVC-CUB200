import sys
sys.path.append('.')

import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict

import cvt
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms

from apcs import Config
from mmcv.runner import Runner

from transforms import train_transform, eval_transform
import datasets
import networks


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
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


def batch_processor(model, data, train_mode):
    log_vars = OrderedDict()

    input, target = data
    input = input.cuda()
    target = target.cuda()

    feature_center = model.module.center if hasattr(model, 'module') else model.center
    outputs, feature_matrix, attention_map = model(input)
    if train_mode:
        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[target], dim=-1)
        feature_center[target] += 5e-2 * \
            (feature_matrix.detach() - feature_center_batch)

        # Attention Cropping
        with torch.no_grad():
            crop_images = networks.ws_dan2.batch_augment(
                input, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)

        # crop images forward
        pred_crop, _, _ = model(crop_images)

        # Attention Dropping
        with torch.no_grad():
            drop_images = batch_augment(
                input, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))

        # drop images forward
        pred_drop, _, _ = model(drop_images)

        # loss
        loss = F.cross_entropy(outputs, target) / 3. + \
            F.cross_entropy(pred_crop, target) / 3. + \
            F.cross_entropy(pred_drop, target) / 3. + \
            F.mse_loss(feature_matrix, feature_center_batch)
    else:
        # Object Localization and Refinement
        crop_images = networks.ws_dan2.batch_augment(
            input, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
        pred_crop, _, _ = model(crop_images)

        # Final prediction
        outputs = (outputs + pred_crop) / 2.

        # loss
        loss = F.cross_entropy(pred, target)

    acc_top1, acc_top5 = accuracy(outputs, target, topk=(1, 5))
    log_vars['loss'] = loss.item()
    log_vars['acc'] = acc_top1.item()

    outputs = dict(loss=loss, log_vars=log_vars, num_samples=input.size(0))
    return outputs


def main():
    parser, cfg = Config.auto_argparser()
    args = parser.parse_args()
    cfg.merge_from_args(args)

    # build datasets and dataloaders

    train_dataset = datasets.CUB200(
        root=cfg.data_root,
        train=True,
        transforms=cvt.from_file(cfg.train_transforms)
    )
    val_dataset = datasets.CUB200(
        root=cfg.data_root,
        train=False,
        transforms=cvt.from_file(cfg.train_transforms)
    )

    num_workers = cfg.data_workers * len(cfg.gpus)
    batch_size = cfg.batch_size
    train_sampler = None
    val_sampler = None
    shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers
    )

    # build model
    model = getattr(networks, cfg.model)(**cfg.model_kwargs)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.gpus))
    cfg.gpus = list(range(len(cfg.gpus)))
    model = DataParallel(model, device_ids=cfg.gpus).cuda()

    # build runner and register hooks
    runner = Runner(
        model,
        batch_processor,
        cfg.optimizer,
        cfg.work_dir,
        log_level=cfg.log_level)

    runner.register_training_hooks(
        lr_config=cfg.lr_config,
        optimizer_config=cfg.optimizer_config,
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config)

    # load param (if necessary) and run
    if cfg.get('resume_from') is not None:
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from') is not None:
        runner.load_checkpoint(cfg.load_from)

    runner.run([train_loader, val_loader], cfg.workflow, cfg.total_epochs)


if __name__ == '__main__':
    main()
