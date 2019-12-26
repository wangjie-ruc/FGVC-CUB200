import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict

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
    img, label = data
    label = label.cuda(non_blocking=True)
    img = img.cuda()
    pred, theta = model(img)
    loss = F.cross_entropy(pred, label)
    acc_top1, acc_top5 = accuracy(pred, label, topk=(1, 5))
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    log_vars['acc'] = acc_top1.item()

    outputs = dict(loss=loss, log_vars=log_vars, num_samples=img.size(0))
    return outputs


def main():
    parser, cfg = Config.auto_argparser()
    args = parser.parse_args()
    cfg.merge_from_args(args)

    # build datasets and dataloaders

    train_dataset = datasets.CUB200(
        root=cfg.data_root,
        train=True,
        transforms=train_transform
    )
    val_dataset = datasets.CUB200(
        root=cfg.data_root,
        train=False,
        transforms=eval_transform
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

    # import itertools
    # model_without_module = model.module if hasattr(model, 'module') else model
    # optimizer = torch.optim.SGD([{"params": model_without_module.stn.parameters()},
    #                              {"params": model_without_module.fc.parameters()}],
    #                             lr=0.001, momentum=0.9, weight_decay=1e-4)
    # runner.optimizer = optimizer

    # load param (if necessary) and run
    if cfg.get('resume_from') is not None:
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from') is not None:
        runner.load_checkpoint(cfg.load_from)

    runner.run([train_loader, val_loader], cfg.workflow, cfg.total_epochs)


if __name__ == '__main__':
    main()
