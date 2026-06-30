"""
RTG CenterPoint training entry point.

Based on CenterPoint/tools/train.py, adapted for RTG BEV scenario.
"""

import argparse
import json
import os
import sys

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import numpy as np
import torch
from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
import torch.distributed as dist
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Train a CenterPoint detector for RTG')
    parser.add_argument(
        '--config',
        default='configs/rtg/rtg_centerpoint_voxelnet.py',
        help='train config file path (default: RTG CenterPoint config)')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Resolve config path relative to CenterPoint root
    cp_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(cp_root, config_path)

    cfg = Config.fromfile(config_path)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join(cp_root, 'work_dirs', 'rtg_centerpoint_voxelnet')

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        if args.launcher == 'pytorch':
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            cfg.local_rank = args.local_rank
        elif args.launcher == 'slurm':
            proc_id = int(os.environ['SLURM_PROCID'])
            ntasks = int(os.environ['SLURM_NTASKS'])
            node_list = os.environ['SLURM_NODELIST']
            num_gpus = torch.cuda.device_count()
            cfg.gpus = num_gpus
            torch.cuda.set_device(proc_id % num_gpus)
            addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
            port = None
            if port is not None:
                os.environ['MASTER_PORT'] = str(port)
            elif 'MASTER_PORT' in os.environ:
                pass
            else:
                os.environ['MASTER_PORT'] = '29501'
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = addr
            os.environ['WORLD_SIZE'] = str(ntasks)
            os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
            os.environ['RANK'] = str(proc_id)
            dist.init_process_group(backend='nccl')
            cfg.local_rank = int(os.environ['LOCAL_RANK'])
        cfg.gpus = dist.get_world_size()
    else:
        cfg.local_rank = args.local_rank

    if args.autoscale_lr:
        cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus

    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))
    logger.info(f'torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}')

    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(config=cfg.text, CLASSES=datasets[0].CLASSES)

    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model, datasets, cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger,
    )


if __name__ == '__main__':
    main()
