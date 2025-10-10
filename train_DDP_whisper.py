import os
import sys
sys.path.append('./')

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader as Loader
from torch.utils.data import DistributedSampler
from data_loader.Dataset import Datasets
import model
from logger import set_logger
import logging
from config import option
import argparse
import torch
import trainer_whisper
import importlib
from model.whisper_loss import ASRLoss
import torch.distributed as dist

def setup_dist(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

def make_dataloader(opt, rank, world_size):
    # make train's dataloader
    
    train_dataset = Datasets(
        opt['datasets']['train']['dataroot_mix'],
        [opt['datasets']['train']['dataroot_targets'][0],
         opt['datasets']['train']['dataroot_targets'][1]],
        **opt['datasets']['audio_setting'])
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_dataloader = Loader(train_dataset,
                              batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                              num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                              sampler=train_sampler)
    
    # make validation dataloader
    
    val_dataset = Datasets(
        opt['datasets']['val']['dataroot_mix'],
        [opt['datasets']['val']['dataroot_targets'][0],
         opt['datasets']['val']['dataroot_targets'][1]],
        **opt['datasets']['audio_setting'])
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    val_dataloader = Loader(val_dataset,
                            batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                            num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                            sampler=val_sampler)
    
    return train_dataloader, val_dataloader


def make_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        optimizer = optimizer(
            params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
    elif opt['optim']['name'] == 'AdamW':
        optimizer = optimizer(
            params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
    else:
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']
                              ['weight_decay'], momentum=opt['optim']['momentum'])
        
    return optimizer


def train(rank, world_size, opt):
    if rank == 0:
        set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
    logger = logging.getLogger(opt['logger']['name'])
    model_name = opt['name']
    # if torch.cuda.is_available():
    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cudnn.enabled = True
    #     # torch.backends.cudnn.deterministic = True
    #     logger.info("启动cuDNN自动优化器")

    # build model
    logger.info(f"Building the model of {model_name}")
    model_class = getattr(model, model_name)
    model_params = opt[model_name]
    model_common = model_class(**model_params)
    trainer_common = importlib.import_module(f'trainer_whisper.trainer_whisperen')
    # build optimizer
    logger.info(f"Building the optimizer of {model_name}")
    optimizer = make_optimizer(model_common.parameters(), opt)
    # build dataloader
    logger.info(f'Building the dataloader of {model_name}')
    train_dataloader, val_dataloader = make_dataloader(opt, rank, world_size)

    logger.info('Train Datasets Length: {}, Val Datasets Length: {}'.format(
        len(train_dataloader), len(val_dataloader)))
    # build scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=opt['scheduler']['factor'],
        patience=opt['scheduler']['patience'],
        verbose=True, min_lr=opt['scheduler']['min_lr'])
    
    # build trainer
    logger.info(f'Building the Trainer of {model_name}')
    asr_loss = ASRLoss()
    trainer = trainer_common.Trainer(rank, world_size, train_dataloader, val_dataloader, model_common, optimizer, scheduler, opt, asr_loss)
    trainer.run()

def main():
    parser = argparse.ArgumentParser(description='Parameters for training Separation Models')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup_dist(rank, world_size)
    train(rank, world_size, opt)

if __name__ == "__main__":
    main()
