import sys
sys.path.append('../')

from utils.util import check_parameters
import time
import logging
from model.loss import Loss
import torch
import os
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer(object):
    def __init__(self, rank, world_size, train_dataloader, val_dataloader, Dual_RNN, optimizer, scheduler, opt, asrloss):
        super(Trainer).__init__()
        self.rank = rank
        self.world_size = world_size
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        self.num_spks = opt['num_spks']
        self.cur_epoch = 0
        self.total_epoch = opt['train']['epoch']
        self.early_stop = opt['train']['early_stop']
        self.print_freq = opt['logger']['print_freq']
        self.logger = logging.getLogger(opt['logger']['name'])
        self.checkpoint = opt['train']['path']
        self.name = opt['name']
        self.asrloss = asrloss

        self.device = torch.device(f'cuda:{self.rank}')
        self.dualrnn = Dual_RNN.to(self.device)
        if world_size > 1:
            self.dualrnn = DDP(self.dualrnn, device_ids=[rank])
        if self.asrloss is not None:
            self.asrloss.model.to(self.device)

        self.logger.info(
                'Loading Dual-Path-RNN parameters: {:.3f} Mb'.format(check_parameters(self.dualrnn)))
        
        if opt['resume']['state']:
            ckp = torch.load(opt['resume']['path'], map_location='cpu')
            self.cur_epoch = ckp['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:.3f}".format(
                opt['resume']['path'], self.cur_epoch))
            has_module_prefix = any(key.startswith('module.') for key in ckp['model_state_dict'].keys())
            if not has_module_prefix:
                state_dict = {f"module.{k}": v for k,v in ckp['model_state_dict'].items()}
            else:
                state_dict = ckp['model_state_dict']
            self.dualrnn.load_state_dict(state_dict)  # Ensure DDP state is loaded
            optimizer.load_state_dict(ckp['optim_state_dict'])
            self.optimizer = optimizer
            # lr = self.optimizer.param_groups[0]['lr']
            # self.adjust_learning_rate(self.optimizer, lr*0.5)
        else:
            self.optimizer = optimizer

        if opt['optim']['clip_norm']:
            self.clip_norm = opt['optim']['clip_norm']
            self.logger.info(
                "Gradient clipping by {}, default L2".format(self.clip_norm))
        else:
            self.clip_norm = 0

    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def train(self, epoch):
        if self.rank == 0:
            self.logger.info(
                'Start training from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.dualrnn.train()
        num_batchs = len(self.train_dataloader)
        total_loss = 0.0
        total_asr_loss = 0.0
        num_index = 1
        start_time = time.time()
        for mix, ref in self.train_dataloader:
            mix = mix.to(self.device)
            ref = [ref[i].to(self.device) for i in range(self.num_spks)]
            self.optimizer.zero_grad()

            out = self.dualrnn(mix)

            l = Loss(out, ref)
            l_asr = self.asrloss(out, ref)
            epoch_loss = l + l_asr*0.1
            epoch_loss.backward()
            total_loss += l.item()
            total_asr_loss += l_asr.item()
            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(self.dualrnn.parameters(), self.clip_norm)
            self.optimizer.step()
            if self.rank == 0 and num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, sisnrloss:{:.3f}, asrloss:{:.3f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index, total_asr_loss/num_index)
                self.logger.info(message)
            num_index += 1

        end_time = time.time()
        total_loss = total_loss/num_index
        total_asr_loss = total_asr_loss/num_index
        if self.rank == 0:
            message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, asrloss:{:.3f}, Total time:{:.3f} min> '.format(
                epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, total_asr_loss, (end_time-start_time)/60)
            self.logger.info(message)
        return total_loss
    
    def validation(self, epoch):
        if self.rank == 0:
            self.logger.info(
                'Start Validation from epoch: {:d}, iter: {:d}'.format(epoch, 0))
            self.dualrnn.eval()
        num_batchs = len(self.val_dataloader)
        num_index = 1
        total_loss = 0.0
        start_time = time.time()
        with torch.no_grad():
            for mix, ref in self.val_dataloader:
                mix = mix.to(self.device)
                ref = [ref[i].to(self.device) for i in range(self.num_spks)]
                self.optimizer.zero_grad()

                out = self.dualrnn(mix)
                
                l = Loss(out, ref)
                epoch_loss = l
                total_loss += epoch_loss.item()
                if self.rank == 0 and num_index % self.print_freq == 0:
                    message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                        epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                    self.logger.info(message)
                num_index += 1
        end_time = time.time()
        if self.rank == 0:
            total_loss = total_loss/num_index
            message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
                epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, (end_time-start_time)/60)
            self.logger.info(message)
        return total_loss

    def run(self):
        train_loss = []
        val_loss = []
        if self.rank == 0:
            self.save_checkpoint(self.cur_epoch, mode='epoch')
            v_loss = self.validation(self.cur_epoch)
            best_loss = v_loss
            # best_loss=100
            self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(
                self.cur_epoch, best_loss))
            no_improve = 0

        # starting training part
        while self.cur_epoch < self.total_epoch:
            self.cur_epoch += 1
            t_loss = self.train(self.cur_epoch)
            v_loss = self.validation(self.cur_epoch)

            train_loss.append(t_loss)
            val_loss.append(v_loss)

            # schedule here
            self.scheduler.step(v_loss)

            self.save_checkpoint(epoch=self.cur_epoch, mode='epoch')

            if self.rank == 0:
                if v_loss >= best_loss:
                    no_improve += 1
                    self.logger.info(
                        'No improvement, Best Loss: {:.4f}'.format(best_loss))
                else:
                    best_loss = v_loss
                    no_improve = 0
                    self.save_checkpoint(self.cur_epoch, mode='epoch')
                    self.logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}'.format(
                        self.cur_epoch, best_loss))

                if no_improve == self.early_stop:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_improve))
                    break
        if self.rank == 0:
            self.save_checkpoint(self.cur_epoch, best=False)
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, self.total_epoch))

        if self.rank == 0:
            # draw loss image
            plt.title("Loss of train and test")
            x = [i for i in range(self.cur_epoch)]
            plt.plot(x, train_loss, 'b-', label=u'train_loss', linewidth=0.8)
            plt.plot(x, val_loss, 'c-', label=u'val_loss', linewidth=0.8)
            plt.legend()
            #plt.xticks(l, lx)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig('loss.png')

    def save_checkpoint(self, epoch, mode='epoch'):
        if self.rank == 0:
            os.makedirs(os.path.join(self.checkpoint, self.name), exist_ok=True)
            if mode == 'best':
                filename = 'best.pt'
            elif mode == 'last':
                filename = 'last.pt'
            elif mode == 'epoch':
                filename = f'epoch_{epoch}.pt'
            else:
                raise ValueError(f"Invalid mode: {mode}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.dualrn.state_dict(),
                'optim_state_dict': self.optimizer.state_dict()
            }, 
                os.path.join(self.checkpoint, self.name, filename))
