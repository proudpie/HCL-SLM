import os
import torch
from data_loader.AudioReader import AudioReader, write_wav
import argparse
from torch.nn.parallel import data_parallel
from model.DPRNN import Dual_RNN_model
from logger.set_logger import setup_logger
import logging
from config import option
import tqdm

class Separation():
    def __init__(self, config_path):
        super(Separation, self).__init__()
        self.opt = option.parse(config_path)
        self.mix = AudioReader(
            self.opt['mix_scp'],
            sample_rate = self.opt['audio']['sample_rate']
        )

        self.net = Dual_RNN_model(**self.opt['TFLocoformer'])
        dicts = torch.load(self.opt['model_path'], map_location='cpu')
        model_state_dict = dicts["model_state_dict"]
        new_state_dict = {}
        for k, v in model_state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        self.net.load_state_dict(new_state_dict)

        setup_logger(self.opt['logger']['name'], self.opt['logger']['path'],
                     screen=self.opt['logger']['screen'], tofile=self.opt['logger']['tofile'])
        self.logger = logging.getLogger(self.opt['logger']['name'])
        self.logger.info('Load checkpoint from {}, epoch {: d}'.format(self.opt['model_path'], dicts["epoch"]))
        gupid = self.opt['gpub']
        self.device = torch.device('cuda:{}'.format(gupid))
        self.net = self.net.to(self.device)
        self.net.eval()

    def inference(self, file_path):
        with torch.no_grad():
            self.logger.info("Start: Compute over {:d} utterence".format(len(self.mix)))
            for key, egs in tqdm.tqdm(self.mix):
                #self.logger.info("Compute on utterance {}...".format(key))
                egs=egs.to(self.device)
                norm = torch.norm(egs,float('inf'))
                if egs.dim() == 1:
                    egs = torch.unsqueeze(egs, 0)
                ests=self.net(egs)
                spks=[torch.squeeze(s.detach().cpu()) for s in ests]

                index=0
                for s in spks:
                    s = s[:egs.shape[1]]
                    s = s - torch.mean(s)
                    s = s/torch.max(torch.abs(s))
                    #norm
                    #s = s*norm/torch.max(torch.abs(s))
                    s = s.unsqueeze(0)
                    index += 1
                    os.makedirs(self.opt['save_path']+'/spk'+str(index), exist_ok=True)
                    filename=self.opt['save_path']+'/spk'+str(index)+'/'+key
                    write_wav(filename, s, self.opt['audio']['sample_rate'])
            self.logger.info("Compute over {:d} utterances".format(len(self.mix)))


def main():
    parser = argparse.ArgumentParser(description='Speech Separation Inference')
    parser.add_argument(
        '--yaml', type=str, required=True, help='Path to Config yaml file.')
    args = parser.parse_args()

    separation = Separation(args.yaml)
    separation.inference()

if __name__ == "__main__":
    main()

# python3 test_DPRNN.py --yaml config/DPRNN/test.yml
