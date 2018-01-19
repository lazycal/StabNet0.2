import argparse
import os
from util import util
import torch
import json


class Options(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--root', default='../frames', help='path to images')
        self.parser.add_argument('--max_matches', type=int, default=1000, help='max feature matches')
        self.parser.add_argument('--train_source', default='../frames/train-list.txt', help='path to images list')
        self.parser.add_argument('--val_source', default='../frames/val-list.txt', help='path to images list')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--max_epoch', type=int, default=50, help='max epochs')
        self.parser.add_argument('--start_epoch', type=int, default=None, help='start epochs')
        self.parser.add_argument('--height', type=int, default=288, help='scale images to height')
        self.parser.add_argument('--width', type=int, default=512, help='scale images to width')
        # self.parser.add_argument('--nc', type=int, default=3, help='number of googlenet channels')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='LRCN',
                                 help='chooses which model to use. LRCN, Simple, ConvLSTM')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        # self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        # self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        # self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        # self.parser.add_argument('--no_dropout', action='store_true', help='no dropout')
        self.parser.add_argument('--id_loss_weight', type=float, default=0.1, help='identity loss weight')
        self.parser.add_argument('--id_loss_step', type=float, default=1000, help='step to disable identity loss')
        self.parser.add_argument('--pix_loss_weight', type=float, default=1, help='pix_loss_weight')
        self.parser.add_argument('--feature_loss_weight', type=float, default=1, help='feature_loss_weight')        
        self.parser.add_argument('--temp_loss_weight', type=float, default=1, help='temp_loss_weight')           
        self.parser.add_argument('--temp_loss_epoch', type=float, default=0, help='epoch to enable temporal loss')        
        self.parser.add_argument('--pretrained', action='store_true', dest='pretrained', help='pretrained')          
        self.parser.add_argument('--no_pretrained', action='store_false', dest='pretrained', help='no pretrained')  
        self.parser.set_defaults(pretrained=True)    
        self.parser.add_argument('--rnn', type=str, default='lstm', help='lstm, convlstm')
        self.parser.add_argument('--cnn', type=str, default='googlenet', help='resnet50, googlenet')
        self.parser.add_argument('--rnn_chn', type=int, default=256, help='rnn channels')
        self.parser.add_argument('--rnn_layers', type=int, default=1, help='number of rnn layers')
        self.parser.add_argument('--freeze_epochs', type=int, default=1, help='when to finetune cnn')
        # self.parser.add_argument('--freeze', action='store_true', help='whether to finetune cnn')
        self.parser.add_argument('--use_l2', action='store_true', help='use l2 pixel loss. default l1')
        self.parser.add_argument('--isTrain', dest='isTrain', action='store_true')
        self.parser.add_argument('--isDeploy', dest='isTrain', action='store_false')
        self.parser.set_defaults(isTrain=True)
        self.parser.add_argument('--data_augment', action='store_true', help='data_augment')
        self.parser.add_argument('--no_data_augment', dest='data_augment', action='store_false')
        self.parser.set_defaults(data_augment=True)
        # self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--log_freq', type=int, default=250, help='frequency of logging into tensorboard')
        self.parser.add_argument('--print_freq', type=int, default=500, help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--val_freq', type=int, default=1000, help='frequency of validation')
        self.parser.add_argument('--continue_train', help='continue training: load the latest model')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--lr_decay', type=float, default=0.1, help='lr multiply by what')
        self.parser.add_argument('--decay_epochs', type=int, nargs='*', default=[], help='when to decay')
        self.parser.add_argument('--val_iters', type=int, default=100)
        self.parser.add_argument('--fake_rate', type=float, default=0.3)
        self.parser.add_argument('--fake_vel', type=float, default=60)

        self.initialized = True

    def parse(self, cmd=None):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(cmd)

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        print(file_name)
        with open(file_name, 'w') as fout:
            json.dump(args, fout, indent=2)
        return self.opt
