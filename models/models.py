import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from collections import OrderedDict, namedtuple
from torch.autograd import Variable
import itertools
import util.util as util
import sys
from torchvision import models
from .convlstmcell import ConvLSTMCell
from . import inception, resnet

# __all__ = ['LRCNModel', 'Output']

Output = namedtuple('Output', ['warpped', 'mask','thetas'])

def stn(img, theta):
    theta = theta.view(-1, 2, 3)

    grid = F.affine_grid(theta, img.size())
    img = F.grid_sample(img, grid)

    return img

def create_cnn(opt):
    if opt.cnn == 'googlenet':
        return inception.inception_v3(pretrained=opt.pretrained)
    elif opt.cnn == 'resnet50':
        return resnet.resnet50(pretrained=opt.pretrained)
    else:
        raise ValueError('Unrecognized argument {}'.format(opt.cnn))

def get_shape(cnn, h, w):
    return cnn.forward(Variable(torch.zeros(1, 3, h, w), volatile=True)).shape

class LRCNModel(nn.Module):
    def name(self):
        return 'LRCNModel'

    def __init__(self, opt):
        super(LRCNModel, self).__init__()
        self.opt = opt
        # googlenet = models.inception_v3(pretrained=True, aux_logits=False, transform_input=True)
        # self.cnn = nn.Sequential(*list(googlenet.children())[:-1])
        self.cnn = create_cnn(opt)
        shape = get_shape(self.cnn, opt.height, opt.width) # [1, 2048, 7, 14] = 200704a
        self.cnn_output_shape = shape
        print('cnn output shape={}'.format(shape))
        self.rnn = opt.rnn
        self.rnn_layers = opt.rnn_layers
        self.rnn_chn = opt.rnn_chn
        self.rnn_input_dim = shape[1] #shape[1] * shape[2] * shape[3]
        self.lstm = nn.LSTM(self.rnn_input_dim, opt.rnn_chn, num_layers=opt.rnn_layers, batch_first=False)
        
        self.fc_loc = nn.Sequential(
            nn.Linear(self.rnn_chn * self.rnn_layers + self.rnn_input_dim, self.rnn_chn),
            # nn.Linear(self.rnn_input_dim * 2, self.rnn_chn),
            # nn.Linear(self.rnn_chn, 32),
            nn.ReLU(False),
            nn.Linear(self.rnn_chn, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.fill_(0)
        self.fc_loc[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def freeze_cnn(self, freeze):
        for p in self.cnn.parameters():
            p.requires_grad = not freeze

    def forward(self, sample):
        # CNN
        seq = sample.prefix + sample.unstable
        batch_size = seq[0].shape[0]
        num_prefix = len(sample.prefix)
        features = []
        for i in seq:
            output = self.cnn(i)
            if self.rnn == 'lstm':
                output = F.avg_pool2d(output, kernel_size=self.cnn_output_shape[2:])
            features.append(output)

        # RNN
        packed = torch.stack(features[:num_prefix], dim=0)
        out, (h, c) = self.lstm(packed.view(packed.shape[0], packed.shape[1], -1))
        avg = torch.mean(out, dim=0)

        # STN
        warpped, mask, thetas = [], [], []
        for idx, unstable in enumerate(features[num_prefix:]):
            unstable_flat = unstable.view(batch_size, -1)
            # before_fc_loc = torch.cat((unstable_flat, *h), dim=1)
            before_fc_loc = torch.cat((unstable_flat, avg), dim=1)
            thetas.append(self.fc_loc(before_fc_loc))
            warpped.append(stn(sample.unstable[idx], thetas[idx]))
            img_shape = sample.unstable[idx].shape
            mask.append(stn(Variable(output.data.new(img_shape[0], 1, img_shape[2], img_shape[3]).zero_() + 1), thetas[idx]))
            
        return Output(warpped, mask, thetas)


class ConvLSTM(nn.Module):
    def name(self):
        return 'ConvLSTM'

    def __init__(self, opt):
        super(ConvLSTM, self).__init__()
        self.opt = opt
        self.cnn = create_cnn(opt)
        shape = get_shape(self.cnn, opt.height, opt.width) # [1, 2048, 7, 14] = 200704a
        self.cnn_output_shape = shape
        print('cnn output shape={}'.format(shape))
        self.rnn = opt.rnn
        self.rnn_layers = opt.rnn_layers
        self.rnn_chn = opt.rnn_chn
        self.rnn_input_dim = shape[1] #shape[1] * shape[2] * shape[3]
        self.convlstm = ConvLSTMCell(shape[1], self.rnn_chn)# nn.LSTM(self.rnn_input_dim, opt.rnn_chn, num_layers=opt.rnn_layers, batch_first=False)
        
        self.fc_loc = nn.Sequential(
            nn.Conv2d(self.rnn_chn + self.cnn_output_shape[1], 512, (1, 1)), # [n, 512, 7, 14]
            nn.ReLU(),
            nn.AvgPool2d(self.cnn_output_shape[2:]),
            nn.Conv2d(512, 6, (1, 1)), # [n, 6, 1, 1]
            # nn.ReLU(),
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.fill_(0)
        self.fc_loc[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def freeze_cnn(self, freeze):
        for p in self.cnn.parameters():
            p.requires_grad = not freeze

    def forward(self, sample):
        # CNN
        seq = sample.prefix + sample.unstable
        batch_size = seq[0].shape[0]
        num_prefix = len(sample.prefix)
        features = []
        for i in seq:
            output = self.cnn(i)
            features.append(output)
        
        # RNN
        state = None
        for t in range(0, num_prefix):
            state = self.convlstm(features[t], state)
        h, c = state

        # STN
        warpped, mask, thetas = [], [], []
        for idx, unstable in enumerate(features[num_prefix:]):
            before_fc_loc = torch.cat((unstable, h), dim=1)
            output = self.fc_loc(before_fc_loc)
            thetas.append(output)
            warpped.append(stn(sample.unstable[idx], thetas[idx]))
            img_shape = sample.unstable[idx].shape
            mask.append(stn(Variable(output.data.new(img_shape[0], 1, img_shape[2], img_shape[3]).zero_() + 1), thetas[idx]))
            
        return Output(warpped, mask, thetas)

# class SimpleModel(nn.Module):
#     def name(self):
#         return 'SimpleModel'

#     def __init__(self, opt):
#         super(SimpleModel, self).__init__()
#         self.cnn = create_cnn(opt)
#         shape = get_shape(self.cnn, opt.height, opt.width) # [1, 2048, 7, 14] = 200704a
#         self.cnn_output_shape = shape
#         print('cnn output shape={}'.format(shape))

#         self.fc_loc = nn.Sequential(
#             nn.Linear(self.cnn_output_shape[1] * 2, 256),
#             # nn.Linear(256, 32),
#             nn.ReLU(False),
#             nn.Linear(256, 3 * 2)
#         )
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[-1].weight.data.fill_(0)
#         self.fc_loc[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

#     def freeze_cnn(self, freeze):
#         for p in self.cnn.parameters():
#             p.requires_grad = not freeze

#     def forward(self, sample):
#         # CNN
#         seq = sample.prefix + sample.unstable
#         batch_size = seq[0].shape[0]
#         num_prefix = len(sample.prefix)
#         features = []
#         for i in seq:
#             output = self.cnn(i)
#             output = F.avg_pool2d(output, kernel_size=self.cnn_output_shape[2:]).view(batch_size, -1)
#             features.append(output)
        
#         # STN
#         pred = features[num_prefix]
#         warpped, thetas = [], []
#         for idx, unstable_flat in enumerate(features[num_prefix:]):
#             before_fc_loc = torch.cat((unstable_flat, pred), dim=1)
#             thetas.append(self.fc_loc(before_fc_loc))
#             warpped.append(stn(sample.unstable[idx], thetas[idx]))

#         return Output(warpped, thetas)
