import torch.utils.data
import random
import torchvision.transforms as transforms
import torch
from PIL import Image
import json
from collections import namedtuple, Iterable
from torchvision.transforms import functional as F
import os
import scipy.io
import re
import numpy as np
from copy import deepcopy
import cv2
import math
import struct
import traceback
import scipy.misc

BaseDir = './feature_matches/'
regexp = re.compile(r'(\d+)(\.mp4)?\.avi')
regexp_flow = re.compile(r'unstable/(?P<video_name>\d+)/image-(?P<idx>\d+)\.(png|jpg)')

ori_width, ori_height = 1280, 720
FLOW_HEIGHT = 288
FLOW_WIDTH = 512

def norm_axis(x, len):
    return (x * 1. / len - 0.5) * 2

def fetch_fm(root, path):
    # video_name = regexp.match(video_name).group(1)
    path = os.path.join(root, BaseDir, path)
    mat = scipy.io.loadmat(path)
    # print('Read {}. Shape={}'.format(path, mat['res'].shape))
    x = norm_axis(mat['res'], [ori_width, ori_height, ori_width, ori_height]) \
        if mat['res'].shape != (0,0) else np.zeros((0, 4), np.float32) #mat['res'].astype(np.float32)
    return x#np.stack((x[..., 1], x[..., 0], x[..., 3], x[..., 2]), axis=1)

def fetch_flow_npy(opt, video_name, idx):
    return np.load(os.path.join(opt.root, 'flow-npy', video_name, '{:04d}.npy'.format(idx)))
def fetch_flow(opt, video_name, idx):
    root = opt.root
    height = FLOW_HEIGHT
    width = FLOW_WIDTH
    with open(os.path.join(root, 'flow', video_name + '.bin'), 'rb') as flowfile:
        flowdata = flowfile.read()
    float_cnt = 4
    cnt = 2 * height * width * float_cnt * idx
    #calc flow_x
    flow = np.zeros((height, width, 2), dtype=np.float32)
    for xx in range(height):
        for yy in range(width):
            try:
                bit = float(struct.unpack('f', flowdata[cnt:cnt+float_cnt])[0])
            except Exception as e:
                traceback.print_exc()
                print('=======>1', video_name, idx, xx, yy)
            cnt += float_cnt
            flow[xx, yy, 0]=bit + yy
    flow[:, :, 0] = flow[:, :, 0] / (width - 1) * 2 - 1
    #calc flow_y
    for xx in range(height):
        for yy in range(width):
            try:
                bit = float(struct.unpack('f', flowdata[cnt:cnt+float_cnt])[0])
            except Exception as e:
                traceback.print_exc()
                print('=======>2', video_name, idx, xx, yy)
            cnt += float_cnt
            flow[xx, yy, 1]=bit + xx
    flow[:, :, 1] = flow[:, :, 1] / (height - 1) * 2 - 1
    # flow = np.stack([
    #     scipy.misc.imresize(flow[..., 0], (ori_height, ori_width), mode='F'),
    #     scipy.misc.imresize(flow[..., 1], (ori_height, ori_width), mode='F'),
    # ], axis=2)
    return flow

Data = namedtuple('Data', ['prefix', 'unstable', 'target', 'fm', 'fm_mask', 'flow'])

def create_empty_data():
    return Data(prefix=[], unstable=[], target=[], fm=[], fm_mask=[], flow=[])

def map_data(fn, data):
    new_data = create_empty_data()
    for i in range(len(data)):
        img_list = data[i]
        for img in img_list:
            new_data[i].append(fn(img))
    return new_data


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, opt, source, transform):
        self.root = opt.root
        self.source = source
        self.opt = opt
        with open(self.source) as fin:
            self.lst = json.load(fin)
        # like
        # [
        #     {"prefix":   ["img1.png", "img2.png"],
        #      "unstable": ["img3.png", "img4.png"],
        #      "target":   ["img5.png", "img6.png"]


        #     },
        #     {...}
        # ]
        self.transform = transform
        self.max_matches = opt.max_matches

    def __len__(self):
        return len(self.lst)

    def rand(self):
        self.ang = random.uniform(0, math.pi * 2)
        # self.ang = 0
        self.vel = random.uniform(0, self.opt.fake_vel)

    def move(self, img, idx):
        ang = self.ang
        vel = self.vel * idx
        theta = [
            [1, 0, vel * math.cos(ang)],
            [0, 1, vel * math.sin(ang)],
        ]
        theta = np.array(theta, dtype=np.float)
        img = np.array(img)
        img = cv2.warpAffine(img, theta, (img.shape[1], img.shape[0]), borderValue=(0.485 * 255, 0.456 * 255, 0.406 * 255))
        img = Image.fromarray(img, "RGB")
        # img = img.transform(img.size, Image.AFFINE, theta, Image.BICUBIC)
        return img

    def __getitem__(self, idx):
        img_names = self.lst[idx]
        sample = create_empty_data()._asdict()
        for img_class in ["prefix", "unstable", "target"]:
            for img_name in img_names[img_class]:
                img_path = os.path.join(self.root, img_name)
                sample[img_class].append(Image.open(img_path).resize([self.opt.width, self.opt.height], Image.BILINEAR))
        assert len(img_names['unstable']) % 2 == 0
        n = len(img_names['unstable']) // 2
        for img_name in img_names['unstable'][:n]:
            match = regexp_flow.match(img_name)
            sample['flow'].append(fetch_flow_npy(self.opt, match['video_name'], int(match['idx']) - 1))
        # padding and gen initial mask
        fm_masks = []
        for i in img_names['fm']:
            fm = fetch_fm(self.root, i)
            fm = fm[:self.max_matches, ...]
            fm_mask = np.pad(np.ones(fm.shape[0]),  (0, self.max_matches - fm.shape[0]), mode='constant').astype(np.bool)
            fm = np.pad(fm, ((0, self.max_matches - fm.shape[0]), (0,0)), mode='constant').astype(np.float32)
            sample['fm'].append(fm)
            fm_masks.append(fm_mask)

        if random.uniform(0, 1) < self.opt.fake_rate:
            self.rand()
            for i in range(len(fm_masks)):
                fm_masks[i] = np.logical_and(fm_masks[i], False)
            base_img = sample['prefix'][0]
            offset = random.uniform(-7, 7)
            for i in range(len(sample['prefix'])):
                sample['prefix'][i] = self.move(base_img, len(sample['prefix']) - i + offset)
            for i in range(len(sample['target'])):
                sample["target"][i] = self.move(base_img, -i + offset)
                sample['unstable'][i] = base_img
            sample['flow'] = []
        if self.transform:
            sample = self.transform(sample)
        # gen final mask
        for idx in range(len(sample['fm'])):
            fm = sample['fm'][idx]
            fm_mask = np.logical_and(fm_masks[idx], np.all(np.logical_and(fm >= -1, fm <= 1), axis=1))
            sample['fm_mask'].append(torch.from_numpy(fm_mask.astype(np.float32)))
        sample = Data(**sample)
        return sample

def get_transform(opt, isTrain):
    transform_list = []
    if isTrain and opt.data_augment:
        input_ratio = opt.height * 1.0 / opt.width
        # if input_ratio > 1: 
            # input_ratio = 1. / input_ratio
        C = input_ratio
        length = C * 2
        l = length / (math.e ** (length / C) - 1)
        # transform_list.append(BatchRandomCrop(ratio=(3. / 4. * input_ratio, 4. / 3. / input_ratio)))
        transform_list.append(BatchRandomCrop(ratio=(l, l + length)))
        transform_list.append(BatchRandomHorizontalFlip())

    transform_list.append(ResizeToTensorAndNormalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225],
                                                     size=[opt.height, opt.width]))
    return transforms.Compose(transform_list)

def create_data_loader(opt):
    train_dataset = VideoDataset(opt, opt.train_source, get_transform(opt, isTrain=opt.isTrain))
    val_dataset = VideoDataset(opt, opt.val_source, get_transform(opt, isTrain=opt.isTrain))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                            shuffle=not opt.serial_batches, num_workers=int(opt.nThreads))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size,
                            shuffle=not opt.serial_batches, num_workers=int(opt.nThreads))
    return train_dataloader, val_dataloader


class BatchRandomCrop(transforms.RandomResizedCrop):
    def __init__(self, size=(1, 1), scale=(0.7, 1.0), ratio=(276. / 512., 300. / 512.), interpolation=Image.BILINEAR):
        super(BatchRandomCrop, self).__init__(size, scale, ratio, interpolation)
        # size is useless
        self.param_updated = False
        self.height = None
        self.width = None

    def get_or_update_params(self, img, *args, **argv):
        if not self.param_updated:
            self.param_updated = True
            self.tmp = self.get_params(img, *args, **argv)
            self.height = img.height
            self.width = img.width
        return self.tmp

    def __call__(self, sample):
        self.param_updated = False
        return self.transform(sample)

    def transform(self, sample):
        res = create_empty_data()._asdict()
        for img_class in ["prefix", "unstable", "target"]:
            for img in sample[img_class]:
                i, j, h, w = self.get_or_update_params(img, self.scale, self.ratio)
                res[img_class].append(F.crop(img, i, j, h, w))
        # i1, j1, h1, w1 = i * FLOW_HEIGHT // ori_height, j * FLOW_WIDTH //ori_width, \
        #                  h * FLOW_HEIGHT // ori_height, w * FLOW_WIDTH //ori_width
        i1, j1, h1, w1 = i, j, h, w
        i, j = norm_axis(i, self.height - 1), norm_axis(j, self.width - 1)
        # 
        h, w = (self.height - 1) / (h - 1), (self.width - 1) / (w - 1)
        for idx in range(len(sample['fm'])):
            fm = (sample['fm'][idx] - [j, i, j, i]) * [w, h, w, h] - 1
            res['fm'].append(fm)
        for idx in range(len(sample['flow'])):
            flow = (sample['flow'][idx] - [j, i]) * [w, h] - 1
            flow = flow[i1:i1+h1, j1:j1+w1, ...]
            res['flow'].append(flow)
        return res

class BatchRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        res = sample
        if random.random() < 0.5:
            res = create_empty_data()._asdict()
            for img_class in ["prefix", "unstable", "target"]:
                for img in sample[img_class]:
                    res[img_class].append(F.hflip(img))
            for i in range(len(sample['fm'])):
                fm = sample['fm'][i] * [-1, 1, -1, 1]
                res['fm'].append(fm)
            for i in range(len(sample['flow'])):
                flow = sample['flow'][i] * [-1, 1]
                # flow -= [1. / flow.shape[1], 0]
                flow = flow[:, ::-1, :]
                res['flow'].append(flow)
        return res

class ResizeToTensorAndNormalize(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, mean, std, size, interpolation=Image.BILINEAR):
        self.mean = mean
        self.std = std
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        res = create_empty_data()._asdict()
        for img_class in ["prefix", "unstable", "target"]:
            for img in sample[img_class]:
                img = F.resize(img, self.size, self.interpolation)
                res[img_class].append(F.normalize(F.to_tensor(img), self.mean, self.std))
        for i in range(len(sample['fm'])):
            fm = sample['fm'][i]
            res['fm'].append(torch.from_numpy(fm.astype(np.float32)))
        for i in range(len(sample['flow'])):
            flow = sample['flow'][i]
            flow = np.stack([
                scipy.misc.imresize(flow[..., 0], self.size, mode='F'),
                scipy.misc.imresize(flow[..., 1], self.size, mode='F'),
            ], axis=2)
            flow = torch.from_numpy(flow.astype(np.float32))
            res['flow'].append(flow)
        return res

