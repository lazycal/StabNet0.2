import argparse

import torch.utils.data
import random
import torchvision.transforms as transforms
import torch
from PIL import Image
import json
from collections import namedtuple, Iterable
from torchvision.transforms import functional as F
from torch.autograd import Variable
import os
import scipy.io
import re
import numpy as np
import time
import math
import cv2

from data_loader import create_empty_data, Data, get_transform, map_data
from test_options import TestOptions
from train import create_model, visualize
from util.util import AverageMeter
from util.util import tensor2im


def get_images(video_dir):
    videoCap = cv2.VideoCapture(video_dir)
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
    if int(major_ver) < 3:
        fps = videoCap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = videoCap.get(cv2.CAP_PROP_FPS)
    img_list = []
    success, image = videoCap.read()
    while success:
        image = np.stack((image[..., 2], image[... ,1], image[..., 0]), axis=2)
        img = Image.fromarray(np.array(image), 'RGB')
        img_list.append(img)
        success, image = videoCap.read()
    # print(img_list[0])
    return img_list, fps


def generate_video(images, fps, opt):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(os.path.join(expr_dir, "{}.avi".format(opt.video_index)), 
        fourcc, fps, (images[0].shape[1], images[0].shape[0]))
    for image in images:
        # print(image)
        # print(image[0])
        # print(image[0].data)
        # print(img)
        # print(image.shape)
        # print(opt.width)
        # print(opt.height)
        # out.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        out.write(np.stack((image[..., 2], image[..., 1], image[..., 0]), axis=2))
    out.release()

def draw_imgs(net_output, stable_frame, unstable_frame, last_frame):
    net_output = np.array(net_output)
    stable_frame = cv2.resize(np.array(stable_frame), (net_output.shape[1], net_output.shape[0]))
    unstable_frame = cv2.resize(np.array(unstable_frame), (net_output.shape[1], net_output.shape[0]))
    last_frame = cv2.resize(np.array(last_frame), (net_output.shape[1], net_output.shape[0]))
    net_output = cv2.resize(net_output, (net_output.shape[1], net_output.shape[0]))
    output_minus_input  = abs(net_output*1. - unstable_frame).astype(np.uint8)
    output_minus_stable = abs(net_output*1. - stable_frame).astype(np.uint8)
    output_minus_last   = abs(net_output*1. - last_frame).astype(np.uint8)
    img_top    = np.concatenate([net_output,         output_minus_stable], axis=1)
    img_bottom = np.concatenate([output_minus_input, output_minus_last], axis=1)
    img = np.concatenate([img_top, img_bottom], axis=0).astype(np.uint8)
    return cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def output_to_input(image, opt):
    img = tensor2im(image[0].data, imtype=np.float)
    img = ((np.reshape(img, (opt.height, opt.width, 3))) * 255).astype(np.uint8)
    image = Image.fromarray(np.array(img), 'RGB')
    return image


class PreprocessDataSet(torch.utils.data.Dataset):
    def __init__(self, stable_frames, unstable_frames, pred_frames, opt):
        self.stable_frames = stable_frames
        self.unstable_frames = unstable_frames
        self.opt = opt
        self.pred_frames = pred_frames
        if opt.fake_test:
            self.base_img = self.unstable_frames[0]
            self.rand()

    def __len__(self):
        return 1

    def add(self, frame):
        self.pred_frames.append(frame)

    def rand(self):
        self.ang = random.uniform(0, math.pi * 2)
        # self.ang = 0
        self.vel = random.uniform(0, self.opt.fake_vel)
        self.offset = random.uniform(-3, 3)

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
        return img

    def __getitem__(self, idx):
        sample = create_empty_data()._asdict()
        cur = len(self.pred_frames)
        if self.opt.fake_test:
            for i in range(0, len(self.opt.prefix)):
                j = self.opt.prefix[- 1 - i]
                sample['prefix'].append(self.move(self.base_img, len(self.opt.prefix) - i + self.offset))

            sample['unstable'].append(self.move(self.base_img, random.uniform(-5, 5)))
            sample["target"].append(self.move(self.unstable_frames[0], self.offset))
        else:
            for i in range(0, len(self.opt.prefix)):
                j = self.opt.prefix[len(self.opt.prefix) - 1 - i]
                if j > len(self.pred_frames):
                    sample["prefix"].append(self.unstable_frames[0])
                else:
                    sample["prefix"].append(self.pred_frames[len(self.pred_frames) - j])
            sample["unstable"].append(self.unstable_frames[len(self.pred_frames)])
            sample["target"].append(self.stable_frames[len(self.pred_frames)])
        sample = get_transform(self.opt, isTrain=self.opt.isTrain)(sample)
        sample = Data(**sample)
        return sample


def main():
    opt = TestOptions().parse()
    # preprocess data
    all_stable_frames, fps = get_images(opt.video_root + 'stable/' + str(opt.video_index) + '.avi')
    all_unstable_frames, fps = get_images(opt.video_root + 'unstable/' + str(opt.video_index) + '.avi')

    # generate data flow
    pred_frames_for_input = []
    singleVideoData = PreprocessDataSet(all_stable_frames, all_unstable_frames, pred_frames_for_input, opt)
    eval_data_loader = torch.utils.data.DataLoader(singleVideoData)
    model, criterion = create_model(opt)
    checkpoint = torch.load(opt.checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    data_time = AverageMeter()
    end = time.time()
    # go through model to get output
    idx = 0
    pred_frames = []
    if opt.instnorm:
        model.train()
    else:
        model.eval()
    if opt.fake_test:
        print("fake test")
        pred_frames = []
        for i in range(50):
            for j, data in enumerate(eval_data_loader):
                if opt.gpu_ids:
                    data = map_data(lambda x: Variable(x.cuda(), volatile=True), data)
                else:
                    data = map_data(lambda x: Variable(x, volatile=True), data)
                data_time.update(time.time() - end)
                data = Data(*data)
                output = model.forward(data)
                warpped = output.warpped
                pred_frames += data.prefix
                for u, w, t in zip(data.unstable, warpped, data.target):
                    pred_frames += (u, w, torch.abs(w - t))
                # visualize(data, warpped, i, 0, opt, 'save')

        pred_frames = list(map(lambda x: tensor2im(x.data), pred_frames))

    else:
        for i in range(0, len(all_stable_frames) - 1):
            if i % 100 == 0: print("=====> %d/%d"%(i, len(all_stable_frames)))
            for j, data in enumerate(eval_data_loader):
                if opt.gpu_ids:
                    data = map_data(lambda x: Variable(x.cuda(), volatile=True), data)
                else:
                    data = map_data(lambda x: Variable(x, volatile=True), data)
                data_time.update(time.time() - end)
                data = Data(*data)
                # print(data)
                output = model.forward(data)
                warpped = output.warpped
                # save outputs
                # if (i < opt.prefix[0]):
                #     last_frame = all_stable_frames[0]
                # else:
                #     last_frame = pred_frames_for_input[len(pred_frames_for_input) + 1 - opt.prefix[0]]
                # print(data.prefix[-1][0].data.shape)
                last_frame = output_to_input([data.prefix[-1]], opt)
                pred_frames.append(draw_imgs(output_to_input(warpped, opt), all_stable_frames[i], all_unstable_frames[i], last_frame))
                pred_frames_for_input.append(output_to_input(warpped, opt))
                eval_data_loader = torch.utils.data.DataLoader(PreprocessDataSet(all_stable_frames, all_unstable_frames, pred_frames_for_input, opt))
                # if i < 100: visualize(data, warpped, i, 0, opt, 'save')

    # print video
    generate_video(pred_frames, fps, opt)


if __name__ == '__main__':
    main()
