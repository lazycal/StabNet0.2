import random
from PIL import Image
import json
from collections import namedtuple, Iterable
import os
import scipy.io
import re
import numpy as np
from copy import deepcopy
import math
import struct
import traceback
import scipy.misc
from multiprocessing import Pool

regexp_flow = re.compile(r'unstable/(?P<video_name>\d+)/image-(?P<idx>\d+)\.(png|jpg)')
BaseDir = './flow'
OutputDir = './flow-npy'
FLOW_HEIGHT = 288
FLOW_WIDTH = 512

def fetch_flow(flowdata, video_name, idx):
    height = FLOW_HEIGHT
    width = FLOW_WIDTH
    float_cnt = 4
    cnt = 2 * height * width * float_cnt * idx
    #calc flow_x
    flow = np.zeros((height, width, 2), dtype=np.float32)
    for xx in range(height):
        for yy in range(width):
            bit = float(struct.unpack('f', flowdata[cnt:cnt+float_cnt])[0])
            cnt += float_cnt
            flow[xx, yy, 0]=bit + yy
    flow[:, :, 0] = flow[:, :, 0] / (width - 1) * 2 - 1
    #calc flow_y
    for xx in range(height):
        for yy in range(width):
            bit = float(struct.unpack('f', flowdata[cnt:cnt+float_cnt])[0])
            cnt += float_cnt
            flow[xx, yy, 1]=bit + xx
    flow[:, :, 1] = flow[:, :, 1] / (height - 1) * 2 - 1
    # flow = np.stack([
    #     scipy.misc.imresize(flow[..., 0], (ori_height, ori_width), mode='F'),
    #     scipy.misc.imresize(flow[..., 1], (ori_height, ori_width), mode='F'),
    # ], axis=2)
    return flow
def run(i):
    print('=======>'+str(i))
    parent = os.path.join(OutputDir, str(i))
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(os.path.join(BaseDir, str(i) + '.bin'), 'rb') as flowfile:
        flowdata = flowfile.read()
    for j in range(0, 999999):
        try:
            flow = fetch_flow(flowdata, str(i), j)
        except:
            traceback.print_exc()
            break
        path = os.path.join(parent, '{:04d}.npy'.format(j))
        np.save(path, flow)

p = Pool(24)
videos = range(1, 45)
p.map(run, videos)
