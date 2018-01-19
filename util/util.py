from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.float):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * std + mean
    return image_numpy.astype(imtype)

def train2show(image_batch):
    assert(image_batch.dim() == 4)
    assert(image_batch.shape[1] == 3)
    image_batch = image_batch.permute(0, 2, 3, 1)
    # image_numpy = image_tensor[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    mean = image_batch.new([0.485, 0.456, 0.406])
    std = image_batch.new([0.229, 0.224, 0.225])
    image_batch = image_batch * std + mean
    return image_batch.permute(0, 3, 1, 2)


def diagnose_network(net, name='network'):
    # print('------- diagnose_network begin -------')
    mean = 0.0
    count = 0
    for param in net.parameters():
        # print(param.data)
        # if param.grad is not None:
        mean += torch.mean(torch.abs(param.data))
        count += 1
    if count > 0:
        mean = mean / count
    print('diagnose: {}:{}'.format(name, mean))
    # print('------- diagnose_network end -------')


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(state, is_best, expr_dir, filename='checkpoint.pth.tar'):
    import shutil
    filename = os.path.join(expr_dir, filename + '-' + str(state['epoch'] + 1))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(expr_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DictAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = 0

    def update(self, val, n=1):
        self.count += n
        for k, v in val.items():
            self.val[k] = v
            self.sum[k] = self.sum.get(k, 0) + v * n
            self.avg[k] = self.sum[k] / self.count
