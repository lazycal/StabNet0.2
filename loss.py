from torch import nn
import torch
from torch.autograd import Variable
EPS = 1e-10

class ZeroLoss(object):
    def __call__(self, *args):
        return 0

class PixelLoss(object):
    def __init__(self, use_l1=True):
        self.name = 'PixelLoss'
        if use_l1:
            self.loss = nn.L1Loss(reduce=False)
        else:
            self.loss = nn.MSELoss(reduce=False)

    @staticmethod
    def to_gray(x):
        return x[:, 0, :, :] * 0.299 + x[:, 1, :, :] * 0.587 + x[:, 2, :, :] * 0.114

    def __call__(self, net_output, data):
        return self._subcall(net_output.warpped, net_output.mask, data.target)

    def _subcall(self, warpped, mask, target):
        output = warpped
        output = list(map(PixelLoss.to_gray, output))
        target = list(map(PixelLoss.to_gray, target))
        mask   = (torch.cat(mask, dim=1) == 1).float().detach()
        # print('mask.shape={}, output[0].shape={}'.format(mask.shape, output[0].shape))
        # print(mask, mask.mean())
        loss_per_pix = self.loss(torch.stack(output, dim=1), torch.stack(target, dim=1)) * mask
        self.loss_val = (loss_per_pix.sum(dim=3).sum(dim=2) / (mask.sum(dim=3).sum(dim=2) + EPS) ).mean()
        # print('loss_per_pix.max={}, loss_per_pix.min={}, loss_per_pix.mean={}'.format(loss_per_pix.max(), loss_per_pix.min(), loss_per_pix.mean()))
        # print('output.max={}, output.min={}, output.mean={}'.format(output[0].max(), output[0].min(), output[0].mean()))
        return self.loss_val


class IdentityLoss(object):
    def __init__(self):
        self.name = 'IdentityLoss'

    def __call__(self, output, data):
        thetas = output.thetas
        target = Variable(thetas[0].data.new([1, 0, 0, 0, 1, 0]))
        self.loss_val = torch.mean(torch.abs(torch.cat(thetas, dim=0) - target))
        return self.loss_val

class FeatureLoss(object):
    def __init__(self):
        self.name = 'FeatureLoss'

    def __call__(self, output, data):
        mask = data.fm_mask
        thetas = output.thetas
        err = 0
        for i in range(len(thetas)):
            theta = thetas[i].view((-1, 2, 3))
            w = theta[:, :, :2].transpose(1, 2)
            b = theta[:, :, 2]
            # print('theta.shape={}'.format(theta.shape))
            # print('w={},b={}'.format(w,b))
            # print('torch.matmul(data.fm[i][:, :, :2], w).shape={}'.format(torch.matmul(data.fm[i][:, :, :2], w).shape))
            stable = torch.matmul(data.fm[i][:, :, :2], w) + b[:, None, :]
            unstable = data.fm[i][:, :, 2:]
            # print('stable.shape={}'.format(stable.shape))
            # print('stable-unstable={}'.format(stable-unstable))
            batch_err = torch.sum(torch.mean(torch.abs(stable - unstable), dim=2) * mask[i], dim=1) \
                         / (torch.sum(mask[i], dim=1) + 1e-10)
            # print('batch_err={}'.format(batch_err))
            err += torch.mean(batch_err)
        return err

class Loss(object):
    def __init__(self, opt):
        self.opt = opt
        self.pixelLoss = PixelLoss(not opt.use_l2)# if opt.pix_loss_weight != 0 else ZeroLoss()
        self.identityLoss = IdentityLoss()# if opt.id_loss_weight != 0 else ZeroLoss()
        self.featureLoss = FeatureLoss()# if opt.feature_loss_weight != 0 else ZeroLoss()
        self.loss_val = {}
        self.name = 'Loss'

    def __call__(self, output, data):
        total_loss = 0
        self.loss_val['PixelLoss'] = self.pixelLoss(output, data)
        self.loss_val['IdentityLoss'] = self.identityLoss(output, data)
        self.loss_val['FeatureLoss'] = self.featureLoss(output, data)
        total_loss = self.loss_val['PixelLoss'] * self.opt.pix_loss_weight + \
                    self.loss_val['IdentityLoss'] * self.opt.id_loss_weight + \
                    self.loss_val['FeatureLoss'] * self.opt.feature_loss_weight

        self.loss_val['Loss'] = total_loss
        return total_loss

    def summary(self):
        return {k:v.data.cpu().float().numpy()[0] if hasattr(v, 'data') else v \
                for k, v in self.loss_val.items()}
