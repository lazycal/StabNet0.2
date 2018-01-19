import time
from options import Options
from data_loader import create_data_loader, Data, map_data, create_empty_data
from util.util import AverageMeter, DictAverageMeter
from util import util
from models import models
# from util.visualizer import Visualizer
import torchvision.utils
import torch
from torch.autograd import Variable
import os
import loss
import numpy as np
import cv2
import sys
import tensorboard_logger as tl
import torch.nn.functional as F

def to_gray(x):
    return x[:, 0, :, :] * 0.299 + x[:, 1, :, :] * 0.587 + x[:, 2, :, :] * 0.114

def draw_flow(warpped1, warpped2, flow):
    warpped_target = F.grid_sample(warpped2, flow).data
    return torch.abs(warpped_target - warpped1)

def visualize(data2, output2, global_step, sid, opt, output1=None, flow=None, mode='both', name=''):

    def draw(img, pts, mask, color=None):
        res = img.copy()
        assert(pts.shape[0] == opt.max_matches)
        assert(mask.shape[0] == opt.max_matches)
        pts = (pts / 2 + .5) * (img.shape[:2])[::-1]
        # print('pts={}'.format(pts))
        pts = pts.astype(np.int32)
        for i in range(pts.shape[0]):
            if not mask[i]: continue
            cv2.circle(res, tuple(pts[i]), 5, tuple(np.random.rand(3)) if color is None else color)
        return res
    warpped2 = output2.warpped
    mask2 = output2.mask
    prefix = util.train2show(torch.stack(list(map(lambda x: x[sid], data2.prefix)), dim=0).data)
    unstable = util.train2show(torch.stack(list(map(lambda x: x[sid], data2.unstable)), dim=0).data)
    target = util.train2show(torch.stack(list(map(lambda x: x[sid], data2.target)), dim=0).data)
    warpped2 = util.train2show(torch.stack(list(map(lambda x: x[sid], warpped2)), dim=0).data)
    mask2 = (torch.stack(list(map(lambda x: x[sid], mask2)), dim=0).data > .9).float()
    if flow is not None: 
        warpped1 = output1.warpped
        warpped1 = util.train2show(torch.stack(list(map(lambda x: x[sid], warpped1)), dim=0).data)
        flow = torch.stack(list(map(lambda x: x[sid], flow)), dim=0).data
        diff_flow = draw_flow(warpped1, warpped2, flow)
    diff = torch.abs(warpped2 - target)
    diff = to_gray(diff)
    diff = torch.stack([diff, diff, diff], dim=1)
    fm = list(map(lambda x: x.data, data2.fm))
    fm_mask = list(map(lambda x: x.data, data2.fm_mask))
    for i in range(len(fm)):
        pts = fm[i][sid].cpu().numpy()
        a_fm_mask = fm_mask[i][sid].cpu().numpy()
        img = target[i].cpu().numpy().transpose([1, 2, 0])
        # print(img.shape, pts.shape, a_fm_mask.shape)
        # print('a_fm_mask={}'.format(a_fm_mask))
        img = draw(img, pts[:, :2], a_fm_mask) # stable
        target[i].copy_(torch.from_numpy(img.transpose([2, 0, 1])))
        img = unstable[i].cpu().numpy().transpose([1, 2, 0])
        img = draw(img, pts[:, 2:], a_fm_mask) # unstable
        unstable[i].copy_(torch.from_numpy(img.transpose([2, 0, 1])))
    if flow is not None:
        vis = torch.cat(
            (unstable, warpped2, target, diff, diff_flow),
            dim=0
        )
    else:
        vis = torch.cat(
            (unstable, warpped2, target, diff),
            dim=0
        )
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    prefix_grid = torchvision.utils.make_grid(prefix, nrow=1)
    vis_grid = torchvision.utils.make_grid(vis, nrow=vis.shape[0] // 4)
    if name != '': name += '-'
    if mode == 'both' or mode == 'save':
        torchvision.utils.save_image(prefix, 
            os.path.join(expr_dir, name + 'prefix-{:0>4}-{:0>3}.png'.format(global_step, sid)), nrow=1)
        torchvision.utils.save_image(vis, 
            os.path.join(expr_dir, name + 'input-output-target{:0>4}-{:0>3}.png'.format(global_step, sid)), nrow=vis.shape[0] // 4)
    if name != '': name = name[:-1] + '/'
    if mode == 'both' or mode == 'log':
        tl.log_images(name + 'prefix/{}'.format(sid), [prefix_grid.cpu().numpy()], step=global_step)
        tl.log_images(name + 'input-output-target/{}'.format(sid), [vis_grid.cpu().numpy()], step=global_step)
        tl.log_images(name + 'diff/{}'.format(sid), diff[:, 0, ...].cpu().numpy(), step=global_step)
        tl.log_images(name + 'mask/{}'.format(sid), mask2[:, 0, ...].cpu().numpy(), step=global_step)
        if flow is not None:
            tl.log_images(name + 'flow-diff/{}'.format(sid), diff_flow[:, 0, ...].cpu().numpy(), step=global_step)

def train(epoch):
    global global_step, criterion
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_dataloader):
        criterion.id_loss_weight = 0 if global_step >= opt.id_loss_step else opt.id_loss_weight
        # measure data loading time
        if opt.gpu_ids:
            data = map_data(lambda x: Variable(x.cuda()), data)
        else:
            data = map_data(lambda x: Variable(x), data)

        data_time.update(time.time() - end)
        data1 = {}
        data2 = {}
        data = data._asdict()
        for class_name in ['prefix', 'unstable', 'target', 'fm', 'fm_mask']:
            size = len(data[class_name]) // 2
            data1[class_name] = data[class_name][:size]
            data2[class_name] = data[class_name][size:]
          
        data = Data(**data)
        data1 = Data(**data1, flow=[])
        data2 = Data(**data2, flow=[])
        output1 = model.forward(data1)
        output2 = model.forward(data2)
        loss = criterion(output1, output2, data.flow, data1, data2)
        # measure accuracy and record loss
        losses.update(loss.data[0], opt.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if (global_step + 1) % opt.print_freq == 0:
            all_loss = criterion.summary()
            util.diagnose_network(model.cnn)
            util.diagnose_network(model.fc_loc)
            visualize(data2, output2, global_step, 0, opt, output1, data.flow, 
                mode='save', name='train2')
            visualize(data1, output1, global_step, 0, opt,
                mode='save', name='train1')
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Learning Rate {learning_rate}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\n\t'
                  'ALl Loss {all_loss}'.format(
                   epoch, i, len(train_dataloader), batch_time=batch_time, learning_rate=scheduler.get_lr(),
                   data_time=data_time, loss=losses, all_loss=all_loss))

        if (global_step + 1) % opt.log_freq == 0:
            all_loss = criterion.summary()
            tl.log_value('train/Loss', losses.val, global_step)
            tl.log_value('train/Learning Rate', scheduler.get_lr()[0], global_step)
            # tl.log_value('train/Batch Time', batch_time.val, global_step)
            tl.log_value('train/Data Time', data_time.val, global_step)
            for k, v in all_loss.items():
                tl.log_value('train/loss/' + k, v, global_step)
            for sid in range(data.fm[0].shape[0]):
                visualize(data2, output2, global_step, sid, opt, output1, data.flow,
                    mode='log', name='train2')
                visualize(data1, output1, global_step, sid, opt,
                    mode='log', name='train1')
        
        if (global_step + 1) % opt.val_freq == 0:
            validate(epoch)
            validate(epoch, False)

        global_step += 1
        end = time.time()

def validate(epoch, isEval=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dict_losses = DictAverageMeter()
    # switch to train mode
    evalStr = 'NoEval'
    if isEval:
        model.eval()
        evalStr = ''

    end = time.time()
    for i, data_raw in enumerate(val_dataloader):
        if i == opt.val_iters: break
        data = data_raw
        if opt.gpu_ids:
            data = map_data(lambda x: Variable(x.cuda(), volatile=True), data)
        else:
            data = map_data(lambda x: Variable(x, volatile=True), data)

        data_time.update(time.time() - end)
        data1 = {}
        data2 = {}
        data = data._asdict()
        for class_name in ['prefix', 'unstable', 'target', 'fm', 'fm_mask']:
            size = len(data[class_name]) // 2
            data1[class_name] = data[class_name][:size]
            data2[class_name] = data[class_name][size:]
          
        data = Data(**data)
        data1 = Data(**data1, flow=[])
        data2 = Data(**data2, flow=[])
        output1 = model.forward(data1)
        output2 = model.forward(data2)
        loss = criterion(output1, output2, data.flow, data1, data2)

        # measure accuracy and record loss
        losses.update(loss.data[0], opt.batch_size)
        dict_losses.update(criterion.summary(), opt.batch_size)
    
    all_loss = dict_losses.avg
    print('{evalStr}Validation: Epoch: [{0}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Total Time {1:.3f}\n\t'
            'ALl Loss {all_loss}'.format(
            epoch, time.time() - end, loss=losses, all_loss=all_loss, evalStr=evalStr))
    for sid in range(data.fm[0].shape[0]):
        visualize(data2, output2, global_step, sid, opt, output1, data.flow,
            mode='both', name='{}val2'.format(evalStr))
        visualize(data1, output1, global_step, sid, opt,
            mode='both', name='{}val1'.format(evalStr))
    tl.log_value('{}val/Loss'.format(evalStr), losses.val, global_step)
    tl.log_value('{}val/Learning Rate'.format(evalStr), scheduler.get_lr()[0], global_step)
    # tl.log_value('val/Batch Time', batch_time.val, global_step)
    tl.log_value('{}val/Data Time'.format(evalStr), data_time.val, global_step)
    for k, v in all_loss.items():
        tl.log_value('{}val/loss/'.format(evalStr) + k, v, global_step)
    model.train()
    return losses.val

def create_model(opt):
    if opt.model == 'LRCN':
        model = models.LRCNModel(opt)
    # elif opt.model == 'Simple':
    #     model = models.SimpleModel(opt)
    elif opt.model == 'ConvLSTM':
        model = models.ConvLSTM(opt)
    else:
        raise ValueError('Unrecognized opt.mode={}'.format(opt.model))
    criterion = loss.Loss(opt)
    if opt.gpu_ids:
        model.cuda()
        torch.backends.cudnn.benchmark = True
    return model, criterion

def main():
    global opt, train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, global_step
    opt = Options().parse()
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    tl.configure(expr_dir)
    train_dataloader, val_dataloader = create_data_loader(opt)
    dataset_size = len(train_dataloader)
    print('#training images = %d' % dataset_size)

    model, criterion = create_model(opt)

    print('--------- model begin ----------')
    print(model)
    print('--------- model end ----------')
    print('--------- criterion begin ----------')
    print(criterion)
    print('--------- criterion end ----------')

    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    cnn_params = model.cnn.parameters()
    model.freeze_cnn(True)
    rest_params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam([
        {'params': rest_params, 'lr': opt.lr}
    ])
    if opt.continue_train:
        if os.path.isfile(opt.continue_train):
            print("=> loading checkpoint '{}'".format(opt.continue_train))
            checkpoint = torch.load(opt.continue_train, map_location=lambda storage, loc: storage)
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.continue_train, checkpoint['epoch']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(opt.continue_train))
    if opt.start_epoch is not None:
        start_epoch = opt.start_epoch
        global_step = dataset_size // opt.batch_size * start_epoch
    if start_epoch > opt.freeze_epochs:
        print('finetune enable')
        model.freeze_cnn(False)
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.decay_epochs, opt.lr_decay)
        
    for epoch in range(start_epoch, opt.max_epoch):
        criterion.temp_loss_weight = 0 if epoch < opt.temp_loss_epoch else opt.temp_loss_weight
        # TODO decay lr
        if epoch == opt.freeze_epochs:
            print('finetune enable')
            model.freeze_cnn(False)
            # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
            optimizer.add_param_group({
                'params': cnn_params,
                'lr': opt.lr
            })
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.decay_epochs, opt.lr_decay)
        scheduler.step(epoch)
        epoch_start_time = time.time()
        train(epoch)

        if (epoch + 1) % opt.save_epoch_freq == 0:
            print('saving checkpoint')
            util.save_checkpoint({
                'epoch': epoch,
                'global_step': global_step,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                # 'optimizer' : optimizer.state_dict(),
            }, False, expr_dir)

if __name__ == '__main__':
    main()
