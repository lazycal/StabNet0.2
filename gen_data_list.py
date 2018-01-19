import sys, random, os
import argparse
from collections import namedtuple
# from data_loader import Data, create_empty_data
import json

parser = argparse.ArgumentParser()
parser.add_argument('--root', default = '/home/lazycal/workspace/qudou/frames')
parser.add_argument('--prefix', default = [6, 12, 18, 24, 30], type=int, nargs='+')
parser.add_argument('--suffix', default = [0, 6, 12, 18, 24], type=int, nargs='+')
parser.add_argument('--num', default = -1, type=int)
parser.add_argument('--filename', default = '', help = 'Suffix that appends to generated file name.')
parser.add_argument('--val-list', default = None, type=int, nargs='*')
args = parser.parse_args()
Data = namedtuple('Data', ['prefix', 'unstable', 'target', 'fm'])

def create_empty_data():
    return Data(prefix=[], unstable=[], target=[], fm=[])
def stab(vid):
    return os.path.join(args.root, 'stable', vid)
def unst(vid):
    return os.path.join(args.root, 'unstable', vid)
def imna(iid):
    return 'image-{:04d}.png'.format(iid + 1)

def gen_samples(vid):
    stab_frame_root = stab(vid)
    # unst_frame_root = unst(vid)
    frame_list = []
    frame_fm_list = []
    for i in range(0, 10000):
        img_name = imna(i)
        if not os.path.exists(os.path.join(stab_frame_root, img_name)):
            break
        frame_list.append(os.path.join(vid, img_name))
        frame_fm_list.append(os.path.join(vid, '{:04d}.mat'.format(i)))
    n = len(frame_list)
    res = []
    for i in range(args.prefix[-1], n - args.suffix[-1]):
        asample = create_empty_data()
        for p in args.prefix[::-1]:
            asample.prefix.append('stable/' + frame_list[i - p])
        for s in args.suffix:
            asample.unstable.append('unstable/' + frame_list[i + s])
            asample.target.append('stable/' + frame_list[i + s])
            asample.fm.append(frame_fm_list[i + s])
        res.append(asample._asdict())
    return res

def main():
    videos = list(filter(lambda x: x.isdigit(), os.listdir(os.path.join(args.root, 'stable'))))
    n = len(videos)
    if args.val_list is None:
        random.shuffle(videos)
        m = int(n * 0.9)
        train_videos = videos[:m]
        val_videos = videos[m:]
    else:
        val_set = set(args.val_list)
        val_videos = list(filter(lambda x: int(x) in val_set, videos))
        train_videos = list(filter(lambda x: int(x) not in val_set, videos))
        m = len(val_videos)
    print('train_videos={}'.format(train_videos))
    print('val_videos={}'.format(val_videos))
    train_list = []
    for i in train_videos:
        train_list += gen_samples(i)
    if args.num != -1:
        train_list = train_list[:args.num]
    with open('train-list{}.txt'.format('-' + args.filename if args.filename else ''), 'w') as fout:
        json.dump(train_list, fout, indent=2)
    val_list = []
    for i in val_videos:
        val_list += gen_samples(i)
    if args.num != -1:
        val_list = train_list[:args.num]
    with open('val-list{}.txt'.format('-' + args.filename if args.filename else ''), 'w') as fout:
        json.dump(val_list, fout, indent=2)
    print('len_train_list={}\nlen_val_list={}'.format(len(train_list), len(val_list)))

if __name__ == '__main__':
    main()