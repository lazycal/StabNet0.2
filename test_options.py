from options import Options


class TestOptions(Options):
    def initialize(self):
        Options.initialize(self)
        self.parser.add_argument('--checkpoint_path', default = './checkpoints/lstm-0.0001/checkpoint.pth.tar-9')
        self.parser.add_argument('--video_root', default='/home/cxjyxx_me/workspace/spatial-transformer-tensorflow/data_video/')
        self.parser.add_argument('--video_index', default=1, type=int)
        self.parser.add_argument('--prefix', default=[6, 12, 18, 24, 30], type=int, nargs='+')
        self.parser.add_argument('--fake_test', action='store_true')
        self.parser.add_argument('--instnorm', action='store_true')