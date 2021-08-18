import argparse
import os
import torch
import util.utils as util

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataset_name', default='JSTL', help='SHA|SHB|QNRF')
        parser.add_argument('--base_mae', type=str, default='64,15,110,100', help='something like, 64,15,110,100')
        parser.add_argument('--batch_size', type=int, default=5, help='input batch size')
        parser.add_argument('--net_name', type=str, default='res_unet_aspp', help='res_unet|res_unet_leaky')
        parser.add_argument('--fine_size', type=int, default=400, help='cropped size')
        parser.add_argument('--name', type=str, default='Res_unet_aspp', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
        parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./output', help='models are saved here')
        parser.add_argument('--mode', type=str, default='crop', help='crop')
        parser.add_argument('--is_flip', type=int, default=1, help='whether perform flipping data augmentation')
        parser.add_argument('--is_random_hsi', type=int, default=1, help='whether perform random hsi data augmentation')
        parser.add_argument('--is_color_aug', type=int, default=1, help='whether perform color augmentation')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer [sgd|adam|adamW]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.01, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay of adam')
        parser.add_argument('--amsgrad', type=int, default=0, help='weight using amsgrad of adam')
        parser.add_argument('--eval_per_step', type=int, default=float("inf"), help='When detailed change super-parameter, may need it, step of evaluation')
        parser.add_argument('--eval_per_epoch', type=int, default=1, help='epoch step of evaluation')
        parser.add_argument('--start_eval_epoch', type=int, default=100, help='beginning epoch of evaluation')
        parser.add_argument('--print_step', type=int, default=10, help='print step of loss')
        parser.add_argument('--max_epochs', type=int, default=500, help='Epochs of training')
        self.initialized = True
        return parser

    def gather_options(self, options=None):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)


        self.parser = parser
        if options == None:
            return parser.parse_args()
        else:
            return parser.parse_args(options)

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.dataset_name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, options=None):

        opt = self.gather_options(options=options)
        opt.isTrain = self.isTrain   # train or test


        self.print_options(opt)

        # set gpu ids
        os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # re-order gpu ids
        opt.gpu_ids = [i.item() for i in torch.arange(len(opt.gpu_ids))]
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
