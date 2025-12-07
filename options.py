import argparse
import os
from datetime import datetime
import time
import torch
import random
import numpy as np
import sys


class Options(object):
    """docstring for Options"""

    def __init__(self):
        super(Options, self).__init__()

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--mode', type=str, default='train', help='Mode of code. [train|test]')
        parser.add_argument('--model', type=str, default='convnext_tiny',
                            help='[resnet18|MMNet], see model.__init__ from more details.')
        parser.add_argument('--input_type', type=str, default='flow', help='[flow|apex|on_apex]')
        parser.add_argument('--loso', type=bool, default=True, help='loso.')
        parser.add_argument('--dataset', type=str, default='cde', help='name of dataset.[casme2|samm|smic|mmew|cde]')
        parser.add_argument('--num_classes', type=int, default=3, help='num_classes.')
        parser.add_argument('--pretrained', type=str, default='imagenet', help='[imagenet|miex].')
        parser.add_argument('--lucky_seed', type=int, default=42, help='seed for random initialize, 0 to use current time.')

        parser.add_argument('--data_root', default="/data/xiaohang/ME_DATA/", help='paths to data set.')
        parser.add_argument('--data_apex_frame_path', default="datasets/cde_apex_optical_flow.csv")
        parser.add_argument('--data_n_frames_path', default="datasets/cde_4_frames_optical_flow.csv")
        parser.add_argument('--scale_factor', type=float, default=1.0)
        parser.add_argument('--use_ema', type=bool, default=False, help='use EMA.')
        parser.add_argument('--ema_decay', type=float, default=0.9, help='EMA decay.')

        parser.add_argument('--sub_val', default=0, type=int, help='the subject used to validate in LOSO')

        parser.add_argument('--batch_size', type=int, default=32, help='input batch size.')
        parser.add_argument('--n_workers', type=int, default=4, help='number of workers to load data.')

        parser.add_argument('--gpu_ids', type=str, default='7', help='gpu ids, eg. 0,1,2; -1 for cpu.')
        parser.add_argument('--ckpt_dir', type=str, default="/NAS/xiaohang/MER/checkpoints", help='directory to save check points.')
        parser.add_argument('--results', type=str, default='./results', help='directory to save result.')
        parser.add_argument('--log_file', type=str, default="logs.txt", help='logs file')
        parser.add_argument('--opt_file', type=str, default="opt.txt", help='options file')

        # train options
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--norm', type=str, default='batch',
                            help='instance normalization or batch normalization [batch|instance|none] resnet:none==batch')
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer: adam|adamw')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay term of adam|adamw')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        parser.add_argument('--lr_policy', type=str, default='exponential',
                            help='learning rate policy: lambda|step|plateau|cosine|exponential')
        parser.add_argument('--exponential_gamma', type=float, default=0.987)

        parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run')

        return parser

    def parse(self):
        parser = self.initialize()
        parser.set_defaults(name=datetime.now().strftime("%y%m%d_%H%M%S"))
        opt = parser.parse_args()

        # update checkpoint dir
        if opt.mode == 'train':
            opt.ckpt_dir = os.path.join(opt.ckpt_dir, opt.dataset, opt.model, opt.name)
            if not os.path.exists(opt.ckpt_dir):
                os.makedirs(opt.ckpt_dir)

        if not os.path.exists(opt.results):
            os.makedirs(opt.results)

        # set gpu device
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            cur_id = int(str_id)
            if cur_id >= 0:
                opt.gpu_ids.append(cur_id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        opt.device = torch.device('cuda:%d' % opt.gpu_ids[0] if opt.gpu_ids else 'cpu')
        print('device:', opt.device)

        # set seed
        if opt.lucky_seed == 0:
            opt.lucky_seed = int(time.time())
        random.seed(a=opt.lucky_seed)
        np.random.seed(seed=opt.lucky_seed)
        torch.manual_seed(opt.lucky_seed)
        if len(opt.gpu_ids) > 0:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(opt.lucky_seed)
            torch.cuda.manual_seed_all(opt.lucky_seed)

        # write command to file
        script_dir = opt.ckpt_dir
        with open(os.path.join(os.path.join(script_dir, "run_script.sh")), 'a+') as f:
            f.write("[%5s][%s]python %s\n" % (opt.mode, opt.name, ' '.join(sys.argv)))

        # print and write options file
        msg = ''
        msg += '------------------- [%5s][%s]Options --------------------\n' % (opt.mode, opt.name)
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_v = parser.get_default(k)
            if v != default_v:
                comment = '\t[default: %s]' % str(default_v)
            msg += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        msg += '--------------------- [%5s][%s]End ----------------------\n' % (opt.mode, opt.name)
        print(msg)
        with open(os.path.join(os.path.join(script_dir, "opt.txt")), 'a+') as f:
            f.write(msg + '\n\n')

        return opt


if __name__ == '__main__':
    opt = Options().parse()
    print(type(vars(opt)))
    print(vars(opt))
