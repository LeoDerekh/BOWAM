import torch
import os
import sys
current_working_directory = os.getcwd()
# print("Current working directory:", current_working_directory)
sys.path.insert(0, current_working_directory)
sys.path.append(current_working_directory)
from models import convnext_timm, convnext_tiny, convnext, resnet
from models.SFFNet.mymodel import MyModel
from models.mymodel2 import MyModel2
from models.test.backbone_vit_timm import vit_small_patch16_224
from models.test.testmodel import TestModel
from models.test import backbone_convnext_timm, dualmodel
from models.test.wavevit import wavevit_s

def init_net(net, gpu_ids):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net


def get_model(opt):
    if opt.model == 'resnet18':
        net = resnet.resnet18(opt)
    elif opt.model == 'resnet50':
        net = resnet.resnet50(opt)
    elif opt.model == 'convnext_tiny':
        net = convnext_timm.convnext_tiny(opt)
        # net = backbone_convnext_timm.convnext_tiny(opt)
    elif opt.model == 'mymodel':
        net = MyModel(opt)
    elif opt.model == 'mymodel2':
        net = MyModel2(opt)
    elif opt.model == 'testmodel':
        net = TestModel(opt)
    elif opt.model == 'dualmodel':
        net = dualmodel.DualModel(opt)
    elif opt.model == 'vit_s':
        net = vit_small_patch16_224(opt)
    elif opt.model == 'wave_vit':
        net = wavevit_s(opt)

    else:
        raise NotImplementedError('model [%s] is not implemented' % opt.model)
    return init_net(net, opt.gpu_ids)
