import torch

from .ops import *
from . import pack as torch_pack

################################################################
# String to torch.nn.Module
################################################################
module_map = {
    'fc'          : torch.nn.Linear,
    'linear'      : torch.nn.Linear,
    'conv2d'      : torch.nn.Conv2d,

    'maxpool2d'   : torch.nn.MaxPool2d,
    'avgpool2d'   : torch.nn.AvgPool2d,
    
    'batchnorm2d' : torch.nn.BatchNorm2d,

    'relu'        : torch.nn.ReLU,

    'flatten'     : torch.nn.Flatten
}

################################################################
# String to argument packing for a torch.nn.Module
################################################################
pack_config_map = {
    'fc'          : torch_pack.transforms.pack_linear_config,
    'linear'      : torch_pack.transforms.pack_linear_config,
    'conv2d'      : torch_pack.transforms.pack_conv2d_config,
    
    'maxpool2d'   : torch_pack.pooling.pack_maxpool2d_config,
    'avgpool2d'   : torch_pack.pooling.pack_avgpool2d_config,
    
    'batchnorm2d' : torch_pack.norms.pack_batchnorm2d_config,

    'relu'        : torch_pack.activations.pack_relu_config,

    'flatten'     : torch_pack.utility.pack_flatten_config
}