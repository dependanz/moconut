import torch

from . import pack as torch_pack

################################################################
# keyword to torch.nn.Module
################################################################
module_map = {
    'fc'          : torch.nn.Linear,
    'linear'      : torch.nn.Linear,
    'conv1d'      : torch.nn.Conv1d,
    'conv2d'      : torch.nn.Conv2d,
    'convT1d'         : torch.nn.ConvTranspose1d,
    'convtranspose1d' : torch.nn.ConvTranspose1d,

    'maxpool2d'   : torch.nn.MaxPool2d,
    'avgpool2d'   : torch.nn.AvgPool2d,
    
    'batchnorm2d' : torch.nn.BatchNorm2d,

    'relu'        : torch.nn.ReLU,
    'leakyrelu'   : torch.nn.LeakyReLU,
    'tanh'        : torch.nn.Tanh,

    'flatten'     : torch.nn.Flatten
}

################################################################
# String to argument packing for a torch.nn.Module
#
# TODO: Multimap
################################################################
pack_config_map = {
    'fc'          : torch_pack.transforms.pack_linear_config,
    'linear'      : torch_pack.transforms.pack_linear_config,
    'conv1d'      : torch_pack.transforms.pack_conv1d_config,
    'conv2d'      : torch_pack.transforms.pack_conv2d_config,
    'convT1d'         : torch_pack.transforms.pack_convtranspose1d_config,
    'convtranspose1d' : torch_pack.transforms.pack_convtranspose1d_config,
    
    'maxpool2d'   : torch_pack.pooling.pack_maxpool2d_config,
    'avgpool2d'   : torch_pack.pooling.pack_avgpool2d_config,
    
    'batchnorm2d' : torch_pack.norms.pack_batchnorm2d_config,

    'relu'        : torch_pack.activations.pack_relu_config,
    'leakyrelu'   : torch_pack.activations.pack_leakyrelu_config,
    'tanh'        : torch_pack.activations.pack_tanh_config,
    
    'flatten'     : torch_pack.utility.pack_flatten_config
}