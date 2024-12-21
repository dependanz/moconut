import torch
import moconut

################################################################
# keyword to torch.nn.Module
################################################################
module_map = {
    'fc'              : torch.nn.Linear,
    'linear'          : torch.nn.Linear,
    'conv1d'          : torch.nn.Conv1d,
    'conv2d'          : torch.nn.Conv2d,
    'convT1d'         : torch.nn.ConvTranspose1d,
    'convtranspose1d' : torch.nn.ConvTranspose1d,

    'maxpool2d'       : torch.nn.MaxPool2d,
    'avgpool2d'       : torch.nn.AvgPool2d,
    
    'batchnorm2d'     : torch.nn.BatchNorm2d,

    'relu'            : torch.nn.ReLU,
    'leakyrelu'       : torch.nn.LeakyReLU,
    'tanh'            : torch.nn.Tanh,

    'flatten'         : torch.nn.Flatten
}

################################################################
# String to argument packing for a torch.nn.Module
#
# TODO: Multimap
################################################################
pack_config_map = {
    'fc'              : moconut.pack.transforms.pack_linear_config,
    'linear'          : moconut.pack.transforms.pack_linear_config,
    'conv1d'          : moconut.pack.transforms.pack_conv1d_config,
    'conv2d'          : moconut.pack.transforms.pack_conv2d_config,
    'convT1d'         : moconut.pack.transforms.pack_convtranspose1d_config,
    'convtranspose1d' : moconut.pack.transforms.pack_convtranspose1d_config,
    
    'maxpool2d'       : moconut.pack.pooling.pack_maxpool2d_config,
    'avgpool2d'       : moconut.pack.pooling.pack_avgpool2d_config,
    
    'batchnorm2d'     : moconut.pack.norms.pack_batchnorm2d_config,

    'relu'            : moconut.pack.activations.pack_relu_config,
    'leakyrelu'       : moconut.pack.activations.pack_leakyrelu_config,
    'tanh'            : moconut.pack.activations.pack_tanh_config,
    
    'flatten'         : moconut.pack.utility.pack_flatten_config
}