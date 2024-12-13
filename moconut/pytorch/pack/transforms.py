from moconut.utils import pack_args_in_order_with_defaults

################################################################
# Transformations
################################################################
def pack_linear_config(config : dict):
    arg_order = ['in_features', 'out_features', 'bias']
    arg_defaults = {'bias' : True}

    return pack_args_in_order_with_defaults(config, arg_order, arg_defaults)

def pack_conv1d_config(config : dict):
    arg_order = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']
    arg_defaults = {
        'stride'       : 1,
        'padding'      : 0, 
        'dilation'     : 1, 
        'groups'       : 1, 
        'bias'         : True, 
        'padding_mode' : 'zeros'
    }

    return pack_args_in_order_with_defaults(config, arg_order, arg_defaults)

def pack_conv2d_config(config : dict):
    arg_order = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']
    arg_defaults = {
        'stride'       : 1,
        'padding'      : 0, 
        'dilation'     : 1, 
        'groups'       : 1, 
        'bias'         : True, 
        'padding_mode' : 'zeros'
    }

    return pack_args_in_order_with_defaults(config, arg_order, arg_defaults)

def pack_convtranspose1d_config(config : dict):
    arg_order = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'groups', 'bias', 'dilation', 'padding_mode']
    arg_defaults = {
        'stride'         : 1,
        'padding'        : 0,
        'output_padding' : 0,
        'groups'         : 1,
        'bias'           : True,
        'dilation'       : 1,
        'padding_mode'   : 'zeros'
    }

    return pack_args_in_order_with_defaults(config, arg_order, arg_defaults)