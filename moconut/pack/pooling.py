from moconut.pack.utils import pack_args_in_order_with_defaults

################################################################
# Pooling
################################################################
def pack_maxpool2d_config(config : dict):
    arg_order = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']
    arg_defaults = {
        'stride'         : None, 
        'padding'        : 0, 
        'dilation'       : 1, 
        'return_indices' : False, 
        'ceil_mode'      : False
    }

    return pack_args_in_order_with_defaults(config, arg_order, arg_defaults)

def pack_avgpool2d_config(config : dict):
    arg_order = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override']
    arg_defaults = {
        'stride' : None, 
        'padding' : 0, 
        'ceil_mode' : False,
        'count_include_pad' : True, 
        'divisor_override' : None
    }

    return pack_args_in_order_with_defaults(config, arg_order, arg_defaults)